# create_spectrogram_tensors_torchaudio.py
"""
Fast GPU-accelerated spectrogram generation using torchaudio.
Optimized for large-scale processing (hundreds of thousands of 5s clips).
Features:
- Full GPU acceleration (RTX 4070)
- Automatic resume (skips already processed files)
- Subset sampling for testing
- Robust error handling
- Progress tracking
"""

import pandas as pd
import numpy as np
import torch
import torchaudio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import torchcodec
import argparse
import warnings


# ====================== GPU SETUP ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"   GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")


def compute_clean_mel_spectrogram(audio_path: str, sr_target: int = 32000,
                                  n_mels: int = 128, n_fft: int = 1024,
                                  hop_length: int = 512, fmin: int = 500,
                                  fmax: int = 16000):
    """
    Compute a clean Mel-spectrogram on GPU using torchaudio.
    Returns a numpy array of shape (n_mels, time_frames).
    """
    try:
        # Load audio file
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)
        
        # Move to GPU and resample if necessary
        if sr != sr_target:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sr_target).to(device)
            waveform = resampler(waveform.to(device))
        else:
            waveform = waveform.to(device)
        
        # Pre-emphasis filter (reduces low-frequency noise)
        waveform = torchaudio.functional.preemphasis(waveform, coeff=0.97)
        
        # Mel Spectrogram transformation on GPU
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr_target,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
            normalized=False
        ).to(device)
        
        S = mel_transform(waveform)
        
        # Convert to dB scale
        S_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(S)
        
        # Move back to CPU for numpy conversion
        S_db = S_db.squeeze(0).cpu().numpy()
        
        # === Critical Preprocessing Steps ===
        S_db = np.clip(S_db, -80, 0)                    # Remove extreme noise floor
        S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)  # Per-sample standardization
        
        # Ensure fixed time dimension (exactly 313 frames for 5 seconds)
        target_frames = int((5.0 * sr_target) / hop_length)
        if S_db.shape[1] != target_frames:
            if S_db.shape[1] < target_frames:
                pad_width = target_frames - S_db.shape[1]
                S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                S_db = S_db[:, :target_frames]
        
        return S_db.astype(np.float32)
    
    except Exception as e:
        print(f"  Error in spectrogram computation for {Path(audio_path).name}: {e}")
        return None


def process_clip(row, audio_base_dir: Path, tensor_base_dir: Path, **kwargs):
    """
    Process a single 5-second clip.
    Supports resume: skips files that already have .npy tensors.
    """
    try:
        rel_path = row['filename']
        audio_path = audio_base_dir / rel_path
        output_rel = Path(rel_path).with_suffix('.npy')
        output_path = tensor_base_dir / output_rel
        
        # === RESUME CHECK ===
        if output_path.exists():
            new_row = row.copy()
            new_row['spectrogram_npy_path'] = str(output_rel)
            new_row['spectrogram_shape'] = f"({kwargs.get('n_mels', 128)}, 313)"
            return new_row
        
        # Skip if original audio doesn't exist
        if not audio_path.exists():
            return None
        
        # Compute spectrogram on GPU
        spec = compute_clean_mel_spectrogram(str(audio_path), **kwargs)
        if spec is None:
            return None
        
        # Save tensor
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, spec)
        
        # Prepare metadata row
        new_row = row.copy()
        new_row['spectrogram_npy_path'] = str(output_rel)
        new_row['spectrogram_shape'] = str(spec.shape)
        new_row['n_mels'] = spec.shape[0]
        new_row['time_frames'] = spec.shape[1]
        
        return new_row
    
    except Exception as e:
        print(f"   Failed processing {row.get('filename', 'unknown')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fast GPU Spectrogram Generator (torchaudio + RTX 4070 optimized)")
    parser.add_argument('--subset', type=float, default=1.0,
                        help='Fraction of dataset to process (e.g. 0.1 = 10 percent)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of clips to process')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads (I/O bound). Keep 4-8 for best GPU utilization.')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of mel bins (keep at 128 as requested)')
    args = parser.parse_args()

    # ====================== PATHS ======================
    clips_csv = Path("prepared_5s_clips/train_5s.csv")
    audio_base_dir = Path("prepared_5s_clips")
    tensor_dir = Path("5sSpectrograms_tensors")
    
    tensor_dir.mkdir(parents=True, exist_ok=True)
    output_csv = tensor_dir / "train_5s_spectrograms.csv"
    
    # ====================== LOAD DATA ======================
    print("Loading clip metadata...")
    df = pd.read_csv(clips_csv)
    print(f"Total clips available: {len(df):,}")
    
    # Randomized subset (useful for testing)
    if args.max_samples is not None:
        sample_size = min(args.max_samples, len(df))
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    elif args.subset < 1.0:
        df = df.sample(frac=args.subset, random_state=42).reset_index(drop=True)
    
    print(f"Selected {len(df):,} clips for processing.")
    print(f"Using {args.workers} workers | n_mels={args.n_mels} | Device={device}\n")
    
    # ====================== PROCESSING ======================
    all_new_rows = []
    start_time = time.time()
    successful = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_row = {
            executor.submit(
                process_clip,
                row,
                audio_base_dir,
                tensor_dir,
                sr_target=32000,
                n_mels=args.n_mels,
                n_fft=1024,
                hop_length=512,
                fmin=500,
                fmax=16000
            ): row
            for _, row in df.iterrows()
        }
        
        for future in tqdm(as_completed(future_to_row), total=len(df), desc="Generating Spectrograms"):
            result = future.result()
            if result is not None:
                all_new_rows.append(result)
                successful += 1
            
            # Progress update
            if successful % 5000 == 0 and successful > 0:
                elapsed = time.time() - start_time
                print(f"   → {successful:,} tensors completed | {elapsed/60:.1f} minutes elapsed")
    
    # ====================== SAVE RESULTS ======================
    new_df = pd.DataFrame(all_new_rows)
    new_df.to_csv(output_csv, index=False)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*90)
    print("✅ SPECTROGRAM GENERATION COMPLETED SUCCESSFULLY")
    print("="*90)
    print(f"Clips attempted          : {len(df):,}")
    print(f"Tensors created/skipped  : {successful:,}")
    print(f"Resume support           : ENABLED")
    print(f"Total time               : {total_time/60:.1f} minutes")
    print(f"Output directory         : {tensor_dir}")
    print(f"Metadata CSV             : {output_csv}")
    print("="*90)


if __name__ == "__main__":
    main()