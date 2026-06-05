# Create_Spectrogram_Tensors.py
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import argparse


def compute_clean_mel_spectrogram(audio_path: Path, sr_target: int = 32000,
                                  n_mels: int = 128, n_fft: int = 2048,
                                  hop_length: int = 512, fmin: int = 500,
                                  fmax: int = 16000):
    """Compute clean, standardized Mel-spectrogram tensor."""
    try:
        y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
        
        # Pre-emphasis to reduce low-frequency noise
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # Mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            power=2.0
        )
        
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = np.clip(S_db, -80, 0)                    # Noise floor clipping
        S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)  # Standardization
        
        # Ensure consistent time dimension (~313 frames for 5s)
        target_frames = int((5.0 * sr) / hop_length)
        if S_db.shape[1] != target_frames:
            S_db = librosa.util.fix_length(S_db, size=target_frames, axis=1)
        
        return S_db.astype(np.float32)
    
    except Exception as e:
        print(f"Error computing spectrogram for {audio_path.name}: {e}")
        return None


def process_clip(row, audio_base_dir: Path, tensor_base_dir: Path, **kwargs):
    """Process one chunk and save its spectrogram tensor."""
    try:
        rel_path = row['filename']
        audio_path = audio_base_dir / rel_path
        output_rel = Path(rel_path).with_suffix('.npy')
        output_path = tensor_base_dir / output_rel
        
        # Resume support: skip if tensor already exists
        if output_path.exists():
            new_row = row.copy()
            new_row['spectrogram_npy_path'] = str(output_rel)
            new_row['spectrogram_shape'] = f"({kwargs.get('n_mels', 128)}, 313)"
            new_row['status'] = 'skipped_existing'
            return new_row
        
        if not audio_path.exists():
            return None
        
        spec = compute_clean_mel_spectrogram(audio_path, **kwargs)
        if spec is None:
            return None
        
        # Save tensor
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, spec)
        
        # Metadata
        new_row = row.copy()
        new_row['spectrogram_npy_path'] = str(output_rel)
        new_row['spectrogram_shape'] = str(spec.shape)
        new_row['n_mels'] = spec.shape[0]
        new_row['time_frames'] = spec.shape[1]
        new_row['status'] = 'new'
        
        return new_row
    
    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create spectrogram tensors from 5s chunks")
    parser.add_argument('--subset', type=float, default=1.0,
                        help='Fraction of chunks to process')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--n_mels', type=int, default=128)
    args = parser.parse_args()

    # ========================= CONFIGURATION =========================
    clips_csv = Path("prepared_5s_clips/train_5s.csv")
    audio_base_dir = Path("prepared_5s_clips")
    tensor_dir = Path("5sSpectrograms_tensors")
    
    sr_target = 32000
    n_mels = args.n_mels
    n_fft = 2048
    hop_length = 512
    fmin = 500
    fmax = 16000
    max_workers = args.workers

    tensor_dir.mkdir(parents=True, exist_ok=True)
    output_csv = tensor_dir / "train_5s_spectrograms.csv"

    # Load chunk metadata
    print("Loading 5s chunks CSV...")
    df = pd.read_csv(clips_csv)
    print(f"Total available 5s chunks: {len(df):,}")

    # Subset sampling
    if args.max_samples is not None:
        sample_size = min(args.max_samples, len(df))
    else:
        sample_size = int(len(df) * args.subset)
    
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Randomly selected {len(df):,} chunks ({args.subset*100:.1f}%)")

    print(f"n_mels         : {n_mels}")
    print(f"Expected shape : ({n_mels}, ~313)")
    print(f"Workers        : {max_workers}\n")

    all_new_rows = []
    start_time = time.time()
    successful = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(
                process_clip, 
                row, 
                audio_base_dir, 
                tensor_dir,
                sr_target=sr_target,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax
            ): row 
            for _, row in df.iterrows()
        }
        
        for future in tqdm(as_completed(future_to_row), total=len(df), desc="Creating spectrogram tensors"):
            result = future.result()
            if result is not None:
                all_new_rows.append(result)
                if result.get('status') == 'new':
                    successful += 1
                else:
                    skipped += 1
            
            if (successful + skipped) % 2000 == 0 and (successful + skipped) > 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {successful+skipped:,} processed | New: {successful:,} | Skipped: {skipped:,}")

    # Save final metadata
    new_df = pd.DataFrame(all_new_rows)
    new_df.to_csv(output_csv, index=False)

    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("SPECTROGRAM TENSOR CREATION COMPLETE!")
    print("="*80)
    print(f"Chunks processed     : {len(df):,}")
    print(f"New tensors created  : {successful:,}")
    print(f"Skipped (existing)   : {skipped:,}")
    print(f"Output CSV           : {output_csv}")
    print(f"Tensor folder        : {tensor_dir}")
    print(f"Total time           : {total_time/60:.1f} minutes")
    print(f"Resume support       : ENABLED")
    print("="*80)


if __name__ == "__main__":
    main()