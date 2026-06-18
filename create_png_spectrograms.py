# create_png_spectrograms.py
"""
Stable single-threaded Librosa PNG generator (matplotlib safe).
"""

import pandas as pd
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time
import argparse
import warnings
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
plt.rcParams['figure.max_open_warning'] = 0

# ====================== SPECTROGRAM ======================
def compute_clean_mel_spectrogram(audio_path: str, sr_target: int = 32000,
                                  n_mels: int = 128, hop_length: int = 512,
                                  fmin: int = 500, fmax: int = 16000):
    try:
        y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
        
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
        )
        
        S_db = librosa.power_to_db(S, ref=1.0, top_db=80)
        S_db = np.clip(S_db, -80, 0)
        
        # Fixed normalization for ML consistency
        S_db = (S_db + 40.0) / 13.333
        S_db = np.clip(S_db, -3.0, 3.0)
        
        # Force exact shape
        target_frames = int((5.0 * sr_target) / hop_length)
        if S_db.shape != (n_mels, target_frames):
            new_spec = np.full((n_mels, target_frames), -3.0, dtype=np.float32)
            h = min(n_mels, S_db.shape[0])
            w = min(target_frames, S_db.shape[1])
            new_spec[:h, :w] = S_db[:h, :w]
            S_db = new_spec
        
        return S_db.astype(np.float32)
    
    except Exception as e:
        print(f"  Error computing {Path(audio_path).name}: {e}")
        return None


# ====================== SAVE PNG ======================
def save_raw_png(spec: np.ndarray, output_path: Path, dpi: int = 150):
    try:
        if spec.shape != (128, 313):
            new_spec = np.full((128, 313), -3.0, dtype=np.float32)
            h, w = min(128, spec.shape[0]), min(313, spec.shape[1])
            new_spec[:h, :w] = spec[:h, :w]
            spec = new_spec
        
        fig = plt.figure(figsize=(3.13, 1.28), dpi=dpi, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ax.imshow(spec, aspect='auto', cmap='viridis', origin='lower',
                  vmin=-3.0, vmax=3.0, interpolation='nearest')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi, format='png')
        plt.close('all')
        return True
    except Exception as e:
        print(f"  Save error {output_path.name}: {e}")
        plt.close('all')
        return False


# ====================== PROCESS CLIP ======================
def process_clip(row, audio_base_dir: Path, png_base_dir: Path, **kwargs):
    try:
        rel_path = row['filename']
        audio_path = audio_base_dir / rel_path
        output_rel = Path(rel_path).with_suffix('.png')
        output_path = png_base_dir / output_rel

        if output_path.exists():
            return row.copy()  # Already done

        if not audio_path.exists():
            return None

        spec = compute_clean_mel_spectrogram(str(audio_path), **{k: v for k, v in kwargs.items() if k != 'dpi'})
        if spec is None:
            return None

        if save_raw_png(spec, output_path, kwargs.get('dpi', 150)):
            new_row = row.copy()
            new_row['png_path'] = str(output_rel)
            new_row['png_shape'] = "(128, 313)"
            new_row['n_mels'] = 128
            new_row['time_frames'] = 313
            return new_row
        return None

    except Exception as e:
        print(f"   Failed {row.get('filename', 'unknown')}: {e}")
        return None


# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=float, default=1.0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    clips_csv = Path("prepared_5s_clips/train_5s.csv")
    audio_base_dir = Path("prepared_5s_clips")
    png_dir = Path("pngSpectrograms")
    png_dir.mkdir(parents=True, exist_ok=True)
    output_csv = png_dir / "train_5s_with_png.csv"

    print("Loading metadata...")
    df = pd.read_csv(clips_csv)
    print(f"Total clips: {len(df):,}")

    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42).reset_index(drop=True)
    elif args.subset < 1.0:
        df = df.sample(frac=args.subset, random_state=42).reset_index(drop=True)

    print(f"Processing {len(df):,} clips (single-threaded for stability)...")

    all_new_rows = []
    start_time = time.time()
    successful = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating PNGs"):
        result = process_clip(row, audio_base_dir, png_dir, dpi=args.dpi)
        if result is not None:
            all_new_rows.append(result)
            successful += 1

        if successful % 1000 == 0 and successful > 0:
            elapsed = (time.time() - start_time) / 60
            print(f"   → {successful:,} PNGs completed | {elapsed:.1f} min")

    new_df = pd.DataFrame(all_new_rows)
    new_df.to_csv(output_csv, index=False)

    print("\n✅ Completed!")
    print(f"Successful: {successful:,} / {len(df):,}")
    print(f"PNG folder: {png_dir}")


if __name__ == "__main__":
    main()