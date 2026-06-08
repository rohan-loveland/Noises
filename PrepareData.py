# PrepareData.py
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import soundfile as sf
import argparse


def process_file(row, audio_base_dir: Path, output_base_dir: Path, 
                 chunk_duration: float = 5.0, overlap: float = 0.5,
                 save_audio: bool = True, sr_target: int = 32000):
    """
    Split one audio file into 5-second chunks with optional overlap.
    """
    try:
        rel_path = row['filename']
        full_audio_path = audio_base_dir / rel_path
        
        if not full_audio_path.exists():
            return []

        # Load audio
        y, sr = librosa.load(full_audio_path, sr=None, mono=True)
        
        # Resample if necessary
        if sr != sr_target:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        duration = librosa.get_duration(y=y, sr=sr)
        if duration < chunk_duration * 0.5:   # Skip very short files
            return []

        chunk_samples = int(chunk_duration * sr)
        hop_samples = int(chunk_duration * (1 - overlap) * sr)
        
        new_rows = []
        start = 0
        chunk_idx = 0

        while start + chunk_samples <= len(y):
            end = start + chunk_samples
            chunk = y[start:end]
            
            # Create chunk filename
            stem = Path(rel_path).stem
            parent = Path(rel_path).parent
            chunk_filename = f"{stem}_chunk{chunk_idx:03d}.ogg"
            chunk_rel_path = str(parent / chunk_filename)
            
            # New metadata row
            new_row = row.copy()
            new_row['filename'] = chunk_rel_path
            new_row['original_filename'] = rel_path
            new_row['chunk_idx'] = chunk_idx
            new_row['chunk_start_sec'] = float(start / sr)
            new_row['original_duration'] = float(duration)
            new_row['chunk_duration'] = chunk_duration
            new_row['overlap'] = overlap
            new_row['hop_duration'] = chunk_duration * (1 - overlap)
            
            new_rows.append(new_row)
            
            # Save audio chunk
            if save_audio:
                chunk_path = output_base_dir / chunk_rel_path
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(chunk_path, chunk, sr, format='OGG')
            
            start += hop_samples
            chunk_idx += 1
        
        return new_rows

    except Exception as e:
        print(f"Error processing {row.get('filename', 'unknown')}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Prepare overlapping 5-second audio chunks")
    parser.add_argument('--subset', type=float, default=1.0,
                        help='Fraction of data to process (0.0-1.0)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of original files to process')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of worker threads')
    parser.add_argument('--no_save_audio', action='store_true',
                        help='Only create metadata, do not save .ogg files')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap fraction (0.5 = 50 percent overlap)')
    args = parser.parse_args()

    # ========================= CONFIGURATION =========================
    csv_path = Path("Data/train.csv")
    audio_base_dir = Path(r"C:\PHXResearch\Noises\Data\train_audio")
    output_dir = Path("prepared_5s_clips")
    
    chunk_duration = 5.0
    sr_target = 32000
    save_audio_chunks = not args.no_save_audio
    max_workers = args.workers
    overlap = args.overlap

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "train_5s.csv"

    # Load data
    print("Loading train.csv...")
    df = pd.read_csv(csv_path)
    print(f"Original recordings: {len(df):,}")

    # Subset sampling
    if args.max_samples is not None:
        sample_size = min(args.max_samples, len(df))
    else:
        sample_size = int(len(df) * args.subset)
    
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Randomly selected {len(df):,} files ({args.subset*100:.1f}%)")

    print(f"Chunk duration : {chunk_duration}s")
    print(f"Overlap        : {overlap*100:.0f}% → hop = {chunk_duration*(1-overlap):.1f}s")
    print(f"Save audio     : {save_audio_chunks}")
    print(f"Workers        : {max_workers}\n")

    all_new_rows = []
    start_time = time.time()
    processed = 0
    total_chunks = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(
                process_file, 
                row, 
                audio_base_dir, 
                output_dir,
                chunk_duration,
                overlap,
                save_audio_chunks,
                sr_target
            ): row 
            for _, row in df.iterrows()
        }
        
        for future in tqdm(as_completed(future_to_row), total=len(df), desc="Creating overlapping chunks"):
            new_rows = future.result()
            all_new_rows.extend(new_rows)
            total_chunks += len(new_rows)
            processed += 1
            
            if processed % 400 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {processed}/{len(df)} files | "
                      f"Chunks created: {total_chunks:,}")

    # Save final CSV
    new_df = pd.DataFrame(all_new_rows)
    new_df.to_csv(output_csv, index=False)

    total_time = time.time() - start_time
    expansion = len(new_df) / len(df) if len(df) > 0 else 0

    print("\n" + "="*75)
    print("5-SECOND OVERLAPPING CHUNK PREPARATION COMPLETE!")
    print("="*75)
    print(f"Original files     : {len(df):,}")
    print(f"Total 5s chunks    : {len(new_df):,}")
    print(f"Expansion factor   : {expansion:.2f}x")
    print(f"Output CSV         : {output_csv}")
    print(f"Audio saved to     : {output_dir}")
    print(f"Total time         : {total_time/60:.1f} minutes")
    print(f"Resume support     : Built-in (skips existing files)")
    print("="*75)


if __name__ == "__main__":
    main()