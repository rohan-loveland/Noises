import os
import random
import librosa
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time


def get_audio_metadata(file_path: Path):
    """Process a single audio file: return duration, size, sr, class_name (if available)."""
    try:
        # File size
        size = file_path.stat().st_size

        # Load audio (keep original sr). Librosa is generally thread-safe for independent file reads.
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Get class from CSV mapping (filename -> class_name)
        filename = file_path.name
        class_name = filename_to_class.get(filename, "Unknown")

        return {
            'duration': duration,
            'size': size,
            'sr': sr,
            'class_name': class_name,
            'file_path': str(file_path),
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'file_path': str(file_path),
            'error': str(e)
        }


def analyze_audio_dataset(data_dir: str, max_workers: int = 8):
    """
    Analyze all .ogg files with multithreading: duration, class stats from train.csv.
    Outputs per-class averages, min/max, counts. Sorted by class name.
    Includes progress bar and periodic status updates.
    """
    global filename_to_class
    filename_to_class = {}

    # Load class mapping from train.csv (keyed by filename)
    csv_path = Path("Data/train.csv")
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'filename' in row and 'class_name' in row:
                    fname = row['filename'].split('/')[-1] if '/' in row['filename'] else row['filename']
                    filename_to_class[fname] = row['class_name']
        print(f"Loaded class mappings for {len(filename_to_class)} files from train.csv")
    else:
        print("Warning: train.csv not found. Class categorization will be limited.")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    audio_files = list(data_path.rglob("*.ogg"))  # recursive search
    if not audio_files:
        raise FileNotFoundError(f"No .ogg files found in {data_dir}")

    print(f"Found {len(audio_files)} audio files. Starting multithreaded analysis with {max_workers} workers...\n")

    # Multithreaded processing with progress and status updates
    durations = []
    file_sizes = []
    sample_rates = []
    class_stats = defaultdict(lambda: {'durations': [], 'count': 0})
    errors = []
    processed_count = 0
    last_update_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(get_audio_metadata, f): f for f in audio_files}

        # Use tqdm for live progress bar
        for future in tqdm(as_completed(future_to_file), total=len(audio_files), desc="Processing audio files"):
            result = future.result()
            processed_count += 1

            if result['success']:
                durations.append(result['duration'])
                file_sizes.append(result['size'])
                sample_rates.append(result['sr'])

                cls = result['class_name']
                class_stats[cls]['durations'].append(result['duration'])
                class_stats[cls]['count'] += 1
            else:
                errors.append(result)

            # Occasional status update to show it's not stalled (every ~5s or 500 files)
            current_time = time.time()
            if (processed_count % 500 == 0) or (current_time - last_update_time > 5):
                print(f"  Status: Processed {processed_count}/{len(audio_files)} files. "
                      f"Current avg duration: {np.mean(durations) if durations else 0:.2f}s")
                last_update_time = current_time

    print(f"\nCompleted processing. Errors: {len(errors)}")

    if not durations:
        print("No audio files processed successfully.")
        return

    # Convert to numpy for stats
    durations = np.array(durations)
    file_sizes_mb = np.array(file_sizes) / (1024 * 1024)

    # ============== Overall Statistics ==============
    print("\n=== OVERALL AUDIO STATISTICS ===")
    print(f"Total files analyzed : {len(durations)}")
    print(f"Total duration       : {durations.sum()/3600:.2f} hours")
    print(f"Average duration     : {durations.mean():.3f} seconds")
    print(f"Min duration         : {durations.min():.3f} seconds")
    print(f"Max duration         : {durations.max():.3f} seconds")
    print(f"Median duration      : {np.median(durations):.3f} seconds")
    print(f"Std dev duration     : {durations.std():.3f} seconds\n")

    print("=== FILE SIZE STATISTICS ===")
    print(f"Average size         : {file_sizes_mb.mean():.2f} MB")
    print(f"Min size             : {file_sizes_mb.min():.3f} MB")
    print(f"Max size             : {file_sizes_mb.max():.2f} MB")
    print(f"Total dataset size   : {file_sizes_mb.sum():.2f} MB\n")

    print("=== SAMPLE RATE STATS ===")
    unique_srs = np.unique(sample_rates)
    print(f"Sample rates found   : {unique_srs}")
    sr_counts = Counter(sample_rates)
    for sr in sorted(unique_srs):
        print(f"  {sr} Hz : {sr_counts[sr]} files")

    # ============== Per-Class Statistics (sorted by class name) ==============
    print("\n=== PER-CLASS STATISTICS (sorted by class name) ===")
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        cls_durs = np.array(stats['durations'])
        print(f"\nClass: {cls} ({stats['count']} files)")
        print(f"  Avg duration : {cls_durs.mean():.3f} seconds")
        print(f"  Min duration : {cls_durs.min():.3f} seconds")
        print(f"  Max duration : {cls_durs.max():.3f} seconds")
        print(f"  Total duration: {cls_durs.sum()/3600:.2f} hours")

    if errors:
        print(f"\nNote: {len(errors)} files had errors (see console for details).")

    # ============== Visualizations ==============
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.hist(durations, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Audio Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of files')

    plt.subplot(2, 2, 2)
    plt.hist(file_sizes_mb, bins=50, color='lightgreen', edgecolor='black')
    plt.title('Distribution of File Sizes')
    plt.xlabel('Size (MB)')
    plt.ylabel('Number of files')

    plt.subplot(2, 2, 3)
    plt.boxplot([durations], labels=['Duration'])
    plt.title('Duration Box Plot')
    plt.ylabel('Seconds')

    plt.tight_layout()
    plt.show()

    # Optional: save summary to file
    summary = {
        "total_files": len(durations),
        "total_hours": float(durations.sum()/3600),
        "avg_duration": float(durations.mean()),
        "min_duration": float(durations.min()),
        "max_duration": float(durations.max()),
        "avg_size_mb": float(file_sizes_mb.mean()),
        "total_size_mb": float(file_sizes_mb.sum()),
        "per_class": {
            cls: {
                "count": stats['count'],
                "avg_duration": float(np.array(stats['durations']).mean()),
                "min_duration": float(np.array(stats['durations']).min()),
                "max_duration": float(np.array(stats['durations']).max())
            } for cls, stats in class_stats.items()
        }
    }

    import json
    with open("birdclef_audio_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to birdclef_audio_stats.json")

    # Convert to numpy for easy stats
    durations = np.array(durations)
    file_sizes_mb = np.array(file_sizes) / (1024 * 1024)

    # ============== Statistics ==============
    print("=== AUDIO STATISTICS ===")
    print(f"Total files analyzed : {len(durations)}")
    print(f"Total duration       : {durations.sum()/3600:.2f} hours")
    print(f"Average duration     : {durations.mean():.3f} seconds")
    print(f"Min duration         : {durations.min():.3f} seconds")
    print(f"Max duration         : {durations.max():.3f} seconds")
    print(f"Median duration      : {np.median(durations):.3f} seconds")
    print(f"Std dev duration     : {durations.std():.3f} seconds\n")

    print("=== FILE SIZE STATISTICS ===")
    print(f"Average size         : {file_sizes_mb.mean():.2f} MB")
    print(f"Min size             : {file_sizes_mb.min():.3f} MB")
    print(f"Max size             : {file_sizes_mb.max():.2f} MB")
    print(f"Total dataset size   : {file_sizes_mb.sum():.2f} MB\n")

    print("=== SAMPLE RATE STATS ===")
    unique_srs = np.unique(sample_rates)
    print(f"Sample rates found   : {unique_srs}")
    for sr in unique_srs:
        count = sum(1 for s in sample_rates if s == sr)
        print(f"  {sr} Hz : {count} files")

    print("\n=== TOP SPECIES (by folder) ===")
    for species, count in sorted(species_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {species}: {count} recordings")

    # ============== Visualizations ==============
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.hist(durations, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Audio Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of files')

    plt.subplot(2, 2, 2)
    plt.hist(file_sizes_mb, bins=50, color='lightgreen', edgecolor='black')
    plt.title('Distribution of File Sizes')
    plt.xlabel('Size (MB)')
    plt.ylabel('Number of files')

    plt.subplot(2, 2, 3)
    plt.boxplot([durations], labels=['Duration'])
    plt.title('Duration Box Plot')
    plt.ylabel('Seconds')

    plt.tight_layout()
    plt.show()

    # Optional: save summary to file
    summary = {
        "total_files": len(durations),
        "total_hours": float(durations.sum()/3600),
        "avg_duration": float(durations.mean()),
        "min_duration": float(durations.min()),
        "max_duration": float(durations.max()),
        "avg_size_mb": float(file_sizes_mb.mean()),
        "total_size_mb": float(file_sizes_mb.sum())
    }
    
    import json
    with open("birdclef_audio_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to birdclef_audio_stats.json")


# ============== Usage ==============
if __name__ == "__main__":
    DATA_DIR = "Data/train_audio"   # Updated to match actual directory structure
    # Multithreading is safe here because each thread processes independent files.
    # Librosa.load() with sr=None is I/O and CPU bound but doesn't share mutable state.
    # ThreadPoolExecutor with 4-8 workers provides good speedup on typical hardware.
    # Progress bar + periodic status prints ensure visibility that processing is active.
    analyze_audio_dataset(DATA_DIR, max_workers=6)  # Adjust workers based on your CPU (6 is balanced)