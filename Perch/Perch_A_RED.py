"""
Fast Perch v2 Embedding Extraction (single-process + threaded audio I/O)

Key improvements for 8 GB VRAM laptops + 443k clip scale:
- Uses Perch/perch_embedder.py (single model load, no multiprocessing of TF).
- Threaded audio loading (CPU) overlapped with GPU inference.
- Resume support: re-run the exact command after Ctrl-C / reboot and it continues.
- Tunable batch size (default 4, drop to 2 or 1 if you see OOM on 4070 laptop).
- Same robust Linux paths + model auto-detection as before.

Usage examples:
    python Perch/Perch_A_RED.py --max-samples 256 --batch-size 4
    python Perch/Perch_A_RED.py --batch-size 4 --io-threads 6          # full run (resumable)
    python Perch/Perch_A_RED.py --cpu-only --max-samples 32

After extraction you get:
    5sSpectrograms_tensors/perch_embeddings.npy   (rows align with the linux CSV order; D = detected embedding dim)
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# The reusable engine (single TF session, safe for 8 GB)
from perch_embedder import PerchEmbedder
import tensorflow as tf  # only for tf.stack in the overlapped batch loop (embedder owns the model)

# ====================== CONFIG ======================
BATCH_SIZE = 4                    # Smaller per process
NUM_WORKERS = 4                    # Good for 16 cores (adjust between 6-12)

# Default relative (will be overridden by robust resolution below for Linux/Windows/CWD safety)
CSV_PATH = "5sSpectrograms_tensors/train_5s_spectrograms_linux.csv"
AUDIO_DIR = "prepared_5s_clips"
OUTPUT_PATH = "5sSpectrograms_tensors/perch_embeddings.npy"

# === Perch v2 model selection ===
# The GPU version you want (tarball):
#   curl -L -o ~/Downloads/model.tar.gz \
#     https://www.kaggle.com/api/v1/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2/download
#
# After extracting, set LOCAL_MODEL_PATH to the directory containing saved_model.pb.
#
# We now have two URLs because the GPU variant requires a CUDA-enabled TensorFlow.
GPU_MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"

# Known good CPU-only variant (much more likely to work if your env only sees CPU).
CPU_MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/perch_v2_cpu/versions/1"

LOCAL_MODEL_PATH = None   # Explicit override, e.g. "/home/wes/Downloads/perch_v2_gpu" 

# Default we try (will be overridden by smart selection below based on visible GPUs)
MODEL_URL = GPU_MODEL_URL

# Limit for smoke-testing the full pipeline (set to small number like 16 or 64, or None for all)
MAX_TEST_SAMPLES = None

# Will be auto-detected from a successful model load (see main()).
# Used only for the error fallback path when a batch completely fails.
PERCH_EMBEDDING_DIM = 1536  # fallback; actual value comes from embedder at runtime (may be 1536 or 1280 depending on saved model variant)

# (load_audio_waveform + batch inference now live in Perch/perch_embedder.py)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Perch v2 embeddings (single-process, resumable, 8 GB VRAM friendly)."
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Process only the first N rows (useful for testing / smoke)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Inference batch size (default 4; try 2 or 1 on 8 GB laptop)")
    parser.add_argument("--io-threads", type=int, default=4,
                        help="Number of threads for parallel audio loading (CPU overlap)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU model even if a GPU is visible")
    args = parser.parse_args()

    # Apply overrides to module globals
    global BATCH_SIZE, NUM_WORKERS, MAX_TEST_SAMPLES
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.max_samples is not None:
        MAX_TEST_SAMPLES = args.max_samples   # reuse the old constant for limit

    io_threads = max(1, args.io_threads)
    test_limit = args.max_samples or MAX_TEST_SAMPLES

    print("=== Fast Perch Embedding Extraction (single-process + threaded I/O) ===")

    # Robust path resolution
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    csv_path = project_root / CSV_PATH
    if not csv_path.exists():
        fallback = project_root / "5sSpectrograms_tensors" / "train_5s_spectrograms.csv"
        if fallback.exists():
            print(f"WARNING: {csv_path.name} not found, falling back to {fallback.name}")
            csv_path = fallback
        else:
            print(f"ERROR: Could not find CSV at {csv_path} or fallback.")
            return

    audio_dir = project_root / AUDIO_DIR
    output_path = project_root / OUTPUT_PATH

    print(f"CSV: {csv_path}")
    print(f"Audio dir: {audio_dir}")
    print(f"Output: {output_path}")
    print(f"Batch size: {BATCH_SIZE} | IO threads: {io_threads} | CPU-only: {args.cpu_only}")

    # === Create the single embedder (this does the old "early load test" + dim detection) ===
    try:
        embedder = PerchEmbedder(
            batch_size=BATCH_SIZE,
            force_cpu=args.cpu_only,
        )
        emb_dim = embedder.get_embedding_dim()
        global PERCH_EMBEDDING_DIM
        PERCH_EMBEDDING_DIM = emb_dim
    except Exception as e:
        print(f"[Perch] Failed to create embedder: {e}")
        return

    # Load CSV
    df = pd.read_csv(csv_path)

    # === Resume logic (only for full runs; --max-samples always starts fresh for safety) ===
    start_row = 0
    if test_limit is None and output_path.exists():
        try:
            existing = np.load(str(output_path), mmap_mode="r")
            start_row = existing.shape[0]
            print(f"[Perch] Found existing embeddings with {start_row:,} rows — resuming.")
        except Exception as e:
            print(f"[Perch] Could not read existing {output_path} for resume: {e}")
            start_row = 0

    if start_row > 0:
        df = df.iloc[start_row:].reset_index(drop=True)
        print(f"[Perch] {len(df):,} clips remaining after resume offset.")

    if test_limit:
        # For test runs we intentionally ignore any prior full embeddings and slice from current df
        df = df.head(test_limit).reset_index(drop=True)
        print(f"Limited to first {len(df):,} clips for this run (test mode — resume disabled)")

    if len(df) == 0:
        print("[Perch] Nothing to do (already complete or limit reached).")
        return

    print(f"Total clips to process this run: {len(df):,}")

    # Build full paths (in stable order)
    audio_paths = []
    for _, row in df.iterrows():
        p = audio_dir / str(row.get("filename", ""))
        audio_paths.append(str(p))

    start_time = time.time()
    new_embeddings_list = []

    # Threaded audio load (CPU) overlapped with inference (GPU via embedder)
    # We load one batch-worth of audio in parallel, then infer, repeat.
    # This is simple, correct, and gives excellent overlap without complex queues.
    with ThreadPoolExecutor(max_workers=io_threads) as io_exec:
        for b_start in tqdm(range(0, len(audio_paths), BATCH_SIZE), desc="Embedding batches"):
            batch_paths = audio_paths[b_start : b_start + BATCH_SIZE]

            # Parallel load raw waveforms for this batch (returns tf.Tensor from embedder)
            fut_to_local = {
                io_exec.submit(embedder.load_audio_waveform, p): i
                for i, p in enumerate(batch_paths)
            }
            batch_audio = [None] * len(batch_paths)
            for fut in as_completed(fut_to_local):
                local_idx = fut_to_local[fut]
                batch_audio[local_idx] = fut.result()

            batch_input = tf.stack(batch_audio)
            batch_emb = embedder.embed_batch(batch_input)
            new_embeddings_list.append(batch_emb)

            # Light periodic status (helps on very long runs)
            if (b_start // BATCH_SIZE) % 50 == 0 and b_start > 0:
                elapsed = time.time() - start_time
                done = b_start + len(batch_paths)
                rate = done / max(elapsed, 1e-6)
                remaining = len(audio_paths) - done
                eta_min = (remaining / rate) / 60.0 if rate > 0 else 0
                print(f"  progress: {done:,}/{len(audio_paths):,} | ~{rate:.1f} clips/s | ETA ~{eta_min:.1f} min")

    # Combine only the newly computed embeddings
    if not new_embeddings_list:
        print("[Perch] No new embeddings produced.")
        return

    new_embs = np.vstack(new_embeddings_list)

    # z-score normalize *only the new block* (old block was already normalized when written)
    new_embs = (new_embs - new_embs.mean(axis=1, keepdims=True)) / \
               (new_embs.std(axis=1, keepdims=True) + 1e-8)

    # Final assembly + save
    if start_row > 0 and output_path.exists():
        old = np.load(str(output_path), mmap_mode="r")
        final = np.vstack([old, new_embs])
    else:
        final = new_embs

    np.save(str(output_path), final)

    total_time = (time.time() - start_time) / 60.0
    print(f"\n✅ Perch embeddings saved to: {output_path}")
    print(f"Shape: {final.shape}")
    print(f"New rows this run: {len(new_embs):,}")
    print(f"Total time (this run): {total_time:.1f} minutes")
    if len(audio_paths) > 0:
        print(f"Average rate: {len(audio_paths) / max((time.time() - start_time), 1e-6):.1f} clips/sec")


if __name__ == "__main__":
    main()