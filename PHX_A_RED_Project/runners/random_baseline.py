"""
Random baseline for class discovery (pure random labeling, no ARED).

This runner does *not* use ARED at all.
It loads *only* the label column from the CSV (lightweight, no embeddings or spectrograms),
applies the same shuffle+max_samples logic as the data streams (for reproducibility),
then simulates drawing points at random (without replacement) and records the exact
query count at which each new class is first hit.

This produces clean "queries to discover each class" data for comparison against ARED.

The random baseline deliberately does **not** go through PerchDataStream / Dino etc.
It just needs the labels.

Usage examples:
    python -m PHX_A_RED_Project.runners.random_baseline --num-points 500 --label-column class_name --seed 42
    python -m PHX_A_RED_Project.runners.random_baseline --num-points -1 --label-column scientific_name --seed 123
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from ..utils.paths import resolve_project_paths


def _load_label_pool(label_column: str, max_samples: Optional[int], shuffle: bool, seed: int) -> List[str]:
    """
    Lightweight loader: only reads the label column from the CSV.
    Applies the same shuffle + head(max_samples) logic the DataStreams use,
    so that a random baseline run with the same seed + max_samples sees the
    same *set* of labels (in the same arrival order) that an ARED run would have.
    No embeddings or spectrogram tensors are loaded.
    """
    _, csv_path, _, _ = resolve_project_paths()
    df = pd.read_csv(csv_path)

    # Robust label column resolution (same spirit as _resolve_label)
    col = label_column
    if col not in df.columns:
        for candidate in (label_column, "scientific_name", "common_name", "class_name", "primary_label"):
            if candidate in df.columns:
                col = candidate
                break
    if col not in df.columns:
        # last resort
        col = df.columns[0]

    # Work on a minimal frame to mimic stream sampling
    tmp = pd.DataFrame({"_lab": df[col].fillna("unknown").astype(str)})

    if shuffle:
        tmp = tmp.sample(frac=1, random_state=seed).reset_index(drop=True)
    if max_samples is not None and max_samples > 0:
        tmp = tmp.head(max_samples).reset_index(drop=True)

    labels = tmp["_lab"].tolist()
    # Clean obvious junk
    labels = [l for l in labels if l and l.lower() != "nan"]
    return labels


def run_random_baseline(
    num_points: int = 500,
    label_column: str = "class_name",
    shuffle: bool = True,
    seed: int = 42,
    save: bool = True,
    results_dir: str = "results",
    **kwargs,   # accepted for CLI compatibility (e.g. old --frontend), but ignored
) -> dict:
    """
    Pure random labeling baseline.

    Loads only labels (very fast, no heavy feature data), applies the same
    deterministic shuffle + max_samples as the real streams, then simulates
    drawing at random without replacement.

    Returns (and optionally saves) a record with:
      - discoveries: {class: query_ordinal when first drawn}
      - num_classes_in_pool
      - num_classes_found
    """
    max_s = num_points if (num_points is not None and num_points > 0) else None

    labels = _load_label_pool(label_column, max_s, shuffle, seed)
    n = len(labels)
    if n == 0:
        raise RuntimeError("No labels found — nothing to sample from.")

    # To simulate "random selection" (as opposed to just arrival order),
    # we draw a fresh random permutation of the (already shuffled+truncated) pool.
    # Using the provided seed keeps runs reproducible.
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)

    seen = set()
    discoveries = {}
    query_count = 0

    for pos in order:
        lab = labels[pos]
        query_count += 1
        if lab not in seen:
            seen.add(lab)
            discoveries[lab] = query_count
            if len(seen) == len(set(labels)):
                break

    pool_set = set(labels)
    found_set = set(discoveries.keys())
    missed = sorted(pool_set - found_set)

    record = {
        "method": "random",
        "label_column": label_column,
        "N": n,
        "queries": max(discoveries.values()) if discoveries else 0,
        "seed": seed,
        "shuffle": shuffle,
        "discoveries": discoveries,
        "queries_to_find_all_present_classes": max(discoveries.values()) if discoveries else 0,
        "num_classes_in_pool": len(pool_set),
        "num_classes_found": len(discoveries),
        "num_classes_missed": len(missed),
        "missed_classes": missed,
    }

    if save:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"random_{label_column}_N{n}_seed{seed}.json"
        fpath = out_dir / fname
        with open(fpath, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[random] Saved discovery record -> {fpath}")

    # Summary
    print("\nRandom baseline discovery (queries to first hit each class):")
    for lab, q in sorted(discoveries.items(), key=lambda kv: kv[1]):
        print(f"  {lab}: query {q}")
    print(f"\nTotal random queries to discover all {len(discoveries)} classes present in the {n}-point pool: {record['queries_to_find_all_present_classes']}")

    return record


def main():
    parser = argparse.ArgumentParser(description="Pure random class discovery baseline (no ARED).")
    parser.add_argument("--num-points", type=int, default=500, help="Size of the pool to sample from (-1 for full dataset)")
    parser.add_argument("--label-column", type=str, default="class_name")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true", help="Do not write JSON result file")
    parser.add_argument("--results-dir", type=str, default="results")
    # accepted for backward compat with old scripts / run_test, but ignored
    parser.add_argument("--frontend", "-f", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--ensure-min-class-presence", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    shuffle = not args.no_shuffle

    run_random_baseline(
        num_points=args.num_points,
        label_column=args.label_column,
        shuffle=shuffle,
        seed=args.seed,
        save=not args.no_save,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()