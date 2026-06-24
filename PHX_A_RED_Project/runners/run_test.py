"""
Run Test program for automated ARED (perch) vs pure Random baseline comparisons.

Purpose:
- Run ARED (via the perch runner) on the full (or capped) dataset
- Then run a *pure random label* baseline (no feature loaders at all)
- Use identical seeds
- Support --label-column scientific_name (species) or class_name
- Save per-class discovery records for both (with num_classes_in_pool + num_classes_found)
- Generate basic charts: cumulative classes discovered vs. number of queries

The random baseline now simply reads labels from the CSV and does random draws.
It no longer goes through any DataStream / embedding loader.

Intended for overnight runs to generate comparable ARED vs Random data + plots.

Usage examples:
    python -m PHX_A_RED_Project.runners.run_test --seed 42
    python -m PHX_A_RED_Project.runners.run_test --seed 123 --label-column scientific_name
    python -m PHX_A_RED_Project.runners.run_test --num-points 500 --seed 99   # small test

    # Run 30 different seeds starting from 1000 (great for overnight batch)
    python -m PHX_A_RED_Project.runners.run_test --num-seeds 30 --seed 1000 --label-column scientific_name

    # Resume safely (it will skip seeds that already have results)
    python -m PHX_A_RED_Project.runners.run_test --num-seeds 30 --seed 1000 --label-column scientific_name
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt


def find_record(results_dir: Path, prefix: str, label_column: str, seed: int) -> Path | None:
    """Find the most recently modified matching record file."""
    # ared uses: ared_{label}_N..._seed...
    # random uses: random_perch_{label}_N..._seed...
    # Use broad glob that still anchors on label and seed.
    pattern = str(results_dir / f"{prefix}*{label_column}*seed{seed}*.json")
    candidates = glob.glob(pattern)
    if not candidates:
        # Fallback: try without extra wildcards between
        pattern2 = str(results_dir / f"{prefix}*_{label_column}_*seed{seed}*.json")
        candidates = glob.glob(pattern2)
    if not candidates:
        return None
    # Prefer most recent
    candidates.sort(key=os.path.getmtime, reverse=True)
    return Path(candidates[0])


def build_cumulative_curve(discoveries: dict) -> tuple[list[int], list[int]]:
    """Return (query_points, cumulative_classes) suitable for step plotting."""
    if not discoveries:
        return [0], [0]
    ordinals = sorted(discoveries.values())
    xs = [0]
    ys = [0]
    for i, q in enumerate(ordinals, 1):
        xs.append(q)
        ys.append(i)
    return xs, ys


def generate_charts(ared_record: dict, random_record: dict, seed: int, label_column: str, plots_dir: Path,
                    is_full: bool = True, num_points: int = -1) -> list[Path]:
    """Generate and save the basic comparison charts."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    ared_disc = ared_record.get("discoveries", {})
    rand_disc = random_record.get("discoveries", {})

    ared_x, ared_y = build_cumulative_curve(ared_disc)
    rand_x, rand_y = build_cumulative_curve(rand_disc)

    # 1. Main chart: cumulative classes discovered vs queries
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.step(ared_x, ared_y, where="post", label="ARED (perch)", linewidth=2.0)
    ax.step(rand_x, rand_y, where="post", label="Random baseline", linewidth=2.0)
    ax.set_xlabel("Number of queries (labels)")
    ax.set_ylabel("Cumulative classes discovered")
    max_x = max(ared_x[-1] if ared_x else 0, rand_x[-1] if rand_x else 0)
    max_y = max(ared_y[-1] if ared_y else 0, rand_y[-1] if rand_y else 0)
    ax.set_xlim(left=0, right=max(1, int(max_x * 1.02)))
    ax.set_ylim(bottom=0, top=max(1, int(max_y * 1.05) + 1))
    dataset_label = "Full Dataset" if is_full else f"Capped ({num_points} points)"
    ax.set_title(
        f"Class Discovery vs Queries — {dataset_label}\n"
        f"seed={seed}   |   label_column={label_column}"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    main_plot = plots_dir / f"discovery_curve_seed{seed}_{label_column}.png"
    fig.savefig(main_plot, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(main_plot)
    print(f"[plot] Saved: {main_plot}")

    # 2. Secondary chart: query cost to reach each successive new class (rank plot)
    ared_sorted = sorted(ared_disc.values())
    rand_sorted = sorted(rand_disc.values())
    min_len = min(len(ared_sorted), len(rand_sorted))

    if min_len >= 1:
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        ranks = list(range(1, min_len + 1))
        ax2.plot(ranks, ared_sorted[:min_len], marker="o", markersize=3, linestyle="-", label="ARED (perch)")
        ax2.plot(ranks, rand_sorted[:min_len], marker="o", markersize=3, linestyle="-", label="Random baseline")
        ax2.set_xlabel("Class rank (order of discovery)")
        ax2.set_ylabel("Query ordinal when class was first discovered")
        ax2.set_title(f"Query cost per successive new class — seed={seed} | {label_column}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        rank_plot = plots_dir / f"rank_query_cost_seed{seed}_{label_column}.png"
        fig2.savefig(rank_plot, dpi=160, bbox_inches="tight")
        plt.close(fig2)
        saved.append(rank_plot)
        print(f"[plot] Saved: {rank_plot}")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Automated test runner: ARED (perch) + Random baseline. Supports multiple seeds."
    )
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed. When using --num-seeds, this is the first seed.")
    parser.add_argument("--num-seeds", type=int, default=1, help="Number of different seeds to run (default 1). Seeds will be seed, seed+1, ...")
    parser.add_argument(
        "--label-column",
        type=str,
        default="class_name",
        help="Label column to use (e.g. class_name or scientific_name for species-level)",
    )
    parser.add_argument("--num-points", type=int, default=-1, help="Pool size. Use -1 for the full dataset (default)")
    parser.add_argument("--results-dir", type=str, default="results", help="Where JSON records are written")
    parser.add_argument("--plots-dir", type=str, default=None, help="Where to write PNG charts (defaults to results/plots)")
    parser.add_argument("--skip-perch", action="store_true", help="Skip running the ARED/perch step (use existing records)")
    parser.add_argument("--skip-random", action="store_true", help="Skip running the random baseline step")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate charts (still run the experiments)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands that would be run but do not execute")
    parser.add_argument("--force", action="store_true", help="Force re-run even if results for a seed already exist")

    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(args.plots_dir).resolve() if args.plots_dir else (results_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    label_col = args.label_column
    num_points = args.num_points
    is_full = num_points <= 0

    num_seeds = max(1, args.num_seeds)
    start_seed = args.seed

    print("=" * 70)
    print("RUN TEST — ARED (perch) vs Random baseline")
    if num_seeds > 1:
        print(f"  Running {num_seeds} seeds starting from {start_seed}")
    print(f"  label_column   : {label_col}")
    print(f"  num_points     : {num_points}  ({'FULL DATASET' if is_full else 'capped'})")
    print(f"  results_dir    : {results_dir}")
    print(f"  plots_dir      : {plots_dir}")
    print("=" * 70)

    seeds = [start_seed + i for i in range(num_seeds)]

    for seed in seeds:
        run_single_test(
            seed=seed,
            label_col=label_col,
            num_points=num_points,
            is_full=is_full,
            results_dir=results_dir,
            plots_dir=plots_dir,
            skip_perch=args.skip_perch,
            skip_random=args.skip_random,
            no_plots=args.no_plots,
            dry_run=args.dry_run,
            force=args.force,
        )


def run_single_test(seed, label_col, num_points, is_full, results_dir, plots_dir,
                    skip_perch, skip_random, no_plots, dry_run, force):
    """Run one seed (ARED + random) + charts if needed."""

    # Check if we can skip
    if not force and not dry_run:
        ared_rec = find_record(results_dir, "ared", label_col, seed)
        rand_rec = find_record(results_dir, "random", label_col, seed)
        if ared_rec and rand_rec:
            print(f"\n[skip] Results already exist for seed {seed} (label={label_col}). Use --force to re-run.")
            if not no_plots:
                try:
                    with open(ared_rec) as f: ared_data = json.load(f)
                    with open(rand_rec) as f: rand_data = json.load(f)
                    generate_charts(ared_data, rand_data, seed, label_col, plots_dir,
                                    is_full=is_full, num_points=num_points)
                    print(f"[skip] Regenerated charts for seed {seed}")
                except Exception as e:
                    print(f"[skip] Could not regenerate charts: {e}")
            return

    print(f"\n{'='*60}")
    print(f">>> Starting seed {seed}")
    print(f"{'='*60}")

    # Build flags per runner
    common_base = [
        "--num-points", str(num_points),
        "--seed", str(seed),
        "--label-column", label_col,
        "--results-dir", str(results_dir),
    ]

    # 1. ARED perch
    if not skip_perch:
        perch_flags = common_base + ["--save-results"]
        cmd = [sys.executable, "-m", "PHX_A_RED_Project.runners.perch"] + perch_flags
        print("\n>>> Running ARED (perch)")
        print("    " + " ".join(cmd))
        if dry_run:
            print("    [dry-run] skipped")
        else:
            subprocess.run(cmd, check=True)

    # 2. Random baseline
    if not skip_random:
        rand_flags = common_base
        cmd = [sys.executable, "-m", "PHX_A_RED_Project.runners.random_baseline"] + rand_flags
        print("\n>>> Running Random baseline")
        print("    " + " ".join(cmd))
        if dry_run:
            print("    [dry-run] skipped")
        else:
            subprocess.run(cmd, check=True)

    if dry_run:
        print(f"[dry-run] Finished seed {seed}")
        return

    # 3. Find records
    ared_rec = find_record(results_dir, "ared", label_col, seed)
    rand_rec = find_record(results_dir, "random", label_col, seed)

    if not ared_rec or not rand_rec:
        print(f"\n[warning] Could not locate results for seed {seed}")
        print(f"  ared: {ared_rec}")
        print(f"  rand: {rand_rec}")
        return

    print(f"\nFound records for seed {seed}:\n  ARED   : {ared_rec}\n  Random : {rand_rec}")

    with open(ared_rec) as f:
        ared_data = json.load(f)
    with open(rand_rec) as f:
        rand_data = json.load(f)

    # 4. Charts
    if not no_plots:
        saved = generate_charts(ared_data, rand_data, seed, label_col, plots_dir,
                                is_full=is_full, num_points=num_points)
        print(f"  Generated {len(saved)} chart(s)")

    # 5. Summary for this seed
    ared_disc = ared_data.get("discoveries", {})
    rand_disc = rand_data.get("discoveries", {})
    ared_q = ared_data.get("queries", len(ared_disc))
    rand_q = rand_data.get("queries", len(rand_disc))

    ds_label = "full dataset" if is_full else f"{num_points}-point pool"
    print(f"\n  [{seed}] SUMMARY ({ds_label})")
    print(f"    ARED:   {len(ared_disc)} classes | last query {max(ared_disc.values()) if ared_disc else 0} | total Q {ared_q}")
    print(f"    Random: {len(rand_disc)} classes | last query {max(rand_disc.values()) if rand_disc else 0} | total Q {rand_q}")


if __name__ == "__main__":
    main()