"""
Thin runner for spectrogram (pooled raw) frontend.

Usage:
    python -m PHX_A_RED_Project.runners.spectrogram --num-points 500 --label-column class_name
"""

import argparse

from ..experiment import run_ared


def main():
    parser = argparse.ArgumentParser(description="ARED over pooled spectrogram features.")
    parser.add_argument("--num-points", type=int, default=500, help="Number of points (-1 for all)")
    parser.add_argument("--label-column", type=str, default="class_name")
    parser.add_argument("--kappa", type=float, default=.25)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--k-comp", type=int, default=5)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-results", action="store_true", help="Write per-class discovery query ordinals to results/ folder")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    run_ared(
        "spectrogram",
        num_points=args.num_points,
        label_column=args.label_column,
        kappa=args.kappa,
        data_window_size=args.window,
        k_comparison_clusters=args.k_comp,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        save_results=args.save_results,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
