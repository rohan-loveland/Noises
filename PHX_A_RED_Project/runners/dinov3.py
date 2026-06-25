"""
Thin runner for DinoV3 spectrogram embeddings frontend.

Usage:
    python -m PHX_A_RED_Project.runners.dinov3 --num-points 500
"""

import argparse

from ..experiment import run_ared


def main():
    parser = argparse.ArgumentParser(description="ARED over DinoV3 (or DINO) spectrogram embeddings.")
    parser.add_argument("--num-points", type=int, default=500, help="Number of points (-1 for all)")
    parser.add_argument("--label-column", type=str, default="class_name")
    parser.add_argument("--kappa", type=float, default=.38)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--k-comp", type=int, default=5)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-results", action="store_true", help="Write per-class discovery query ordinals to results/ folder")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    run_ared(
        "dinov3",
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
