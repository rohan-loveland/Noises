"""
Thin runner for Perch v2 embeddings frontend.

Usage:
    python -m PHX_A_RED_Project.runners.perch --num-points 500 --label-column scientific_name
    python -m PHX_A_RED_Project.runners.perch --num-points 2000
"""

import argparse

from ..experiment import run_ared


def main():
    parser = argparse.ArgumentParser(description="ARED over Perch v2 embeddings (precomp or live).")
    parser.add_argument("--num-points", type=int, default=500, help="Number of points (-1 for all)")
    parser.add_argument("--label-column", type=str, default="class_name")
    parser.add_argument("--kappa", type=float, default=1.15)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--k-comp", type=int, default=5)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--batch-size", type=int, default=4, help="Live extraction batch size")
    args = parser.parse_args()

    run_ared(
        "perch",
        num_points=args.num_points,
        label_column=args.label_column,
        kappa=args.kappa,
        data_window_size=args.window,
        k_comparison_clusters=args.k_comp,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        live_batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
