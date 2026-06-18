"""
Runner that demonstrates the live ShiftingKappaController.

Matches historical behavior of:
    Spectrogram_A_RED/main_spectrogram_shifting_kappa.py
    Spectrogram_A_RED/main_spectrogram_shifting_kappa_species.py

Usage examples:
    python -m PHX_A_RED_Project.runners.with_shifting_kappa --num-points 2000 --target 4 --label-column class_name
    python -m PHX_A_RED_Project.runners.with_shifting_kappa --num-points 5000 --target 8 --label-column scientific_name --frontend perch
    python -m PHX_A_RED_Project.runners.with_shifting_kappa --frontend dinov3 --fast
"""

import argparse

from ..data.base_stream import SpectrogramDataStream, PerchDataStream, Dinov3DataStream
from ..data.oracle import NoRelevanceOracle
from ..experiment import AREDExperiment
from ..utils import ClassDiscoveryCounter, ShiftingKappaController


def build_stream(frontend: str, num_points: int, label_column: str, shuffle: bool, seed: int, fast: bool, **kw):
    max_s = num_points if num_points > 0 else None
    if frontend == "spectrogram":
        return SpectrogramDataStream(
            max_samples=max_s, shuffle=shuffle, seed=seed, label_column=label_column
        )
    elif frontend == "perch":
        return PerchDataStream(
            max_samples=max_s, shuffle=shuffle, seed=seed, label_column=label_column,
            live_batch_size=kw.get("batch_size", 4),
        )
    elif frontend == "dinov3":
        return Dinov3DataStream(
            max_samples=max_s, shuffle=shuffle, seed=seed, label_column=label_column
        )
    else:
        raise ValueError(f"Unknown frontend {frontend}")


def main():
    parser = argparse.ArgumentParser(description="ARED + live shifting-kappa controller")
    parser.add_argument("--frontend", "-f", choices=["spectrogram", "perch", "dinov3"], default="spectrogram")
    parser.add_argument("--num-points", type=int, default=2000)
    parser.add_argument("--label-column", type=str, default="class_name")
    parser.add_argument("--target", type=float, default=4.0, help="Target queries per 100 points")
    parser.add_argument("--initial-kappa", type=float, default=None)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="fast_mode for ClassDiscoveryCounter (long runs)")
    parser.add_argument("--status-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    shuffle = not args.no_shuffle
    stream = build_stream(
        args.frontend, args.num_points, args.label_column, shuffle, args.seed, args.fast,
        batch_size=args.batch_size,
    )

    counter = ClassDiscoveryCounter(fast_mode=args.fast)

    oracle = NoRelevanceOracle(stream, discovery_tracker=counter)

    initial_k = args.initial_kappa if args.initial_kappa is not None else (2.1 if args.frontend == "spectrogram" else 1.15)

    exp = AREDExperiment(
        stream,
        oracle,
        kappa=initial_k,
        data_window_size=250,
        k_comparison_clusters=5 if args.frontend != "dinov3" else 8,
    )

    controller = ShiftingKappaController(
        target_queries_per_100=args.target,
        initial_kappa=initial_k,
        verbose=True,
        keep_history=not args.fast,
    )

    print(f"=== ARED + ShiftingKappa ({args.frontend}) ===")
    print(f"target={args.target}/100, start_kappa={initial_k}, label_column={args.label_column}")
    print()

    exp.run(
        num_points=args.num_points,
        controller=controller,
        status_every=args.status_every,
        verbose=True,
    )

    exp.print_report()

    # Extra controller summary
    print("\n" + controller.get_summary())


if __name__ == "__main__":
    main()
