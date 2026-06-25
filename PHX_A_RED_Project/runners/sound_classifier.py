"""
Thin runner for the standard SoundClassifier.
Usage examples:
    python -m PHX_A_RED_Project.runners.sound_classifier --label-column scientific_name --num-points 30000
    python -m PHX_A_RED_Project.runners.sound_classifier --model-type prototype
    python -m PHX_A_RED_Project.runners.sound_classifier --model-type logistic --group-split
    python -m PHX_A_RED_Project.runners.sound_classifier --load-model models/best_model.pkl --evaluate-only
"""

import argparse
from pathlib import Path

from PHX_A_RED_Project.classifier.sound_classifier import SoundClassifier
from PHX_A_RED_Project.data.base_stream import PerchDataStream


def main():
    parser = argparse.ArgumentParser(description="Standard Sound Classifier Runner")
    parser.add_argument("--num-points", type=int, default=30000, help="Max samples to use for training pool (larger => better coverage of 200+ classes)")
    parser.add_argument("--label-column", type=str, default="scientific_name",
                        choices=["scientific_name", "class_name", "common_name"])
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--model-type", type=str, default="lightgbm",
                        choices=["randomforest", "lightgbm", "logistic", "prototype"])
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-split", action="store_true", help="Split at recording level to avoid train/test leakage from chunks of same file")

    # Save / Load options
    parser.add_argument("--save-model", type=str, default=None, help="Path to save model (e.g. models/sound_clf.pkl)")
    parser.add_argument("--load-model", type=str, default=None, help="Load and evaluate a saved model")
    parser.add_argument("--evaluate-only", action="store_true")

    # External test override (if omitted or <=0 we prefer the internal held-out set from fit)
    parser.add_argument("--test-samples", type=int, default=0, help="If >0, build an external test stream of this size instead of using internal held-out")

    args = parser.parse_args()

    if args.load_model:
        print(f"Loading model from {args.load_model}")
        clf = SoundClassifier.load(args.load_model)
    else:
        clf = SoundClassifier(
            random_state=args.seed,
            n_estimators=args.n_estimators,
            use_gpu=args.use_gpu,
            model_type=args.model_type,
            group_split=args.group_split,
        )

        clf.fit(
            num_points=args.num_points,
            label_column=args.label_column,
            train_frac=args.train_frac
        )

        if args.save_model:
            save_path = clf.save(args.save_model)
            print(f"Model saved successfully!")

    # === Evaluation ===
    if not args.evaluate_only or args.load_model:
        print("\n" + "="*60)
        print("Running Evaluation on Held-out Test Set")
        print("="*60)

        if args.test_samples and args.test_samples > 0:
            # User explicitly wants an external (possibly larger / differently sampled) test
            test_stream = PerchDataStream(
                label_column=args.label_column,
                shuffle=False,
                seed=args.seed + 1,
                max_samples=args.test_samples
            )
            clf.evaluate(test_stream)
        else:
            # Preferred path: use the internal held-out set created during fit
            clf.evaluate()


if __name__ == "__main__":
    main()