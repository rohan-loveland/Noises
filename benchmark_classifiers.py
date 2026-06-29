#!/usr/bin/env python3
"""
Benchmark script for comparing classifiers, with strong focus on ARED.

Structure of experiments:
- Baseline (full supervision): sound_prototype, sound_logistic
- ARED used as a classifier: ared_k{kappa}  (locates test points in the "feature space" ARED built)
- Random selection baselines:
    - random_Q=NNN                  (1NN on randomly chosen supports)
    - random_Q=NNN_train_*          (train prototype/logistic on randomly chosen supports)
- ARED selection + train:
    - ared_select_train_k{kappa}_*  (use exactly the points ARED queried to train prototype/logistic)

All use the same fixed test set. Full per-class + averaged precision/recall/f1 are saved.

Usage examples:

    # Run a standard comparison suite
    python benchmark_classifiers.py --num-points 100000 --test-size 15000 --seed 42

    # Full data
    python benchmark_classifiers.py --num-points -1 --test-size 20000 --seed 123

    # Only ARED related (includes ared-as-classifier + ared-select-train + matching randoms)
    python benchmark_classifiers.py --num-points 80000 --kappas 0.25 0.35 0.5 --only-ared --seed 42

    # Display comparison from previous results (shows weighted+macro P/R/F1 and other avgs)
    python benchmark_classifiers.py --compare results/benchmarks/2026-06-29_...

The script saves detailed JSON files (with full classification_report dict) for every experiment.
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from PHX_A_RED_Project.classifier import SoundClassifier, AREDClassifier
from PHX_A_RED_Project.data.base_stream import PerchDataStream


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def compute_rich_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Return the complete set of metrics for a prediction: scalars + the FULL classification_report.

    This ensures we capture precision, recall, f1 per class, support, and all averages.
    """
    from sklearn.metrics import (
        classification_report,
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
    )

    # Always use zero_division=0 so rare classes don't blow up
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    rich = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_true)),
        "classes_in_test": int(len(np.unique(y_true))),
        "classification_report": report,   # <-- the complete thing
    }
    return rich


def create_fixed_test_stream(
    label_column: str = "scientific_name",
    test_size: int = 15000,
    seed: int = 42,
) -> tuple[PerchDataStream, np.ndarray, np.ndarray]:
    """Create one fixed test set used for all methods in a benchmark run."""
    print(f"Creating fixed test set (size={test_size}, seed={seed}) ...")
    test_stream = PerchDataStream(
        label_column=label_column,
        max_samples=test_size,
        shuffle=True,
        seed=seed,
    )
    X_test = test_stream.processed_vectors
    y_test = np.array(test_stream.labels_cache)
    print(f"  Test set ready: {len(y_test)} samples, {len(np.unique(y_test))} classes")
    return test_stream, X_test, y_test


def get_pool_for_random(
    label_column: str,
    num_points: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a pool of data that we can sample from for random baselines."""
    max_s = num_points if (num_points is not None and num_points > 0) else None
    pool = PerchDataStream(
        label_column=label_column,
        max_samples=max_s,
        shuffle=True,
        seed=seed,
    )
    X = pool.processed_vectors
    y = np.array(pool.labels_cache)
    return X, y


def run_sound_experiment(
    name: str,
    model_type: str,
    num_points: Optional[int],
    label_column: str,
    seed: int,
    fixed_test_stream: PerchDataStream,
) -> Dict[str, Any]:
    print(f"\n=== Running {name} ===")
    start = time.perf_counter()

    clf = SoundClassifier(
        random_state=seed,
        model_type=model_type,
        use_gpu=True,
    )
    clf.fit(
        num_points=num_points if num_points and num_points > 0 else -1,
        label_column=label_column,
    )

    # We call evaluate for its side-effects (printing the report + any internal stats)
    _ = clf.evaluate(fixed_test_stream)

    # But for COMPLETE data we re-predict and build the rich metrics ourselves
    X_test = fixed_test_stream.processed_vectors
    y_test = np.array(fixed_test_stream.labels_cache)
    y_pred = clf.predict(X_test)

    rich_metrics = compute_rich_metrics(y_test, y_pred)

    elapsed = time.perf_counter() - start

    record = {
        "name": name,
        "type": "sound",
        "model_type": model_type,
        "seed": seed,
        "num_points": num_points,
        "label_column": label_column,
        "Q": "full",
        "elapsed_sec": round(elapsed, 2),
        "metrics": rich_metrics,
        "extra": {
            "n_train_used": num_points,
        },
    }
    return record


def run_ared_experiment(
    name: str,
    kappa: float,
    num_points: Optional[int],
    label_column: str,
    seed: int,
    fixed_test_stream: PerchDataStream,
    include_random_baseline: bool = True,
) -> List[Dict[str, Any]]:
    """Run ARED + optionally a random-Q baseline using the *exact* query count ARED used."""
    records: List[Dict[str, Any]] = []
    print(f"\n=== Running {name} (kappa={kappa}) ===")
    start = time.perf_counter()

    clf = AREDClassifier(
        kappa=kappa,
        data_window_size=10000,
        random_state=seed,
    )
    clf.fit(
        num_points=num_points if num_points and num_points > 0 else -1,
        label_column=label_column,
    )

    # Side-effect print + ARED-specific fields (anom rate etc.)
    ared_eval_metrics = clf.evaluate(fixed_test_stream, mode="ared_space")

    # Authoritative label budget = number of queries ARED actually decided to make
    Q = int(getattr(clf, "num_queries_used_", 0) or clf.n_prototypes_)

    # Re-predict to get full rich metrics + complete classification_report
    X_test = fixed_test_stream.processed_vectors
    y_test = np.array(fixed_test_stream.labels_cache)
    y_pred = clf.predict(X_test, mode="ared_space")
    rich = compute_rich_metrics(y_test, y_pred)

    # Merge ARED-specific diagnostics into the rich metrics for convenience
    rich.update({
        "n_prototypes": getattr(clf, "n_prototypes_", Q),
        "n_queries_during_training": Q,
        "n_classes_in_supports": getattr(clf, "n_classes_covered_", 0),
        "fraction_test_would_be_anomalous": ared_eval_metrics.get("fraction_test_would_be_anomalous", None),
    })

    elapsed = time.perf_counter() - start

    ared_record = {
        "name": name,
        "type": "ared",
        "kappa": kappa,
        "seed": seed,
        "num_points": num_points,
        "label_column": label_column,
        "Q": Q,
        "elapsed_sec": round(elapsed, 2),
        "metrics": rich,
        "extra": {
            "clusters_meta_count": len(getattr(clf, "clusters_meta_", [])),
        },
    }
    records.append(ared_record)

    # --- Random baseline at *exactly* the same query budget Q ---
    if include_random_baseline and Q > 0:
        print(f"  Running random baseline with exactly Q={Q} labels (matching ARED queries)...")
        rand_start = time.perf_counter()

        X_pool, y_pool = get_pool_for_random(label_column, num_points, seed + 999)

        if len(X_pool) <= Q:
            rand_idx = np.arange(len(X_pool))
        else:
            rng = np.random.default_rng(seed + 42)
            rand_idx = rng.choice(len(X_pool), Q, replace=False)

        X_rand = X_pool[rand_idx]
        y_rand = y_pool[rand_idx]

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(X_rand)

        X_test = fixed_test_stream.processed_vectors
        y_test = np.array(fixed_test_stream.labels_cache)

        def l2(x):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return x / n

        _, ind = nn.kneighbors(l2(X_test))
        y_pred_rand = y_rand[ind[:, 0]]

        # FULL rich metrics + print the complete report for the random baseline
        print("\n--- Full classification report for RANDOM baseline ---")
        from sklearn.metrics import classification_report as sk_classification_report
        print(sk_classification_report(y_test, y_pred_rand, zero_division=0))
        print("--- End random baseline report ---\n")

        rich_rand = compute_rich_metrics(y_test, y_pred_rand)

        rand_record = {
            "name": f"random_Q={Q}",
            "type": "random_baseline",
            "seed": seed,
            "Q": Q,
            "elapsed_sec": round(time.perf_counter() - rand_start, 2),
            "metrics": rich_rand,
            "extra": {
                "source_Q_from": name,
                "pool_size": len(X_pool),
                "selection": "random",
            },
        }
        records.append(rand_record)

        # Also produce random_Q_train_<head> versions using the exact same random supports
        random_train_recs = _make_selection_train_records(
            base_name=f"random_Q={Q}_train",
            supports=X_rand,
            support_labels=y_rand,
            Q=Q,
            fixed_test_stream=fixed_test_stream,
            heads=["prototype", "logistic"],
            seed=seed,
            elapsed_so_far=time.perf_counter() - rand_start,
            extra={"source_Q_from": name, "selection": "random"},
        )
        records.extend(random_train_recs)

    # --- NEW: ARED selection + train normal classifier head ---
    # After ARED has chosen its points, train prototype/logistic on *exactly* those points.
    if getattr(clf, "support_vectors_", None) is not None and len(clf.support_vectors_) > 0:
        ared_train_recs = _make_selection_train_records(
            base_name=f"ared_select_train_k{kappa}",
            supports=clf.support_vectors_,
            support_labels=clf.support_labels_,
            Q=Q,
            fixed_test_stream=fixed_test_stream,
            heads=["prototype", "logistic"],
            seed=seed,
            elapsed_so_far=time.perf_counter() - start,
            extra={
                "source_Q_from": name,
                "selection": "ared",
                "n_queries_during_training": Q,
            },
        )
        records.extend(ared_train_recs)

    return records


# ------------------------------------------------------------------
# New experiment type: ARED (or random) selects exactly Q points,
# then we train a *normal* classifier head on those points.
# This is the "use ARED to select the training examples" mode.
# ------------------------------------------------------------------

def _train_and_predict_head(
    head: str,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Train a simple head on the given supports and return predictions on X_test."""
    if head == "prototype":
        # Per-class mean vectors (same spirit as SoundClassifier "prototype")
        classes = np.unique(y_support)
        proto = {}
        for c in classes:
            m = X_support[y_support == c].mean(axis=0)
            n = np.linalg.norm(m)
            proto[c] = m / n if n > 0 else m

        # cosine similarity on L2-normalized test
        def l2(x):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return x / n

        Xt = l2(X_test)
        # vectorized cosine
        protos = np.stack([proto[c] for c in classes])
        sims = Xt @ protos.T
        idx = np.argmax(sims, axis=1)
        return np.array([classes[i] for i in idx])

    elif head == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_support, y_support)
        return clf.predict(X_test)

    else:
        raise ValueError(f"Unknown head: {head}")


def _make_selection_train_records(
    base_name: str,
    supports: np.ndarray,
    support_labels: np.ndarray,
    Q: int,
    fixed_test_stream: PerchDataStream,
    heads: List[str],
    seed: int,
    elapsed_so_far: float,
    extra: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Given a support set (from ARED or random), train heads and produce rich records."""
    recs = []
    X_test = fixed_test_stream.processed_vectors
    y_test = np.array(fixed_test_stream.labels_cache)

    for h in heads:
        y_pred = _train_and_predict_head(h, supports, support_labels, X_test, random_state=seed)

        print(f"\n--- Full classification report for {base_name}_{h} (trained on {len(supports)} supports) ---")
        from sklearn.metrics import classification_report as sk_cr
        print(sk_cr(y_test, y_pred, zero_division=0))
        print("--- End report ---\n")

        rich = compute_rich_metrics(y_test, y_pred)
        rec = {
            "name": f"{base_name}_{h}",
            "type": "ared_select_train" if "ared" in base_name else "random_train",
            "seed": seed,
            "Q": Q,
            "head": h,
            "elapsed_sec": round(elapsed_so_far, 2),
            "metrics": rich,
            "extra": extra,
        }
        recs.append(rec)
    return recs


def print_comparison_table(records: List[Dict[str, Any]]):
    """Pretty text table of the most important columns."""
    if not records:
        print("No records to compare.")
        return

    # Collect rows
    rows = []
    for r in records:
        m = r.get("metrics", {})
        report = m.get("classification_report") or {}
        wavg = report.get("weighted avg") or {}
        mavg = report.get("macro avg") or {}
        rows.append({
            "name": r.get("name", "?"),
            "Q": r.get("Q", "-"),
            "acc": m.get("accuracy"),
            "bal_acc": m.get("balanced_accuracy"),
            "f1": m.get("f1_weighted"),
            "f1_m": (mavg.get("f1-score") if mavg else None),
            "prec_w": wavg.get("precision"),
            "rec_w": wavg.get("recall"),
            "prec_m": mavg.get("precision"),
            "rec_m": mavg.get("recall"),
            "classes_test": m.get("classes_in_test_full") or m.get("classes_in_test") or m.get("n_test_filtered"),
            "anom": m.get("fraction_test_would_be_anomalous"),
            "seed": r.get("seed"),
            "type": r.get("type"),
        })

    # Header
    headers = ["Name", "Q", "Acc", "BalAcc", "F1w", "F1m", "P_w", "R_w", "P_m", "R_m", "Cls", "Anom", "Seed"]
    col_widths = [34, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 6]

    def fmt_row(vals):
        return " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    print("\n" + "=" * 105)
    print("BENCHMARK COMPARISON")
    print("=" * 105)
    print(fmt_row(headers))
    print("-" * 105)

    for row in rows:
        vals = [
            row["name"][:33],
            str(row["Q"]),
            f"{row['acc']:.4f}" if row["acc"] is not None else "-",
            f"{row['bal_acc']:.4f}" if row["bal_acc"] is not None else "-",
            f"{row['f1']:.4f}" if row["f1"] is not None else "-",
            f"{row['f1_m']:.4f}" if row["f1_m"] is not None else "-",
            f"{row['prec_w']:.3f}" if row["prec_w"] is not None else "-",
            f"{row['rec_w']:.3f}" if row["rec_w"] is not None else "-",
            f"{row['prec_m']:.3f}" if row["prec_m"] is not None else "-",
            f"{row['rec_m']:.3f}" if row["rec_m"] is not None else "-",
            str(row["classes_test"]) if row["classes_test"] else "-",
            f"{row['anom']:.3f}" if row["anom"] is not None else "-",
            str(row["seed"]),
        ]
        print(fmt_row(vals))

    print("=" * 105)
    print("Notes:")
    print("  - Q = number of labels used (prototypes for ARED, 'full' for supervised)")
    print("  - F1w = weighted F1; F1m = macro F1   |   P_w/R_w = weighted avg Prec/Rec ; P_m/R_m = macro avg Prec/Rec")
    print("  - Cls = # classes in the (fixed) test set for this run")
    print("  - Anom = fraction of test points outside ARED's learned cluster regions (ared_* rows only)")
    print("  - All methods use the exact same fixed test set within one benchmark run.")
    print("  - Full per-class + all averages (prec/rec/f1/support) are in the JSONs: metrics.classification_report")
    print("=" * 105 + "\n")


def run_full_suite(args):
    timestamp = get_timestamp()
    out_dir = Path(args.output_dir) / f"benchmark_{timestamp}"
    ensure_dir(out_dir)

    print(f"Results will be saved to: {out_dir}")

    label_column = args.label_column
    num_points = args.num_points if args.num_points and args.num_points > 0 else None
    test_size = args.test_size
    base_seed = args.seed

    # Fixed test set for the whole suite
    fixed_test_stream, _, _ = create_fixed_test_stream(
        label_column=label_column,
        test_size=test_size,
        seed=base_seed + 1000,
    )

    all_records: List[Dict[str, Any]] = []

    # === Sound / normal supervised baselines ===
    if not args.only_ared:
        for model in ["prototype", "logistic"]:
            if model == "logistic" and args.skip_logistic:
                continue
            rec = run_sound_experiment(
                name=f"sound_{model}",
                model_type=model,
                num_points=num_points,
                label_column=label_column,
                seed=base_seed,
                fixed_test_stream=fixed_test_stream,
            )
            all_records.append(rec)
            save_json(rec, out_dir / f"{rec['name']}_seed{base_seed}.json")

    # === ARED runs + random baselines ===
    kappas = args.kappas
    if not kappas:
        kappas = [0.25, 0.35, 0.50]

    for kappa in kappas:
        name = f"ared_k{kappa}"
        recs = run_ared_experiment(
            name=name,
            kappa=kappa,
            num_points=num_points,
            label_column=label_column,
            seed=base_seed,
            fixed_test_stream=fixed_test_stream,
            include_random_baseline=not args.no_random_baseline,
        )
        for rec in recs:
            all_records.append(rec)
            if rec["type"] == "random_baseline":
                fname = f"{rec['name']}_from_{name}_seed{base_seed}.json"
            else:
                fname = f"{rec['name']}_seed{base_seed}.json"
            save_json(rec, out_dir / fname)

    # Save a manifest with all results for this run
    manifest = {
        "timestamp": timestamp,
        "args": vars(args),
        "num_experiments": len(all_records),
        "records": [r["name"] for r in all_records],
    }
    save_json(manifest, out_dir / "manifest.json")

    # Final comparison table
    print_comparison_table(all_records)

    print(f"\nAll results saved under: {out_dir}")
    print("You can re-run with different seeds or parameters and compare the folders manually.")


def load_and_compare(compare_dir: str):
    p = Path(compare_dir)
    jsons = sorted(p.glob("*.json"))
    records = []
    for j in jsons:
        if j.name == "manifest.json":
            continue
        try:
            with open(j) as f:
                records.append(json.load(f))
        except Exception as e:
            print(f"Warning: could not load {j}: {e}")

    if not records:
        print("No experiment JSONs found in that directory.")
        return

    print_comparison_table(records)


def main():
    parser = argparse.ArgumentParser(description="Classifier Benchmark Runner")
    parser.add_argument("--num-points", type=int, default=100000,
                        help="Training pool size (-1 or large number for full data)")
    parser.add_argument("--test-size", type=int, default=15000,
                        help="Size of the fixed held-out test set used for all methods")
    parser.add_argument("--label-column", type=str, default="scientific_name",
                        choices=["scientific_name", "class_name", "common_name"])
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--kappas", type=float, nargs="*", default=None,
                        help="List of kappa values for ARED (default: 0.25 0.35 0.50)")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks",
                        help="Where to write result folders")
    parser.add_argument("--only-ared", action="store_true",
                        help="Only run ARED experiments (skip normal SoundClassifier)")
    parser.add_argument("--skip-logistic", action="store_true",
                        help="Skip the logistic SoundClassifier variant")
    parser.add_argument("--no-random-baseline", action="store_true",
                        help="Do not run random-Q baselines for ARED runs")
    parser.add_argument("--compare", type=str, default=None,
                        help="Instead of running, just load and display results from this directory")

    args = parser.parse_args()

    if args.compare:
        load_and_compare(args.compare)
    else:
        run_full_suite(args)


if __name__ == "__main__":
    main()