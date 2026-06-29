"""
Test script for ARED as a classifier.

Core idea (per user):
  ARED runs on training data and *builds a feature space* (its clusters + the specific
  labeled points it decided to query).
  After training, we take new points and ask: "where does this point lie in the space
  ARED built?" and take the label of the region (cluster) it lands in.

This is different from "use ARED to pick some points, then train a completely separate
traditional classifier on them".
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PHX_A_RED_Project.classifier import AREDClassifier
from PHX_A_RED_Project.data.base_stream import PerchDataStream
import numpy as np

print("=== Building ARED 'feature space' on training data ===")
clf = AREDClassifier(
    kappa=0.35,
    data_window_size=5000,
    n_neighbors=1,
    random_state=42,
)

clf.fit(
    num_points=30000,
    label_column="scientific_name",
    train_frac=0.8
)

print("\nSpace ARED built:")
print("  Labeled points (Q):", clf.n_prototypes_)
print("  Classes covered:", clf.n_classes_covered_)
print("  Model summary:", clf.get_ared_model_summary())

print("\n=== Testing: feed new points and see where they lie in the built space ===")
test_stream = PerchDataStream(
    label_column="scientific_name",
    shuffle=True,
    seed=123,
    max_samples=5000
)

# This calls locate_in_built_space internally and reports anomaly diagnostics
metrics = clf.evaluate(test_stream, mode="ared_space")

# Direct "where does it lie" for a few examples
sample_vecs = test_stream.processed_vectors[:8]
labels, dists, comp_ds, would_anom = clf.locate_in_built_space(sample_vecs)

print("\nExample new points located in ARED-built space:")
for i in range(len(labels)):
    print(f"  pred={labels[i]:30s}  dist={dists[i]:.4f}  comp_d={comp_ds[i]:.4f}  would_anomalous={would_anom[i]}")

# --- Simple same-budget comparison sketch (highly recommended for your goal) ---
print("\n=== Rough same-budget comparison (illustrative) ===")
Q = clf.n_prototypes_
print(f"ARED used Q={Q} labels via its anomaly mechanism.")

# For a real comparison you would:
# 1. From the *same* train df used inside fit, sample Q points randomly (stratified or not).
# 2. Build a plain 1NN or the prototype classifier from SoundClassifier on exactly those Q labeled points.
# 3. Evaluate both on the identical test_stream.
#
# Example skeleton (you can expand this):
# from PHX_A_RED_Project.classifier import SoundClassifier
# ... obtain the original train vectors/labels used by ARED ...
# random_idx = np.random.choice(len(train_vecs), Q, replace=False)
# random_supports = train_vecs[random_idx]
# random_labs = train_labs[random_idx]
# Then train a small 1NN on (random_supports, random_labs) and compare accuracies.

print("See the plan / your own comparison harness for a full random-Q vs ARED-Q head-to-head.")