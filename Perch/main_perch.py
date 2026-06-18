"""
Perch A_RED Runner

Runs the original A_RED algorithm (untouched in A_REDimplementation/A_RED)
on Google Perch v2 embeddings (semantic, auto-detected dim) instead of raw spectrograms.

This file deliberately does **not** modify anything under A_REDimplementation/.

It follows the exact same adapter pattern as:
    Spectrogram_A_RED/main_spectrogram.py
    Spectrogram_A_RED/SpectrogramDataStream.py

Two ways to get embeddings (both supported automatically):
1. Precompute once with:  python Perch/Perch_A_RED.py
   (supports resume, small batches for 8 GB VRAM, threaded I/O)
2. Live / on-demand: just run this script (or with --num-points N).
   Only the embeddings for the points you actually stream are computed.

CLI:
  python Perch/main_perch.py --num-points 128 --label-column scientific_name
  python Perch/main_perch.py --num-points -1 --no-shuffle   # full CSV order (very long)

The Oracle is configured exactly like the spectrogram version:
- relevance=False for *all* classes (including birds).
- This produces the desired "query only on true anomaly + near-zero queries
  after initial points" behavior.
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

# Add paths so we can import the untouched A_RED core + our local adapters
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "A_REDimplementation" / "A_RED"))
sys.path.insert(0, str(script_dir))

from PerchDataStream import PerchDataStream, PerchOracle
from A_RED import ARED
from Stats import Stats


class ClassDiscoveryCounter:
    """
    Tracks for each class (label):
      - The processed point index when the class first appeared in the incoming data stream.
      - The processed point index when the class was first actually queried (oracle called for it).
      - How many instances ("how many of them") of the class were seen (via direct label peek
        on *every* processed point) before the first query for that class. These are the points
        that arrived and were typically absorbed as o_pts under other clusters without spending
        an oracle query on the new label. Useful for measuring "missed" examples under a query budget.
    This allows measuring both discovery *latency* (point delay) and *volume seen before discovery*
    (count of same-class points before the query that revealed it).
    Uses the ARED processed-point index space (0,1,2,...) so delays are in "algorithm time".
    Also tracks total queries per class (every time the oracle returns that label) so we can
    compute enrichment vs. random chance (query share / prevalence).
    """

    def __init__(self):
        self.first_appearance = {}       # label -> first processed_idx
        self.first_queried = {}          # label -> first processed_idx (only when oracle was called)
        self.first_queried_query_count = {} # label -> query number (1-based) when first discovered
        self.total_seen = {}             # label -> total # of points of this class seen in the whole run
        self.seen_before_queried = {}    # label -> # of this class seen (on appearance) strictly before first query
        self.queries_per_class = {}      # label -> total # of queries (oracle calls) that returned this label

    def record_appearance(self, label, processed_idx: int):
        if label not in self.first_appearance:
            self.first_appearance[label] = processed_idx
        self.total_seen[label] = self.total_seen.get(label, 0) + 1

        # Only count toward "before queried" while we have not yet spent a query on this label.
        if label not in self.first_queried:
            self.seen_before_queried[label] = self.seen_before_queried.get(label, 0) + 1

    def record_query(self, label, processed_idx: int, query_count: int):
        if label not in self.first_queried:
            self.first_queried[label] = processed_idx
            self.first_queried_query_count[label] = query_count
            # The appearance for the discovering point itself has already incremented seen_before_queried.
            # "How many we see before they are queried" should exclude the point that caused the query.
            pre = self.seen_before_queried.get(label, 0)
            self.seen_before_queried[label] = max(0, pre - 1)
        # Always count every query for the enrichment calculation ("number of queries wherein a class was found").
        self.queries_per_class[label] = self.queries_per_class.get(label, 0) + 1

    def get_seen_count(self) -> int:
        return len(self.first_appearance)

    def get_discovered_count(self) -> int:
        return len(self.first_queried)

    def get_total_seen(self, label: str) -> int:
        return self.total_seen.get(label, 0)

    def get_seen_before_queried(self, label: str) -> int:
        return self.seen_before_queried.get(label, 0)

    def get_queries_for_class(self, label: str) -> int:
        return self.queries_per_class.get(label, 0)

    def get_first_query_count(self, label: str) -> int:
        return self.first_queried_query_count.get(label,0)
# ====================== CONFIG ======================
# Same "near-zero query" defaults that work well for the spectrogram frontend.
# Perch embeddings are semantic (much cleaner clusters than pooled spectrograms),
# so these values are a great starting point. You can raise K_COMP_CLUST or
# DATA_WINDOW_SIZE if you want more aggressive merging / longer memory.
KAPPA = 1.15
DATA_WINDOW_SIZE = 250
K_COMP_CLUST = 5
QS_VAR = 0
REL_PROC_VAR = 0
VERBOSE_FLAGS = []

NUM_POINTS_TO_PROCESS = 5000     # Same quick-test default as main_spectrogram.py; override with --num-points
N_REL_CLASSES = 5

def main():
    parser = argparse.ArgumentParser(description="Run A_RED (untouched) over Perch v2 embeddings.")
    parser.add_argument("--num-points", type=int, default=None, help="Override NUM_POINTS_TO_PROCESS (use -1 for all in the CSV order after shuffle)")
    parser.add_argument("--label-column", type=str, default=None, help="e.g. scientific_name or class_name")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffle (use CSV order)")
    args = parser.parse_args()

    global NUM_POINTS_TO_PROCESS
    if args.num_points is not None:
        NUM_POINTS_TO_PROCESS = args.num_points
    label_col = args.label_column or "class_name"
    do_shuffle = not args.no_shuffle

    print("=== Perch A_RED Implementation ===")
    print("Using Google Perch v2 embeddings (semantic bird vocalization space; dim auto-detected)")
    print("Data: prepared_5s_clips + 5sSpectrograms_tensors/train_5s_spectrograms_linux.csv")
    print(f"Parameters: kappa={KAPPA}, window={DATA_WINDOW_SIZE}, k_comp={K_COMP_CLUST}, qs_var={QS_VAR}")
    print()

    # Initialize data stream
    # - If a full (or large enough) perch_embeddings.npy exists next to the CSV, it is used (fast).
    # - Otherwise only the embeddings needed for NUM_POINTS_TO_PROCESS (+shuffle) are computed live.
    data_stream = PerchDataStream(
        csv_path=str(project_root / "5sSpectrograms_tensors" / "train_5s_spectrograms_linux.csv"),
        embeddings_path=str(project_root / "5sSpectrograms_tensors" / "perch_embeddings.npy"),
        max_samples=NUM_POINTS_TO_PROCESS if NUM_POINTS_TO_PROCESS > 0 else None,
        shuffle=do_shuffle,
        seed=552,
        label_column=label_col,   # change to "scientific_name" for species-level experiments
        live_batch_size=4,           # safe for 8 GB VRAM during live preload
    )

    # Discovery counter (always on for the basic runner; mirrors the pattern from the spectrogram
    # shifting-kappa mains). Passed to the oracle so record_query fires on every label reveal.
    discovery_counter = ClassDiscoveryCounter()
    oracle = PerchOracle(data_stream, discovery_tracker=discovery_counter)

    ared = ARED(
        oracle=oracle,
        kappa=KAPPA,
        data_window_size=DATA_WINDOW_SIZE,
        k_comparison_clusters=K_COMP_CLUST,
        QS_VAR=QS_VAR,
        REL_PROC_VAR=REL_PROC_VAR,
        VERBOSE_FLAGS=VERBOSE_FLAGS,
    )
    print(f"Initialized A_RED with kappa={KAPPA}, QS_VAR={QS_VAR}, REL_PROC_VAR={REL_PROC_VAR}")
    print("Oracle returns relevance=False for ALL classes (same policy as the spectrogram runner).")
    print("Queries will be near-zero after the first few points (only true anomalies query).")
    print()

    print(f"Starting A_RED on {data_stream.n_samples} Perch-embedded samples...")
    start_time = time.time()

    # First point (always creates the initial cluster + one query)
    try:
        first_point = data_stream.stream_new_data_point()
        # Record appearance using *correct source row* (stream_counter-1) so any future skips
        # don't mis-map labels. Use processed_idx=0 to match ARED abs_idx space used by oracle for record_query.
        first_source_idx = data_stream.stream_counter - 1
        first_label = data_stream.get_true_label_for_idx(first_source_idx)
        discovery_counter.record_appearance(first_label, 0)

        ared.process_first_point(first_point)
        print("First point processed. Initial cluster created.")
        initial_cluster = ared.subspace_partition.cluster_list[0]
        print(f"Initial cluster comp_distance: {initial_cluster.comp_distance:.4f}")
    except Exception as e:
        print(f"Error on first point: {e}")
        return

    points_processed = 1
    queries = 1

    while (NUM_POINTS_TO_PROCESS == -1 or points_processed < NUM_POINTS_TO_PROCESS) and data_stream.get_remaining_num_points() > 0:
        try:
            data_point = data_stream.stream_new_data_point()

            # Record appearance for *every* point using proper source row for label (handles any skips)
            # and the ARED processed_idx (current points_processed value == this point's abs_idx in ARED).
            current_idx = points_processed
            source_idx = data_stream.stream_counter - 1
            label = data_stream.get_true_label_for_idx(source_idx)
            discovery_counter.record_appearance(label, current_idx)

            ared.process_point(data_point)
            points_processed += 1

            if points_processed % 20 == 0 or points_processed == NUM_POINTS_TO_PROCESS:
                current_queries = len(ared.labeled_data.abs_idx_array)
                print(f"Processed {points_processed:,}/{data_stream.n_samples:,} | "
                      f"Queries: {current_queries} | "
                      f"Known classes: {len(ared.subspace_partition.set_of_known_labels)} | "
                      f"Query rate: {current_queries/points_processed*100:.1f}%")
        except Exception as e:
            print(f"Error at point {points_processed}: {e}")
            break

    total_time = time.time() - start_time
    final_queries = len(ared.labeled_data.abs_idx_array)

    print("\n" + "=" * 60)
    print("A_RED (Perch) COMPLETE")
    print("=" * 60)
    print(f"Points processed: {points_processed:,}")
    print(f"Queries made: {final_queries} ({final_queries/points_processed*100:.2f}% of points)")
    print(f"Known classes discovered: {len(ared.subspace_partition.set_of_known_labels)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Queries per second: {final_queries/total_time:.2f}")

    # Class Discovery Report (first appearance in processed stream vs. first oracle query + volume seen before query).
    # Also includes the enrichment metric: (queries_for_class / total_queries) / (total_class / N) vs random chance.
# After the main loop, replace the "Class Discovery Report" section with:

    print("\nClass Discovery Report (vs Random Baseline):")
    seen = discovery_counter.get_seen_count()
    disc = discovery_counter.get_discovered_count()
    N = points_processed
    Q = final_queries
    print(f"  Classes seen: {seen} | Discovered: {disc} | Total queries: {Q}")
    print(f"  Dataset size processed: {N:,} points")
    print()

    if seen > 0:
        print("  Per-class (sorted by appearance order):")
        items = []
        for label, appear_idx in discovery_counter.first_appearance.items():
            discover_idx = discovery_counter.first_queried.get(label, None)
            query_count = discovery_counter.get_first_query_count(label)
            tot = discovery_counter.get_total_seen(label)
            pre = discovery_counter.get_seen_before_queried(label)
            q_c = discovery_counter.get_queries_for_class(label)
            
            prevalence = tot / N if N > 0 else 0
            expected_random = 1.0 / prevalence if prevalence > 0 else float('inf')
            
            if query_count is not None:
                actual_queries = query_count  # 1-based for readability
                lift = expected_random / actual_queries if actual_queries > 0 else float('inf')
                lift_str = f"{lift:.1f}x"
            else:
                actual_queries = "never"
                lift_str = "N/A"
            
            items.append((appear_idx, discover_idx, label, actual_queries, expected_random, lift_str, prevalence, pre, tot, q_c))
        
        items.sort(key=lambda x: x[0])
        
        for appear_idx, discover_idx, label, act_q, exp_rand, lift_str, prev, pre, tot, q_c in items:
            print(f"  {label:8} | prevelance={prev*100:5.2f}% | "
                  f"seen index={appear_idx} | queried index={discover_idx} | discovery query={act_q} | expected random≈{exp_rand:.0f} | "
                  f"lift={lift_str} | seen before={pre} | (total={tot})")
    
    print("\nInterpretation:")
    print("  • lift > 1.0  = discovered faster than random (good)")
    print("  • lift >> 1.0 = strong active discovery of rare classes")
    print("  • For dominant class (~99%), lift near 1.0 is expected")
    print("  • Geometric model: random expected queries = 1 / prevalence")


    print("\nCluster Summary:")
    for i, cluster in enumerate(ared.subspace_partition.cluster_list):
        if cluster.label is not None:
            n_l = len(cluster.l_pts)
            n_o = len(cluster.o_pts)
            print(f"  Cluster {i}: label={cluster.label}, relevance={cluster.relevance}, "
                  f"l_pts={n_l}, o_pts={n_o}, comp_dist={cluster.comp_distance:.4f}")

    print(f"\nOracle queries: {oracle.get_query_count()}")

    stats = Stats(ared)
    print("\nDetailed stats available in Stats object.")

    print("\nNote: True labels were ONLY used by the Oracle when A_RED decided to query.")
    print("Relevance=False for every class (including Aves) → no re-querying after discovery.")
    print("This satisfies: query *only* on anomaly + 'once classes are discovered we do not query again'.")
    print("Perch embeddings give a strong semantic space — expect good separation of bird vs. non-bird")
    print("and among bird species when using label_column='scientific_name'.")


if __name__ == "__main__":
    main()