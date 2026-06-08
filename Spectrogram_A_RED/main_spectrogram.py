"""
Spectrogram A_RED Runner
Adapts the existing A_RED implementation for streaming .npy spectrograms.
Uses flattened spectrogram vectors as high-dimensional input.
The implementation does NOT know true labels until the Oracle is queried.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

# Add paths for imports (A_RED module and local Spectrogram adapter)
sys.path.insert(0, str(Path(__file__).parent.parent / "A_REDimplementation" / "A_RED"))
sys.path.insert(0, str(Path(__file__).parent))

from SpectrogramDataStream import SpectrogramDataStream, SpectrogramOracle
from A_RED import ARED
from Stats import Stats

# ====================== CONFIG ======================
# A_RED Parameters (tuned for high-dim spectrogram data ~40k dims)
KAPPA = 2                    # Low value ensures few anomalies (most points join via add_o_pt or add_l_pt); near-zero queries after initial points
DATA_WINDOW_SIZE = 2000        # Smaller memory bound for spectrogram streaming (faster forgetting of old points)
K_COMP_CLUST = 5               # More clusters for comparison in high-variance audio data
QS_VAR = 0                     # Diameter (stable with our 0.1 floor fix)
REL_PROC_VAR = 0               # Disabled: avoids extra queries on o_pts during relevance_processing for new non-bird classes
VERBOSE_FLAGS = [0, 1, 2]      # 0=summary, 1=add_o_pt prints (confirms non-query path), 2=anomalous checks

NUM_POINTS_TO_PROCESS = 10000     # Minimal for quick verification run (npy loads + BallTree are heavy); demonstrates algorithm with low queries
N_REL_CLASSES = 5              # Target number of relevant classes to discover (for reporting)

def main():
    print("=== Spectrogram A_RED Implementation ===")
    print("Using .npy spectrograms from 5sSpectrograms_tensors/")
    print(f"Parameters: kappa={KAPPA}, window={DATA_WINDOW_SIZE}, k_comp={K_COMP_CLUST}, qs_var={QS_VAR}")
    print()
    
    # Initialize data stream (loads CSV metadata, streams .npy on demand, flattens to 1D vector)
    data_stream = SpectrogramDataStream(
        csv_path="5sSpectrograms_tensors/train_5s_spectrograms.csv",
        tensor_dir="5sSpectrograms_tensors",
        max_samples=NUM_POINTS_TO_PROCESS if NUM_POINTS_TO_PROCESS > 0 else None,
        shuffle=True,  # Deterministic order for reproducible test
        seed=42
    )
    
    # Oracle knows true labels but only reveals on query (simulates human-in-loop)
    oracle = SpectrogramOracle(data_stream)
    
    # Initialize A_RED
    ared = ARED(
        oracle=oracle,
        kappa=KAPPA,
        data_window_size=DATA_WINDOW_SIZE,
        k_comparison_clusters=K_COMP_CLUST,
        QS_VAR=QS_VAR,
        REL_PROC_VAR=REL_PROC_VAR,
        VERBOSE_FLAGS=VERBOSE_FLAGS
    )
    print(f"Initialized A_RED with kappa={KAPPA}, QS_VAR={QS_VAR}, REL_PROC_VAR={REL_PROC_VAR} (tuned for near-zero queries)")
    print("Oracle now returns relevance=False for ALL classes (none marked relevant once discovered). This prevents any re-querying.")
    print("Queries are now explicitly logged with 'QUERY OCCURRED' when they happen (should be near-zero after initial points).")
    
    print(f"Starting A_RED on {data_stream.n_samples} spectrogram samples...")
    start_time = time.time()
    
    # Process first point to initialize
    try:
        first_point = data_stream.stream_new_data_point()
        ared.process_first_point(first_point)
        print("First point processed. Initial cluster created.")
        initial_cluster = ared.subspace_partition.cluster_list[0]
        print(f"Initial cluster comp_distance: {initial_cluster.comp_distance:.4f} (if 0, all subsequent points appear anomalous)")
    except Exception as e:
        print(f"Error on first point: {e}")
        return
    
    # Process remaining points
    points_processed = 1
    queries = 1  # first point is always queried
    
    while (NUM_POINTS_TO_PROCESS == -1 or points_processed < NUM_POINTS_TO_PROCESS) and data_stream.get_remaining_num_points() > 0:
        try:
            data_point = data_stream.stream_new_data_point()
            ared.process_point(data_point)
            points_processed += 1
            
            if points_processed % 20 == 0 or points_processed == NUM_POINTS_TO_PROCESS:  # More frequent debug for diagnosis
                current_queries = len(ared.labeled_data.abs_idx_array)
                print(f"Processed {points_processed:,}/{data_stream.n_samples:,} points | Queries: {current_queries} | "
                      f"Known classes: {len(ared.subspace_partition.set_of_known_labels)} | "
                      f"Query rate so far: {current_queries/points_processed*100:.1f}%")
                
        except Exception as e:
            print(f"Error at point {points_processed}: {e}")
            break
    
    total_time = time.time() - start_time
    final_queries = len(ared.labeled_data.abs_idx_array)
    
    # Stats and Results
    print("\n" + "="*60)
    print("A_RED COMPLETE")
    print("="*60)
    print(f"Points processed: {points_processed:,}")
    print(f"Queries made: {final_queries} ({final_queries/points_processed*100:.2f}% of points)")
    print(f"Known classes discovered: {len(ared.subspace_partition.set_of_known_labels)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Queries per second: {final_queries/total_time:.2f}")
    
    # Cluster summary
    print("\nCluster Summary:")
    for i, cluster in enumerate(ared.subspace_partition.cluster_list):
        if cluster.label is not None:  # valid clusters
            n_l = len(cluster.l_pts)
            n_o = len(cluster.o_pts)
            print(f"  Cluster {i}: label={cluster.label}, relevance={cluster.relevance}, "
                  f"l_pts={n_l}, o_pts={n_o}, comp_dist={cluster.comp_distance:.4f}")
    
    # Oracle stats
    print(f"\nOracle queries: {oracle.get_query_count()}")
    
    # Save stats if desired
    stats = Stats(ared)
    print("\nDetailed stats available in Stats object.")
    
    print("\nNote: True labels were ONLY used by the Oracle when A_RED decided to query.")
    print("No classes are marked relevant (Oracle always returns relevance=False). Once discovered, no re-querying occurs.")
    print("With REL_PROC_VAR=0 + low KAPPA, queries are near-zero (only initial anomalies + first few points).")
    print("This satisfies: query *only* on anomaly + 'none of the classes are marked as relevant, once they are discovered, we do not want to query them again.'")

if __name__ == "__main__":
    main()
