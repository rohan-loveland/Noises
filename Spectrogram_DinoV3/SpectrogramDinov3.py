"""
Spectrogram DinoV3 (DINOv2) Runner
Uses DINOv2-small embeddings (~384-dim semantic features) from existing .npy Mel-spectrograms.
No audio regeneration needed. Provides massive speedup + better accuracy vs raw/high-dim spectrograms.
Separate folder/process from Spectrogram_A_RED.
"""
import sys
import time
from pathlib import Path
import numpy as np
from collections import Counter

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "A_REDimplementation" / "A_RED"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # for Spectrogram_A_RED modules
sys.path.insert(0, str(Path(__file__).parent))  # for local Dinov3DataStream

from Dinov3DataStream import Dinov3DataStream
from Spectrogram_A_RED.SpectrogramDataStream import SpectrogramOracle
from A_RED import ARED
from Stats import Stats

# ====================== CONFIG ======================
# A_RED Parameters (tuned for low-dim Dino embeddings)
KAPPA = 2                    # Adjusted for semantic embeddings (fewer anomalies expected)
DATA_WINDOW_SIZE = 1000      # Larger window viable with low-dim data
K_COMP_CLUST = 5
QS_VAR = 0
REL_PROC_VAR = 0
VERBOSE_FLAGS = []

NUM_POINTS_TO_PROCESS = 500   # Small for quick verification + model download
N_REL_CLASSES = 5

def main():
    print("=== Spectrogram DinoV3 (DINOv2) A_RED Implementation ===")
    print("Using DINOv2 embeddings on existing .npy spectrograms (no regeneration needed)")
    print(f"Parameters: kappa={KAPPA}, window={DATA_WINDOW_SIZE}, k_comp={K_COMP_CLUST}, qs_var={QS_VAR}")
    print("DinoV2 provides 384-dim semantic features -> faster NN queries, better clustering.")
    print()
    
    # Initialize DinoV2 data stream (embeddings from existing tensors)
    data_stream = Dinov3DataStream(
        csv_path="5sSpectrograms_tensors/train_5s_spectrograms.csv",
        tensor_dir="5sSpectrograms_tensors",
        max_samples=NUM_POINTS_TO_PROCESS if NUM_POINTS_TO_PROCESS > 0 else None,
        shuffle=True,
        seed=69,
        dino_model_name="facebook/dinov2-small"
    )
    
    # Hidden tracker (reused from discovery work)
    discovery_tracker = data_stream.discovery_tracker if hasattr(data_stream, 'discovery_tracker') else None
    
    # Oracle (reused exactly; passes tracker for hidden metrics)
    oracle = SpectrogramOracle(data_stream, discovery_tracker=discovery_tracker)
    
    # Initialize A_RED (works with any flattened vector input)
    ared = ARED(
        oracle=oracle,
        kappa=KAPPA,
        data_window_size=DATA_WINDOW_SIZE,
        k_comparison_clusters=K_COMP_CLUST,
        QS_VAR=QS_VAR,
        REL_PROC_VAR=REL_PROC_VAR,
        VERBOSE_FLAGS=VERBOSE_FLAGS
    )
    print(f"Initialized A_RED with DinoV2 embeddings (dim={data_stream.embed_dim if hasattr(data_stream, 'embed_dim') else 'N/A'})")
    print("Oracle returns relevance=False for ALL classes (no re-querying after discovery).")
    
    print(f"Starting A_RED on {data_stream.n_samples} Dino-embedded samples...")
    start_time = time.time()
    
    # Process first point to initialize
    try:
        first_point = data_stream.stream_new_data_point()
        ared.process_first_point(first_point)
        print("First point processed. Initial cluster created.")
    except Exception as e:
        print(f"Error on first point: {e}")
        return
    
    # Process remaining points
    points_processed = 1
    queries = 1
    
    while (NUM_POINTS_TO_PROCESS == -1 or points_processed < NUM_POINTS_TO_PROCESS) and data_stream.get_remaining_num_points() > 0:
        try:
            data_point = data_stream.stream_new_data_point()
            ared.process_point(data_point)
            points_processed += 1
            
            if points_processed % 50 == 0 or points_processed == NUM_POINTS_TO_PROCESS:
                current_queries = len(ared.labeled_data.abs_idx_array)
                print(f"Processed {points_processed:,}/{data_stream.n_samples:,} points | Queries: {current_queries} | "
                      f"Known classes: {len(ared.subspace_partition.set_of_known_labels)} | "
                      f"Query rate: {current_queries/points_processed*100:.1f}%")
                
        except Exception as e:
            print(f"Error at point {points_processed}: {e}")
            break
    
    total_time = time.time() - start_time
    final_queries = len(ared.labeled_data.abs_idx_array)
    
    # Stats and Results
    print("\n" + "="*70)
    print("DinoV3 A_RED COMPLETE")
    print("="*70)
    print(f"Points processed: {points_processed:,}")
    print(f"Queries made: {final_queries} ({final_queries/points_processed*100:.2f}% of points)")
    print(f"Known classes discovered: {len(ared.subspace_partition.set_of_known_labels)}")
    print(f"Total time: {total_time:.2f}s")
    
    # Cluster summary
    print("\nCluster Summary (Dino embeddings):")
    for i, cluster in enumerate(ared.subspace_partition.cluster_list):
        if cluster.label is not None:
            n_l = len(cluster.l_pts)
            n_o = len(cluster.o_pts)
            print(f"  Cluster {i}: label={cluster.label}, relevance={cluster.relevance}, "
                  f"l_pts={n_l}, o_pts={n_o}")
    
    print(f"\nOracle queries: {oracle.get_query_count()}")
    
    # Hidden discovery tracking report (reused)
    if discovery_tracker:
        print("\n" + "="*60)
        print("DISCOVERY TRACKER REPORT (DinoV3 - HIDDEN FROM ALGORITHM)")
        print("="*60)
        report = discovery_tracker.get_discovery_report()
        print(f"Total classes seen: {report['total_classes_seen']}")
        print(f"Total classes first-queried: {report['total_classes_queried']}")
        print(f"Total points examined: {report['total_points_examined']}")
        print(f"Total queries: {report['total_queries']}")
        
        print("\nPer-class first discovery (with Dino embeddings):")
        for cls, info in report.get('classes', {}).items():
            seen = info.get('first_seen_query_num')
            queried = info.get('first_queried_query_num')
            print(f"  {cls}: first_seen_query#{seen} (idx={info.get('first_seen_stream_idx')})")
            if queried is not None:
                print(f"           first_queried_query#{queried} (idx={info.get('first_queried_stream_idx')})")
                print(f"           queries_before_query: {info.get('queries_before_first_query', 'N/A')}")
            print("  ---")
        
        discovery_tracker.save_report("dinov3_discovery_report.json")
    
    stats = Stats(ared)
    print("\nDetailed stats available in Stats object.")
    print("\nNote: DinoV2 embeddings used on *existing* .npy files (feasible, no regeneration).")
    print("Semantic features should improve accuracy and reduce queries vs raw spectrograms.")
    print("New process isolated in Spectrogram_DinoV3/ folder.")

if __name__ == "__main__":
    main()
