import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
import json
from datetime import datetime

class SpectrogramDataStream:
    def __init__(self, csv_path: str = "5sSpectrograms_tensors/train_5s_spectrograms.csv", tensor_dir: str = "5sSpectrograms_tensors", max_samples=None, shuffle=True, seed=42):
        self.csv_path = Path(csv_path)
        self.tensor_dir = Path(tensor_dir)
        self.max_samples = max_samples
        self.stream_counter = 0
        
        # Hidden discovery tracker (for testing/reference only - not visible to ARED algorithm)
        self.discovery_tracker = DiscoveryTracker()
        
        # Load metadata
        self.df = pd.read_csv(self.csv_path)
        if 'spectrogram_npy_path' not in self.df.columns:
            raise ValueError("CSV must contain 'spectrogram_npy_path' column")
        
        # Optional shuffle for streaming order (simulates random arrival)
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # High-dim spectrograms (~40k dims). Normalization helps, but for best performance with A_RED we can add
        # dimensionality reduction (e.g. SparseRandomProjection like in the parking-lot example, or mean-pooling).
        # DINOv2 embeddings would be excellent (semantic features, much lower dim ~768), but requires model download/inference.
        
        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)
        
        self.n_samples = len(self.df)
        print(f"Loaded {self.n_samples:,} spectrogram samples for streaming")
        print(f"Example path: {self.df.iloc[0]['spectrogram_npy_path']}")
        
    def stream_new_data_point(self):
        """Load next .npy spectrogram as flattened vector"""
        if self.stream_counter >= self.n_samples:
            raise StopIteration("No more data points")
        
        row = self.df.iloc[self.stream_counter]
        npy_rel_path = row['spectrogram_npy_path']
        npy_path = self.tensor_dir / npy_rel_path
        
        #print(f"Examining spectrogram: {npy_rel_path} (counter={self.stream_counter})")  # Debug: show which file is being processed
        
        if not npy_path.exists():
            print(f"  WARNING: File not found - {npy_path}")
            self.stream_counter += 1
            if self.stream_counter >= self.n_samples:
                raise StopIteration("No more data points")
            return self.stream_new_data_point()
        
        # Record for hidden discovery tracking BEFORE algorithm sees the point
        true_label = self.get_true_label_for_idx(self.stream_counter)
        self.discovery_tracker.record_examined_point(true_label, self.stream_counter)
        
        # Load as float32 (keep 2D for optional pooling)
        spec = np.load(npy_path).astype(np.float32)
        
        # Mean-pool time axis (handles variable lengths 312/313) to ~2.5k dims. Critical fix for high-dim curse:
        # - Slows KDTree/BallTree NN queries on raw 40k-dim data.
        # - Makes distances stable post-L2 norm.
        # DINOv2 (semantic ~768-dim) would be ideal upgrade (see comment in main).
        if spec.ndim == 2:
            time_steps = spec.shape[1]
            pool_size = 16
            num_pools = time_steps // pool_size
            if num_pools > 0:
                spec = np.mean(spec[:, :num_pools*pool_size].reshape(spec.shape[0], num_pools, pool_size), axis=2)
            # fallback: keep full if too short (rare)
        
        data_point = spec.flatten()
        
        # L2 normalization (unit vector) - critical for high-dim data
        #norm = np.linalg.norm(data_point)
        #if norm > 0:
        #    data_point = data_point / norm
        #else:
        #    print("  WARNING: Zero-norm spectrogram encountered")
        
        self.stream_counter += 1
        return data_point
    
    def get_remaining_num_points(self):
        return self.n_samples - self.stream_counter
    
    def get_true_label_for_idx(self, stream_idx):
        """For oracle or evaluation only - program should not use during normal streaming"""
        if stream_idx >= len(self.df):
            return None
        row = self.df.iloc[stream_idx]
        return row.get('class_name', row.get('primary_label', 'unknown'))


class DiscoveryTracker:
    """
    Tracks when classes are FIRST SEEN in the data stream (every point examined)
    and when they are FIRST QUERIED by the algorithm.
    This info is HIDDEN from the ARED algorithm - purely for post-run testing/analysis.
    Does NOT affect any decisions or behavior.
    """
    def __init__(self):
        self.first_seen = {}          # class -> (query_number_when_first_seen, stream_idx, timestamp)
        self.first_queried = {}       # class -> (query_number_when_first_queried, stream_idx, timestamp)
        self.query_counter = 0        # Increments only on actual oracle queries
        self.seen_counter = 0         # Increments on every data point examined
        self.class_to_first_seen_query_num = {}  # For easy lookup: class -> first query# when seen
    
    def record_examined_point(self, true_label: str, stream_idx: int):
        """Called on EVERY data point (before algorithm decides to query or not)."""
        self.seen_counter += 1
        if true_label not in self.first_seen:
            self.first_seen[true_label] = (self.seen_counter, stream_idx, datetime.now().isoformat())
            self.class_to_first_seen_query_num[true_label] = self.seen_counter
            # print(f"[TRACKER] First seen: {true_label} at stream idx {stream_idx} (query #{self.seen_counter})")
    
    def record_query(self, true_label: str, stream_idx: int):
        """Called only when oracle is queried (i.e. algorithm decides to query)."""
        self.query_counter += 1
        if true_label not in self.first_queried:
            self.first_queried[true_label] = (self.query_counter, stream_idx, datetime.now().isoformat())
            # print(f"[TRACKER] First queried: {true_label} at stream idx {stream_idx} (query #{self.query_counter})")
    
    def get_discovery_report(self):
        """Returns comprehensive report for testing/reference. Sorted by first seen."""
        report = {}
        all_classes = set(self.first_seen.keys()) | set(self.first_queried.keys())
        
        for cls in sorted(all_classes):
            seen_info = self.first_seen.get(cls, (None, None, None))
            queried_info = self.first_queried.get(cls, (None, None, None))
            report[cls] = {
                "first_seen_query_num": seen_info[0],
                "first_seen_stream_idx": seen_info[1],
                "first_queried_query_num": queried_info[0],
                "first_queried_stream_idx": queried_info[1],
                "queries_before_first_query": (queried_info[0] - seen_info[0] if seen_info[0] and queried_info[0] else None),
                "first_seen_timestamp": seen_info[2],
                "first_queried_timestamp": queried_info[2]
            }
        
        summary = {
            "total_classes_seen": len(self.first_seen),
            "total_classes_queried": len(self.first_queried),
            "total_points_examined": self.seen_counter,
            "total_queries": self.query_counter,
            "classes": report
        }
        return summary
    
    def save_report(self, filepath="discovery_report.json"):
        """Save report to JSON for easy analysis."""
        report = self.get_discovery_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[TRACKER] Saved discovery report to {filepath}")
        return report


class SpectrogramOracle:
    def __init__(self, data_stream, discovery_tracker=None):
        self.data_stream = data_stream  # reference to access true labels
        self.query_count = 0
        self.discovery_tracker = discovery_tracker  # hidden reference for testing only
    
    def answer_query(self, abs_index):
        """Oracle returns true label and relevance. 
        Relevance=False for *all* non-Aves (noise, Amphibia, Insecta, etc.). This ensures the `if not comp_cluster_relevant and not is_anomalous`
        branch in process_point() ALWAYS hits add_o_pt() WITHOUT querying for non-birds (goal: near-zero queries).
        Only true bird points trigger relevance=True + query. Directly satisfies 'only query on anomaly' + 'as close to zero queries as possible'."""
        self.query_count += 1
        true_label = self.data_stream.get_true_label_for_idx(abs_index)
        
        # Record query for hidden discovery tracking (does NOT affect algorithm)
        if self.discovery_tracker:
            self.discovery_tracker.record_query(true_label, abs_index)
        
        # Relevance = False for *all* classes (including discovered birds). This ensures no cluster.relevance=True, so comp_cluster_relevant=False always.
        # After initial discovery, no further queries (add_o_pt path for non-anomalous points). Matches "none of the classes are marked as relevant, once they are discovered, we do not want to query them again."
        relevance = False
        return true_label, relevance
    
    def get_query_count(self):
        return self.query_count
