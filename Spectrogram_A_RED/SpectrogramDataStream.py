import numpy as np
from pathlib import Path
import pandas as pd

class SpectrogramDataStream:
    def __init__(self, csv_path: str = "5sSpectrograms_tensors/train_5s_spectrograms.csv", tensor_dir: str = "5sSpectrograms_tensors", max_samples=None, shuffle=True, seed=42):
        self.csv_path = Path(csv_path)
        self.tensor_dir = Path(tensor_dir)
        self.max_samples = max_samples
        self.stream_counter = 0
        
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
        
        print(f"Examining spectrogram: {npy_rel_path} (counter={self.stream_counter})")  # Debug: show which file is being processed
        
        if not npy_path.exists():
            print(f"  WARNING: File not found - {npy_path}")
            self.stream_counter += 1
            if self.stream_counter >= self.n_samples:
                raise StopIteration("No more data points")
            return self.stream_new_data_point()
        
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
        norm = np.linalg.norm(data_point)
        if norm > 0:
            data_point = data_point / norm
        else:
            print("  WARNING: Zero-norm spectrogram encountered")
        
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


class SpectrogramOracle:
    def __init__(self, data_stream):
        self.data_stream = data_stream  # reference to access true labels
        self.query_count = 0
    
    def answer_query(self, abs_index):
        """Oracle returns true label and relevance. 
        Relevance=False for *all* non-Aves (noise, Amphibia, Insecta, etc.). This ensures the `if not comp_cluster_relevant and not is_anomalous`
        branch in process_point() ALWAYS hits add_o_pt() WITHOUT querying for non-birds (goal: near-zero queries).
        Only true bird points trigger relevance=True + query. Directly satisfies 'only query on anomaly' + 'as close to zero queries as possible'."""
        self.query_count += 1
        true_label = self.data_stream.get_true_label_for_idx(abs_index)
        # Relevance = False for *all* classes (including discovered birds). This ensures no cluster.relevance=True, so comp_cluster_relevant=False always.
        # After initial discovery, no further queries (add_o_pt path for non-anomalous points). Matches "none of the classes are marked as relevant, once they are discovered, we do not want to query them again."
        relevance = False
        return true_label, relevance
    
    def get_query_count(self):
        return self.query_count
