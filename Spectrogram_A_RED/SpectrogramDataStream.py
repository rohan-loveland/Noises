import numpy as np
from pathlib import Path
import pandas as pd
import torchvision.transforms as transforms

class SpectrogramDataStream:
    def __init__(self, csv_path: str = "5sSpectrograms_tensors/train_5s_spectrograms_linux.csv", tensor_dir: str = "5sSpectrograms_tensors", max_samples=None, shuffle=True, seed=42, label_column: str = "class_name"):
        self.csv_path = Path(csv_path)
        self.tensor_dir = Path(tensor_dir)
        self.max_samples = max_samples
        self.stream_counter = 0
        self.label_column = label_column   # which column to use for true labels (e.g. "class_name" for broad, "scientific_name" for species)
        
        # Load metadata
        self.df = pd.read_csv(self.csv_path)
        if 'spectrogram_npy_path' not in self.df.columns:
            raise ValueError("CSV must contain 'spectrogram_npy_path' column")
        
        # Optional shuffle for streaming order (simulates random arrival)
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # High-dim spectrograms (~40k dims raw). We preload + pool here for speed.
        # Preloading all tensors at startup removes thousands of per-point np.load + pandas + pooling costs
        # that cause the "slows to a crawl after a few thousand points" behavior.
        # Pooling reduces to ~2.5k dims. L2 norm. DINOv2 embeddings (~768d semantic) would be an even bigger win
        # for both speed (lower D in BallTree/dists) and cluster quality (fewer spurious singletons).
        
        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)
        
        # === PRELOAD + PROCESS ALL VECTORS INTO MEMORY (major perf win) ===
        # We filter to only files that actually exist, apply the mean-pool + L2 once up front,
        # and cache the resulting vectors + labels. stream_new_data_point then becomes a fast list lookup.
        # Memory: ~10KB per point after pooling (2560*float32). 20k pts ≈ 200MB — fine. For 200k use --num-points to limit.
        print("Preloading spectrogram tensors into memory for fast streaming (eliminates per-point disk I/O)...")
        processed = []
        labels = []
        valid_rows = []
        pool_size = 16

        for idx, row in self.df.iterrows():
            npy_rel_path = row['spectrogram_npy_path']
            npy_path = self.tensor_dir / npy_rel_path
            if not npy_path.exists():
                continue  # skip missing (preserves old skip behavior but done once at start)

            spec = np.load(npy_path).astype(np.float32)
            if spec.ndim == 2:
                time_steps = spec.shape[1]
                num_pools = time_steps // pool_size
                if num_pools > 0:
                    spec = np.mean(spec[:, :num_pools*pool_size].reshape(spec.shape[0], num_pools, pool_size), axis=2)

            data_point = spec.flatten()
            norm = np.linalg.norm(data_point)
            if norm > 0:
                data_point = data_point / norm
            else:
                data_point = data_point  # keep zero vector; rare

            # Resolve label using same fallback logic as before (but done once)
            lab = None
            if label_column in row and pd.notna(row[label_column]):
                lab = str(row[label_column])
            if lab is None or lab == "nan":
                for col in (label_column, "scientific_name", "common_name", "class_name", "primary_label"):
                    if col in row and pd.notna(row[col]):
                        val = row[col]
                        lab = str(val) if not pd.isna(val) else "unknown"
                        break
            if lab is None or lab == "nan":
                lab = "unknown"

            processed.append(data_point.astype(np.float32))
            labels.append(lab)
            valid_rows.append(idx)

        if not processed:
            raise RuntimeError("No valid spectrogram .npy files found after filtering missing paths.")

        self.labels_cache = labels
        # Keep a minimal df view for any other uses (rare); n_samples is now the valid count
        if valid_rows:
            self.df = self.df.iloc[valid_rows].reset_index(drop=True)
        self.n_samples = len(processed)

        # Store as contiguous ndarray when possible (uniform dim after pooling) for fastest access in hot loop
        if processed:
            try:
                self.processed_vectors = np.stack(processed).astype(np.float32)
            except Exception:
                self.processed_vectors = processed  # ragged fallback (rare)
        else:
            self.processed_vectors = []

        print(f"Preloaded {self.n_samples:,} valid spectrogram samples (after dropping missing files).")
        if self.n_samples > 0:
            print(f"Example path: {self.df.iloc[0]['spectrogram_npy_path'] if len(self.df) > 0 else 'N/A'}")
        print(f"Using label column: '{self.label_column}' (e.g. broad class vs. scientific_name for species)")
        
    def stream_new_data_point(self):
        """Return next pre-processed (pooled + L2-normalized) vector from in-memory cache.
        This is extremely fast compared to repeated np.load + pooling per point."""
        if self.stream_counter >= self.n_samples:
            raise StopIteration("No more data points")
        
        data_point = self.processed_vectors[self.stream_counter]
        self.stream_counter += 1
        return data_point
    
    def get_remaining_num_points(self):
        return self.n_samples - self.stream_counter
    
    def get_true_label_for_idx(self, stream_idx):
        """Fast path: labels pre-cached at preload time. No pandas iloc in the hot loop."""
        if stream_idx >= len(self.labels_cache):
            return None
        return self.labels_cache[stream_idx]


class SpectrogramOracle:
    def __init__(self, data_stream, discovery_tracker=None):
        self.data_stream = data_stream  # reference to access true labels
        self.query_count = 0
        self.discovery_tracker = discovery_tracker  # hidden tracker for testing only
    
    def answer_query(self, abs_index):
        """Oracle returns true label and relevance. 
        Relevance=False for *all* non-Aves (noise, Amphibia, Insecta, etc.). This ensures the `if not comp_cluster_relevant and not is_anomalous`
        branch in process_point() ALWAYS hits add_o_pt() WITHOUT querying for non-birds (goal: near-zero queries).
        Only true bird points trigger relevance=True + query. Directly satisfies 'only query on anomaly' + 'as close to zero queries as possible'."""
        self.query_count += 1
        true_label = self.data_stream.get_true_label_for_idx(abs_index)
        if self.discovery_tracker:
            self.discovery_tracker.record_query(true_label, abs_index)
        # Relevance = False for *all* classes (including discovered birds). This ensures no cluster.relevance=True, so comp_cluster_relevant=False always.
        # After initial discovery, no further queries (add_o_pt path for non-anomalous points). Matches "none of the classes are marked as relevant, once they are discovered, we do not want to query them again."
        #if true_label != "Aves":
        #    #print(true_label)
        #    relevance = True
        #else:
        #    relevance = False
        relevance = False
        return true_label, relevance
    
    def get_query_count(self):
        return self.query_count
