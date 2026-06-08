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
        
        # For high-dim spectrograms (128*313 ~40k dims), optionally reduce dimensionality if needed
        # Current impl uses full flattened vector; A_RED's BallTree/KDTree handles it but may be slow
        
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
        npy_path = self.tensor_dir / row['spectrogram_npy_path']
        
        if not npy_path.exists():
            # Skip missing files
            self.stream_counter += 1
            return self.stream_new_data_point()
        
        # Load as float32 and flatten (spectrogram is 2D, e.g. (128, 313) -> 1D vector of ~40k dims)
        spec = np.load(npy_path).astype(np.float32)
        data_point = spec.flatten()
        
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
        """Oracle returns true label and relevance (for now, all are 'relevant' or based on class; customize as needed)"""
        self.query_count += 1
        true_label = self.data_stream.get_true_label_for_idx(abs_index)
        # For bird/noise detection, perhaps relevance based on whether it's a target class; for now all relevant=True
        relevance = True  # or customize e.g. if 'target_bird' in true_label
        return true_label, relevance
    
    def get_query_count(self):
        return self.query_count
