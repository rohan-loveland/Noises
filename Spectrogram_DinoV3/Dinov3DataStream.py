"""
Dinov3DataStream.py - Fully compatible with SpectrogramOracle
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm

class DiscoveryTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_points_examined = 0
        self.total_queries = 0
        self.classes_first_seen = {}
        self.classes_first_queried = {}
        self.first_seen_stream_idx = {}
        self.first_queried_stream_idx = {}

    def record_point(self, true_label, stream_idx, is_query=False):
        self.total_points_examined += 1
        if true_label not in self.classes_first_seen:
            self.classes_first_seen[true_label] = stream_idx
            self.first_seen_stream_idx[true_label] = stream_idx
        if is_query and true_label not in self.classes_first_queried:
            self.classes_first_queried[true_label] = stream_idx
            self.first_queried_stream_idx[true_label] = stream_idx

    # === Added for compatibility with SpectrogramOracle ===
    def record_query(self, true_label, stream_idx):
        """Called by SpectrogramOracle when a query happens."""
        self.total_queries += 1
        self.record_point(true_label, stream_idx, is_query=True)

    def get_discovery_report(self):
        report = {
            'total_points_examined': self.total_points_examined,
            'total_queries': self.total_queries,
            'total_classes_seen': len(self.classes_first_seen),
            'total_classes_queried': len(self.classes_first_queried),
            'classes': {}
        }
        for cls in self.classes_first_seen:
            report['classes'][cls] = {
                'first_seen_stream_idx': self.first_seen_stream_idx.get(cls),
                'first_queried_stream_idx': self.first_queried_stream_idx.get(cls),
                'queries_before_first_query': self.first_queried_stream_idx.get(cls, 0) - 
                                             (self.first_seen_stream_idx.get(cls, 0) if cls in self.first_seen_stream_idx else 0)
            }
        return report

    def save_report(self, path="dinov3_discovery_report.json"):
        import json
        with open(path, 'w') as f:
            json.dump(self.get_discovery_report(), f, indent=2)


class Dinov3DataStream:
    def __init__(self, 
                 csv_path="5sSpectrograms_tensors/train_5s_spectrograms.csv",
                 tensor_dir="5sSpectrograms_tensors",
                 max_samples=None,
                 shuffle=True,
                 seed=42,
                 dino_model_name="vit_small_patch16_224",
                 use_pretrained=None, #"dinov3_pretrained_final.pth",
                 embed_dim=384,
                 device=None):
        
        self.csv_path = Path(csv_path)
        self.tensor_dir = Path(tensor_dir)
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.embed_dim = embed_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.df = pd.read_csv(self.csv_path)
        if self.max_samples:
            self.df = self.df.sample(n=min(self.max_samples, len(self.df)), random_state=seed).reset_index(drop=True)
        
        self.n_samples = len(self.df)
        print(f"Loaded {self.n_samples:,} 5s spectrogram samples for streaming.")

        self.discovery_tracker = DiscoveryTracker()

        # Load DINOv3 model
        self.model = None
        if use_pretrained and Path(use_pretrained).exists():
            print(f"Loading custom DINOv3 checkpoint: {use_pretrained}")
            self.model = timm.create_model(dino_model_name, pretrained=False, num_classes=0).to(self.device)
            ckpt = torch.load(use_pretrained, map_location=self.device)
            self.model.load_state_dict(ckpt['student_state_dict'])
            self.model.eval()
            print("✅ Model loaded successfully")
        else:
            print("Warning: No custom checkpoint found.")

        self.embeddings = None
        self._try_load_precomputed_embeddings()

        self.current_idx = 0
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.indices)

    def _try_load_precomputed_embeddings(self):
        embed_path = self.tensor_dir / "dinov3_embeddings.npy"
        if embed_path.exists():
            print(f"✅ Loading precomputed embeddings from {embed_path}")
            self.embeddings = np.load(embed_path, mmap_mode='r')
        else:
            print("Precomputed embeddings not found → extracting on-the-fly.")

    @torch.no_grad()
    def _extract_embedding(self, spec_npy_path):
        if self.model is None:
            spec = np.load(self.tensor_dir / spec_npy_path, mmap_mode='r').astype(np.float32)
            return spec.flatten()[:self.embed_dim]

        spec = np.load(self.tensor_dir / spec_npy_path, mmap_mode='r').astype(np.float32)
        
        if spec.ndim == 2:
            spec_norm = ((spec - spec.min()) / (spec.max() - spec.min() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(spec_norm).convert('RGB')
        else:
            image = Image.fromarray(spec.squeeze().astype(np.uint8)).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)

        features = self.model(img_tensor)
        embedding = features.squeeze(0).cpu().numpy()
        embedding = (embedding - embedding.mean()) / (embedding.std() + 1e-8)
        return embedding

    def stream_new_data_point(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration("End of stream")

        idx = self.indices[self.current_idx]
        row = self.df.iloc[idx]

        if self.embeddings is not None:
            embedding = self.embeddings[idx]
        else:
            embedding = self._extract_embedding(row['spectrogram_npy_path'])

        self.discovery_tracker.record_point(self.get_true_label_for_idx(idx), self.current_idx)
        self.current_idx += 1
        return embedding.astype(np.float32)

    # === Required by SpectrogramOracle ===
    def get_true_label_for_idx(self, idx):
        row = self.df.iloc[idx]
        return row.get('class_name') or row.get('primary_label') or f"unknown_{idx}"

    def get_remaining_num_points(self):
        return self.n_samples - self.current_idx

    def reset(self):
        self.current_idx = 0
        self.discovery_tracker.reset()
        if self.shuffle:
            np.random.shuffle(self.indices)


# Quick test
if __name__ == "__main__":
    stream = Dinov3DataStream(max_samples=100, shuffle=False)
    print("Testing stream...")
    for i in range(5):
        emb = stream.stream_new_data_point()
        print(f"Point {i}: shape={emb.shape}, norm={np.linalg.norm(emb):.2f}")
    print("Stream test complete.")