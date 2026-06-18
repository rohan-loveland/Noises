"""
BaseDataStream + concrete implementations.

Unified, reusable streaming interface for ARED.

All streams:
- Preload vectors into memory (fast stream_new_data_point)
- L2-normalize every vector
- Support shuffle + deterministic seed + max_samples
- Robust label resolution with fallback across common columns
- Provide the duck-typed API expected by ARED frontends:
    stream_new_data_point() -> np.ndarray (unit vector)
    get_remaining_num_points() -> int
    get_true_label_for_idx(stream_idx) -> str or None

Concrete frontends:
- SpectrogramDataStream: raw/pooled spectrogram .npy -> flatten
- PerchDataStream: precomputed perch_embeddings.npy (fast) OR live PerchEmbedder
- Dinov3DataStream: precomputed dinov3_embeddings.npy OR live DinoV3Extractor
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

from ..utils.paths import resolve_project_paths, resolve_embeddings_path

# Heavy feature extractors are imported lazily inside the classes that need them
# so that "spectrogram only" runs do not pull tensorflow/torch.
PerchEmbedder = None
DinoV3Extractor = None


def _resolve_label(row: pd.Series, preferred_column: str) -> str:
    """Robust label fallback used by all streams (matches historical behavior)."""
    lab = None
    if preferred_column in row and pd.notna(row[preferred_column]):
        lab = str(row[preferred_column])
    if lab is None or lab == "nan":
        for col in (preferred_column, "scientific_name", "common_name", "class_name", "primary_label"):
            if col in row and pd.notna(row[col]):
                val = row[col]
                lab = str(val) if not pd.isna(val) else "unknown"
                break
    if lab is None or lab == "nan":
        lab = "unknown"
    return lab


class BaseDataStream:
    """
    Common base for all ARED data streams.

    Subclasses should typically call super().__init__ and then call
    self._preload() (or override _preload entirely for complex cases like Perch).
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        tensor_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        label_column: str = "class_name",
    ):
        self.label_column = label_column
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.stream_counter = 0

        root, default_csv, default_tensor, _ = resolve_project_paths()

        self.csv_path = Path(csv_path) if csv_path else default_csv
        if not self.csv_path.exists():
            fb = root / "5sSpectrograms_tensors" / "train_5s_spectrograms.csv"
            if fb.exists():
                print(f"[BaseDataStream] WARNING: {self.csv_path.name} not found, using {fb.name}")
                self.csv_path = fb

        self.tensor_dir = Path(tensor_dir) if tensor_dir else default_tensor

        self.df = pd.read_csv(self.csv_path)
        self._prepare_df()

        self.labels_cache: List[str] = []
        self.processed_vectors = None  # np.ndarray or list
        self.n_samples = 0

    def _prepare_df(self):
        """Shuffle + truncate. Subclasses may add _orig_idx before calling super or after."""
        df = self.df
        if self.shuffle:
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        if self.max_samples is not None and self.max_samples > 0:
            df = df.head(self.max_samples).reset_index(drop=True)
        self.df = df

    def _preload(self):
        """Subclasses implement. Must set self.processed_vectors, self.labels_cache, self.n_samples."""
        raise NotImplementedError

    # -------------- Common public API (duck-typed for ARED usage) --------------

    def stream_new_data_point(self):
        if self.stream_counter >= self.n_samples:
            raise StopIteration("No more data points")
        vec = self.processed_vectors[self.stream_counter]
        self.stream_counter += 1
        return vec

    def get_remaining_num_points(self):
        return self.n_samples - self.stream_counter

    def get_true_label_for_idx(self, stream_idx: int):
        if stream_idx >= len(self.labels_cache):
            return None
        return self.labels_cache[stream_idx]

    def reset(self):
        """Reset stream position (useful for some tests / repeated runs)."""
        self.stream_counter = 0


class SpectrogramDataStream(BaseDataStream):
    """
    Raw spectrogram frontend.

    Loads .npy files under tensor_dir, applies 16-frame mean-pool on time axis,
    flattens, L2 normalizes. Preloads everything for speed.
    """

    POOL_SIZE = 16

    def __init__(
        self,
        csv_path: Optional[str] = None,
        tensor_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        label_column: str = "class_name",
    ):
        super().__init__(
            csv_path=csv_path,
            tensor_dir=tensor_dir,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
            label_column=label_column,
        )
        print("Preloading spectrogram tensors into memory for fast streaming...")
        self._preload()
        print(f"Preloaded {self.n_samples:,} spectrogram samples. label_column='{self.label_column}'")

    def _load_vector(self, row: pd.Series) -> np.ndarray:
        npy_rel = row["spectrogram_npy_path"]
        npy_path = self.tensor_dir / npy_rel
        if not npy_path.exists():
            return None
        spec = np.load(npy_path).astype(np.float32)
        if spec.ndim == 2:
            T = spec.shape[1]
            num_pools = T // self.POOL_SIZE
            if num_pools > 0:
                spec = np.mean(
                    spec[:, : num_pools * self.POOL_SIZE].reshape(spec.shape[0], num_pools, self.POOL_SIZE),
                    axis=2,
                )
        vec = spec.flatten()
        return vec

    def _preload(self):
        processed = []
        labels = []
        valid_rows = []

        for local_i, (_, row) in enumerate(self.df.iterrows()):
            if "spectrogram_npy_path" not in row:
                continue
            vec = self._load_vector(row)
            if vec is None:
                continue

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vec = vec.astype(np.float32)

            lab = _resolve_label(row, self.label_column)
            processed.append(vec)
            labels.append(lab)
            valid_rows.append(local_i)

        if not processed:
            raise RuntimeError("No valid spectrogram .npy files found after filtering.")

        self.labels_cache = labels
        if valid_rows:
            self.df = self.df.iloc[valid_rows].reset_index(drop=True)
        self.n_samples = len(processed)

        try:
            self.processed_vectors = np.stack(processed).astype(np.float32)
        except Exception:
            self.processed_vectors = processed


class PerchDataStream(BaseDataStream):
    """
    Perch v2 embedding frontend.

    - Fast path: uses precomputed 5sSpectrograms_tensors/perch_embeddings.npy when it covers
      the required _orig_idx range (after shuffle/max).
    - Live path: falls back to PerchEmbedder on only the needed audio files.

    Always L2-normalizes. Uses _orig_idx to correctly map shuffled rows back to
    CSV-order precomputed embeddings.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        audio_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        label_column: str = "class_name",
        live_batch_size: int = 4,
    ):
        # We need _orig_idx before shuffle/max, so do special df prep
        root, default_csv, default_tensor, default_audio = resolve_project_paths()

        self.label_column = label_column
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.stream_counter = 0
        self.embedder = None
        self.live = False
        self.live_batch_size = live_batch_size

        self.csv_path = Path(csv_path) if csv_path else default_csv
        if not self.csv_path.exists():
            fb = root / "5sSpectrograms_tensors" / "train_5s_spectrograms.csv"
            if fb.exists():
                self.csv_path = fb

        self.embeddings_path = Path(embeddings_path) if embeddings_path else (default_tensor / "perch_embeddings.npy")
        self.audio_dir = Path(audio_dir) if audio_dir else default_audio

        self.df = pd.read_csv(self.csv_path)
        if "filename" not in self.df.columns:
            raise ValueError("CSV must contain 'filename' for Perch audio lookup")

        # Preserve original CSV order indices for precomp lookup
        orig_df = self.df.copy()
        orig_df["_orig_idx"] = np.arange(len(orig_df))

        if shuffle:
            orig_df = orig_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        if max_samples is not None and max_samples > 0:
            orig_df = orig_df.head(max_samples).reset_index(drop=True)

        self.df = orig_df

        print("Preloading Perch embeddings into memory for fast streaming...")
        print(f"  (embeddings_path={self.embeddings_path})")
        self._preload()
        print(f"Preloaded {self.n_samples:,} Perch vectors ({'live' if self.live else 'precomputed'}). label_column='{label_column}'")

    def _preload(self):
        processed = []
        labels = []
        valid_rows = []

        full_emb = None
        if self.embeddings_path.exists():
            try:
                cand = np.load(self.embeddings_path, mmap_mode="r")
                max_needed = int(self.df["_orig_idx"].max()) + 1 if len(self.df) > 0 else 0
                if len(cand) >= max_needed:
                    full_emb = cand
                    print(f"  Using precomputed embeddings (shape {full_emb.shape}) — fast path.")
                else:
                    print(f"  Precomputed too small ({len(cand)} < {max_needed}); falling back to live.")
            except Exception as e:
                print(f"  Failed to load precomputed ({e}); will use live extraction.")

        if full_emb is None:
            # LIVE
            global PerchEmbedder
            if PerchEmbedder is None:
                from ..features.perch_embedder import PerchEmbedder as _PE
                PerchEmbedder = _PE
            self.embedder = PerchEmbedder(batch_size=self.live_batch_size)
            self.live = True

            audio_paths = []
            for _, row in self.df.iterrows():
                fn = str(row.get("filename", ""))
                audio_paths.append(str(self.audio_dir / fn))

            print(f"  Live-extracting {len(audio_paths)} Perch embeddings (batch={self.live_batch_size})...")
            live_embs = self.embedder.embed_audio_files(audio_paths)

            for i, (_, row) in enumerate(self.df.iterrows()):
                emb = live_embs[i].astype(np.float32).copy()
                norm = np.linalg.norm(emb)
                vec = (emb / norm) if norm > 0 else emb
                lab = _resolve_label(row, self.label_column)
                processed.append(vec.astype(np.float32))
                labels.append(lab)
                valid_rows.append(i)
        else:
            # PRECOMP FAST PATH
            for local_i, (_, row) in enumerate(self.df.iterrows()):
                oidx = int(row["_orig_idx"])
                emb = full_emb[oidx].astype(np.float32).copy()
                norm = np.linalg.norm(emb)
                vec = (emb / norm) if norm > 0 else emb
                lab = _resolve_label(row, self.label_column)
                processed.append(vec.astype(np.float32))
                labels.append(lab)
                valid_rows.append(local_i)

        if not processed:
            raise RuntimeError("No valid Perch embeddings could be prepared.")

        self.labels_cache = labels
        if valid_rows:
            self.df = self.df.iloc[valid_rows].reset_index(drop=True)
        self.n_samples = len(processed)

        try:
            self.processed_vectors = np.stack(processed).astype(np.float32)
        except Exception:
            self.processed_vectors = processed


class Dinov3DataStream(BaseDataStream):
    """
    DinoV3 / DINO spectrogram embedding frontend.

    - Fast path: dinov3_embeddings.npy (full CSV order) if present and large enough.
    - Live path: DinoV3Extractor (timm vit + spectrogram -> PIL RGB + ImageNet norm).
      Falls back to truncated raw flatten if no model available (legacy behavior).

    Vectors are L2-normalized (or near-unit after extractor normalization).
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        tensor_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        label_column: str = "class_name",
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        embed_dim: int = 384,
        pretrained_path: Optional[str] = None,
        device=None,
    ):
        super().__init__(
            csv_path=csv_path,
            tensor_dir=tensor_dir,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
            label_column=label_column,
        )

        self.embed_dim = embed_dim
        self.extractor = None
        self.precomputed_embeddings = None

        # Try precomputed first (full CSV order, not shuffled)
        emb_path = self.tensor_dir / "dinov3_embeddings.npy"
        if emb_path.exists():
            try:
                cand = np.load(emb_path, mmap_mode="r")
                # We will index using original df row numbers before our shuffle/max.
                # Because we already shuffled the df, we need the original indices.
                # Reconstruct orig order mapping using the same logic as Perch.
                # For simplicity and robustness we store the orig indices on the current df.
                # But since super() already shuffled, we do a second pass here for precomp support.
                # Better approach: re-resolve from the *unshuffled* view when precomp exists.
                print(f"  Found dinov3_embeddings.npy (shape {cand.shape})")
                self.precomputed_embeddings = cand
            except Exception as e:
                print(f"  Could not load dinov3 precomp: {e}")

        # Prepare extractor only if we will need live extraction
        need_live = self.precomputed_embeddings is None
        if need_live:
            try:
                global DinoV3Extractor
                if DinoV3Extractor is None:
                    from ..features.dino_extractor import DinoV3Extractor as _DE
                    DinoV3Extractor = _DE
                self.extractor = DinoV3Extractor(
                    model_name=model_name,
                    embed_dim=embed_dim,
                    device=device,
                    pretrained_path=pretrained_path,
                )
            except Exception as e:
                print(f"[Dinov3DataStream] WARNING: Could not create DinoV3Extractor ({e}). Will fall back to truncated flatten.")
                self.extractor = None

        print("Preloading DinoV3 embeddings into memory...")
        self._preload()
        mode = "precomputed" if self.precomputed_embeddings is not None else ("live-dino" if self.extractor is not None else "raw-fallback")
        print(f"Preloaded {self.n_samples:,} DinoV3 vectors ({mode}). label_column='{self.label_column}'")

    def _preload(self):
        processed = []
        labels = []
        valid_rows = []

        # We need original row indices for precomputed array (which is in CSV order).
        # Rebuild a clean original-index view.
        # Because we shuffled in super, we saved no orig_idx. Re-read strategy:
        # - If precomp exists we must map from the *current* rows back to original CSV positions.
        # Simplest robust way matching historical Dino: the dinov3 precomp (when written) follows the CSV
        # order of the file that was used to produce it. We therefore:
        #   - Re-load the CSV without shuffle to get "orig order"
        #   - Build a map from spectrogram_npy_path (or row number) to position in precomp
        # But many historical runs just did iloc after their own shuffle. To be safe and match Perch style,
        # we attach _orig_idx *before* shuffle in a local copy for precomp lookup.

        # For this implementation we take a pragmatic approach that works with the existing precomp files:
        # If precomputed_embeddings exists, we assume it is indexed by the *order in the CSV as read before any shuffle*.
        # So we:
        #   1. Read the raw CSV again (no shuffle)
        #   2. Build a position map from iloc position in raw CSV -> embedding row
        #   3. For each row in our (already shuffled) self.df, find its original iloc in the full CSV and fetch.

        use_precomp = self.precomputed_embeddings is not None

        raw_df = pd.read_csv(self.csv_path)
        if use_precomp:
            # Map (spectrogram_npy_path, or iloc) -> row in precomp
            # We will use positional: the precomp row i corresponds to raw_df.iloc[i]
            # Our current self.df rows came from a shuffle of a (possibly filtered) view.
            # The safest is to match on the npy path when possible.
            path_to_precomp_idx = {}
            for i, r in raw_df.iterrows():
                p = str(r.get("spectrogram_npy_path", f"__row_{i}"))
                path_to_precomp_idx[p] = i

        for local_i, (_, row) in enumerate(self.df.iterrows()):
            vec = None
            if use_precomp:
                p = str(row.get("spectrogram_npy_path", ""))
                oidx = path_to_precomp_idx.get(p)
                if oidx is not None and oidx < len(self.precomputed_embeddings):
                    vec = self.precomputed_embeddings[oidx].astype(np.float32).copy()

            if vec is None:
                # LIVE extraction or fallback
                if self.extractor is not None:
                    try:
                        npy_rel = row.get("spectrogram_npy_path")
                        if npy_rel:
                            vec = self.extractor.extract_from_npy(self.tensor_dir / npy_rel)
                        else:
                            vec = None
                    except Exception as ex:
                        print(f"  Dino live extract failed for a file, falling back: {ex}")
                        vec = None

                if vec is None:
                    # Legacy fallback: truncated flatten of raw spec (rare)
                    npy_rel = row.get("spectrogram_npy_path")
                    if npy_rel:
                        try:
                            spec = np.load(self.tensor_dir / npy_rel, mmap_mode="r").astype(np.float32)
                            vec = spec.flatten()[: self.embed_dim]
                        except Exception:
                            vec = np.zeros(self.embed_dim, dtype=np.float32)
                    else:
                        vec = np.zeros(self.embed_dim, dtype=np.float32)

            # Final normalization to be consistent with other streams
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vec = vec.astype(np.float32)

            lab = _resolve_label(row, self.label_column)
            processed.append(vec)
            labels.append(lab)
            valid_rows.append(local_i)

        if not processed:
            raise RuntimeError("No DinoV3 vectors could be prepared.")

        self.labels_cache = labels
        if valid_rows:
            self.df = self.df.iloc[valid_rows].reset_index(drop=True)
        self.n_samples = len(processed)

        try:
            self.processed_vectors = np.stack(processed).astype(np.float32)
        except Exception:
            self.processed_vectors = processed
