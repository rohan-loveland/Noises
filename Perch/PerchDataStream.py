"""
Perch A_RED Data Stream + Oracle.

Drop-in replacement for SpectrogramDataStream / SpectrogramOracle when you want
to run A_RED on Google Perch v2 embeddings (semantic, e.g. 1536-d or 1280-d depending
on the exact saved_model variant) instead of raw/pooled spectrograms.

Features:
- Exact same public API as the spectrogram version (duck-typed for ARED).
- Preload + in-memory cache (fast stream_new_data_point).
- L2 normalization of embeddings (consistent with how spectrograms are fed to ARED).
- Supports precomputed embeddings (fast path) **or** live on-demand Perch inference
  for the exact subset you ask for (max_samples). This is the key to avoiding
  unreasonable full-dataset precompute time during development.
- Resume-friendly: if you have a full (or partial) perch_embeddings.npy from the
  improved Perch/Perch_A_RED.py, it is used automatically.
- shuffle + max_samples + deterministic seed supported (uses _orig_idx to correctly
  index into a CSV-order embeddings array).

Usage (typical from a runner):
    ds = PerchDataStream(
        csv_path="5sSpectrograms_tensors/train_5s_spectrograms_linux.csv",
        max_samples=2000,
        shuffle=True,
        seed=420,
        label_column="class_name"   # or "scientific_name"
    )
    oracle = PerchOracle(ds)
    vec = ds.stream_new_data_point()   # (D,) float32 L2 unit vector (D depends on Perch model variant)
"""

from pathlib import Path
import numpy as np
import pandas as pd

# We import the embedder only when we actually need the live path (lazy)
# This keeps "pure numpy fast path" runs from importing TF at all.
PerchEmbedder = None  # filled on first live use


def _resolve_project_paths():
    """Return (project_root, default_csv, default_emb, default_audio_dir)."""
    here = Path(__file__).resolve().parent          # Perch/
    root = here.parent
    csv = root / "5sSpectrograms_tensors" / "train_5s_spectrograms_linux.csv"
    emb = root / "5sSpectrograms_tensors" / "perch_embeddings.npy"
    audio = root / "prepared_5s_clips"
    return root, csv, emb, audio


class PerchDataStream:
    def __init__(
        self,
        csv_path: str = None,
        embeddings_path: str = None,
        audio_dir_for_live: str = None,
        max_samples=None,
        shuffle=True,
        seed=42,
        label_column: str = "class_name",
        live_batch_size: int = 4,
    ):
        self.label_column = label_column
        self.stream_counter = 0
        self.embedder = None  # only set in live mode
        self.live = False

        root, default_csv, default_emb, default_audio = _resolve_project_paths()

        csv_path = Path(csv_path) if csv_path else default_csv
        if not csv_path.exists():
            fallback = root / "5sSpectrograms_tensors" / "train_5s_spectrograms.csv"
            if fallback.exists():
                print(f"[PerchDataStream] WARNING: {csv_path.name} not found, trying {fallback.name}")
                csv_path = fallback
        self.csv_path = csv_path

        if embeddings_path is None:
            embeddings_path = default_emb
        self.embeddings_path = Path(embeddings_path)

        if audio_dir_for_live is None:
            audio_dir_for_live = default_audio
        self.audio_dir = Path(audio_dir_for_live)

        # Load metadata
        self.df = pd.read_csv(self.csv_path)
        if "filename" not in self.df.columns:
            # The spectrogram CSV also has it (from prepared clips stage)
            raise ValueError("CSV must contain a 'filename' column for audio lookup (live mode)")

        # Preserve original row indices so we can correctly index a CSV-order embeddings.npy
        orig_df = self.df.copy()
        orig_df["_orig_idx"] = np.arange(len(orig_df))

        # Optional shuffle (simulates random arrival order for A_RED)
        if shuffle:
            orig_df = orig_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        if max_samples is not None:
            orig_df = orig_df.head(max_samples).reset_index(drop=True)

        self.df = orig_df
        self.n_samples = len(self.df)

        print("Preloading Perch embeddings into memory for fast streaming...")
        print(f"  (embeddings_path={self.embeddings_path})")

        # Decide fast vs live
        full_emb = None
        if self.embeddings_path.exists():
            try:
                cand = np.load(self.embeddings_path, mmap_mode="r")
                if len(cand) >= (self.df["_orig_idx"].max() + 1):
                    full_emb = cand
                    print(f"  Using precomputed embeddings (shape {full_emb.shape}) — fast path.")
                else:
                    print(f"  Precomputed embeddings exist but are smaller than needed "
                          f"({len(cand)} vs required at least {self.df['_orig_idx'].max()+1}). "
                          "Falling back to live extraction for this subset.")
            except Exception as e:
                print(f"  Could not load precomputed embeddings ({e}). Will use live mode.")

        processed = []
        labels = []
        valid_rows = []

        if full_emb is None:
            # LIVE PATH — only for the rows we actually kept after shuffle+max
            global PerchEmbedder
            if PerchEmbedder is None:
                from perch_embedder import PerchEmbedder as _PE
                PerchEmbedder = _PE

            self.embedder = PerchEmbedder(batch_size=live_batch_size)
            self.live = True

            # Collect audio paths in the *current stream order*
            live_audio_paths = []
            for _, row in self.df.iterrows():
                fn = str(row.get("filename", ""))
                ap = self.audio_dir / fn
                live_audio_paths.append(str(ap))

            print(f"  Live-extracting {len(live_audio_paths)} Perch embeddings "
                  f"(batch_size={live_batch_size}). This is fast for a few thousand points.")
            live_embs = self.embedder.embed_audio_files(live_audio_paths)

            for i, (_, row) in enumerate(self.df.iterrows()):
                emb = live_embs[i].astype(np.float32).copy()
                norm = np.linalg.norm(emb)
                data_point = (emb / norm) if norm > 0 else emb

                lab = self._resolve_label(row)
                processed.append(data_point)
                labels.append(lab)
                valid_rows.append(i)  # local to this df slice
        else:
            # FAST PRECOMP PATH
            for local_i, (_, row) in enumerate(self.df.iterrows()):
                oidx = int(row["_orig_idx"])
                emb = full_emb[oidx].astype(np.float32).copy()

                norm = np.linalg.norm(emb)
                data_point = (emb / norm) if norm > 0 else emb

                lab = self._resolve_label(row)
                processed.append(data_point)
                labels.append(lab)
                valid_rows.append(local_i)

        if not processed:
            raise RuntimeError("No valid Perch embeddings could be prepared (missing files or CSV issues).")

        self.labels_cache = labels

        # Keep a trimmed df view (rarely used after preload)
        if valid_rows:
            self.df = self.df.iloc[valid_rows].reset_index(drop=True)
        self.n_samples = len(processed)

        try:
            self.processed_vectors = np.stack(processed).astype(np.float32)
        except Exception:
            self.processed_vectors = processed

        print(f"Preloaded {self.n_samples:,} Perch embedding vectors "
              f"({'live' if self.live else 'precomputed'}).")
        print(f"Using label column: '{self.label_column}'")

    def _resolve_label(self, row):
        """Same robust fallback logic used by the spectrogram stream."""
        lab = None
        if self.label_column in row and pd.notna(row[self.label_column]):
            lab = str(row[self.label_column])
        if lab is None or lab == "nan":
            for col in (self.label_column, "scientific_name", "common_name", "class_name", "primary_label"):
                if col in row and pd.notna(row[col]):
                    val = row[col]
                    lab = str(val) if not pd.isna(val) else "unknown"
                    break
        if lab is None or lab == "nan":
            lab = "unknown"
        return lab

    def stream_new_data_point(self):
        if self.stream_counter >= self.n_samples:
            raise StopIteration("No more data points")
        data_point = self.processed_vectors[self.stream_counter]
        self.stream_counter += 1
        return data_point

    def get_remaining_num_points(self):
        return self.n_samples - self.stream_counter

    def get_true_label_for_idx(self, stream_idx):
        """stream_idx is position in the (shuffled + max) order we are streaming."""
        if stream_idx >= len(self.labels_cache):
            return None
        return self.labels_cache[stream_idx]


class PerchOracle:
    """
    Oracle that returns (true_label, relevance).

    Exact same contract and "near-zero queries" policy as SpectrogramOracle:
    - Always returns relevance=False for every class.
    - Therefore after the very first point(s), almost everything goes through
      the add_o_pt (no query) path in ARED as long as it is not anomalous.
    - This directly implements the project requirement:
      "only query on anomaly" + "once classes are discovered we do not want to
      query them again".
    """

    def __init__(self, data_stream, discovery_tracker=None):
        self.data_stream = data_stream
        self.query_count = 0
        self.discovery_tracker = discovery_tracker

    def answer_query(self, abs_index):
        self.query_count += 1
        true_label = self.data_stream.get_true_label_for_idx(abs_index)
        if self.discovery_tracker:
            self.discovery_tracker.record_query(true_label, abs_index, self.query_count)
        #if true_label != "Aves":
        #    relevance = True
        #else:
        #    relevance = False
        relevance = False   # <--- the key design choice (identical to spectro setup)
        return true_label, relevance

    def get_query_count(self):
        return self.query_count