"""
Reusable Perch v2 Embedding Engine.

- Single TF session / single model load (critical for 8 GB VRAM laptops).
- Same robust model resolution + diagnostics as the original Perch_A_RED.py.
- Clean batch embedding API used by both the bulk extractor and live DataStream.
- Audio loading is CPU-side and can be called from threads by callers.
- Robustly selects the flat 'embedding' (B, D) even if the SavedModel also exports spatial_embedding (B,16,4,D).

Intended usage:
    embedder = PerchEmbedder(batch_size=4)          # auto GPU/local/hub
    embs = embedder.embed_audio_files(list_of_paths)  # (N, D) float32
    dim = embedder.get_embedding_dim()

The heavy I/O pipelining + resume logic lives in the caller (Perch_A_RED.py or PerchDataStream preload).
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf

try:
    import librosa
except Exception:
    librosa = None

try:
    import scipy.signal as signal
except Exception:
    signal = None


# ====================== CONFIG (sensible 8 GB VRAM defaults) ======================
GPU_MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"
CPU_MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/perch_v2_cpu/versions/1"

# Local folder that may contain the extracted GPU tarball (placed inside Perch/)
LOCAL_MODEL_CANDIDATE = Path(__file__).parent / "bird-vocalization-classifier-tensorflow2-perch_v2-v2"


class PerchEmbedder:
    def __init__(
        self,
        model_ref: Optional[Union[str, Path]] = None,
        batch_size: int = 4,
        force_cpu: bool = False,
        local_model_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize (and test-load) the Perch v2 model exactly once.

        Args:
            model_ref: explicit override (path or URL). If None, auto-resolve.
            batch_size: inference batch size (tune down to 1-2 on 8 GB if needed).
            force_cpu: if True, prefer CPU model URL even if GPU is visible.
            local_model_path: explicit local dir containing saved_model.pb (highest priority).
        """
        self.batch_size = max(1, int(batch_size))
        self.embedding_dim: Optional[int] = None
        self._serving_fn = None

        # GPU setup (same as original)
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[PerchEmbedder] Detected {len(gpus)} GPU(s) — memory growth enabled.")
            else:
                print("[PerchEmbedder] No GPUs visible to TensorFlow.")
        except Exception as e:
            print("[PerchEmbedder] GPU config note:", e)

        # Model reference resolution (priority: explicit > local_model_path arg > LOCAL_CANDIDATE if GPU > URLs)
        gpus_visible = len(tf.config.list_physical_devices("GPU")) > 0
        has_gpu = gpus_visible and not force_cpu

        if model_ref is not None:
            self.model_ref = str(model_ref)
            print(f"[PerchEmbedder] Using explicit model_ref: {self.model_ref}")
        elif local_model_path:
            self.model_ref = str(local_model_path)
            print(f"[PerchEmbedder] Using explicit local_model_path: {self.model_ref}")
        else:
            candidate = LOCAL_MODEL_CANDIDATE
            if has_gpu and candidate.exists():
                self.model_ref = str(candidate)
                print("[PerchEmbedder] GPUs detected — using local GPU model dir.")
            elif has_gpu:
                self.model_ref = GPU_MODEL_URL
                print("[PerchEmbedder] Using GPU Perch v2 via TF Hub (download on first use).")
            else:
                self.model_ref = CPU_MODEL_URL
                print("[PerchEmbedder] Falling back to CPU-only Perch v2 model.")

        print(f"[PerchEmbedder] Model reference: {self.model_ref}")
        print("[PerchEmbedder] TensorFlow devices:", tf.config.list_physical_devices())

        # One-time load + dimension probe (this is the "early test" that used to live in main())
        print("[PerchEmbedder] Loading model (first time may download/extract)...")
        try:
            model = hub.load(self.model_ref)
            self._serving_fn = model.signatures["serving_default"]

            # Probe real output dimension with a tiny dummy batch
            dummy = tf.zeros([1, 160000], dtype=tf.float32)
            out = self._serving_fn(dummy)
            emb_tensor = self._select_embedding_tensor(out)
            self.embedding_dim = int(emb_tensor.shape[-1])
            print(f"[PerchEmbedder] Model ready. Embedding dim = {self.embedding_dim}")
            # Free the probe model reference (we keep only the serving_fn)
            del model, out
        except Exception as load_err:
            err_str = str(load_err)
            print("\n[PerchEmbedder] *** MODEL LOAD FAILED ***")
            print("Error:", type(load_err).__name__, "-", err_str[:400])
            print("\nCommon fixes:")
            print("  - If you have the GPU tarball, extract it under Perch/ so the folder contains saved_model.pb")
            print("  - Or set local_model_path=... when constructing PerchEmbedder")
            print("  - For a pure CPU run: PerchEmbedder(..., force_cpu=True)")
            print("  - The original diagnostic messages from Perch_A_RED.py explain the CUDA vs CPU wheel issues.")
            raise

    def get_embedding_dim(self) -> int:
        if self.embedding_dim is None:
            raise RuntimeError("Embedding dim not known (model load failed)")
        return self.embedding_dim

    def _select_embedding_tensor(self, outputs):
        """Robustly pick the canonical (pooled) embedding tensor.

        This model exports both 'embedding' (B, D) and 'spatial_embedding' (B, 16, 4, D).
        We must use the flat pooled one for A_RED (L2 unit vectors in semantic space).
        Selection is order-independent and prefers the flat embedding.
        """
        keys = list(outputs.keys())
        if "embedding" in keys:
            return outputs["embedding"]
        # Fallback: among embed-like keys, choose the lowest-rank tensor (2D over 4D spatial)
        candidates = [(k, outputs[k]) for k in keys if "embed" in k.lower() or "output" in k.lower()]
        if candidates:
            candidates.sort(key=lambda kv: len(kv[1].shape))
            return candidates[0][1]
        return outputs[keys[0]]

    def load_audio_waveform(
        self, audio_path: Union[str, Path], target_sr: int = 32000, target_len: int = 160000
    ) -> tf.Tensor:
        """Load audio (ogg/wav/etc) -> mono 32 kHz float32 waveform of exact target_len. Returns zeros on failure."""
        p = Path(audio_path)
        if not p.exists():
            return tf.zeros([target_len], dtype=tf.float32)

        try:
            audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1).astype(np.float32)

            if sr != target_sr:
                if librosa is not None:
                    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
                elif signal is not None:
                    num_samples = int(round(len(audio) * float(target_sr) / sr))
                    audio = signal.resample(audio, num_samples).astype(np.float32)
                else:
                    # Crude fallback (not ideal but keeps us running)
                    ratio = target_sr / sr
                    indices = (np.arange(int(len(audio) * ratio)) / ratio).astype(int)
                    indices = np.clip(indices, 0, len(audio) - 1)
                    audio = audio[indices].astype(np.float32)

            if len(audio) > target_len:
                audio = audio[:target_len]
            else:
                audio = np.pad(audio, (0, target_len - len(audio)))

            return tf.convert_to_tensor(audio, dtype=tf.float32)

        except Exception as e:
            print(f"  [PerchEmbedder] WARNING: Audio load failed for {p.name}: {e}")
            return tf.zeros([target_len], dtype=tf.float32)

    def embed_batch(self, waveform_batch: tf.Tensor) -> np.ndarray:
        """Embed an already-prepared (B, 160000) float32 waveform tensor. Returns (B, D)."""
        outputs = self._serving_fn(waveform_batch)
        emb_tensor = self._select_embedding_tensor(outputs)
        return emb_tensor.numpy().astype(np.float32)

    def embed_audio_files(
        self, audio_paths: List[Union[str, Path]], show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed a list of audio files. Returns float32 array of shape (len(audio_paths), embedding_dim).
        Loads audio on the fly (caller can parallelize loads with threads if desired).
        Uses the configured batch_size.
        """
        if not audio_paths:
            return np.zeros((0, self.get_embedding_dim()), dtype=np.float32)

        embeddings = []
        n = len(audio_paths)

        for start in range(0, n, self.batch_size):
            batch_paths = audio_paths[start : start + self.batch_size]
            batch_audio = [self.load_audio_waveform(p) for p in batch_paths]
            batch_input = tf.stack(batch_audio)

            batch_emb = self.embed_batch(batch_input)
            embeddings.append(batch_emb)

            if show_progress:
                print(f"  embedded {min(start + self.batch_size, n)}/{n}")

        return np.vstack(embeddings).astype(np.float32)


# Convenience: allow "python -m Perch.perch_embedder" for a tiny self-test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-audio", type=str, default=None, help="Path to a 5s .ogg to embed (smoke test)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    embedder = PerchEmbedder(batch_size=args.batch_size, force_cpu=args.force_cpu)
    print("Dim:", embedder.get_embedding_dim())

    if args.test_audio:
        emb = embedder.embed_audio_files([args.test_audio])
        print("Test embedding shape:", emb.shape)
        print("Norm (should be reasonable):", np.linalg.norm(emb[0]))