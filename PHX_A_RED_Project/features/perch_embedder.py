"""
Reusable Perch v2 Embedding Engine (cleaned & self-contained).

- Single TF session / single model load.
- Thread-safe audio loading for callers that want to overlap I/O.
- Robust selection of the flat 'embedding' tensor.
- Used by both bulk precompute and live PerchDataStream.

This is a re-implementation of the logic originally in Perch/perch_embedder.py
placed here so the new package is self-sufficient.
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


GPU_MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2"
CPU_MODEL_URL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/perch_v2_cpu/versions/1"

# Local candidate inside the old Perch/ folder (we still look for it for convenience)
LOCAL_MODEL_CANDIDATE = Path(__file__).resolve().parents[2] / "Perch" / "bird-vocalization-classifier-tensorflow2-perch_v2-v2"


class PerchEmbedder:
    def __init__(
        self,
        model_ref: Optional[Union[str, Path]] = None,
        batch_size: int = 4,
        force_cpu: bool = False,
        local_model_path: Optional[Union[str, Path]] = None,
    ):
        self.batch_size = max(1, int(batch_size))
        self.embedding_dim: Optional[int] = None
        self._serving_fn = None

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

        gpus_visible = len(tf.config.list_physical_devices("GPU")) > 0
        has_gpu = gpus_visible and not force_cpu

        if model_ref is not None:
            self.model_ref = str(model_ref)
        elif local_model_path:
            self.model_ref = str(local_model_path)
        else:
            candidate = LOCAL_MODEL_CANDIDATE
            if has_gpu and candidate.exists():
                self.model_ref = str(candidate)
            elif has_gpu:
                self.model_ref = GPU_MODEL_URL
            else:
                self.model_ref = CPU_MODEL_URL

        print(f"[PerchEmbedder] Model reference: {self.model_ref}")

        print("[PerchEmbedder] Loading model (first time may download)...")
        try:
            model = hub.load(self.model_ref)
            self._serving_fn = model.signatures["serving_default"]

            dummy = tf.zeros([1, 160000], dtype=tf.float32)
            out = self._serving_fn(dummy)
            emb_tensor = self._select_embedding_tensor(out)
            self.embedding_dim = int(emb_tensor.shape[-1])
            print(f"[PerchEmbedder] Model ready. Embedding dim = {self.embedding_dim}")
            del model, out
        except Exception as load_err:
            print("\n[PerchEmbedder] *** MODEL LOAD FAILED ***")
            print("Error:", type(load_err).__name__, str(load_err)[:400])
            raise

    def get_embedding_dim(self) -> int:
        if self.embedding_dim is None:
            raise RuntimeError("Embedding dim not known (model load failed)")
        return self.embedding_dim

    def _select_embedding_tensor(self, outputs):
        keys = list(outputs.keys())
        if "embedding" in keys:
            return outputs["embedding"]
        candidates = [(k, outputs[k]) for k in keys if "embed" in k.lower() or "output" in k.lower()]
        if candidates:
            candidates.sort(key=lambda kv: len(kv[1].shape))
            return candidates[0][1]
        return outputs[keys[0]]

    def load_audio_waveform(
        self, audio_path: Union[str, Path], target_sr: int = 32000, target_len: int = 160000
    ) -> tf.Tensor:
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
                    num = int(round(len(audio) * float(target_sr) / sr))
                    audio = signal.resample(audio, num).astype(np.float32)
                else:
                    ratio = target_sr / sr
                    idx = (np.arange(int(len(audio) * ratio)) / ratio).astype(int)
                    idx = np.clip(idx, 0, len(audio) - 1)
                    audio = audio[idx].astype(np.float32)

            if len(audio) > target_len:
                audio = audio[:target_len]
            else:
                audio = np.pad(audio, (0, target_len - len(audio)))

            return tf.convert_to_tensor(audio, dtype=tf.float32)
        except Exception as e:
            print(f"  [PerchEmbedder] WARNING: Audio load failed for {p.name}: {e}")
            return tf.zeros([target_len], dtype=tf.float32)

    def embed_batch(self, waveform_batch: tf.Tensor) -> np.ndarray:
        outputs = self._serving_fn(waveform_batch)
        emb_tensor = self._select_embedding_tensor(outputs)
        return emb_tensor.numpy().astype(np.float32)

    def embed_audio_files(
        self, audio_paths: List[Union[str, Path]], show_progress: bool = False
    ) -> np.ndarray:
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
