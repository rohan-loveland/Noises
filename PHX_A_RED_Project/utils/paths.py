"""
Robust path resolution helpers for the PHX ARED project.

These replicate (and clean up) the repeated "find CSV / embeddings / audio dir"
logic that was scattered across Perch/, Spectrogram_A_RED/, etc.
"""
from pathlib import Path
from typing import Tuple


def resolve_project_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Return (project_root, default_csv, default_tensor_dir, default_audio_dir).

    Tries to be robust whether you run from the project root, from inside
    PHX_A_RED_Project/, or from a subdirectory.
    """
    here = Path(__file__).resolve()

    # Walk up until we see 5sSpectrograms_tensors or prepared_5s_clips or .git
    root = here
    for _ in range(6):
        if (root / "5sSpectrograms_tensors").exists() or (root / "prepared_5s_clips").exists():
            break
        if (root / ".git").exists():
            break
        root = root.parent
    else:
        # Fallback: assume two levels up from this file reaches project root
        root = here.parents[3] if len(here.parents) > 3 else here.parent.parent.parent

    csv_linux = root / "5sSpectrograms_tensors" / "train_5s_spectrograms_linux.csv"
    csv_default = root / "5sSpectrograms_tensors" / "train_5s_spectrograms.csv"
    csv = csv_linux if csv_linux.exists() else csv_default

    tensor_dir = root / "5sSpectrograms_tensors"
    audio_dir = root / "prepared_5s_clips"

    return root, csv, tensor_dir, audio_dir


def resolve_embeddings_path(name: str = "perch_embeddings.npy") -> Path:
    """Common helper for embedding .npy files next to the spectrogram tensors."""
    root, _, tensor_dir, _ = resolve_project_paths()
    return tensor_dir / name
