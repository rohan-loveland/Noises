"""
verify_dinov3_fidelity.py

Correctness cross-check for visualize_dinov3_feature_space.py artifacts.

Replicates the *exact* DINOv3 embedding extraction used to produce a given
_coords.csv (same model load + min-max uint8 + PIL RGB + Resize+ImageNet norm
+ forward + post-norm), then refits the identical PCA reducer (PCA with the
same random_state seed that was used for the original run) on the re-extracted
embeddings (in the exact row order present in the coords.csv), and compares
the resulting 2D coordinates to the saved (x, y) values.

This proves that the visualized points truly came from the DINOv3 representation
"as used in" Spectrogram_DinoV3/ (Dinov3DataStream.py + SpectrogramDinov3.py).

NOTE: This helper always re-fits PCA(2). It was written to cross-check extraction
fidelity for PCA-based runs of visualize_dinov3_feature_space.py. UMAP (and TSNE)
runs do not require (and are not supported by) this verifier.

Intended to be run after a viz script invocation (e.g. the smoke with seed 420).
It imports load_dinov3_model and extract_dinov3_embedding directly from the
visualize script so there is zero risk of divergence.

Usage (PowerShell):
    python verify_dinov3_fidelity.py --coords viz_dino_test/dinov3_2d_pca_8078pts_s420_20260611_160446_coords.csv --seed 420 --pretrained_ckpt dinov3_pretrained_final.pth

The script will auto-discover a *coords.csv under viz_dino_test/ if --coords omitted.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA

import torch
import torchvision.transforms as transforms

# Import the *exact* functions that produced the original embeddings.
# This guarantees we are testing the same code path used in the viz run.
from visualize_dinov3_feature_space import (
    load_dinov3_model,
    extract_dinov3_embedding,
)

DEFAULT_DINO_MODEL_NAME = "facebook/vit_small_patch16_dinov3.lvd1689m"


def main():
    parser = argparse.ArgumentParser(
        description="Fidelity check: re-extract DINOv3 embeddings for a coords.csv sample and verify 2D projections match after identical PCA (PCA-only; not for UMAP runs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--coords",
        type=str,
        default=None,
        help="Path to a specific *_coords.csv from a previous visualize_dinov3 run. "
             "If omitted, auto-discovers the first *coords.csv under viz_dino_test/.",
    )
    parser.add_argument(
        "--tensor_dir",
        type=str,
        default="5sSpectrograms_tensors",
        help="Directory with the .npy spectrograms (same as used for the viz run).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=420,
        help="random_state passed to PCA (must match the --seed used when the viz run produced the coords; the viz script uses args.seed for both sampling and the reducer). This verifier only supports PCA runs; UMAP runs of the viz scripts do not use it.",
    )
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default=DEFAULT_DINO_MODEL_NAME,
        help="timm model name (must match what was used for the viz).",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="dinov3_pretrained_final.pth",
        help="Custom checkpoint path or name (same as passed to the viz run).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for max |delta| in the 2D coordinates to consider PASS.",
    )

    args = parser.parse_args()

    # Resolve coords file
    if args.coords:
        coords_path = Path(args.coords)
    else:
        candidates = sorted(Path("viz_dino_test").glob("*coords.csv"))
        if not candidates:
            print("ERROR: No *coords.csv found under viz_dino_test/ and --coords not provided.")
            sys.exit(2)
        coords_path = candidates[0]
        print(f"Auto-discovered coords: {coords_path}")

    if not coords_path.exists():
        print(f"ERROR: coords file not found: {coords_path}")
        sys.exit(2)

    tensor_dir = Path(args.tensor_dir)
    if not tensor_dir.exists():
        print(f"ERROR: tensor_dir not found: {tensor_dir}")
        sys.exit(2)

    print("=== DINOv3 Embedding Fidelity Cross-Check ===")
    print(f"Coords file : {coords_path}")
    print(f"Tensor dir  : {tensor_dir}")
    print(f"Reducer seed (PCA random_state): {args.seed}")
    print(f"DINO model  : {args.dino_model_name}")
    print(f"CKPT        : {args.pretrained_ckpt}")
    print(f"Atol for PASS: {args.atol}")
    print()

    # Load the exact sample that was visualized (row order here = draw order for the plot and the final coords)
    df = pd.read_csv(coords_path)
    required = {"x", "y", "class_name", "spectrogram_npy_path", "original_csv_row"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: coords CSV missing required columns: {missing}")
        sys.exit(2)

    n = len(df)
    print(f"Loaded {n:,} points from coords (these define the exact sample + order).")
    print("Per-class counts in this coords file (ordered):")
    print(df["class_name"].value_counts().reindex(
        ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
    ).dropna().astype(int).to_string())
    print()

    # Prepare transform exactly as the viz script does (ImageNet stats, 224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model using the *identical* loader (custom ckpt logic + student_state_dict etc.)
    print("Loading DINOv3 model (via visualize_dinov3_feature_space.load_dinov3_model)...")
    model, model_source = load_dinov3_model(
        args.dino_model_name,
        args.pretrained_ckpt,
        device,
    )
    print(f"  model_source: {model_source}")
    if model is None:
        print("ERROR: Failed to load DINOv3 model. Cannot perform fidelity check.")
        sys.exit(3)
    print()

    # Re-extract in the *exact* order the rows appear in the coords.csv
    # (this order is what the written x/y are aligned to after the final class sort in the viz writer)
    paths_in_order = df["spectrogram_npy_path"].astype(str).tolist()

    print(f"Re-extracting DINOv3 embeddings for all {n:,} rows (order matches coords.csv)...")
    X_list = []
    for rel in tqdm(paths_in_order, desc="re-extract-dino", unit="spec"):
        try:
            vec = extract_dinov3_embedding(rel, tensor_dir, model, device, transform)
            X_list.append(vec)
        except Exception as e:
            print(f"\nERROR while extracting {rel}: {e}")
            # Propagate to fail the check loudly
            raise

    X_check = np.vstack(X_list).astype(np.float32)
    print(f"  Re-extracted matrix: {X_check.shape} (embed_dim={X_check.shape[1]})")
    print()

    # Refit the *exact same* reducer configuration used by the original run
    print(f"Refitting PCA(n_components=2, random_state={args.seed}) on the re-extracted cloud...")
    pca = PCA(n_components=2, random_state=args.seed)
    X2_check = pca.fit_transform(X_check)
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print()

    # Saved coordinates are already in the same row order as df (and thus as X2_check)
    saved = df[["x", "y"]].to_numpy(dtype=np.float32)

    # Per-coordinate absolute differences
    diffs = np.abs(X2_check - saved)
    max_diff = float(diffs.max())
    mean_diff = float(diffs.mean())
    median_diff = float(np.median(diffs))

    print("=== 2D Coordinate Comparison (recomputed vs saved in coords.csv) ===")
    print(f"Max   |delta|: {max_diff:.3e}")
    print(f"Mean  |delta|: {mean_diff:.3e}")
    print(f"Median|delta|: {median_diff:.3e}")
    print()

    # Pick representative rows from different classes (using the df iloc order)
    print("=== Per-sample spot checks (different classes) ===")
    picked = []
    for cls in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        sub = df[df["class_name"] == cls]
        if len(sub) > 0:
            iloc = int(sub.index[0])  # position in the coords file / X2_check
            picked.append((cls, iloc))

    # Also add a late Aves (near end of its block) and the very last point for good measure
    aves_mask = (df["class_name"] == "Aves")
    if aves_mask.any():
        last_aves_iloc = int(df[aves_mask].index[-1])
        if last_aves_iloc not in [p[1] for p in picked]:
            picked.append(("Aves (last in block)", last_aves_iloc))
    picked.append(("LAST (Reptilia tail)", n - 1))

    for label, iloc in picked:
        cls = df.loc[iloc, "class_name"]
        orig_row = int(df.loc[iloc, "original_csv_row"])
        path = df.loc[iloc, "spectrogram_npy_path"]
        sx, sy = saved[iloc]
        cx, cy = X2_check[iloc]
        dx, dy = diffs[iloc]
        print(f"[{label}] iloc={iloc} class={cls} orig_csv_row={orig_row}")
        print(f"    path: {path}")
        print(f"    saved   : ({sx:+.8f}, {sy:+.8f})")
        print(f"    recomputed: ({cx:+.8f}, {cy:+.8f})")
        print(f"    delta   : ({dx:.2e}, {dy:.2e})")
        print()

    # Final verdict
    print("=== VERDICT ===")
    if max_diff <= args.atol:
        print(f"PASS: All 2D coordinates match within atol={args.atol} (max |delta|={max_diff:.3e}).")
        print("The visualized points were produced by the exact DINOv3 embedding pipeline")
        print("(load_dinov3_model + extract_dinov3_embedding from visualize_dinov3_feature_space.py,")
        print("which mirrors Spectrogram_DinoV3/Dinov3DataStream.py + SpectrogramDinov3.py).")
        print("The PCA reduction (random_state) was also reproduced identically.")
        return 0
    else:
        print(f"FAIL: max |delta| = {max_diff:.3e} > atol={args.atol}.")
        print("There is a material difference between the saved 2D coords and a fresh re-extraction + re-projection.")
        print("This would indicate either:")
        print("  - a divergence in the extraction code path,")
        print("  - different model weights / ckpt load,")
        print("  - different transform or device behavior, or")
        print("  - the coords.csv was produced with a different seed for the reducer.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
