"""
visualize_dinov3_feature_space.py

Project DINOv3 embeddings (extracted from 5s mel-spectrograms exactly as done
in Spectrogram_DinoV3/) to 2D so we can visually inspect the learned semantic
feature space.

Primary goal: reuse the sampling, explicit class_name ordering, coloring,
layered plotting, and artifact machinery from visualize_spectrogram_feature_space.py
(the "last program"), but feed it DINOv3 embeddings (~384-dim) instead of raw
pooled+flattened mels. This lets us compare whether taxonomic structure
(e.g. Mammalia near Aves?) emerges more clearly in the DINOv3 space.

DINOv3 embedding creation is taken directly from:
  - Spectrogram_DinoV3/Dinov3DataStream.py  (_extract_embedding and model load)
  - Spectrogram_DinoV3/SpectrogramDinov3.py  (usage example + ckpt + model name)

This script is deliberately standalone (no modifications to other files).
For --reducer umap you must first `pip install umap-learn`.

Usage examples (PowerShell / Windows):
    python visualize_dinov3_feature_space.py --max_aves 400 --reducer pca --pretrained_ckpt dinov3_pretrained_final.pth
    python visualize_dinov3_feature_space.py --max_aves 12000 --reducer tsne --seed 42 --pretrained_ckpt dinov3_pretrained_final.pth
    python visualize_dinov3_feature_space.py --max_aves 400 --reducer umap --seed 420 --pretrained_ckpt dinov3_pretrained_final.pth
    python visualize_dinov3_feature_space.py --max_aves 0 --compare_raw   # minorities only + side-by-side raw flatten

Outputs (in --output_dir):
    - dinov3_2d_....png / .pdf   : the 2D scatter(s)  (DINO primary; optional raw panel)
    - dinov3_2d_..._coords.csv   : the 2D coordinates + class_name + paths (reusable)
    - dinov3_2d_..._summary.txt  : human-readable run metadata
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

try:
    import umap
except ImportError:
    umap = None

import torch
import torchvision.transforms as transforms
from PIL import Image
import timm


# ====================== DEFAULTS (match the rest of the project + DINO refs) ======================
DEFAULT_CSV = "5sSpectrograms_tensors/train_5s_spectrograms.csv"
DEFAULT_TENSOR_DIR = "5sSpectrograms_tensors"
DEFAULT_MAX_AVES = 15000
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "viz_outputs"

# Match SpectrogramDinov3.py:50 and Dinov3DataStream defaults
DEFAULT_DINO_MODEL_NAME = "facebook/vit_small_patch16_dinov3.lvd1689m"
# Common good checkpoints (searched if --pretrained_ckpt not supplied)
DEFAULT_CKPT_CANDIDATES = []

# Fixed color palette and draw order (identical to the raw-flatten viz script).
# Aves is intentionally desaturated / low-alpha so the minority classes remain visible.
CLASS_COLORS = {
    'Aves': "#0077ff",       # soft blue-gray (background layer)
    'Mammalia': '#d62728',   # red
    'Insecta': '#ff7f0e',    # orange
    'Amphibia': '#2ca02c',   # green
    'Reptilia': '#9467bd',   # purple
}

# Draw order: Aves first (so it sits underneath), then the others on top.
# This directly supports the request "first order them by the class name".
CLASS_ORDER = ['Aves', 'Amphibia', 'Insecta', 'Mammalia', 'Reptilia']


def load_and_flatten_spectrogram(
    npy_rel_path: str,
    tensor_dir: Path,
    pool_size: int = 16
) -> np.ndarray:
    """
    (Included only for the optional --compare_raw side-by-side path.)
    Load one .npy spectrogram and turn it into the exact same 1D vector that
    A_RED receives today.

    This is a direct copy of the helper in:
        visualize_spectrogram_feature_space.py (which itself mirrors
        Spectrogram_A_RED/SpectrogramDataStream.py stream_new_data_point ~lines 51-74)
    """
    npy_path = tensor_dir / npy_rel_path
    if not npy_path.exists():
        raise FileNotFoundError(f"Spectrogram tensor not found: {npy_path}")

    spec = np.load(npy_path).astype(np.float32)

    if spec.ndim == 2:
        time_steps = spec.shape[1]
        num_pools = time_steps // pool_size
        if num_pools > 0:
            spec = np.mean(
                spec[:, :num_pools * pool_size].reshape(spec.shape[0], num_pools, pool_size),
                axis=2
            )

    vec = spec.flatten()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    else:
        print(f"  WARNING: zero-norm spectrogram encountered: {npy_rel_path}")

    return vec.astype(np.float32)


def load_dinov3_model(
    dino_model_name: str,
    pretrained_ckpt: str | None,
    device: torch.device
) -> tuple[torch.nn.Module | None, str]:
    """
    Load a DINOv3 ViT exactly as done in:
        Spectrogram_DinoV3/Dinov3DataStream.py:94-101  (custom ckpt path)
        Spectrogram_DinoV3/SpectrogramDinov3.py:44-52  (example values)

    Preference order:
      1. Explicit --pretrained_ckpt (if exists)
      2. Auto-search a few well-known filenames in CWD and Pretrain_DinoV3/
      3. Fall back to timm pretrained=True (generic DINOv3 weights, still useful)

    Returns (model or None, source_description_string)
    """
    model = None
    source = "none"

    ckpt_to_try = None
    if pretrained_ckpt:
        p = Path(pretrained_ckpt)
        if p.exists():
            ckpt_to_try = p
        else:
            print(f"WARNING: --pretrained_ckpt {pretrained_ckpt} not found; will search defaults.")

    if ckpt_to_try is None:
        for cand in DEFAULT_CKPT_CANDIDATES:
            p = Path(cand)
            if p.exists():
                ckpt_to_try = p
                break

    if ckpt_to_try is not None and ckpt_to_try.exists():
        print(f"Loading custom DINOv3 checkpoint: {ckpt_to_try}")
        model = timm.create_model(dino_model_name, pretrained=False, num_classes=0).to(device)
        ckpt = torch.load(ckpt_to_try, map_location=device)
        # The pretrain script saves {'student_state_dict': ..., ...}
        if 'student_state_dict' in ckpt:
            model.load_state_dict(ckpt['student_state_dict'])
        else:
            # Some checkpoints may store the full state under 'model' or directly
            state = ckpt.get('model', ckpt)
            model.load_state_dict(state, strict=False)
        model.eval()
        print("✅ Model loaded successfully (custom pretrained)")
        source = str(ckpt_to_try)
        return model, source

    # Fallback to generic timm pretrained weights (still a real DINOv3 ViT)
    print(f"No usable custom DINOv3 checkpoint found. Falling back to timm pretrained weights for {dino_model_name}.")
    try:
        model = timm.create_model(dino_model_name, pretrained=True, num_classes=0).to(device)
        model.eval()
        print("✅ Loaded generic DINOv3 weights via timm (pretrained=True)")
        source = "timm_pretrained"
        return model, source
    except Exception as e:
        print(f"ERROR: Could not load DINO model ({dino_model_name}) even with pretrained=True: {e}")
        return None, "failed"


@torch.no_grad()
def extract_dinov3_embedding(
    npy_rel_path: str,
    tensor_dir: Path,
    model: torch.nn.Module | None,
    device: torch.device,
    transform: transforms.Compose
) -> np.ndarray:
    """
    Turn one spectrogram .npy into a DINOv3 embedding.

    This function is a direct, self-contained copy of the logic in:
        Spectrogram_DinoV3/Dinov3DataStream.py
        (see _extract_embedding around lines 122-146)

    Also matches the RGB conversion used in Pretrain_DinoV3/Dinov3PretrainDataset.py:40-45.
    """
    if model is None:
        # Should not normally happen for this viz script (we prefer real embeddings)
        spec = np.load(tensor_dir / npy_rel_path, mmap_mode='r').astype(np.float32)
        # Fallback to a tiny slice (keeps script alive but useless for real DINO analysis)
        return spec.flatten()[:384].astype(np.float32)

    spec = np.load(tensor_dir / npy_rel_path, mmap_mode='r').astype(np.float32)

    if spec.ndim == 2:
        spec_norm = ((spec - spec.min()) / (spec.max() - spec.min() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(spec_norm).convert('RGB')
    else:
        image = Image.fromarray(spec.squeeze().astype(np.uint8)).convert('RGB')

    img_tensor = transform(image).unsqueeze(0).to(device)

    features = model(img_tensor)
    embedding = features.squeeze(0).cpu().numpy()
    embedding = (embedding - embedding.mean()) / (embedding.std() + 1e-8)
    return embedding.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Project DINOv3 embeddings of 5s mel-spectrograms to 2D, colored and ordered by class_name (Aves/Mammalia/etc).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_dinov3_feature_space.py --max_aves 400 --reducer pca --pretrained_ckpt dinov3_pretrained_final.pth
  python visualize_dinov3_feature_space.py --max_aves 12000 --reducer tsne --seed 420 --pretrained_ckpt dinov3_pretrained_final.pth
  python visualize_dinov3_feature_space.py --max_aves 400 --reducer umap --seed 420 --pretrained_ckpt dinov3_pretrained_final.pth
  python visualize_dinov3_feature_space.py --max_aves 0 --compare_raw   # minorities only + DINO vs raw-flatten side-by-side
        """
    )
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV,
                        help="Path to the train_5s_spectrograms.csv")
    parser.add_argument("--tensor_dir", type=str, default=DEFAULT_TENSOR_DIR,
                        help="Directory containing the .npy tensors (and optionally dinov3_embeddings.npy)")
    parser.add_argument("--max_aves", type=int, default=DEFAULT_MAX_AVES,
                        help="How many Aves samples to keep (ALL non-Aves are always included). Use 0 to see only minorities.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for the Aves subsample (reproducibility)")
    parser.add_argument("--reducer", choices=["pca", "tsne", "umap"], default="pca",
                        help="'pca' = fast PCA(2). 'tsne' = PCA(50) then TSNE(2) (much slower but often nicer). 'umap' = UMAP(2) (local + global structure, good default for these visualizations).")
    parser.add_argument("--umap_n_neighbors", type=int, default=15,
                        help="n_neighbors for UMAP (default 15). Only used when --reducer=umap.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="min_dist for UMAP (default 0.1). Only used when --reducer=umap.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Where to write the PNG/PDF/CSV artifacts")
    parser.add_argument("--no_save_coords", action="store_true",
                        help="Do not write the _coords.csv (saves a little disk/IO)")

    # DINOv3-specific (modeled on Spectrogram_DinoV3/ usage)
    parser.add_argument("--dino_model_name", type=str, default=DEFAULT_DINO_MODEL_NAME,
                        help="timm model name for DINOv3 ViT (default matches SpectrogramDinov3.py)")
    parser.add_argument("--pretrained_ckpt", type=str, default=None,
                        help="Path to a custom dinov3_pretrained*.pth (student_state_dict). If omitted, script searches common locations then falls back to timm pretrained=True.")
    parser.add_argument("--force_on_the_fly", action="store_true",
                        help="Ignore any dinov3_embeddings.npy and always extract on-the-fly for the sampled points.")
    parser.add_argument("--compare_raw", action="store_true",
                        help="Also extract the raw pooled+flattened spectrograms for the EXACT same sampled points and produce a side-by-side figure (DINO | raw). "
                             "Very useful for directly seeing whether DINOv3 improves taxonomic separation.")
    # pool_size only affects the --compare_raw path
    parser.add_argument("--pool_size", type=int, default=16,
                        help="Time-axis pooling size used ONLY for the --compare_raw flattened vectors (default 16, matches A_RED)")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    tensor_dir = Path(args.tensor_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== DINOv3 2D Feature Space Visualization ===")
    print("Embedding logic reference: Spectrogram_DinoV3/Dinov3DataStream.py (_extract_embedding + model load)")
    print("Sampling / ordering / plotting / artifacts reference: visualize_spectrogram_feature_space.py (the last program)")
    print(f"CSV:            {csv_path}")
    print(f"Tensor dir:     {tensor_dir}")
    print(f"max_aves:       {args.max_aves}")
    print(f"seed:           {args.seed}")
    print(f"reducer:        {args.reducer}")
    print(f"output_dir:     {output_dir}")
    print(f"dino_model_name: {args.dino_model_name}")
    print(f"pretrained_ckpt: {args.pretrained_ckpt or '(auto-search + timm fallback)'}")
    print(f"force_on_the_fly: {args.force_on_the_fly}")
    print(f"compare_raw:    {args.compare_raw}")
    print(f"device:         {device}")
    print()
    print("Points will be ORDERED and COLORED by class_name (Aves first as background layer).")
    print()

    # ====================== 1. LOAD CSV (do not shuffle the master frame) ======================
    print("Loading metadata CSV...")
    t0 = time.time()
    df = pd.read_csv(csv_path)

    required_cols = {'spectrogram_npy_path', 'class_name'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain at least columns: {required_cols}")

    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    print("  class_name distribution in full CSV:")
    print(df['class_name'].value_counts().to_string())
    print()

    # ====================== 2. SAMPLE + EXPLICITLY ORDER BY class_name ======================
    # This entire block is copied nearly verbatim from visualize_spectrogram_feature_space.py:181-223
    # so that ordering, counts, and original_csv_row behavior are identical to the last program.
    non_aves = df[df['class_name'] != 'Aves'].copy()
    aves = df[df['class_name'] == 'Aves'].copy()

    if args.max_aves > 0 and len(aves) > args.max_aves:
        aves_sample = aves.sample(n=args.max_aves, random_state=args.seed).copy()
    else:
        aves_sample = aves.copy()

    # Concatenate (keeping original index for later embedding alignment)
    selected = pd.concat([non_aves, aves_sample], ignore_index=False)

    # === THIS IS THE KEY STEP REQUESTED BY THE USER ===
    # "We want to first order them by the class name. I.e. Aves, insectae, and so on."
    selected = selected.sort_values(
        by=['class_name', 'primary_label'],
        kind='stable'
    ).reset_index(drop=False)

    # After reset, the old positional index (0-based row number in the CSV as read)
    # lives in the column named 'index'. Rename for clarity.
    selected = selected.rename(columns={'index': 'original_csv_row'})

    # Make class_name a categorical with a stable draw order (Aves first)
    selected['class_name'] = pd.Categorical(
        selected['class_name'],
        categories=CLASS_ORDER,
        ordered=True
    )
    selected = selected.sort_values('class_name').reset_index(drop=True)

    n_total = len(selected)
    class_counts = selected['class_name'].value_counts().reindex(CLASS_ORDER).dropna().astype(int)

    print(f"Selected for 2D visualization (ordered by class_name): {n_total:,}")
    for cls in CLASS_ORDER:
        if cls in class_counts.index:
            print(f"  {cls:10s}: {class_counts[cls]:,}")
    print()

    # These original row numbers let us pull the exact same points from any precomputed
    # full-size embedding array (dinov3_embeddings.npy, ...) that was written in CSV row order.
    original_rows = selected['original_csv_row'].to_numpy(dtype=np.int64)

    # ====================== 3. DINOv3 FEATURE EXTRACTION ======================
    print("Extracting DINOv3 embeddings (precomputed or on-the-fly)...")
    start_load = time.time()

    X = None
    kept_paths = []
    kept_class_names = []
    kept_primary = []
    kept_common = []
    kept_orig_rows = []
    model_source = "precomputed"
    emb_dim = None

    precomp_path = tensor_dir / "dinov3_embeddings.npy"
    used_precomp = False

    if not args.force_on_the_fly and precomp_path.exists():
        print(f"Loading precomputed DINOv3 embeddings from {precomp_path} (mmap index via original_csv_row)...")
        try:
            all_emb = np.load(precomp_path, mmap_mode='r')
            if len(all_emb) <= max(original_rows):
                print(f"  WARNING: precomputed embeddings have only {len(all_emb)} rows; some indices may be out of range.")
            emb_subset = np.asarray(all_emb[original_rows])
            X = emb_subset.astype(np.float32)
            used_precomp = True
            model_source = f"precomputed:{precomp_path.name}"
            emb_dim = X.shape[1] if X is not None else None
            # For precomp fast path we still need the kept metadata from selected (we filter later if any issues, but precomp assumes alignment)
            for _, row in selected.iterrows():
                kept_paths.append(str(row['spectrogram_npy_path']))
                kept_class_names.append(str(row['class_name']))
                kept_primary.append(str(row.get('primary_label', '')))
                kept_common.append(str(row.get('common_name', row.get('scientific_name', ''))))
                kept_orig_rows.append(int(row['original_csv_row']))
        except Exception as e:
            print(f"  Could not use precomputed embeddings ({e}). Falling back to on-the-fly extraction.")
            used_precomp = False
            X = None

    if X is None:
        # On-the-fly path (current normal case)
        model, model_source = load_dinov3_model(args.dino_model_name, args.pretrained_ckpt, device)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        X_list = []
        for _, row in tqdm(selected.iterrows(), total=n_total, desc="dino-embed"):
            rel = str(row['spectrogram_npy_path'])
            try:
                vec = extract_dinov3_embedding(rel, tensor_dir, model, device, transform)
                X_list.append(vec)
                kept_paths.append(rel)
                kept_class_names.append(str(row['class_name']))
                kept_primary.append(str(row.get('primary_label', '')))
                kept_common.append(str(row.get('common_name', row.get('scientific_name', ''))))
                kept_orig_rows.append(int(row['original_csv_row']))
            except FileNotFoundError:
                print(f"  WARNING: file not found, skipping: {rel}")
            except Exception as e:
                print(f"  WARNING: failed to embed {rel}: {e}")

        if not X_list:
            print("ERROR: No spectrograms were successfully embedded. Nothing to visualize.")
            return

        X = np.vstack(X_list).astype(np.float32)
        emb_dim = X.shape[1]

    load_time = time.time() - start_load
    print(f"  Extracted {len(X):,} vectors of dim {emb_dim} in {load_time:.1f}s (source: {model_source})")
    print()

    # Update the "kept" lists (in case any files were missing during on-the-fly)
    labels = kept_class_names
    primary_labels = kept_primary
    common_names = kept_common
    paths = kept_paths
    original_rows = np.array(kept_orig_rows, dtype=np.int64)
    n_total = len(labels)

    # ====================== 4. 2D REDUCTION ======================
    print(f"Running {args.reducer.upper()} to 2D...")
    start_dr = time.time()

    if args.reducer == "pca":
        reducer = PCA(n_components=2, random_state=args.seed)
        coords = reducer.fit_transform(X)
        reducer_label = "PCA(2)"
        print(f"  Explained variance ratio: {reducer.explained_variance_ratio_}")
    elif args.reducer == "tsne":
        pre = PCA(n_components=min(50, X.shape[1] - 1), random_state=args.seed)
        X_pre = pre.fit_transform(X)
        print(f"  Pre-reduced with PCA(50), total explained ~{pre.explained_variance_ratio_.sum():.3f}")
        tsne = TSNE(
            n_components=2,
            random_state=args.seed,
            perplexity=min(30, max(5, (len(X_pre) - 1) // 3)),
            max_iter=1000,
            verbose=0
        )
        coords = tsne.fit_transform(X_pre)
        reducer_label = "PCA(50)+TSNE(2)"
    elif args.reducer == "umap":
        if umap is None:
            print("ERROR: --reducer umap requires the umap-learn package.\n"
                  "  Install with:  pip install umap-learn")
            return
        print(f"  Running UMAP(n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist}) to 2D...")
        u = umap.UMAP(
            n_components=2,
            random_state=args.seed,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            verbose=0
        )
        coords = u.fit_transform(X)
        reducer_label = "UMAP(2)"
    else:
        print(f"ERROR: unknown reducer {args.reducer!r}")
        return

    dr_time = time.time() - start_dr
    print(f"  2D reduction finished in {dr_time:.1f}s")
    print()

    # ====================== 5. OPTIONAL: RAW FLATTENED ON IDENTICAL SAMPLES (for side-by-side) ======================
    raw_coords = None
    raw_reducer_label = None

    if args.compare_raw:
        print("Extracting raw flattened spectrograms (identical sample set) for side-by-side comparison...")
        start_raw = time.time()
        raw_list = []
        for rel in tqdm(paths, desc="raw-flatten"):
            try:
                v = load_and_flatten_spectrogram(rel, tensor_dir, pool_size=args.pool_size)
                raw_list.append(v)
            except Exception as e:
                print(f"  WARNING: raw flatten failed for {rel}: {e}")
                # pad with zeros to keep alignment (rare)
                raw_list.append(np.zeros(2432, dtype=np.float32))  # approximate dim; will be overwritten by real first success
        if raw_list:
            raw_X = np.vstack(raw_list).astype(np.float32)
            if args.reducer == "pca":
                r_red = PCA(n_components=2, random_state=args.seed).fit_transform(raw_X)
                raw_reducer_label = "PCA(2)"
            elif args.reducer == "tsne":
                r_pre = PCA(n_components=min(50, raw_X.shape[1] - 1), random_state=args.seed).fit_transform(raw_X)
                r_tsne = TSNE(n_components=2, random_state=args.seed,
                              perplexity=min(30, max(5, (len(r_pre)-1)//3)), max_iter=1000, verbose=0)
                r_red = r_tsne.fit_transform(r_pre)
                raw_reducer_label = "PCA(50)+TSNE(2)"
            elif args.reducer == "umap":
                if umap is None:
                    print("ERROR: --reducer umap requires the umap-learn package (for --compare_raw).")
                else:
                    print(f"  (compare) Running UMAP(n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist}) on raw flattened...")
                    u = umap.UMAP(
                        n_components=2,
                        random_state=args.seed,
                        n_neighbors=args.umap_n_neighbors,
                        min_dist=args.umap_min_dist,
                        verbose=0
                    )
                    r_red = u.fit_transform(raw_X)
                    raw_reducer_label = "UMAP(2)"
            else:
                r_red = None
            if r_red is not None:
                raw_coords = r_red
                print(f"  Raw flattened side-by-side ready ({raw_X.shape[1]}D -> 2D) in {time.time()-start_raw:.1f}s")
        else:
            print("  --compare_raw requested but no raw vectors could be produced.")

    # ====================== 6. VISUALIZATION (class_name ordered + styled) ======================
    print("Generating plot(s)...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"dinov3_2d_{args.reducer}_{n_total}pts_s{args.seed}_{timestamp}"

    # Build a small frame for convenient grouped plotting (DINO is always primary)
    plot_df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'class_name': labels,
    })

    n_panels = 2 if raw_coords is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7), squeeze=False)
    ax0 = axes[0, 0]

    # Draw in CLASS_ORDER so Aves is painted first (background), minorities on top.
    for cls in CLASS_ORDER:
        if cls not in plot_df['class_name'].values:
            continue
        mask = (plot_df['class_name'] == cls)
        color = CLASS_COLORS.get(cls, '#555555')
        count = int(mask.sum())

        if cls == 'Aves':
            ax0.scatter(
                plot_df.loc[mask, 'x'], plot_df.loc[mask, 'y'],
                c=color, s=10, alpha=0.90, linewidths=0,
                label=f"{cls} (n={count})",
                zorder=1
            )
        else:
            ax0.scatter(
                plot_df.loc[mask, 'x'], plot_df.loc[mask, 'y'],
                c=color, s=16, alpha=0.92, linewidths=0,
                label=f"{cls} (n={count})",
                zorder=3
            )

    ax0.set_xlabel(f"{reducer_label} dimension 1")
    ax0.set_ylabel(f"{reducer_label} dimension 2")
    ax0.set_title(
        f"DINOv3 embeddings ({model_source})\n"
        f"{reducer_label}\n"
        f"Ordered & colored by class_name"
    )
    ax0.legend(loc='best', fontsize=9, framealpha=0.95)
    ax0.grid(True, alpha=0.25)

    if raw_coords is not None and raw_reducer_label is not None:
        ax1 = axes[0, 1]
        raw_plot = pd.DataFrame({'x': raw_coords[:, 0], 'y': raw_coords[:, 1], 'class_name': labels})
        for cls in CLASS_ORDER:
            if cls not in raw_plot['class_name'].values:
                continue
            mask = (raw_plot['class_name'] == cls)
            color = CLASS_COLORS.get(cls, '#555555')
            count = int(mask.sum())
            if cls == 'Aves':
                ax1.scatter(
                    raw_plot.loc[mask, 'x'], raw_plot.loc[mask, 'y'],
                    c=color, s=2.5, alpha=0.22, linewidths=0,
                    label=f"{cls} (n={count})", zorder=1
                )
            else:
                ax1.scatter(
                    raw_plot.loc[mask, 'x'], raw_plot.loc[mask, 'y'],
                    c=color, s=16, alpha=0.92, linewidths=0,
                    label=f"{cls} (n={count})", zorder=3
                )
        ax1.set_xlabel(f"{raw_reducer_label} dimension 1")
        ax1.set_ylabel(f"{raw_reducer_label} dimension 2")
        ax1.set_title(f"Raw flattened mel-spectrograms (identical samples)\nraw pool+flatten+L2 → {raw_reducer_label}")
        ax1.legend(loc='best', fontsize=9, framealpha=0.95)
        ax1.grid(True, alpha=0.25)

    fig.suptitle(
        f"5s Spectrogram Feature Space — DINOv3 2D projection by taxonomic class_name\n"
        f"{base}",
        fontsize=10
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    png_path = output_dir / f"{base}.png"
    pdf_path = output_dir / f"{base}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")

    # ====================== 7. ARTIFACTS: coords CSV + summary ======================
    if not args.no_save_coords:
        coords_out = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'class_name': labels,
            'primary_label': primary_labels,
            'common_name': common_names,
            'spectrogram_npy_path': paths,
            'original_csv_row': original_rows,
        })
        # Keep the class_name ordering the user asked for
        coords_out['class_name'] = pd.Categorical(coords_out['class_name'], categories=CLASS_ORDER, ordered=True)
        coords_out = coords_out.sort_values('class_name')

        coords_csv_path = output_dir / f"{base}_coords.csv"
        coords_out.to_csv(coords_csv_path, index=False)
        print(f"  Saved: {coords_csv_path}")

        # Tiny human-readable summary
        summary_path = output_dir / f"{base}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("DINOv3 2D Feature Space Visualization\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"CSV source: {csv_path}\n")
            f.write(f"Tensor dir: {tensor_dir}\n")
            f.write(f"seed:       {args.seed}\n")
            f.write(f"reducer:    {reducer_label}\n")
            f.write(f"points:     {n_total}\n")
            f.write(f"embed_dim:  {emb_dim}\n")
            f.write(f"model_source: {model_source}\n\n")
            f.write("Counts by class_name (as ordered for visualization):\n")
            for cls in CLASS_ORDER:
                c = sum(1 for l in labels if l == cls)
                if c > 0:
                    f.write(f"  {cls:10s} : {c:6d}\n")
            if raw_coords is not None:
                f.write("\nSide-by-side raw flattened comparison was generated.\n")
            f.write("\nNote: DINOv3 embedding logic mirrors Spectrogram_DinoV3/Dinov3DataStream.py exactly\n")
            f.write("      (_extract_embedding + custom/generic model load).\n")
            f.write("      Sampling, class_name ordering, plotting style and artifact layout mirror\n")
            f.write("      visualize_spectrogram_feature_space.py exactly.\n")
        print(f"  Saved: {summary_path}")

    print("\n=== Complete ===")
    print(f"All artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
