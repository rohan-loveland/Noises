"""
visualize_spectrogram_feature_space.py

Flatten 5s mel-spectrograms (exactly as used by A_RED today) and project them
to 2D so we can visually inspect the raw feature-space structure.

Primary goal: color and order points by `class_name` (Aves, Mammalia, Insecta,
Amphibia, Reptilia) from the CSV to see whether any natural grouping or
separation emerges (e.g. are mammals close to avians in this space?).

Data access + flattening logic is taken directly from the reference:
  - Spectrogram_A_RED/main_spectrogram.py
  - Spectrogram_A_RED/SpectrogramDataStream.py  (the pool/flatten/L2 block)

This script is deliberately standalone (no modifications to other files).
For --reducer umap you must first `pip install umap-learn`.

Usage examples (PowerShell / Windows):
    python visualize_spectrogram_feature_space.py --max_aves 2000 --reducer pca
    python visualize_spectrogram_feature_space.py --max_aves 15000 --reducer tsne --seed 42
    python visualize_spectrogram_feature_space.py --max_aves 400 --reducer umap --seed 420
    python visualize_spectrogram_feature_space.py --max_aves 0 --compare_embeddings   # only the rare classes + side-by-side embeddings if available

Outputs (in --output_dir):
    - <name>.png / .pdf   : the 2D scatter(s)
    - <name>_coords.csv   : the 2D coordinates + class_name + paths (reusable)
    - <name>_summary.txt  : human-readable run metadata
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


# ====================== DEFAULTS (match the rest of the project) ======================
DEFAULT_CSV = "5sSpectrograms_tensors/train_5s_spectrograms.csv"
DEFAULT_TENSOR_DIR = "5sSpectrograms_tensors"
DEFAULT_MAX_AVES = 15000
DEFAULT_SEED = 42
DEFAULT_POOL_SIZE = 16
DEFAULT_OUTPUT_DIR = "viz_outputs"

# Fixed color palette for class_name.
# Aves is intentionally desaturated / low-alpha so the minority classes remain visible.
CLASS_COLORS = {
    'Aves': '#8fb8e6',       # soft blue-gray (background layer)
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
    Load one .npy spectrogram and turn it into the exact same 1D vector that
    A_RED receives today.

    This function is a direct, self-contained copy of the transformation in:
        Spectrogram_A_RED/SpectrogramDataStream.py
        (see stream_new_data_point around lines 51-74 and the pooling logic)

    Steps (identical to current production usage):
        1. np.load -> float32, shape (128, 312 or 313)
        2. Mean-pool the time axis using windows of `pool_size` (default 16)
           -> reduces to ~128 x 19-20 ≈ 2.4-2.5k dimensions
        3. Flatten
        4. L2 normalize to unit vector

    Returns a 1-D float32 array (unit length).
    """
    npy_path = tensor_dir / npy_rel_path
    if not npy_path.exists():
        raise FileNotFoundError(f"Spectrogram tensor not found: {npy_path}")

    spec = np.load(npy_path).astype(np.float32)

    # Mean-pool time axis (handles variable lengths 312/313).
    # This is the critical step that makes the ~40k-dim raw spectrograms tractable.
    if spec.ndim == 2:
        time_steps = spec.shape[1]
        num_pools = time_steps // pool_size
        if num_pools > 0:
            spec = np.mean(
                spec[:, :num_pools * pool_size].reshape(spec.shape[0], num_pools, pool_size),
                axis=2
            )
        # If too short for any full pool, we just keep the original (rare).

    vec = spec.flatten()

    # L2 normalization (unit vector) — same as the DataStream.
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    else:
        print(f"  WARNING: zero-norm spectrogram encountered: {npy_rel_path}")

    return vec.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Project flattened 5s mel-spectrograms to 2D, colored and ordered by class_name (Aves/Mammalia/etc).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_spectrogram_feature_space.py --max_aves 800 --reducer pca
  python visualize_spectrogram_feature_space.py --max_aves 15000 --reducer tsne --seed 420
  python visualize_spectrogram_feature_space.py --max_aves 400 --reducer umap --seed 420
  python visualize_spectrogram_feature_space.py --max_aves 0 --compare_embeddings   # minorities only + embeddings if present
        """
    )
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV,
                        help="Path to the train_5s_spectrograms.csv")
    parser.add_argument("--tensor_dir", type=str, default=DEFAULT_TENSOR_DIR,
                        help="Directory containing the .npy tensors (and optionally *embeddings.npy)")
    parser.add_argument("--max_aves", type=int, default=DEFAULT_MAX_AVES,
                        help="How many Aves samples to keep (ALL non-Aves are always included). Use 0 to see only minorities.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for the Aves subsample (reproducibility)")
    parser.add_argument("--pool_size", type=int, default=DEFAULT_POOL_SIZE,
                        help="Time-axis pooling size (must match what A_RED is using, default 16)")
    parser.add_argument("--reducer", choices=["pca", "tsne", "umap"], default="pca",
                        help="'pca' = fast PCA(2). 'tsne' = PCA(50) then TSNE(2) (much slower but often nicer). 'umap' = UMAP(2) (local + global structure, good default for these visualizations).")
    parser.add_argument("--umap_n_neighbors", type=int, default=15,
                        help="n_neighbors for UMAP (default 15). Only used when --reducer=umap.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="min_dist for UMAP (default 0.1). Only used when --reducer=umap.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Where to write the PNG/PDF/CSV artifacts")
    parser.add_argument("--compare_embeddings", action="store_true",
                        help="If any of perch_embeddings.npy / dinov3_embeddings.npy exist in tensor_dir, "
                             "also project the EXACT same selected samples through that embedding space and "
                             "produce side-by-side panels.")
    parser.add_argument("--no_save_coords", action="store_true",
                        help="Do not write the _coords.csv (saves a little disk/IO)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    tensor_dir = Path(args.tensor_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Flattened Spectrogram 2D Feature Space Visualization ===")
    print(f"Reference flattening logic: Spectrogram_A_RED/SpectrogramDataStream.py (pool+flatten+L2)")
    print(f"CSV:            {csv_path}")
    print(f"Tensor dir:     {tensor_dir}")
    print(f"max_aves:       {args.max_aves}")
    print(f"seed:           {args.seed}")
    print(f"pool_size:      {args.pool_size}")
    print(f"reducer:        {args.reducer}")
    print(f"output_dir:     {output_dir}")
    print(f"compare_embeddings: {args.compare_embeddings}")
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
    # full-size embedding array (perch, dinov3, ...) that was written in CSV row order.
    original_rows = selected['original_csv_row'].to_numpy(dtype=np.int64)

    # ====================== 3. FEATURE EXTRACTION (flattened spectrograms) ======================
    print("Extracting flattened spectrogram vectors (pool + flatten + L2)...")
    start_load = time.time()

    X_list = []
    kept_paths = []
    kept_class_names = []
    kept_primary = []
    kept_common = []
    kept_orig_rows = []

    for _, row in tqdm(selected.iterrows(), total=n_total, desc="load+flatten"):
        rel = str(row['spectrogram_npy_path'])
        try:
            vec = load_and_flatten_spectrogram(rel, tensor_dir, pool_size=args.pool_size)
            X_list.append(vec)
            kept_paths.append(rel)
            kept_class_names.append(str(row['class_name']))
            kept_primary.append(str(row.get('primary_label', '')))
            kept_common.append(str(row.get('common_name', row.get('scientific_name', ''))))
            kept_orig_rows.append(int(row['original_csv_row']))
        except FileNotFoundError:
            print(f"  WARNING: file not found, skipping: {rel}")
        except Exception as e:
            print(f"  WARNING: failed to load {rel}: {e}")

    if not X_list:
        print("ERROR: No spectrograms were successfully loaded. Nothing to visualize.")
        return

    X = np.vstack(X_list).astype(np.float32)
    load_time = time.time() - start_load
    print(f"  Extracted {len(X):,} vectors of dim {X.shape[1]} in {load_time:.1f}s")
    print()

    # Update the "kept" lists (in case any files were missing)
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
        # PCA first to 50 dims (standard practice before t-SNE on high-dim data)
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

    # ====================== 5. OPTIONAL: SAME SAMPLES THROUGH EMBEDDING SPACE ======================
    emb_coords = None
    emb_label = None
    emb_source = None

    if args.compare_embeddings:
        emb_candidates = [
            ("perch", tensor_dir / "perch_embeddings.npy"),
            ("dinov3", tensor_dir / "dinov3_embeddings.npy"),
            ("dinov2", tensor_dir / "dinov2_embeddings.npy"),
        ]
        for name, emb_path in emb_candidates:
            if emb_path.exists():
                print(f"Loading precomputed embeddings for side-by-side: {emb_path.name}")
                try:
                    all_emb = np.load(emb_path, mmap_mode='r')
                    if len(all_emb) < len(df):
                        print(f"  WARNING: embeddings have only {len(all_emb)} rows (CSV has {len(df)}). Using what we can.")
                    # Index using the original CSV row numbers we captured earlier
                    emb_subset = np.asarray(all_emb[original_rows])
                    # Run identical-style reduction on the embeddings for a fair visual comparison
                    if args.reducer == "pca":
                        e_red = PCA(n_components=2, random_state=args.seed).fit_transform(emb_subset)
                        emb_label = "PCA(2)"
                    elif args.reducer == "tsne":
                        e_pre = PCA(n_components=min(50, emb_subset.shape[1]), random_state=args.seed).fit_transform(emb_subset)
                        e_tsne = TSNE(n_components=2, random_state=args.seed,
                                      perplexity=min(30, max(5, (len(e_pre)-1)//3)), max_iter=1000)
                        e_red = e_tsne.fit_transform(e_pre)
                        emb_label = "PCA(50)+TSNE(2)"
                    elif args.reducer == "umap":
                        if umap is None:
                            print("ERROR: --reducer umap requires the umap-learn package (for side-by-side).")
                            emb_coords = None
                            break
                        print(f"  (compare) Running UMAP(n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist}) on {name} embeddings...")
                        u = umap.UMAP(
                            n_components=2,
                            random_state=args.seed,
                            n_neighbors=args.umap_n_neighbors,
                            min_dist=args.umap_min_dist,
                            verbose=0
                        )
                        e_red = u.fit_transform(emb_subset)
                        emb_label = "UMAP(2)"
                    else:
                        e_red = None
                    if e_red is not None:
                        emb_coords = e_red
                        emb_source = name
                        print(f"  Successfully projected {len(emb_subset):,} {name} embedding vectors.")
                        break
                except Exception as e:
                    print(f"  Could not use {emb_path.name}: {e}")
        if emb_coords is None:
            print("  --compare_embeddings was requested but no usable *_embeddings.npy was found or loadable.")

    # ====================== 6. VISUALIZATION (class_name ordered + styled) ======================
    print("Generating plot(s)...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"spectrogram_2d_{args.reducer}_{n_total}pts_s{args.seed}_{timestamp}"

    # Build a small frame for convenient grouped plotting
    plot_df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'class_name': labels,
    })

    n_panels = 2 if emb_coords is not None else 1
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
                c=color, s=10, alpha=0.80, linewidths=0,
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
        f"Raw flattened mel-spectrograms\n"
        f"pool_size={args.pool_size} + L2  →  {reducer_label}\n"
        f"Ordered & colored by class_name"
    )
    ax0.legend(loc='best', fontsize=9, framealpha=0.95)
    ax0.grid(True, alpha=0.25)

    if emb_coords is not None and emb_label is not None:
        ax1 = axes[0, 1]
        emb_plot = pd.DataFrame({'x': emb_coords[:, 0], 'y': emb_coords[:, 1], 'class_name': labels})
        for cls in CLASS_ORDER:
            if cls not in emb_plot['class_name'].values:
                continue
            mask = (emb_plot['class_name'] == cls)
            color = CLASS_COLORS.get(cls, '#555555')
            count = int(mask.sum())
            if cls == 'Aves':
                ax1.scatter(
                    emb_plot.loc[mask, 'x'], emb_plot.loc[mask, 'y'],
                    c=color, s=10, alpha=0.90, linewidths=0,
                    label=f"{cls} (n={count})", zorder=1
                )
            else:
                ax1.scatter(
                    emb_plot.loc[mask, 'x'], emb_plot.loc[mask, 'y'],
                    c=color, s=16, alpha=0.92, linewidths=0,
                    label=f"{cls} (n={count})", zorder=3
                )
        ax1.set_xlabel(f"{emb_label} dimension 1")
        ax1.set_ylabel(f"{emb_label} dimension 2")
        ax1.set_title(f"{emb_source.upper()} embeddings (identical samples)\n{emb_label}")
        ax1.legend(loc='best', fontsize=9, framealpha=0.95)
        ax1.grid(True, alpha=0.25)

    fig.suptitle(
        f"5s Spectrogram Feature Space — 2D projection by taxonomic class_name\n"
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
            f.write("Flattened Spectrogram 2D Feature Space Visualization\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"CSV source: {csv_path}\n")
            f.write(f"Tensor dir: {tensor_dir}\n")
            f.write(f"pool_size:  {args.pool_size}\n")
            f.write(f"seed:       {args.seed}\n")
            f.write(f"reducer:    {reducer_label}\n")
            f.write(f"points:     {n_total}\n\n")
            f.write("Counts by class_name (as ordered for visualization):\n")
            for cls in CLASS_ORDER:
                c = sum(1 for l in labels if l == cls)
                if c > 0:
                    f.write(f"  {cls:10s} : {c:6d}\n")
            if emb_source:
                f.write(f"\nSide-by-side embedding source: {emb_source}\n")
                f.write(f"Embedding reducer: {emb_label}\n")
            f.write("\nNote: flattening logic mirrors SpectrogramDataStream.py exactly.\n")
        print(f"  Saved: {summary_path}")

    print("\n=== Complete ===")
    print(f"All artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
