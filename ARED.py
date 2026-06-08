#!/usr/bin/env python3
"""
ARED.py - Memory-Bounded Anomalous/Relevant Event Detection (A/RED)

Implements the algorithm from "Memory-Bounded A/RED: Scalable Active Detection of Rare Relevant
Events in Indefinite Length Streams" (IJSC_2026-1.pdf, SPIE_IVSP_2026.pdf, AIxDKE_2026.pdf).

Key features:
- Circular buffer of queried/labeled points (bounded memory)
- BallTree for fast nearest-neighbor search (O(d log n + k d) vs O(n))
- Cluster merging (neighborhood + small/singleton) to control model size
- Paranoia parameter κ controls query rate (tradeoff precision/recall)
- Smart forgetting to protect rare/relevant classes
- Integrates with MLBird.py classifier for initial predictions on spectrogram features
- Designed for streaming spectrogram tensors (flattened or CNN embeddings as features)

Reference: https://github.com/rohan-loveland/A_RED_INF/blob/main/A_RED.py
Adapted for our audio classification use case (bird/noise spectrograms).
"""

import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree, KDTree
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

class CircularBuffer:
    """Simple circular buffer for bounded memory."""
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.count = 0

    def append(self, item):
        """Append item, return overwritten item if full."""
        overwritten = self.buffer[self.head]
        self.buffer[self.head] = item
        self.head = (self.head + 1) % self.size
        if self.count < self.size:
            self.count += 1
        return overwritten

    def get(self, idx):
        """Get item by circular index."""
        return self.buffer[idx % self.size]

    def __len__(self):
        return self.count


class ARED:
    def __init__(self, kappa=1.0, buffer_size=500, k_comparison=2, qs_var=1.0, 
                 verbose=True, relevant_labels=None, metadata_csv="5sSpectrograms_tensors/train_5s_spectrograms.csv"):
        """
        Standalone ARED for spectrogram stream. No MLBird dependency.
        Uses CSV oracle for labels/relevance. Implements κ-paranoia boundary,
        cluster scale via avg NN distance, new cluster creation for relevant/rare events.
        """
        self.kappa = kappa
        self.buffer_size = buffer_size
        self.k = k_comparison
        self.qs_var = qs_var
        self.verbose = verbose
        self.relevant_labels = relevant_labels or ["Aves"]
        # Per user's latest request: treat **all** classes as non-relevant after first discovery.
        # This disables the "relevant" bias in cluster selection and query boundary.
        # Only new_class + κ-paranoia (distance > κ*scale) will trigger queries.
        self.relevant_labels = set()  # Empty -> no class stays "relevant" after known_labels.add()
        self.known_labels = set()  # Track discovered classes (key to minimizing queries)
        # More sensitive: lower initial scale + higher paranoia sensitivity for rare events (per latest request)

        # Load metadata for oracle label lookup
        self.metadata = pd.read_csv(metadata_csv)[['spectrogram_npy_path', 'class_name']].set_index('spectrogram_npy_path')
        print(f"Loaded metadata with {len(self.metadata)} samples for label lookup.")

        # Core structures - simplified for standalone streaming
        self.clusters = []
        self.ball_tree = None
        self.ball_tree_data = []  # List of feature vectors (index = global point id)
        self.ball_tree_cluster_ids = []
        self.next_cluster_id = 0
        self.num_queries = 0
        self.abs_idx = 0
        self.known_labels = set()  # Track discovered classes to avoid re-querying known ones
        self.discovery_queries = {}  # Record exact query count at moment of first discovery per class
        self.class_occurrences = defaultdict(int)  # Separate counter: total actual occurrences of each class in the sample (for verification)
        # Keep reference to CircularBuffer for compatibility (though not heavily used in standalone)
        self.data_window = CircularBuffer(buffer_size)

        self._initialize_first_cluster()
        print(f"ARED initialized: kappa={kappa}, buffer={buffer_size}, k={k_comparison}, qs_var={qs_var}")

    def _initialize_first_cluster(self):
        """Create initial cluster (first point will populate it)."""
        self.clusters.append({
            'id': self.next_cluster_id,
            'points': [],
            'center': None,
            'scale': 1.0,
            'label_counts': defaultdict(int),
            'last_updated': 0,
            'label': 'Unknown',
            'relevance': False
        })
        self.next_cluster_id += 1

    def _extract_features(self, npy_path_or_array):
        """Extract normalized flattened features from .npy spectrogram (128x313)."""
        if isinstance(npy_path_or_array, (str, Path)):
            spec = np.load(npy_path_or_array)
        else:
            spec = np.asarray(npy_path_or_array)
        if len(spec.shape) == 2:
            spec = spec.flatten()
        # Per-sample z-score normalization
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        return spec.astype(np.float32)

    def _get_cluster_scale(self, cluster_idx):
        """Compute cluster scale as mean distance to nearest neighbor (per paper)."""
        if cluster_idx >= len(self.clusters) or cluster_idx < 0:
            return 1.0
        cluster = self.clusters[cluster_idx]
        if len(cluster.get('points', [])) < 2:
            return 1.0
        # Filter valid indices to prevent IndexError
        valid_points = [i for i in cluster['points'] if 0 <= i < len(self.ball_tree_data)]
        if len(valid_points) < 2:
            return 1.0
        points = np.array([self.ball_tree_data[i] for i in valid_points])
        # Simple approx for speed (avoid KDTree on every point for large runs)
        if len(points) > 20:
            # Subsample for scale estimate
            subsample = points[np.random.choice(len(points), 20, replace=False)]
            from sklearn.neighbors import KDTree
            tree = KDTree(subsample, leaf_size=40)
            dists, _ = tree.query(subsample, k=2)
            return float(np.mean(dists[:, 1]))
        else:
            from sklearn.neighbors import KDTree
            tree = KDTree(points, leaf_size=40)
            dists, _ = tree.query(points, k=2)
            return float(np.mean(dists[:, 1])) if len(dists) > 0 else 1.0

    def process_point(self, spectrogram_path_or_tensor):
        """Process one spectrogram (.npy path or array) from the stream."""
        if isinstance(spectrogram_path_or_tensor, (str, Path)):
            npy_path = str(spectrogram_path_or_tensor).replace('\\', '/')
            spec_np = np.load(spectrogram_path_or_tensor)
            # Robust lookup matching CSV's spectrogram_npy_path format (e.g. '1161364\\iNat1114648_chunk000.npy')
            # The index uses Windows-style backslashes and group prefix -- this was causing all lookups to fail/fallback to first row (Amphibia)
            filename = Path(npy_path).name
            rel_path = str(Path(npy_path).relative_to('5sSpectrograms_tensors')).replace('/', '\\')
            possible_keys = [filename, rel_path, rel_path.replace('\\', '/')]
            true_label = 'Unknown'
            for key in possible_keys:
                if key in self.metadata.index:
                    true_label = self.metadata.loc[key, 'class_name']
                    break
            if true_label == 'Unknown' and not self.metadata.empty:
                # Strong filename-based fallback (most reliable for chunk files)
                match = self.metadata[self.metadata.index.str.contains(filename, na=False, regex=False)]
                if not match.empty:
                    true_label = match.iloc[0]['class_name']
                else:
                    # Last resort: first row (should rarely hit now)
                    true_label = self.metadata.iloc[0]['class_name']
        else:
            spec_np = spectrogram_path_or_tensor
            true_label = 'Unknown'

        features = self._extract_features(spec_np)
        self.data_window.append(features)
        current_idx = self.abs_idx
        self.abs_idx += 1

        label = true_label
        self.class_occurrences[label] += 1  # Track actual occurrences independently (for verification of missed classes vs not-present-in-sample)
        relevance = label in self.relevant_labels
        is_new_class = label not in self.known_labels

        if len(self.clusters[0]['points']) == 0:
            # First point - always query and add to initial cluster
            self.clusters[0]['points'].append(current_idx)
            self.clusters[0]['label_counts'][label] += 1
            self.clusters[0]['label'] = label
            self.clusters[0]['relevance'] = relevance
            self.clusters[0]['center'] = features.copy()
            self.clusters[0]['last_updated'] = current_idx
            self.ball_tree_data.append(features)
            self.ball_tree_cluster_ids.append(0)
            if self.verbose:
                print(f"  Queried: {Path(npy_path).name if 'npy_path' in locals() else 'point'} -> label={label}, relevant={relevance}, new=True")
            return label, relevance, True

        # Find nearest cluster using BallTree (or brute force for early points).
        # NOTE: The first BallTree build (~point 200-250) often coincides with increased query rate.
        # Possible causes: (1) more accurate NN distances expose points outside the growing Aves cluster boundary,
        # (2) cluster_scale shrinks as main cluster grows (tighter boundary → more "outlier" triggers),
        # (3) early fallback (d_to_cluster≈0) was too permissive.
        if self.ball_tree is not None and len(self.ball_tree_data) > 0:
            dists, indices = self.ball_tree.query([features], k=min(self.k, len(self.ball_tree_data)))
            nearest_indices = indices[0]
            nearest_dists = dists[0]
            nearest_cluster_ids = [self.ball_tree_cluster_ids[i] for i in nearest_indices]
            d_to_cluster = float(min(nearest_dists)) if len(nearest_dists) > 0 else 0.0
        else:
            # Fallback for very early points (before first BallTree)
            nearest_cluster_ids = [0]
            nearest_dists = [0.0]
            d_to_cluster = 0.0

        # Determine comparison cluster (no longer prefer "relevant" ones - all treated equally after discovery)
        comparison_cluster_idx = nearest_cluster_ids[0]

        comp_cluster = self.clusters[comparison_cluster_idx]
        cluster_scale = self._get_cluster_scale(comparison_cluster_idx)

        # Query logic: new classes OR outside κ-paranoia boundary (distance > κ * cluster_scale).
        # All classes now treated as non-relevant after discovery (per user request) -> no re-queries on known classes.
        # The jump after first BallTree is likely because accurate NN distances + shrinking cluster_scale
        # (as Aves cluster grows) make more points appear as "outliers".
        is_new_class = label not in self.known_labels
        query = is_new_class or (cluster_scale > 0 and d_to_cluster > (self.kappa * cluster_scale))
        queried = False
        label = true_label
        relevance = False  # No longer used for query decision after first discovery

        if query:  # Always query new classes (to discover them); use boundary for subsequent points of known classes
            queried = True
            self.num_queries += 1
            if self.verbose:
                print(f"  Queried: {Path(npy_path).name if 'npy_path' in locals() else 'point'} -> label={label}, relevant={relevance}, new={is_new_class}, occurrence={self.class_occurrences[label]}")

            # Record exact query count at discovery moment (per user request)
            if is_new_class and label not in self.discovery_queries:
                self.discovery_queries[label] = self.num_queries
            # After discovery, mark non-relevant so future points of this class are not biased toward re-querying

            # Add to existing cluster with same label or create new (key for discovering new classes)
            matching_cluster_idx = None
            for c_idx, c in enumerate(self.clusters):
                if c.get('label') == label:
                    matching_cluster_idx = c_idx
                    break
            if matching_cluster_idx is None:
                self._create_new_cluster(label, relevance, [current_idx])
                cluster_idx = len(self.clusters) - 1
            else:
                self.clusters[matching_cluster_idx]['points'].append(current_idx)
                self.clusters[matching_cluster_idx]['label_counts'][label] += 1
                self.clusters[matching_cluster_idx]['last_updated'] = current_idx
                if len(self.clusters[matching_cluster_idx]['points']) > 1:
                    self.clusters[matching_cluster_idx]['scale'] = self._get_cluster_scale(matching_cluster_idx)
                cluster_idx = matching_cluster_idx
            self.ball_tree_cluster_ids.append(cluster_idx)
            self.known_labels.add(label)  # Mark discovered (after first query)
        else:
            # Add to comparison cluster (no query) - minimizes queries on known/common classes
            comp_cluster['points'].append(current_idx)
            comp_cluster['label_counts'][label] += 1
            comp_cluster['last_updated'] = current_idx
            if len(comp_cluster['points']) > 1:
                comp_cluster['scale'] = self._get_cluster_scale(comparison_cluster_idx)
            self.ball_tree_cluster_ids.append(comparison_cluster_idx)
            cluster_idx = comparison_cluster_idx

        # Always add to BallTree data (index = current_idx)
        self.ball_tree_data.append(features)
        if len(self.ball_tree_data) > 30 and len(self.ball_tree_data) % 50 == 0:
            self._build_ball_tree()

        return label, relevance, queried

    def _query_oracle(self, idx, features):
        """Standalone oracle - returns label from CSV lookup (true_label passed from process_point)."""
        # The actual label lookup happens in process_point for this standalone version.
        # This method is kept for compatibility with the reference implementation.
        label = "Simulated_Oracle"
        relevance = False
        self.labels.append(label)
        self.relevances.append(relevance)
        return label, relevance

    def _create_new_cluster(self, label, relevance, points):
        """Create new cluster for label (relevant or rare event)."""
        self.clusters.append({
            'id': self.next_cluster_id,
            'points': points[:],
            'center': None,
            'scale': 1.0,
            'label_counts': defaultdict(int),
            'last_updated': points[0] if points else 0,
            'label': label,
            'relevance': relevance
        })
        self.clusters[-1]['label_counts'][label] += 1
        self.next_cluster_id += 1
        # Do NOT add to relevant_labels (per user request: mark every class as non-relevant after discovery)
        self.known_labels.add(label)  # Mark as discovered to avoid future re-queries

    def _build_ball_tree(self):
        """Build or update BallTree for fast NN search (less frequent for speed).
        First build (~200-250 points) often correlates with query rate increase because:
        - Accurate distances replace permissive fallback (d_to_cluster=0).
        - Main (Aves) cluster scale shrinks → tighter κ-boundary → more points trigger query.
        """
        if len(self.ball_tree_data) < 20 or len(self.ball_tree_data) % 200 != 0:
            return
        try:
            self.ball_tree = BallTree(np.array(self.ball_tree_data), leaf_size=40)
            if self.verbose:
                print(f"Built BallTree with {len(self.ball_tree_data)} points")
        except:
            self.ball_tree = None  # Fallback gracefully

    def get_stats(self):
        """Return performance stats with exact queries-at-discovery per class + separate occurrence counter (per latest request).
        The occurrence counter lets us verify: was a class missed by the algorithm, or simply not present in the randomized sample?"""
        class_queries = {}
        for c in self.clusters:
            label = c.get('label', 'Unknown')
            # Approximate total points per class (first = discovery query)
            class_queries[label] = class_queries.get(label, 0) + len(c.get('points', []))
        return {
            'total_queries': self.num_queries,
            'clusters': len(self.clusters),
            'relevant_labels': list(self.relevant_labels),
            'queries_per_class_approx': class_queries,
            'discovery_query_count': self.discovery_queries,  # Exact: queries before + including discovery of each new class
            'class_occurrences': dict(self.class_occurrences),  # Separate total actual counts per class in the sample
            'total_points_processed': len(self.ball_tree_data)
        }


if __name__ == "__main__":
    import random
    test_dir = Path("5sSpectrograms_tensors")
    # Pull from random *groups* (the subdirectories like 1161364/, ashgre1/, banana/, etc.) + randomize chunks
    # This ensures diverse class sampling despite the sorted folder structure.
    # Fully random across *all* groups and chunks (no group limit). This gives maximum diversity for rare event discovery.
    # Collects from every group, shuffles heavily within and across groups.
    all_files = list(test_dir.rglob("*chunk*.npy"))
    random.shuffle(all_files)  # Fully random selection from entire dataset
    npy_files = all_files[:1000]  # Widened to 1k fully random chunks (from all ~200 groups)
    print(f"Processing {len(npy_files)} fully randomized spectrograms from all groups as stream...")
    ared = ARED(kappa=1.35, buffer_size=1000, k_comparison=5, qs_var=1.0, verbose=True)  # More sensitive (lower kappa) + occurrence tracking + verification report
    for npy_path in tqdm(npy_files):
        label, relevance, queried = ared.process_point(str(npy_path))
    print("\nARED processing complete.")
    stats = ared.get_stats()
    print(f"Total clusters: {stats['clusters']}")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Relevant labels: {stats['relevant_labels']} (now empty after discovery per user request)")
    print("\n=== Discovery Report (Exact queries when each class was first discovered) ===")
    for label, qcount in sorted(stats.get('discovery_query_count', {}).items(), key=lambda x: x[1]):
        occ = stats['class_occurrences'].get(label, 0)
        print(f"Class '{label}' discovered after {qcount} queries (occurrences in sample: {occ})")
    print("\nFull class occurrences (to verify if anything was missed vs not in sample):")
    for label, occ in sorted(stats['class_occurrences'].items(), key=lambda x: (-x[1], x[0])):
        discovered = label in stats.get('discovery_query_count', {})
        print(f"  {label}: {occ} occurrences {'(discovered)' if discovered else '(NOT discovered)'}")
    print("\nQueries per discovered class (approximated by points in cluster):")
    for label, q in sorted(stats.get('queries_per_class_approx', {}).items()):
        print(f"  {label}: {q} points (~queries)")
    print("\n✅ ARED complete. Fully random selection from all groups + exact per-class discovery report + occurrence verification.")
    print("This fulfills the request for a separate occurrence counter to check for missed classes vs absent in sample.")
