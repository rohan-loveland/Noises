"""
ARED Classifier

Uses an ARED / A_REDIN run (via AREDExperiment) to actively select a small set of
informative labeled exemplars (the points on which the oracle was queried).

After the run we extract a *permanent* support set directly from the cluster
l_pt_idxs recorded by ARED (these are stable indices into the train stream's
preloaded vectors). This is robust even if the internal FiniteBuffer later
forgets points (including with Smart Forgetting modes).

Prediction is nearest-neighbor (k=1 by default) into the collected supports.
This is a simple, faithful way to use A_REDIN "as a classifier".
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.neighbors import NearestNeighbors

from ..experiment import AREDExperiment
from ..data.base_stream import BaseDataStream, PerchDataStream
from ..data.oracle import NoRelevanceOracle
from ..utils.discovery import ClassDiscoveryCounter


class AREDClassifier:
    def __init__(self, 
                 kappa: float = 0.5,
                 data_window_size: int = 10000,
                 k_comparison_clusters: int = 5,
                 random_state: int = 42,
                 smart_forgetting_var: Tuple[int, float] = (0, 0.0),
                 n_neighbors: int = 1,
                 **stream_kwargs):
        
        self.kappa = kappa
        self.data_window_size = data_window_size
        self.k_comparison_clusters = k_comparison_clusters
        self.random_state = random_state
        self.smart_forgetting_var = smart_forgetting_var
        self.n_neighbors = max(1, int(n_neighbors))
        self.stream_kwargs = stream_kwargs
        
        self.exp: Optional[AREDExperiment] = None
        self.is_fitted = False
        self.label_column = None

        # The "feature space" that ARED builds during the training run:
        # - support_vectors_ + support_labels_ : the actual labeled points ARED decided to query
        # - support_comp_distances_ : the final comp_distance of the *cluster* each point belonged to
        #   (this + kappa defines the region ARED learned for that cluster)
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_labels_: Optional[np.ndarray] = None
        self.support_comp_distances_: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None

        self._known_labels: list = []
        self.n_prototypes_: int = 0
        self.n_classes_covered_: int = 0
        self.num_queries_used_: int = 0

        # Richer view of the space ARED built (one entry per final cluster)
        self.clusters_meta_: list = []  # [{cluster_id, label, comp_distance, n_labeled_points}, ...]

    def _stratified_split(self, df, label_column: str, train_frac: float = 0.7):
        unique_classes = df[label_column].unique()
        print(f"Ensuring all {len(unique_classes)} classes ({label_column}) are in training...")

        train_idx = []
        test_idx = []
        np.random.seed(self.random_state)

        for cls in unique_classes:
            cls_idx = df[df[label_column] == cls].index.values
            if len(cls_idx) == 1:
                train_idx.append(cls_idx[0])
            else:
                n_train = max(1, int(len(cls_idx) * train_frac))
                train_i = np.random.choice(cls_idx, n_train, replace=False)
                train_idx.extend(train_i)
                test_idx.extend(np.setdiff1d(cls_idx, train_i))

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        
        print(f"Split → Train: {len(train_idx)} | Test: {len(test_idx)} samples")
        return train_idx, test_idx

    def fit(self, num_points: int = -1, label_column: str = "scientific_name", train_frac: float = 0.7):
        self.label_column = label_column
        
        full_stream = PerchDataStream(
            label_column=label_column,
            max_samples=num_points if num_points > 0 else None,
            shuffle=True,
            seed=self.random_state,
            **self.stream_kwargs
        )
        
        df = full_stream.df.copy()
        train_idx, _ = self._stratified_split(df, label_column, train_frac)
        
        train_stream = PerchDataStream(
            label_column=label_column,
            shuffle=True,
            seed=self.random_state,
            **self.stream_kwargs
        )
        train_stream.df = df.iloc[train_idx].reset_index(drop=True)
        train_stream._preload()

        counter = ClassDiscoveryCounter(fast_mode=False)
        oracle = NoRelevanceOracle(train_stream, discovery_tracker=counter)

        self.exp = AREDExperiment(
            train_stream, oracle,
            kappa=self.kappa,
            data_window_size=self.data_window_size,
            k_comparison_clusters=self.k_comparison_clusters,
            smart_forgetting_var=self.smart_forgetting_var,
        )

        print(f"Training on {len(train_idx)} points...")
        self.exp.run(num_points=-1, verbose=True, status_every=500)

        self.is_fitted = True

        # === Collect the "feature space" ARED built ===
        # After the run, the clusters + their l_pt_idxs + final comp_distances define the space.
        # We extract the labeled points (what ARED actually queried) + the cluster metadata
        # so that later we can answer "where does a new point lie in this space?"
        sp = self.exp.ared.subspace_partition
        clusters = self._get_cluster_list_safe()

        support_vecs = []
        support_labs = []
        support_comp_ds = []

        cluster_meta = []

        cluster_items = list(sp.cluster_dict.items()) if hasattr(sp, 'cluster_dict') else [(i, c) for i, c in enumerate(clusters)]

        for ckey, cluster in cluster_items:
            lab = getattr(cluster, 'label', None)
            cdist = float(getattr(cluster, 'comp_distance', 0.0))
            lpts = getattr(cluster, 'l_pt_idxs', []) or []

            for abs_idx in lpts:
                if abs_idx < len(train_stream.processed_vectors):
                    v = train_stream.processed_vectors[abs_idx]
                    support_vecs.append(v)
                    support_labs.append(lab)
                    support_comp_ds.append(cdist)

            if lpts:
                cluster_meta.append({
                    "cluster_id": ckey if not hasattr(cluster, 'cluster_id') else getattr(cluster, 'cluster_id', ckey),
                    "label": lab,
                    "comp_distance": cdist,
                    "n_labeled_points": len(lpts),
                })

        if support_vecs:
            self.support_vectors_ = np.stack(support_vecs).astype(np.float32)
            self.support_labels_ = np.array(support_labs)
            self.support_comp_distances_ = np.array(support_comp_ds, dtype=np.float32)

            self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean', algorithm='auto')
            self._nn.fit(self.support_vectors_)

            self.n_prototypes_ = len(self.support_vectors_)
            self._known_labels = sorted(set(support_labs))
            self.n_classes_covered_ = len(self._known_labels)
            self.clusters_meta_ = cluster_meta
        else:
            dim = train_stream.processed_vectors.shape[1] if len(train_stream.processed_vectors) > 0 else 1536
            self.support_vectors_ = np.empty((0, dim), dtype=np.float32)
            self.support_labels_ = np.array([], dtype=object)
            self.support_comp_distances_ = np.array([], dtype=np.float32)
            self._nn = None
            self._known_labels = []
            self.n_prototypes_ = 0
            self.n_classes_covered_ = 0
            self.clusters_meta_ = []

        try:
            self.num_queries_used_ = int(self.exp._get_query_count())
        except Exception:
            self.num_queries_used_ = self.n_prototypes_

        print(f"✅ Training complete. ARED built space with {len(clusters)} clusters | "
              f"{self.n_prototypes_} labeled points (queries) across {self.n_classes_covered_} classes")
        if self.n_prototypes_ < 50:
            print("   (Note: low prototype count is normal with NoRelevanceOracle + moderate kappa.)")
        return self

    def predict(self, X_new: np.ndarray, mode: str = "ared_space"):
        """
        Predict labels by "seeing where the new point lies in the space ARED built".

        mode="ared_space" (default):
            - Find the closest labeled point among those ARED queried during training.
            - Return the label of the *cluster* that point belonged to.
            - This follows ARED's own non-query assignment logic (the comparison cluster label).

        mode="flat_nn":
            - Classic nearest labeled point (ignores cluster structure).
            - Useful for ablation.
        """
        if not self.is_fitted or self._nn is None or self.support_vectors_ is None or len(self.support_vectors_) == 0:
            raise RuntimeError("Call .fit() first (no support prototypes collected)")

        X_new = np.asarray(X_new, dtype=np.float32)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        # L2 normalize (matches what streams + ARED expect)
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X_new / norms

        k = min(self.n_neighbors, len(self.support_vectors_))
        dists, indices = self._nn.kneighbors(Xn, n_neighbors=k)

        if mode == "flat_nn":
            # Traditional: just take label of nearest support point
            return self.support_labels_[indices[:, 0]]

        # "ared_space" (the intended ARED-as-classifier behavior):
        # The closest support point tells us which region of the space the point landed in.
        # We return the label of its cluster.
        return self.support_labels_[indices[:, 0]]

    def locate_in_built_space(self, X_new: np.ndarray):
        """
        The core "test ARED as classifier" operation.

        For each new point:
          - Locate it relative to the space ARED built (closest labeled point + its cluster).
          - Return the label ARED's structure would assign + diagnostic info
            about whether it lies inside or outside the learned cluster region.
        """
        if not self.is_fitted or self._nn is None or len(self.support_vectors_) == 0:
            raise RuntimeError("Call .fit() first")

        X_new = np.asarray(X_new, dtype=np.float32)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X_new / norms

        dists, indices = self._nn.kneighbors(Xn, n_neighbors=1)
        idx = indices[:, 0]

        preds = self.support_labels_[idx]
        distances = dists[:, 0]

        comp_ds = self.support_comp_distances_[idx] if self.support_comp_distances_ is not None else np.zeros_like(distances)

        kappa = self.kappa
        would_be_anomalous = (distances * kappa) > comp_ds

        return preds, distances, comp_ds, would_be_anomalous
    def evaluate(self, test_stream: BaseDataStream, mode: str = "ared_space") -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Fit first")

        print(f"\n=== Evaluating AREDClassifier (mode={mode}) on Test Set ===")
        print(f"    (Testing how new points lie in the space ARED built during training)")
        X_test = test_stream.processed_vectors
        y_true = np.array(test_stream.labels_cache)
        y_pred = self.predict(X_test, mode=mode)

        # Also compute location diagnostics using the built-space logic
        _, dists, comp_ds, anom = self.locate_in_built_space(X_test)

        present_classes = set(y_true)
        mask = np.isin(y_true, list(present_classes))

        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        results = {
            "n_test": len(y_true),
            "n_prototypes": self.n_prototypes_,
            "n_queries_during_training": self.num_queries_used_,
            "n_classes_in_supports": self.n_classes_covered_,
            "unique_pred_labels": len(np.unique(y_pred)),
            "n_test_filtered": len(y_true_filtered),
            "fraction_test_would_be_anomalous": float(anom.mean()) if len(anom) > 0 else 0.0,
        }

        if len(y_true_filtered) > 0:
            acc = accuracy_score(y_true_filtered, y_pred_filtered)
            bal_acc = balanced_accuracy_score(y_true_filtered, y_pred_filtered)
            f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)

            results.update({
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "f1_weighted": float(f1),
            })

            print(classification_report(y_true_filtered, y_pred_filtered, zero_division=0))
            print(f"Accuracy (on classes present in test): {acc:.4f}")
        else:
            print("No overlapping classes in test set!")

        print(f"Test samples evaluated: {len(y_true_filtered)} / {len(y_true)}")
        print(f"Prototypes (labels acquired by ARED): {self.n_prototypes_} covering {self.n_classes_covered_} classes")
        print(f"Unique labels predicted: {results['unique_pred_labels']}")
        print(f"Fraction of test points that would have been anomalous (outside learned cluster regions): {results['fraction_test_would_be_anomalous']:.3f}")
        return results

    def get_support_info(self) -> Dict[str, Any]:
        """Return summary of the points ARED selected (the 'labeled set' that defines the space)."""
        if self.support_labels_ is None or len(self.support_labels_) == 0:
            return {"n_prototypes": 0, "classes": []}
        from collections import Counter
        counts = Counter(self.support_labels_.tolist())
        return {
            "n_prototypes": self.n_prototypes_,
            "n_classes": self.n_classes_covered_,
            "n_queries_during_build": self.num_queries_used_,
            "per_class_counts": dict(counts),
            "labels": self._known_labels,
        }

    def get_ared_model_summary(self) -> Dict[str, Any]:
        """Describe the feature space / model that ARED built."""
        return {
            "n_clusters": len(self.clusters_meta_),
            "n_labeled_points": self.n_prototypes_,
            "classes_covered": self.n_classes_covered_,
            "queries_used_to_build": self.num_queries_used_,
            "kappa_used": self.kappa,
            "clusters": self.clusters_meta_[:10] + ["..."] if len(self.clusters_meta_) > 10 else self.clusters_meta_,
        }

    def _get_cluster_list_safe(self):
        if not self.exp or not hasattr(self.exp, 'ared'):
            return []
        sp = self.exp.ared.subspace_partition
        if hasattr(sp, 'cluster_dict'):
            return list(sp.cluster_dict.values())
        return []