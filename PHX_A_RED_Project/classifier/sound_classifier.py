"""
Robust Sound Classifier with improved class coverage in train/test split.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Union

from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..data.base_stream import BaseDataStream, PerchDataStream

try:
    import pandas as pd
except ImportError:
    pd = None


class SoundClassifier:
    def __init__(self, 
                 random_state: int = 42, 
                 n_estimators: int = 400,
                 use_gpu: bool = True,
                 model_type: str = "lightgbm",
                 group_split: bool = False,
                 **kwargs):
        
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.use_gpu = use_gpu and LIGHTGBM_AVAILABLE
        self.model_type = model_type.lower()
        self.group_split = group_split
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.label_column = None
        self.is_fitted = False
        self.metadata = {}

        # Internal held-out test set (populated in fit)
        self.X_test_: Optional[np.ndarray] = None
        self.y_test_: Optional[np.ndarray] = None
        self.train_classes_: set = set()
        self.test_classes_: set = set()

    def _make_group_key(self, row) -> str:
        """Key used to group chunks that came from the same original recording."""
        fn = str(row.get("filename", ""))
        # filename like "..._chunk000" or original_filename
        if pd is not None and "original_filename" in row and pd.notna(row.get("original_filename")):
            return str(row["original_filename"])
        # strip trailing _chunkXXX or similar
        base = fn.rsplit("_chunk", 1)[0]
        return base or fn

    def _safe_split(self, df, label_column: str, train_frac: float = 0.7):
        """Improved split that ensures better class representation in test set.
        If self.group_split, performs the split at recording-group level to avoid leakage.
        """
        import pandas as pd  # local import safe

        print(f"Performing balanced split for {label_column} (group_split={self.group_split})...")

        if self.group_split:
            df = df.copy()
            df["_group"] = df.apply(self._make_group_key, axis=1)
            # Map groups to their labels (use first label seen for the group)
            group_to_label = df.groupby("_group")[label_column].first()
            group_labels = group_to_label.reset_index()
            group_labels.columns = ["_group", label_column]

            # Stratify on the *groups*
            groups = group_labels.groupby(label_column).groups
            train_groups = []
            test_groups = []

            np.random.seed(self.random_state)
            for lab, gidx in groups.items():
                gidx = gidx.to_numpy()
                n = len(gidx)
                if n == 1:
                    train_groups.append(gidx[0])
                else:
                    n_test = max(1, int(n * (1 - train_frac)))
                    n_train = n - n_test
                    shuffled = np.random.permutation(gidx)
                    train_groups.extend(shuffled[:n_train])
                    test_groups.extend(shuffled[n_train:])

            train_mask = df["_group"].isin(train_groups)
            test_mask = df["_group"].isin(test_groups)
            train_idx = df.index[train_mask].to_numpy()
            test_idx = df.index[test_mask].to_numpy()
        else:
            # Original per-sample (but still class-balanced) logic
            groups = df.groupby(label_column).groups
            train_idx = []
            test_idx = []

            np.random.seed(self.random_state)

            for label, idx in groups.items():
                idx = idx.to_numpy()
                n = len(idx)

                if n == 1:
                    train_idx.append(idx[0])
                elif n == 2:
                    train_idx.append(idx[0])
                    test_idx.append(idx[1])
                else:
                    n_test = max(1, int(n * (1 - train_frac)))
                    n_train = n - n_test
                    shuffled = np.random.permutation(idx)
                    train_idx.extend(shuffled[:n_train])
                    test_idx.extend(shuffled[n_train:])

            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)

        # Final shuffle
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)

        n_test_classes = len(np.unique(df.iloc[test_idx][label_column])) if len(test_idx) > 0 else 0
        print(f"Final Split → Train: {len(train_idx)} | Test: {len(test_idx)} samples")
        print(f"Classes in test set: {n_test_classes} / {df[label_column].nunique()}")
        
        return train_idx, test_idx

    def fit(self, 
            num_points: int = 20000, 
            label_column: str = "scientific_name",
            train_frac: float = 0.7,
            **stream_kwargs):
        
        self.label_column = label_column
        print(f"Training {self.model_type.upper()} on '{label_column}'...")

        full_stream = PerchDataStream(
            label_column=label_column,
            max_samples=num_points if num_points > 0 else None,
            shuffle=True,          # Important: global shuffle
            seed=self.random_state,
            **stream_kwargs
        )
        
        df = full_stream.df.copy()
        train_idx, test_idx = self._safe_split(df, label_column, train_frac)

        # === Training data ===
        train_stream = PerchDataStream(
            label_column=label_column,
            shuffle=False,
            seed=self.random_state,
            **stream_kwargs
        )
        train_stream.df = df.iloc[train_idx].reset_index(drop=True)
        train_stream._preload()

        X_train = train_stream.processed_vectors
        y_train = np.array(train_stream.labels_cache)
        y_encoded = self.label_encoder.fit_transform(y_train)

        self.train_classes_ = set(y_train)

        print(f"Training on {len(X_train)} samples | {len(self.label_encoder.classes_)} classes")

        # === Build and store internal held-out test set (the key fix) ===
        if len(test_idx) > 0:
            test_df = df.iloc[test_idx].reset_index(drop=True)
            test_stream = PerchDataStream(
                label_column=label_column,
                shuffle=False,
                seed=self.random_state,
                **stream_kwargs
            )
            test_stream.df = test_df
            test_stream._preload()
            self.X_test_ = test_stream.processed_vectors
            self.y_test_ = np.array(test_stream.labels_cache)
            self.test_classes_ = set(self.y_test_)
        else:
            self.X_test_ = None
            self.y_test_ = None
            self.test_classes_ = set()

        print(f"Internal test set ready: {len(self.y_test_) if self.y_test_ is not None else 0} samples, "
              f"{len(self.test_classes_)} classes")

        # === Model construction ===
        mt = self.model_type

        if mt == "prototype":
            # Class prototype classifier (mean embedding + cosine). Excellent for fixed high-quality embeddings.
            print("Using prototype (mean-per-class + cosine) classifier")
            self._fit_prototypes(X_train, y_train)
            # self.model now holds the prototype matrix
        elif mt == "logistic":
            from sklearn.linear_model import LogisticRegression
            print("Using LogisticRegression (balanced)")
            self.model = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model.fit(X_train, y_encoded)
        elif mt == "lightgbm" and LIGHTGBM_AVAILABLE:
            params = {
                'n_estimators': self.n_estimators,
                'random_state': self.random_state,
                'class_weight': 'balanced',
                'verbose': -1,
                'num_leaves': 63,
                'learning_rate': 0.05,
                'max_bin': 63,
                'min_child_samples': 20,
            }
            if self.use_gpu:
                params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
                print("→ GPU enabled")
            self.model = lgb.LGBMClassifier(**params)
            self.model.fit(X_train, y_encoded)
        else:
            # Fallback / explicit randomforest
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )
            self.model.fit(X_train, y_encoded)

        self.is_fitted = True
        print("✅ Training complete.")
        return self

    def _fit_prototypes(self, X: np.ndarray, y: np.ndarray):
        """Compute one L2-normalized prototype per class."""
        classes = self.label_encoder.fit_transform(y)
        self._proto_labels = self.label_encoder.classes_
        n_classes = len(self._proto_labels)
        dim = X.shape[1]
        protos = np.zeros((n_classes, dim), dtype=np.float32)

        for i, cls in enumerate(self._proto_labels):
            mask = (y == cls)
            if np.any(mask):
                p = X[mask].mean(axis=0)
                n = np.linalg.norm(p)
                protos[i] = p / n if n > 0 else p
        self.model = protos  # store prototypes in .model for save/load simplicity

    def _predict_prototypes(self, X: np.ndarray) -> np.ndarray:
        protos = self.model  # (n_classes, dim)
        # L2 normalize queries
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        Xn = X / norms
        # cosine sim = dot product of unit vectors
        sims = Xn @ protos.T
        idx = np.argmax(sims, axis=1)
        return self._proto_labels[idx]

    def predict(self, X_new: np.ndarray):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first")
        X_new = np.asarray(X_new, dtype=np.float32)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        mt = self.model_type
        if mt == "prototype":
            return self._predict_prototypes(X_new)
        else:
            # tree / linear models use the label_encoder path
            return self.label_encoder.inverse_transform(self.model.predict(X_new))

    def evaluate(self, test_stream: Optional[BaseDataStream] = None):
        if not self.is_fitted:
            raise RuntimeError("Fit first")

        if test_stream is not None:
            X_test = test_stream.processed_vectors
            y_true = np.array(test_stream.labels_cache)
            source = "external test_stream"
        elif self.X_test_ is not None and self.y_test_ is not None:
            X_test = self.X_test_
            y_true = self.y_test_
            source = "internal held-out test set"
        else:
            raise RuntimeError("No test data available. Pass test_stream or ensure fit() produced an internal test set.")

        print(f"\n=== Evaluating {self.model_type.upper()} ({self.label_column}) on {source} ===")
        y_pred = self.predict(X_test)

        present_classes = np.unique(y_true)
        mask = np.isin(y_true, present_classes)

        print(classification_report(y_true[mask], y_pred[mask], zero_division=0))

        results = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            "n_test": len(y_true),
            "classes_in_test": len(present_classes),
            "classes_in_train": len(self.train_classes_),
        }

        print(f"\nOverall Accuracy: {results['accuracy']:.4f} | Balanced: {results['balanced_accuracy']:.4f}")
        print(f"Classes present in test: {results['classes_in_test']} (train had {results['classes_in_train']})")
        return results

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "metadata": self.metadata,
            "label_column": self.label_column,
            "model_type": self.model_type,
            "_proto_labels": getattr(self, "_proto_labels", None),
        }
        joblib.dump(save_dict, path)
        print(f"Model saved to: {path}")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SoundClassifier':
        data = joblib.load(path)
        clf = cls(model_type=data.get("model_type", "lightgbm"))
        clf.model = data["model"]
        clf.label_encoder = data["label_encoder"]
        clf.metadata = data.get("metadata", {})
        clf.label_column = data.get("label_column")
        clf._proto_labels = data.get("_proto_labels")
        clf.is_fitted = True
        print(f"Loaded model from {path}")
        return clf