"""
AREDExperiment — unified high-level runner / facade.

Encapsulates the common boilerplate that was duplicated across every main_*.py:

- Create stream + oracle + ARED
- Wire optional DiscoveryTracker / ClassDiscoveryCounter
- Wire optional ShiftingKappaController
- First-point handling + record_appearance
- Main loop + controller hook
- Basic progress + final reporting

This keeps the core ARED (A_REDimplementation/A_RED/A_REDIN.py) completely untouched
while giving callers a clean, reusable object-oriented interface.

Usage (direct):
    from PHX_A_RED_Project.data import SpectrogramDataStream, NoRelevanceOracle
    from PHX_A_RED_Project.experiment import AREDExperiment
    from PHX_A_RED_Project.utils import ClassDiscoveryCounter, ShiftingKappaController

    stream = SpectrogramDataStream(max_samples=500, shuffle=True, seed=42)
    counter = ClassDiscoveryCounter()
    oracle = NoRelevanceOracle(stream, discovery_tracker=counter)

    exp = AREDExperiment(
        stream, oracle,
        kappa=1.5,
        data_window_size=250,
        k_comparison_clusters=5,
    )
    exp.run(num_points=500, controller=None)   # or pass a ShiftingKappaController
    exp.print_report()
    print(exp.get_discovery_metrics(rare_threshold=0.005))  # RED-focused numbers + lift
    exp.print_rare_event_summary()

Or the one-liner:
    from PHX_A_RED_Project import run_ared
    exp = run_ared("spectrogram", num_points=200, kappa=2.1)
"""

from __future__ import annotations
from typing import Optional, Union, Literal, Any
import time
import sys
from pathlib import Path

from .data.base_stream import BaseDataStream
from .data.oracle import NoRelevanceOracle
from .utils.discovery import ClassDiscoveryCounter
from .utils.controllers import ShiftingKappaController
from .utils.reporting import (
    print_cluster_summary,
    print_class_discovery_report,
    get_class_discovery_table,
)

# The core ARED is imported lazily (see _get_ared_class) so that users can
# import data streams / oracles / utils without needing every dependency
# that the untouched original A_RED implementation pulls in (cv2, etc.).
_ARED = None
_ARED_IMPORT_ERROR = None


def _ensure_ared_on_path():
    """Ensure the original untouched A_RED package location is on sys.path."""
    _ARED_DIR = Path(__file__).resolve().parent.parent / "A_REDimplementation" / "A_RED"
    if str(_ARED_DIR) not in sys.path:
        sys.path.insert(0, str(_ARED_DIR))
    # Some original scripts also relied on running with CWD containing the tree
    alt = Path.cwd() / "A_REDimplementation" / "A_RED"
    if str(alt) not in sys.path:
        sys.path.insert(0, str(alt))


def _inject_minimal_main_stub():
    """
    AREDIN.py does 'from main import QS_VAR' at import time.
    The real main.py has massive side-effect imports (torch, seaborn, all the
    dataset processors, plotting, ...).

    We inject a tiny stub 'main' module containing only the constants that
    AREDIN actually reads at import time. This lets us use AREDIN as the
    core without pulling the entire demo scaffolding.
    """
    import types
    if "main" in sys.modules:
        return  # already provided (real or stub)

    # Reasonable defaults matching the spirit of the original AREDIN main.py
    stub = types.ModuleType("main")
    stub.QS_VAR = 1
    stub.DATA_AUG_VAR = (0, ())
    stub.K_COMP_PTS = 2
    stub.NGHBHOOD_MERGE = False
    stub.SINGLETON_MERGE = False
    stub.SMART_FORGETTING_VAR = (0, 0.0)
    stub.VERBOSE_FLAGS = [0]
    stub.DATA_WINDOW_SIZE = 250
    stub.SMALL_CLUSTER_THRESHOLD = 3
    stub.GRAPH_BATCH_SIZE = 100

    sys.modules["main"] = stub


def _get_ared_class():
    """Lazy import of the untouched core ARED class."""
    global _ARED, _ARED_IMPORT_ERROR
    if _ARED is not None:
        return _ARED
    if _ARED_IMPORT_ERROR is not None:
        raise _ARED_IMPORT_ERROR

    _ensure_ared_on_path()
    _inject_minimal_main_stub()
    try:
        from A_REDIN import ARED as _ARED_CLS  # now using A_REDIN (IN variant)
        _ARED = _ARED_CLS
        return _ARED
    except Exception as e:
        _ARED_IMPORT_ERROR = ImportError(
            "Could not import the core ARED from A_REDimplementation/A_RED (A_REDIN). "
            "This package expects the original untouched A_RED / A_REDIN implementation to stay at that location. "
            f"Underlying error: {e}"
        )
        raise _ARED_IMPORT_ERROR from e


FrontendName = Literal["spectrogram", "perch", "dinov3"]


class AREDExperiment:
    """
    High-level ARED experiment runner.

    Responsibilities:
    - Owns the stream, oracle, ARED instance
    - Owns optional discovery counter and controller
    - Runs the canonical first-point + loop
    - Calls controller.record_and_adjust(ared, did_query) after every point if provided
    - Provides summary / reporting helpers

    Does NOT modify the core ARED implementation.
    """

    def __init__(
        self,
        data_stream: BaseDataStream,
        oracle: NoRelevanceOracle,
        kappa: float = 1.5,
        data_window_size: int = 250,
        k_comparison_clusters: int = 5,
        QS_VAR: int = 1,
        REL_PROC_VAR: int = 0,
        VERBOSE_FLAGS: Optional[list] = None,
        discovery_counter: Optional[ClassDiscoveryCounter] = None,
        smart_forgetting_var: tuple = (0, 0.0),
    ):
        self.stream = data_stream
        self.oracle = oracle
        # Support both explicit discovery_counter and the one already wired into the oracle
        # (matches the historical pattern where mains did: oracle = XXXOracle(ds, discovery_tracker=counter))
        self.discovery_counter = discovery_counter or getattr(oracle, "discovery_tracker", None)

        ARED = _get_ared_class()  # lazy

        # A_REDIN constructor:
        # ARED(oracle, kappa, l_buf_size, K_COMP_PTS, QS_VAR, DATA_AUG_VAR, NGHBHOOD_MERGE, SINGLETON_MERGE, SMART_FORGETTING_VAR, VERBOSE_FLAGS)
        # We map our long-standing high-level params to the IN variant:
        #   data_window_size   -> l_buf_size
        #   k_comparison_clusters -> K_COMP_PTS
        # smart_forgetting_var forwards to AREDIN's SMART_FORGETTING_VAR (modes help retain old/rare class reps in buffer)
        self.ared = ARED(
            oracle,
            float(kappa),
            int(data_window_size),
            int(k_comparison_clusters),
            QS_VAR,
            (0, ()),            # DATA_AUG_VAR: no augmentation
            True,              # NGHBHOOD_MERGE
            True,              # SINGLETON_MERGE
            smart_forgetting_var,   # SMART_FORGETTING_VAR (0,0.0)=disabled; see A_REDIN for modes 1-3
            VERBOSE_FLAGS or [],
        )

        # Give the oracle a back-reference so it can report the authoritative
        # query ordinal from ARED (important for AREDIN which does extra
        # answer_query calls for internal stats even on non-queried points).
        try:
            oracle._ared_ref = self.ared
        except Exception:
            pass

        self.points_processed = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def kappa(self) -> float:
        return self.ared.kappa

    @kappa.setter
    def kappa(self, value: float):
        self.ared.kappa = float(value)

    # ---------------- ARED variant adapters (A_RED.py vs A_REDIN.py) ----------------
    def _get_query_count(self) -> int:
        """Return authoritative number of queries performed so far.
        Works for both the old A_RED (via labeled_data) and A_REDIN (via num_queries).
        Falls back to the oracle's count.
        """
        ared = self.ared
        if hasattr(ared, "num_queries"):
            return int(ared.num_queries)
        ld = getattr(ared, "labeled_data", None)
        if ld is not None and hasattr(ld, "abs_idx_array"):
            return len(ld.abs_idx_array)
        return int(getattr(self.oracle, "query_count", 0))

    def _get_cluster_list(self):
        """Return clusters as a list regardless of whether the core uses
        .cluster_list (old) or .cluster_dict (AREDIN).
        """
        sp = self.ared.subspace_partition
        if hasattr(sp, "cluster_dict"):
            try:
                return list(sp.cluster_dict.values())
            except Exception:
                pass
        if hasattr(sp, "cluster_list"):
            return sp.cluster_list
        return []

    def _record_appearance(self, processed_idx: int):
        """Record every point's label for discovery / enrichment stats."""
        if self.discovery_counter is None:
            return
        # stream_counter has already advanced; the just-yielded point is at stream_counter-1
        src_idx = self.stream.stream_counter - 1
        label = self.stream.get_true_label_for_idx(src_idx)
        self.discovery_counter.record_appearance(label, processed_idx)

    def _did_query_on_last_point(self) -> bool:
        """Heuristic: number of labeled points increased since we last checked."""
        # We compare against the number of times oracle was called.
        # The oracle's query_count is authoritative (one query per labeled point).
        return self.oracle.query_count > (self.points_processed - 1)  # after increment in process

    def run(
        self,
        num_points: int = -1,
        controller: Optional[ShiftingKappaController] = None,
        status_every: int = 100,
        verbose: bool = True,
    ) -> int:
        """
        Run the experiment for up to `num_points`.

        Returns the number of points actually processed.
        """
        if num_points == -1:
            num_points = 10**12  # all

        self.start_time = time.time()
        self.points_processed = 0

        # ---- First point (always creates initial cluster + 1 query) ----
        try:
            first_vec = self.stream.stream_new_data_point()
            # Record appearance *before* process_first_point so indices line up
            first_idx = 0
            self._record_appearance(first_idx)

            self.ared.process_first_point(first_vec)
            self.points_processed = 1

            if controller is not None:
                # First point always queries
                controller.record_and_adjust(self.ared, did_query=True)

            if verbose:
                print("First point processed. Initial cluster created.")
                clusters = self._get_cluster_list()
                if len(clusters) > 0:
                    c0 = clusters[0]
                    print(f"Initial cluster comp_distance: {getattr(c0, 'comp_distance', float('nan')):.4f}")
        except StopIteration:
            if verbose:
                print("Stream exhausted before first point.")
            self.end_time = time.time()
            return 0
        except Exception as e:
            print(f"Error on first point: {e}")
            self.end_time = time.time()
            return 0

        # ---- Main loop ----
        target = num_points
        while self.points_processed < target and self.stream.get_remaining_num_points() > 0:
            try:
                vec = self.stream.stream_new_data_point()

                # Appearance for this point (processed_idx == current count)
                current_idx = self.points_processed
                self._record_appearance(current_idx)

                prev_q = self.oracle.query_count
                self.ared.process_point(vec)
                self.points_processed += 1
                did_query = self.oracle.query_count > prev_q

                if controller is not None:
                    controller.record_and_adjust(self.ared, did_query=did_query)

                if verbose and (self.points_processed % max(1, status_every) == 0 or self.points_processed == target):
                    q = self._get_query_count()
                    known = len(getattr(self.ared.subspace_partition, "set_of_known_labels", []))
                    rate = (q / self.points_processed * 100.0) if self.points_processed else 0.0
                    print(f"Processed {self.points_processed:,} | Queries: {q} | Known: {known} | Rate: {rate:.2f}%")

            except StopIteration:
                break
            except Exception as e:
                print(f"Error at point {self.points_processed}: {e}")
                break

        self.end_time = time.time()
        return self.points_processed

    def print_report(self, include_clusters: bool = False):
        elapsed = (self.end_time or time.time()) - (self.start_time or time.time())
        final_q = self._get_query_count()
        n = self.points_processed or 1
        known = len(getattr(self.ared.subspace_partition, "set_of_known_labels", []))

        print("\n" + "=" * 60)
        print("A_RED COMPLETE")
        print("=" * 60)
        print(f"Points processed: {self.points_processed:,}")
        print(f"Queries made: {final_q} ({final_q / n * 100:.2f}% of points)")
        print(f"Known classes discovered: {known}")
        print(f"Total time: {elapsed:.2f}s")
        if n > 0:
            print(f"Queries per second: {final_q / elapsed:.2f}")

        if self.discovery_counter is not None:
            self._print_discovery_report()

        if include_clusters:
            print_cluster_summary(self._get_cluster_list(), only_labeled=True)

    def _print_discovery_report(self):
        """Delegate to the canonical shared implementation (matches Perch/main_perch.py style)."""
        dc = self.discovery_counter
        N = self.points_processed or 0
        Q = self._get_query_count()
        print_class_discovery_report(dc, N, Q)

    def get_discovery_metrics(self, rare_threshold: float = 0.01) -> dict:
        """
        Return rich discovery statistics for programmatic / RED analysis use.

        Includes:
        - basic counts
        - per-class table (with lift + enrichment)
        - aggregates focused on rare classes (prevalence < rare_threshold)
        """
        if self.discovery_counter is None:
            return {"error": "no discovery_counter attached"}

        dc = self.discovery_counter
        N = self.points_processed or 0
        Q = self._get_query_count()
        table = get_class_discovery_table(dc, N, Q)

        rare_rows = [r for r in table if r["prevalence"] < rare_threshold]
        discovered_rare = [r for r in rare_rows if r["discovery_query"]]
        lifts = [r["lift"] for r in discovered_rare if r["lift"] is not None]

        median_lift = None
        if lifts:
            s = sorted(lifts)
            m = len(s) // 2
            median_lift = s[m] if len(s) % 2 else (s[m-1] + s[m]) / 2

        total_unlabeled_rare = sum(r["seen_before_queried"] for r in rare_rows)

        return {
            "N": N,
            "queries": Q,
            "classes_seen": dc.get_seen_count(),
            "classes_discovered": dc.get_discovered_count(),
            "rare_threshold": rare_threshold,
            "rare_classes_seen": len(rare_rows),
            "rare_classes_discovered": len(discovered_rare),
            "median_lift_rare_discovered": median_lift,
            "total_unlabeled_rare_instances": total_unlabeled_rare,
            "per_class": table,
        }

    def print_rare_event_summary(self, rare_threshold: float = 0.01):
        """Convenience: print the main discovery report + a short rare-focused tail summary."""
        if self.discovery_counter is None:
            print("(no discovery counter)")
            return
        self._print_discovery_report()
        m = self.get_discovery_metrics(rare_threshold=rare_threshold)
        print("\nRare-Event Summary (for tail detection):")
        print(f"  Rare threshold: < {rare_threshold*100:.2f}% prevalence")
        print(f"  Rare classes seen: {m['rare_classes_seen']} | discovered: {m['rare_classes_discovered']}")
        if m["median_lift_rare_discovered"] is not None:
            print(f"  Median lift on discovered rare classes: {m['median_lift_rare_discovered']:.1f}x")
        print(f"  Total rare audio events that went unlabeled before discovery: {m['total_unlabeled_rare_instances']}")

    def get_class_discovery_ordinals(self) -> dict:
        """
        Return {class_label: query_ordinal} for every class that was discovered
        (i.e. the exact query count at the moment the oracle first returned that label).
        This is the key data for comparing ARED vs random baseline.
        Values are clamped to the final authoritative query count from the core.
        """
        dc = self.discovery_counter
        if dc is None:
            return {}
        final_q = max(1, self._get_query_count())
        out = {}
        for lab in dc.get_labels():
            q = dc.get_first_query_count(lab)
            if q and q > 0:
                out[lab] = min(q, final_q)
        return out

    def get_discovery_record(self, method: str = "ared", extra: Optional[dict] = None) -> dict:
        """Structured record suitable for saving and later graphing.

        Now includes:
          - num_classes_in_pool : total distinct labels present in the data this run processed
          - num_classes_found   : how many of them were actually revealed by a query
        This lets you see if ARED "missed" any classes that were in the pool.
        """
        dc = self.discovery_counter
        N = self.points_processed or 0
        Q = self._get_query_count()

        labels_cache = getattr(self.stream, "labels_cache", None) or []
        pool_set = set(labels_cache)
        num_in_pool = len(pool_set)

        discoveries = self.get_class_discovery_ordinals()
        num_found = len(discoveries)

        missed = sorted(pool_set - set(discoveries.keys()))

        rec = {
            "method": method,
            "frontend": None,
            "label_column": getattr(self.stream, "label_column", None),
            "N": N,
            "queries": Q,
            "seed": getattr(self.stream, "seed", None),
            "shuffle": getattr(self.stream, "shuffle", None),
            "discoveries": discoveries,
            "num_classes_in_pool": num_in_pool,
            "num_classes_found": num_found,
            "num_classes_missed": len(missed),
            "missed_classes": missed,
        }
        if extra:
            rec.update(extra)
        return rec

    def save_discovery_record(self, out_dir: str = "results", prefix: str = "", method: str = "ared"):
        """Save a JSON file with per-class discovery query ordinals."""
        import json
        from pathlib import Path as _Path
        rec = self.get_discovery_record(method=method)
        out_path = _Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        n = rec["N"]
        sd = rec.get("seed", "na")
        labcol = rec.get("label_column") or "label"
        fname = f"{prefix}{method}_{labcol}_N{n}_seed{sd}.json"
        fpath = out_path / fname

        with open(fpath, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[save] Wrote discovery record -> {fpath}")
        return fpath

    def get_results(self) -> dict:
        """Return a compact dict of results for programmatic use."""
        final_q = self._get_query_count()
        n = max(1, self.points_processed)
        return {
            "points_processed": self.points_processed,
            "queries": final_q,
            "query_rate": final_q / n,
            "known_classes": len(getattr(self.ared.subspace_partition, "set_of_known_labels", [])),
            "elapsed_sec": (self.end_time or time.time()) - (self.start_time or time.time()),
            "final_kappa": self.ared.kappa,
        }


def run_ared(
    frontend: Union[FrontendName, BaseDataStream],
    num_points: int = 200,
    label_column: str = "class_name",
    kappa: float = 1.5,
    data_window_size: int = 250,
    k_comparison_clusters: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    max_samples: Optional[int] = None,
    controller: Optional[ShiftingKappaController] = None,
    fast: bool = False,
    save_results: bool = False,
    results_dir: str = "results",
    smart_forgetting_var: tuple = (0, 0.0),
    **stream_kwargs,
) -> AREDExperiment:
    """
    Convenience factory + runner.

    frontend: "spectrogram" | "perch" | "dinov3"  OR a ready-made BaseDataStream

    save_results=True will write a JSON file under results_dir containing
    the exact query ordinal at which each class was discovered (key data
    for ARED vs random baseline comparison graphs).
    """
    if isinstance(frontend, BaseDataStream):
        stream = frontend
    else:
        f = frontend.lower()
        stream_kwargs = {k: v for k, v in stream_kwargs.items() if k not in ("verbose",)}
        if f == "spectrogram":
            from .data.base_stream import SpectrogramDataStream as _S

            stream = _S(
                max_samples=max_samples or (num_points if num_points > 0 else None),
                shuffle=shuffle,
                seed=seed,
                label_column=label_column,
                **stream_kwargs,
            )
        elif f == "perch":
            from .data.base_stream import PerchDataStream as _P

            stream = _P(
                max_samples=max_samples or (num_points if num_points > 0 else None),
                shuffle=shuffle,
                seed=seed,
                label_column=label_column,
                **stream_kwargs,
            )
        elif f == "dinov3":
            from .data.base_stream import Dinov3DataStream as _D

            stream = _D(
                max_samples=max_samples or (num_points if num_points > 0 else None),
                shuffle=shuffle,
                seed=seed,
                label_column=label_column,
                **stream_kwargs,
            )
        else:
            raise ValueError(f"Unknown frontend: {frontend}")

    counter = ClassDiscoveryCounter(fast_mode=bool(fast))

    oracle = NoRelevanceOracle(stream, discovery_tracker=counter)

    exp = AREDExperiment(
        stream,
        oracle,
        kappa=kappa,
        data_window_size=data_window_size,
        k_comparison_clusters=k_comparison_clusters,
        smart_forgetting_var=smart_forgetting_var,
    )

    exp.run(num_points=num_points, controller=controller, verbose=True)
    exp.print_report()

    # Fill friendly info for saved records
    if not isinstance(frontend, BaseDataStream):
        try:
            exp._frontend_name = frontend  # type: ignore[attr-defined]
        except Exception:
            pass

    if save_results:
        try:
            extra = {}
            if hasattr(exp, "_frontend_name") and exp._frontend_name:
                extra["frontend"] = exp._frontend_name
            exp.save_discovery_record(out_dir=results_dir, prefix="", method="ared")
            # If we want frontend in the json we can patch after the fact (simple)
            if extra:
                import json as _json
                from pathlib import Path as _P
                n = exp.points_processed or 0
                sd = getattr(exp.stream, "seed", "na")
                labc = getattr(exp.stream, "label_column", "label")
                candidate = _P(results_dir) / f"ared_{labc}_N{n}_seed{sd}.json"
                if candidate.exists():
                    data = _json.loads(candidate.read_text())
                    data.update(extra)
                    candidate.write_text(_json.dumps(data, indent=2))
        except Exception as e:
            print(f"[save] Failed to write ared discovery record: {e}")

    return exp
