"""
AREDExperiment — unified high-level runner / facade.

Encapsulates the common boilerplate that was duplicated across every main_*.py:

- Create stream + oracle + ARED
- Wire optional DiscoveryTracker / ClassDiscoveryCounter
- Wire optional ShiftingKappaController
- First-point handling + record_appearance
- Main loop + controller hook
- Basic progress + final reporting

This keeps the core ARED (A_REDimplementation/A_RED/A_RED.py) completely untouched
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


def _get_ared_class():
    """Lazy import of the untouched core ARED class."""
    global _ARED, _ARED_IMPORT_ERROR
    if _ARED is not None:
        return _ARED
    if _ARED_IMPORT_ERROR is not None:
        raise _ARED_IMPORT_ERROR

    _ensure_ared_on_path()
    try:
        from A_RED import ARED as _ARED_CLS  # untouched
        _ARED = _ARED_CLS
        return _ARED
    except Exception as e:
        _ARED_IMPORT_ERROR = ImportError(
            "Could not import the core ARED from A_REDimplementation/A_RED. "
            "This package expects the original untouched A_RED implementation to stay at that location. "
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
        QS_VAR: int = 0,
        REL_PROC_VAR: int = 0,
        VERBOSE_FLAGS: Optional[list] = None,
        discovery_counter: Optional[ClassDiscoveryCounter] = None,
    ):
        self.stream = data_stream
        self.oracle = oracle
        # Support both explicit discovery_counter and the one already wired into the oracle
        # (matches the historical pattern where mains did: oracle = XXXOracle(ds, discovery_tracker=counter))
        self.discovery_counter = discovery_counter or getattr(oracle, "discovery_tracker", None)

        ARED = _get_ared_class()  # lazy
        self.ared = ARED(
            oracle=oracle,
            kappa=kappa,
            data_window_size=data_window_size,
            k_comparison_clusters=k_comparison_clusters,
            QS_VAR=QS_VAR,
            REL_PROC_VAR=REL_PROC_VAR,
            VERBOSE_FLAGS=VERBOSE_FLAGS or [],
        )

        self.points_processed = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def kappa(self) -> float:
        return self.ared.kappa

    @kappa.setter
    def kappa(self, value: float):
        self.ared.kappa = float(value)

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
                if len(self.ared.subspace_partition.cluster_list) > 0:
                    c0 = self.ared.subspace_partition.cluster_list[0]
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
                    q = len(self.ared.labeled_data.abs_idx_array)
                    known = len(self.ared.subspace_partition.set_of_known_labels)
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
        final_q = len(self.ared.labeled_data.abs_idx_array)
        n = self.points_processed or 1
        known = len(self.ared.subspace_partition.set_of_known_labels)

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
            print_cluster_summary(self.ared.subspace_partition.cluster_list, only_labeled=True)

    def _print_discovery_report(self):
        """Delegate to the canonical shared implementation (matches Perch/main_perch.py style)."""
        dc = self.discovery_counter
        N = self.points_processed or 0
        Q = len(self.ared.labeled_data.abs_idx_array)
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
        Q = len(getattr(self.ared, "labeled_data", None) and self.ared.labeled_data.abs_idx_array or [])
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

    def get_results(self) -> dict:
        """Return a compact dict of results for programmatic use."""
        final_q = len(self.ared.labeled_data.abs_idx_array)
        n = max(1, self.points_processed)
        return {
            "points_processed": self.points_processed,
            "queries": final_q,
            "query_rate": final_q / n,
            "known_classes": len(self.ared.subspace_partition.set_of_known_labels),
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
    **stream_kwargs,
) -> AREDExperiment:
    """
    Convenience factory + runner.

    frontend: "spectrogram" | "perch" | "dinov3"  OR a ready-made BaseDataStream
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
    )

    exp.run(num_points=num_points, controller=controller, verbose=True)
    exp.print_report()
    return exp
