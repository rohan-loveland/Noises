"""
Discovery tracking utilities (unified).

Contains:
- DiscoveryTracker: lightweight hook used by the Oracle to record when
  labels are revealed. Compatible with the older Dinov3 + SpectrogramOracle pattern.
- ClassDiscoveryCounter: richer per-class (or per-species) statistics used
  by the main runners and shifting-kappa experiments. Supports a fast_mode
  that disables per-point overhead for very long runs.
"""
from typing import Dict, Optional


class DiscoveryTracker:
    """
    Minimal tracker that records first-seen and first-queried points.
    Used as the 'discovery_tracker' passed into NoRelevanceOracle.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_points_examined = 0
        self.total_queries = 0
        self.classes_first_seen: Dict[str, int] = {}
        self.classes_first_queried: Dict[str, int] = {}
        self.first_seen_stream_idx: Dict[str, int] = {}
        self.first_queried_stream_idx: Dict[str, int] = {}

    def record_point(self, true_label: str, stream_idx: int, is_query: bool = False):
        self.total_points_examined += 1
        if true_label not in self.classes_first_seen:
            self.classes_first_seen[true_label] = stream_idx
            self.first_seen_stream_idx[true_label] = stream_idx
        if is_query and true_label not in self.classes_first_queried:
            self.classes_first_queried[true_label] = stream_idx
            self.first_queried_stream_idx[true_label] = stream_idx

    def record_query(self, true_label: str, stream_idx: int):
        """Called by NoRelevanceOracle (and compatible old oracles)."""
        self.total_queries += 1
        self.record_point(true_label, stream_idx, is_query=True)

    def get_discovery_report(self) -> dict:
        report = {
            "total_points_examined": self.total_points_examined,
            "total_queries": self.total_queries,
            "total_classes_seen": len(self.classes_first_seen),
            "total_classes_queried": len(self.classes_first_queried),
            "classes": {},
        }
        for cls in self.classes_first_seen:
            report["classes"][cls] = {
                "first_seen_stream_idx": self.first_seen_stream_idx.get(cls),
                "first_queried_stream_idx": self.first_queried_stream_idx.get(cls),
                "queries_before_first_query": (
                    self.first_queried_stream_idx.get(cls, 0)
                    - self.first_seen_stream_idx.get(cls, 0)
                    if cls in self.first_seen_stream_idx
                    else 0
                ),
            }
        return report

    def save_report(self, path: str = "discovery_report.json"):
        import json
        with open(path, "w") as f:
            json.dump(self.get_discovery_report(), f, indent=2)


class ClassDiscoveryCounter:
    """
    Rich per-label statistics for measuring how effectively A_RED discovers
    (especially rare) classes under a limited oracle query budget.

    For each label we track:
      - processed_idx of first appearance in the stream
      - processed_idx when it was first revealed by an oracle query
      - the 1-based query ordinal at discovery time (used for lift)
      - total instances seen across the whole run
      - how many instances of the label arrived *before* we spent a query on it
        (these were typically absorbed as o_pts without labeling cost)
      - total number of times the oracle returned this label

    This data powers the "vs Random Baseline" report and lift calculations.

    The geometric random model:
        If a class has prevalence p, a uniform random querier is expected
        to discover it on query number ~ 1/p.

    Lift for a class = (1 / prevalence) / discovery_query_ordinal
        > 1.0  : discovered earlier (in labeling budget) than random
        >> 1.0 : strong evidence of effective active/rare-event discovery

    "seen_before_queried" is especially important for rare-event audio work:
    it quantifies how many real instances of a tail class flew past unlabeled
    because A_RED had not yet decided they were anomalous.

    fast_mode=True disables the per-point seen_before increments (cheaper
    for very long runs); discovered classes will report seen_before=0 in
    that mode.
    """

    def __init__(self, fast_mode: bool = False):
        self.fast_mode = fast_mode
        self.first_appearance: Dict[str, int] = {}
        self.first_queried: Dict[str, int] = {}
        self.first_queried_query_count: Dict[str, int] = {}
        self.total_seen: Dict[str, int] = {}
        self.seen_before_queried: Dict[str, int] = {}
        self.queries_per_class: Dict[str, int] = {}

    def record_appearance(self, label: str, processed_idx: int):
        if label not in self.first_appearance:
            self.first_appearance[label] = processed_idx
        self.total_seen[label] = self.total_seen.get(label, 0) + 1

        if not self.fast_mode:
            if label not in self.first_queried:
                self.seen_before_queried[label] = self.seen_before_queried.get(label, 0) + 1

    def record_query(self, label: str, processed_idx: int, query_count: int):
        if label not in self.first_queried:
            self.first_queried[label] = processed_idx
            self.first_queried_query_count[label] = query_count
            if not self.fast_mode:
                pre = self.seen_before_queried.get(label, 0)
                self.seen_before_queried[label] = max(0, pre - 1)
        self.queries_per_class[label] = self.queries_per_class.get(label, 0) + 1

    # ---------------- Convenience accessors (used by reports) ----------------
    def get_seen_count(self) -> int:
        return len(self.first_appearance)

    def get_discovered_count(self) -> int:
        return len(self.first_queried)

    def get_total_seen(self, label: str) -> int:
        return self.total_seen.get(label, 0)

    def get_seen_before_queried(self, label: str) -> int:
        return self.seen_before_queried.get(label, 0)

    def get_queries_for_class(self, label: str) -> int:
        return self.queries_per_class.get(label, 0)

    def get_first_query_count(self, label: str) -> int:
        return self.first_queried_query_count.get(label, 0)

    # ---------------- RED / lift helpers (new) ----------------
    def get_prevalence(self, label: str, N: Optional[int] = None) -> float:
        """Fraction of the processed data belonging to this label."""
        tot = self.get_total_seen(label)
        if N is None or N <= 0:
            # best effort: use grand total of everything we have seen so far
            grand = sum(self.total_seen.values()) or 1
            return tot / grand
        return tot / N if N > 0 else 0.0

    def get_expected_random_queries(self, label: str, N: Optional[int] = None) -> float:
        """Under uniform random querying, on which query ordinal would we expect to hit this class?"""
        p = self.get_prevalence(label, N)
        if p <= 0:
            return float("inf")
        return 1.0 / p

    def compute_lift(self, label: str, N: Optional[int] = None) -> Optional[float]:
        """
        Return the lift for this label using the geometric random baseline.

        lift = expected_random_query_ordinal / actual_discovery_query_ordinal

        A value >> 1 means A_RED surfaced the class far earlier (using far
        fewer labeling queries) than a random sampler would have.
        Returns None if the class was never discovered (or no queries yet).
        """
        q = self.get_first_query_count(label)
        if q is None or q <= 0:
            return None
        exp = self.get_expected_random_queries(label, N)
        if exp in (float("inf"), 0):
            return None
        return exp / q

    def get_labels(self) -> list:
        """All labels that have appeared (in no particular order)."""
        return list(self.first_appearance.keys())

    def get_per_class_stats(self, N: Optional[int] = None) -> Dict[str, dict]:
        """
        Structured data for one label (used by reporting and for RED analysis).
        Includes both the classic enrichment factor and the lift (vs random).
        """
        stats = {}
        for label in self.first_appearance:
            tot = self.get_total_seen(label)
            prev = self.get_prevalence(label, N)
            share_q = (self.get_queries_for_class(label) / max(1, 1))  # caller usually normalizes by total Q
            # enrichment = (queries_on_class / total_queries) / prevalence
            # (filled in by reporter with real total Q)
            enrich = None  # computed by caller who knows global Q
            lift = self.compute_lift(label, N)
            stats[label] = {
                "first_appearance": self.first_appearance[label],
                "first_queried": self.first_queried.get(label),
                "first_query_ordinal": self.get_first_query_count(label) or 0,
                "total_seen": tot,
                "seen_before_queried": self.get_seen_before_queried(label),
                "queries": self.get_queries_for_class(label),
                "prevalence": prev,
                "expected_random": self.get_expected_random_queries(label, N),
                "lift": lift,
                "enrichment": enrich,
            }
        return stats
