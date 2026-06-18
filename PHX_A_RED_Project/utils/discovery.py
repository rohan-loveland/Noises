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
    Tracks for each label:
      - first appearance in the processed stream (algorithm time)
      - first actual oracle query that revealed it
      - how many of that label were seen before the discovering query
      - total seen, queries per class, etc.

    Used for enrichment / lift analysis vs random baseline.

    When fast_mode=True the per-point 'seen_before' accounting is skipped
    (saves time and memory on runs of 50k+ points).
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

    # Convenience accessors (used by reports)
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
