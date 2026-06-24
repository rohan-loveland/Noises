"""
Unified Oracle for ARED.

NoRelevanceOracle — the single source of truth for the "near-zero queries" policy.

Contract expected by core ARED (supports both A_RED.py and A_REDIN.py):
    answer_query(abs_index) -> (true_label: str, relevance: bool)

Policy (identical across all historical frontends):
- Always return relevance=False.
- This forces the ARED decision path:
    if not comp_cluster_relevant and not is_anomalous:
        -> add_o_pt (no query)
- Queries only occur for true anomalies (first point + points that fail the
  current kappa anomaly test).
- Once a class is discovered (queried once), subsequent members are absorbed
  as o_pts without additional queries.
- The discovery_tracker (if provided) receives record_query calls so that
  ClassDiscoveryCounter / reporting can track first-seen vs first-queried.

For A_REDIN.py compatibility we also expose:
  - y : list of [label, False] for every point in stream order
  - num_classes
  - int_str_label_bidict : dict[label] -> int
"""

from typing import Optional, List, Dict


class NoRelevanceOracle:
    """
    Single unified oracle.

    Always returns relevance=False.
    Optionally wires a discovery_tracker (DiscoveryTracker or ClassDiscoveryCounter)
    that receives record_query calls.

    We prefer the 3-argument form record_query(label, abs_index, query_count)
    because ClassDiscoveryCounter (the one that powers lift / RED metrics)
    records the exact query ordinal needed for the geometric baseline.
    We fall back to the 2-argument form for lightweight/legacy trackers.

    AREDIN requires y, num_classes and int_str_label_bidict on the oracle.
    """

    def __init__(self, data_stream, discovery_tracker: Optional[object] = None):
        self.data_stream = data_stream
        self.query_count = 0
        self.discovery_tracker = discovery_tracker

        # Build AREDIN-compatible structures (and useful for ARED too)
        self._build_aredin_structs()

        # For distinguishing real query reveals from AREDIN's per-point snoops
        self._last_ared_query_count = 0
        self._answer_calls_per_idx: Dict[int, int] = {}

    def _build_aredin_structs(self):
        """Create y, num_classes, int_str_label_bidict so AREDIN can inspect them directly."""
        n = getattr(self.data_stream, "n_samples", 0) or 0
        labels: List[str] = []
        for i in range(n):
            lab = self.data_stream.get_true_label_for_idx(i)
            if lab is None:
                lab = "unknown"
            labels.append(lab)

        self.y: List[List] = [[lab, False] for lab in labels]

        unique = []
        seen = set()
        for lab in labels:
            if lab not in seen:
                seen.add(lab)
                unique.append(lab)
        self.num_classes = len(unique)
        self.int_str_label_bidict: Dict[str, int] = {lab: idx for idx, lab in enumerate(unique)}

    def answer_query(self, abs_index: int):
        # Track how many times ARED (esp. AREDIN) has called us for this specific point.
        # AREDIN calls answer_query *once per point unconditionally* (to snoop true_label
        # for its internal stats/confusion matrix even on o_pts), and *a second time*
        # only for points it actually decides to query/label.
        self._answer_calls_per_idx[abs_index] = self._answer_calls_per_idx.get(abs_index, 0) + 1
        call_no_for_point = self._answer_calls_per_idx[abs_index]

        true_label = self.data_stream.get_true_label_for_idx(abs_index)

        qc = None

        if hasattr(self, "_ared_ref") and hasattr(self._ared_ref, "num_queries"):
            # AREDIN (and future variants) only increment num_queries on *actual* labeling decisions.
            core_q = int(getattr(self._ared_ref, "num_queries", 0))

            # A real "query spend" reveal is when the core has advanced its num_queries
            # beyond the last one we reported to the discovery tracker.
            # Snoop calls see the old value; the real query call sees the bumped value.
            if core_q > self._last_ared_query_count:
                qc = core_q
                self._last_ared_query_count = core_q
                self.query_count = qc
            else:
                # This is a snoop/peek call (common in AREDIN). Do not treat as a
                # labeling budget query for discovery counting.
                qc = None
        else:
            # Classic A_RED path: answer_query is only invoked for real queries.
            self.query_count += 1
            qc = self.query_count

        if self.discovery_tracker is not None and qc is not None and qc > 0:
            # Only feed the tracker on real query reveals. This ensures each new class
            # gets a distinct, correct query ordinal corresponding to an actual label spend.
            tracker = self.discovery_tracker
            try:
                tracker.record_query(true_label, abs_index, qc)
            except TypeError:
                try:
                    tracker.record_query(true_label, abs_index)
                except Exception:
                    pass
            except Exception:
                pass

        # THE KEY BEHAVIOR: never mark anything relevant
        relevance = False
        return true_label, relevance

    def get_query_count(self) -> int:
        return self.query_count
