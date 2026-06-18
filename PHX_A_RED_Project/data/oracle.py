"""
Unified Oracle for ARED.

NoRelevanceOracle — the single source of truth for the "near-zero queries" policy.

Contract expected by core ARED:
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
"""

from typing import Optional


class NoRelevanceOracle:
    """
    Single unified oracle.

    Always returns relevance=False.
    Optionally wires a discovery_tracker (DiscoveryTracker or ClassDiscoveryCounter)
    that receives record_query(true_label, abs_index).

    Some historical trackers expected (label, idx, query_count). We support both:
    - If the tracker has record_query(self, label, idx, query_count) we call the 3-arg form.
    - Otherwise we call record_query(self, label, idx).
    """

    def __init__(self, data_stream, discovery_tracker: Optional[object] = None):
        self.data_stream = data_stream
        self.query_count = 0
        self.discovery_tracker = discovery_tracker

    def answer_query(self, abs_index: int):
        self.query_count += 1
        true_label = self.data_stream.get_true_label_for_idx(abs_index)

        if self.discovery_tracker is not None:
            # Support both common tracker signatures
            try:
                # Preferred modern signature used by our utils
                self.discovery_tracker.record_query(true_label, abs_index)
            except TypeError:
                try:
                    # Legacy 3-arg version (some older ClassDiscoveryCounter copies)
                    self.discovery_tracker.record_query(true_label, abs_index, self.query_count)
                except Exception:
                    pass  # tracker is best-effort

        # THE KEY BEHAVIOR: never mark anything relevant
        relevance = False
        return true_label, relevance

    def get_query_count(self) -> int:
        return self.query_count
