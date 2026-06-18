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
    that receives record_query calls.

    We prefer the 3-argument form record_query(label, abs_index, query_count)
    because ClassDiscoveryCounter (the one that powers lift / RED metrics)
    records the exact query ordinal needed for the geometric baseline.
    We fall back to the 2-argument form for lightweight/legacy trackers.
    """

    def __init__(self, data_stream, discovery_tracker: Optional[object] = None):
        self.data_stream = data_stream
        self.query_count = 0
        self.discovery_tracker = discovery_tracker

    def answer_query(self, abs_index: int):
        self.query_count += 1
        true_label = self.data_stream.get_true_label_for_idx(abs_index)

        if self.discovery_tracker is not None:
            # Always try to supply the query_count (3-arg) first.
            # The rich ClassDiscoveryCounter (used for lift calculations) needs it.
            # Lightweight DiscoveryTracker and some legacy counters only accept 2 args.
            tracker = self.discovery_tracker
            qc = self.query_count
            try:
                tracker.record_query(true_label, abs_index, qc)
            except TypeError:
                try:
                    tracker.record_query(true_label, abs_index)
                except Exception:
                    pass  # best-effort; never break the ARED loop because of tracking
            except Exception:
                pass

        # THE KEY BEHAVIOR: never mark anything relevant
        relevance = False
        return true_label, relevance

    def get_query_count(self) -> int:
        return self.query_count
