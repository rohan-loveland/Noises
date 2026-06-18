"""
Shared reporting helpers (cluster summaries, etc.).

Kept small on purpose — most detailed reporting still lives in the
experiment or the thin runners so it can be customized per use-case.
"""
from typing import List


def print_cluster_summary(cluster_list: List, only_labeled: bool = True):
    print("\nCluster Summary:")
    for i, cluster in enumerate(cluster_list):
        if only_labeled and cluster.label is None:
            continue
        n_l = len(getattr(cluster, "l_pts", []))
        n_o = len(getattr(cluster, "o_pts", []))
        comp = getattr(cluster, "comp_distance", float("nan"))
        print(f"  Cluster {i}: label={cluster.label}, relevance={cluster.relevance}, "
              f"l_pts={n_l}, o_pts={n_o}, comp_dist={comp:.4f}")
