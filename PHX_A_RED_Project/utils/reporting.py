"""
Shared reporting helpers (cluster summaries, etc.).

Kept small on purpose — most detailed reporting still lives in the
experiment or the thin runners so it can be customized per use-case.
"""
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .discovery import ClassDiscoveryCounter


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


# ----------------------- Class Discovery Report (RED focus) -----------------------

def print_class_discovery_report(
    counter: "ClassDiscoveryCounter",
    N: int,
    Q: int,
    title: str = "Class Discovery Report (vs Random Baseline)",
    rare_threshold: float = 0.01,
) -> None:
    """
    Print the discovery report in the style of Perch/main_perch.py (and later
    enriched versions). This is the authoritative pretty-printer used by
    AREDExperiment and the runners.

    Includes the geometric "lift" metric that is especially meaningful when
    A_RED is used for rare event detection in audio streams:
      - Most species/classes are rare (low prevalence).
      - Labeling budget = number of oracle queries is tiny.
      - We want to know: did we surface the rare classes much earlier than
        a random sampler would have?

    The report also highlights "seen before" counts: real audio events that
    arrived but were not labeled because they were absorbed into existing
    clusters without triggering a query.
    """
    seen = counter.get_seen_count()
    disc = counter.get_discovered_count()

    print(f"\n{title}")
    print(f"  Classes seen: {seen} | Discovered: {disc} | Total queries: {Q}")
    print(f"  Dataset size processed: {N:,} points")
    print()

    if seen == 0:
        return

    print("  Per-class (sorted by appearance order):")
    items = []
    for label in counter.get_labels():
        appear_idx = counter.first_appearance.get(label)
        discover_idx = counter.first_queried.get(label)
        q_ordinal = counter.get_first_query_count(label) or 0
        tot = counter.get_total_seen(label)
        pre = counter.get_seen_before_queried(label)
        q_c = counter.get_queries_for_class(label)

        prevalence = tot / N if N > 0 else 0.0
        expected_random = (1.0 / prevalence) if prevalence > 0 else float("inf")

        if q_ordinal > 0:
            lift = expected_random / q_ordinal if expected_random not in (float("inf"), 0) else float("inf")
            lift_str = f"{lift:.1f}x" if lift != float("inf") else "inf"
            act_q = q_ordinal
        else:
            lift_str = "N/A"
            act_q = "never"

        items.append((
            appear_idx or 0,
            discover_idx,
            label,
            act_q,
            expected_random,
            lift_str,
            prevalence,
            pre,
            tot,
            q_c,
        ))

    items.sort(key=lambda x: x[0])

    for appear_idx, discover_idx, label, act_q, exp_rand, lift_str, prev, pre, tot, q_c in items:
        prev_pct = prev * 100
        exp_str = f"{exp_rand:.0f}" if exp_rand != float("inf") else "inf"
        disc_str = str(discover_idx) if discover_idx is not None else "never"
        print(
            f"  {str(label):8} | prevalence={prev_pct:5.2f}% | "
            f"seen index={appear_idx} | queried index={disc_str} | discovery query={act_q} | "
            f"expected random≈{exp_str} | lift={lift_str} | seen before={pre} | (total={tot})"
        )

    # Interpretation (copied/adapted from the Perch reference implementation)
    print("\nInterpretation:")
    print("  • lift > 1.0  = discovered faster than random (good)")
    print("  • lift >> 1.0 = strong active discovery of rare classes")
    print("  • For dominant class (~99%), lift near 1.0 is expected")
    print("  • Geometric model: random expected queries = 1 / prevalence")
    print("  • 'seen before' = instances of the class that arrived unlabeled (absorbed as o_pts)")

    # Optional rare-class aggregate (always printed for RED utility)
    rare_seen = 0
    rare_disc = 0
    rare_unlabeled = 0  # total seen_before mass for rare classes
    lifts = []
    for appear_idx, discover_idx, label, act_q, exp_rand, lift_str, prev, pre, tot, q_c in items:
        if prev < rare_threshold:
            rare_seen += 1
            if act_q != "never":
                rare_disc += 1
                if isinstance(act_q, int) and act_q > 0:
                    # recompute lift cleanly
                    lft = counter.compute_lift(label, N)
                    if lft is not None:
                        lifts.append(lft)
            rare_unlabeled += pre

    if rare_seen > 0:
        med_lift = None
        if lifts:
            lifts_sorted = sorted(lifts)
            mid = len(lifts_sorted) // 2
            med_lift = lifts_sorted[mid] if len(lifts_sorted) % 2 == 1 else (lifts_sorted[mid-1] + lifts_sorted[mid]) / 2
        med_str = f"{med_lift:.1f}x" if med_lift is not None else "N/A"
        print(f"\n  Rare classes (prevalence < {rare_threshold*100:.1f}%): seen={rare_seen}, discovered={rare_disc}")
        print(f"    Median lift among discovered rare: {med_str}")
        print(f"    Unlabeled rare instances before discovery (total 'seen before'): {rare_unlabeled}")


def get_class_discovery_table(
    counter: "ClassDiscoveryCounter",
    N: int,
    Q: int,
) -> List[dict]:
    """Return structured per-class data (useful for get_discovery_metrics, export, plotting)."""
    table = []
    for label in counter.get_labels():
        tot = counter.get_total_seen(label)
        prev = tot / N if N > 0 else 0.0
        q_ord = counter.get_first_query_count(label) or 0
        exp = (1.0 / prev) if prev > 0 else float("inf")
        lift = (exp / q_ord) if (q_ord > 0 and exp not in (float("inf"), 0)) else None

        table.append({
            "label": label,
            "first_appearance": counter.first_appearance.get(label),
            "first_queried_idx": counter.first_queried.get(label),
            "discovery_query": q_ord or None,
            "total_seen": tot,
            "seen_before_queried": counter.get_seen_before_queried(label),
            "queries": counter.get_queries_for_class(label),
            "prevalence": prev,
            "expected_random": exp if exp != float("inf") else None,
            "lift": lift,
            "enrichment": (counter.get_queries_for_class(label) / Q / prev) if (Q > 0 and prev > 0) else None,
        })
    table.sort(key=lambda r: (r["first_appearance"] or 0))
    return table
