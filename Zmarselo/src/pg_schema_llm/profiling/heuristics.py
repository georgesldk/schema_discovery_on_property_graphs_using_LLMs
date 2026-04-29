from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# Stats-mode heuristics (scalable)
# ============================================================

@dataclass
class StatsHeuristicConfig:
    """
    Configuration container for statistics-based structural heuristics.

    This dataclass defines thresholds and limits used to detect technical
    containers, logical paths, bidirectional patterns, and summary verbosity.
    """

    # container detection thresholds (tuned to avoid false positives on MB6/FIB25)
    max_meaningful_props: int = 1
    meaningful_prop_min_fill: float = 0.20   # property considered meaningful if appears in >= 20% nodes
    min_nodes: int = 50                      # don't classify tiny types aggressively

    # topology-based signals
    min_distinct_in: int = 2                 # join-like nodes have multiple distinct incoming types
    min_distinct_out: int = 2                # ...and multiple outgoing
    max_out_to_in_ratio: float = 4.0         # avoid misclassifying "hub" entities as containers

    # logical path extraction
    min_path_support: int = 5                # votes from A->C and C->B must be >= this

    # summary verbosity
    top_k_containers: int = 20
    top_k_paths: int = 40
    top_k_bidir: int = 30


def _build_type_adjacency_counts(ts) -> Tuple[Dict[str, Counter], Dict[str, Counter]]:
    """
    Build weighted adjacency counts between node types.

    This helper aggregates edge topology statistics across all edge types
    to produce weighted outgoing and incoming adjacency maps between
    node types.

    Args:
        ts: TypeStats object containing edge topology statistics.

    Returns:
        Tuple[Dict[str, Counter], Dict[str, Counter]]:
            - Outgoing adjacency counts per node type
            - Incoming adjacency counts per node type
    """
    out_counts: Dict[str, Counter] = defaultdict(Counter)
    in_counts: Dict[str, Counter] = defaultdict(Counter)

    for _, es in ts.edge_types.items():
        for (src, dst), cnt in es.topology.items():
            if cnt and cnt > 0:
                out_counts[src][dst] += int(cnt)
                in_counts[dst][src] += int(cnt)

    return out_counts, in_counts


def identify_technical_containers_from_stats(ts, cfg: Optional[StatsHeuristicConfig] = None) -> Set[str]:
    """
    Identify technical container or join node types using statistics.

    This function combines property sparsity and topology-based signals
    to detect node types that primarily serve as technical intermediates
    rather than meaningful domain entities.

    Args:
        ts: TypeStats object.
        cfg (Optional[StatsHeuristicConfig]): Heuristic configuration.

    Returns:
        Set[str]: Set of node type names classified as technical containers.
    """
    cfg = cfg or StatsHeuristicConfig()
    out_counts, in_counts = _build_type_adjacency_counts(ts)

    tech: Set[str] = set()

    for nt, ns in ts.node_types.items():
        # small types are unstable; skip aggressive labeling
        if ns.count < cfg.min_nodes:
            continue

        # meaningful properties: appear in >= X% of nodes
        meaningful = 0
        for prop, filled in ns.prop_fill.items():
            if ns.count and (filled / ns.count) >= cfg.meaningful_prop_min_fill:
                meaningful += 1

        # topology signals
        distinct_in = len(in_counts.get(nt, {}))
        distinct_out = len(out_counts.get(nt, {}))

        total_in = sum(in_counts.get(nt, {}).values())
        total_out = sum(out_counts.get(nt, {}).values())
        ratio = (total_out / total_in) if total_in else float("inf")

        # join/container tends to:
        # - have very few stable props
        # - connect multiple types on both sides
        # - not be an extreme hub in one direction only
        sparse_props = meaningful <= cfg.max_meaningful_props
        join_like = (distinct_in >= cfg.min_distinct_in and distinct_out >= cfg.min_distinct_out)
        not_extreme_hub = (ratio <= cfg.max_out_to_in_ratio)

        if sparse_props and join_like and not_extreme_hub:
            tech.add(nt)

    return tech


def analyze_logical_paths_from_stats(ts, tech_containers: Set[str], cfg: Optional[StatsHeuristicConfig] = None) -> List[Tuple[str, str, str, int]]:
    """
    Analyze indirect logical paths through technical containers.

    This function detects A ? C ? B patterns where C is a technical
    container node type and aggregates support counts based on observed
    edge frequencies.

    Args:
        ts: TypeStats object.
        tech_containers (Set[str]): Identified technical container types.
        cfg (Optional[StatsHeuristicConfig]): Heuristic configuration.

    Returns:
        List[Tuple[str, str, str, int]]: Detected logical paths with
        support vote counts.
    """
    cfg = cfg or StatsHeuristicConfig()
    out_counts, in_counts = _build_type_adjacency_counts(ts)

    node_types = set(ts.node_types.keys())
    entity_types = node_types - set(tech_containers)

    results: List[Tuple[str, str, str, int]] = []

    for c in tech_containers:
        preds = set(in_counts.get(c, {}).keys()) & entity_types
        succs = set(out_counts.get(c, {}).keys()) & entity_types

        for a in preds:
            ac = out_counts.get(a, {}).get(c, 0)
            if ac <= 0:
                continue
            for b in succs:
                if a == b:
                    continue
                cb = out_counts.get(c, {}).get(b, 0)
                if cb <= 0:
                    continue

                support = int(min(ac, cb))
                if support >= cfg.min_path_support:
                    results.append((a, c, b, support))

    # strongest first
    results.sort(key=lambda x: x[3], reverse=True)
    return results


def analyze_bidirectional_patterns_from_stats(ts, cfg: Optional[StatsHeuristicConfig] = None) -> List[Tuple[str, str, int]]:
    """
    Detect bidirectional or symmetric relationships between node types.

    This function identifies pairs of node types with observed edges in
    both directions and reports a support score based on edge frequencies.

    Args:
        ts: TypeStats object.
        cfg (Optional[StatsHeuristicConfig]): Heuristic configuration.

    Returns:
        List[Tuple[str, str, int]]: Bidirectional node type pairs with
        support counts.
    """
    cfg = cfg or StatsHeuristicConfig()
    out_counts, _ = _build_type_adjacency_counts(ts)

    pairs: List[Tuple[str, str, int]] = []
    for a, outs in out_counts.items():
        for b, ab in outs.items():
            if a >= b:
                continue
            ba = out_counts.get(b, {}).get(a, 0)
            if ba > 0:
                pairs.append((a, b, int(min(ab, ba))))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def generate_logical_relationship_summary_from_stats(ts, cfg: Optional[StatsHeuristicConfig] = None) -> str:
    """
    Generate a structured summary of inferred logical relationships.

    This function synthesizes detected technical containers, indirect
    logical paths, and bidirectional patterns into a textual summary
    compatible with the LLM prompt format.

    Args:
        ts: TypeStats object.
        cfg (Optional[StatsHeuristicConfig]): Heuristic configuration.

    Returns:
        str: Human-readable logical relationship summary.
    """
    cfg = cfg or StatsHeuristicConfig()

    tech_containers = identify_technical_containers_from_stats(ts, cfg=cfg)
    logical_paths = analyze_logical_paths_from_stats(ts, tech_containers, cfg=cfg)
    bidirectional = analyze_bidirectional_patterns_from_stats(ts, cfg=cfg)

    if not logical_paths and not bidirectional and not tech_containers:
        return ""

    summary = "\n  [STRUCTURAL RELATIONSHIP ANALYSIS]\n"

    if tech_containers:
        tech_list = sorted(tech_containers)
        shown = tech_list[: cfg.top_k_containers]
        summary += f"    - Identified Intermediate/Join Nodes: {', '.join(shown)}\n"
        if len(tech_list) > cfg.top_k_containers:
            summary += f"      * ... +{len(tech_list) - cfg.top_k_containers} more\n"
    else:
        summary += "    - Identified Intermediate/Join Nodes: [None]\n"

    if logical_paths:
        summary += "    - Suggested Direct Logical Relationships (bypassing intermediates):\n"
        grouped: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)

        for a, c, b, support in logical_paths[: cfg.top_k_paths]:
            grouped[(a, b)].append((c, support))

        for (a, b), items in sorted(grouped.items()):
            # keep unique containers, keep max support per container
            best_support: Dict[str, int] = {}
            for c, s in items:
                best_support[c] = max(best_support.get(c, 0), s)

            containers_str = ", ".join(f"{c}(support={best_support[c]})" for c in sorted(best_support.keys()))
            summary += f"      * {a} -> {b} (via: {containers_str})\n"

    if bidirectional:
        summary += "    - Detected Bidirectional/Symmetric Patterns:\n"
        for a, b, support in bidirectional[: cfg.top_k_bidir]:
            summary += f"      * {a} <-> {b} (support={support})\n"

    summary += "    - Observed frequent indirect patterns (via intermediate nodes).\n"
    summary += "      You may propose direct relationships ONLY if they are consistent with observed edge labels and node-type connectivity.\n"
    return summary
