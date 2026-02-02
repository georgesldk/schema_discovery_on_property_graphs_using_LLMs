from __future__ import annotations

import json
import random
from dataclasses import dataclass
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx


# ============================================================
# Config
# ============================================================

@dataclass
class NodeProfileConfig:
    """
    Configuration container for node profiling.

    This dataclass defines sampling parameters, node-type attribute
    resolution, and reporting limits used during node profiling.
    """

    # graph-mode sampling
    sample_size: int = 500
    seed: Optional[int] = 42
    node_type_attr: str = "node_type"

    # stats-mode output
    top_props: int = 60


# ============================================================
# Graph-mode indexing (optional accelerator)
# ============================================================

def build_node_type_index(
    G: nx.MultiDiGraph,
    *,
    node_type_attr: str = "node_type",
) -> Dict[str, List[str]]:
    """
    Build an index mapping node types to node identifiers.

    This optional accelerator constructs a dictionary mapping each
    node type to the list of node identifiers that belong to it. It
    prevents repeated full-graph scans when profiling multiple node
    types.

    Args:
        G (nx.MultiDiGraph): Input property graph.
        node_type_attr (str): Node attribute containing the node type.

    Returns:
        Dict[str, List[str]]: Node type index.
    """

    idx: Dict[str, List[str]] = {}
    for n, attr in G.nodes(data=True):
        for lab in _labels_of(attr, node_type_attr):
            idx.setdefault(lab, []).append(n)
    return idx


# ============================================================
# Small helpers
# ============================================================

def _labels_of(node_attr: dict, node_type_attr: str) -> List[str]:
    """
    Extract node labels from a node attribute dictionary.

    This helper normalizes the node-type attribute into a list of
    string labels, supporting single values as well as collections.

    Args:
        node_attr (dict): Node attribute dictionary.
        node_type_attr (str): Key used to store node type information.

    Returns:
        List[str]: List of node labels.
    """
    v = node_attr.get(node_type_attr)
    if v is None:
        return []
    if isinstance(v, (list, set, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def _stable(v):
    """
    Produce a stable, hashable representation of a value.

    This helper converts dictionaries and lists into a deterministic
    string representation to enable uniqueness and cardinality checks.

    Args:
        v (Any): Input value.

    Returns:
        Any: Stable representation of the value.
    """
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return v


# ============================================================
# Graph-mode profiler
# ============================================================

def profile_node_type(
    G: nx.MultiDiGraph,
    target_type: str,
    *,
    node_index: Optional[Dict[str, List[str]]] = None,
    config: Optional[NodeProfileConfig] = None,
) -> str:
    """
    Profile a node type using a materialized graph.

    This function samples nodes of a given type to infer property
    presence, cardinality characteristics, and value-type distributions.
    It supports deterministic sampling and optional acceleration via
    a precomputed node index.

    Args:
        G (nx.MultiDiGraph): Input property graph.
        target_type (str): Node type to profile.
        node_index (Optional[dict]): Optional node type index.
        config (Optional[NodeProfileConfig]): Profiling configuration.

    Returns:
        str: Human-readable node profile summary.
    """
    cfg = config or NodeProfileConfig()
    target_type = str(target_type)

    # get nodes of this type
    if node_index is not None:
        nodes = node_index.get(target_type, [])
    else:
        # fallback: scan all nodes (slow)
        nodes = []
        for n, attr in G.nodes(data=True):
            if target_type in _labels_of(attr, cfg.node_type_attr):
                nodes.append(n)

    count = len(nodes)
    if count == 0:
        return ""

    rng = random.Random(cfg.seed) if cfg.seed is not None else random
    sample = rng.sample(nodes, k=min(count, cfg.sample_size))

    # property keys excluding label field
    keys: Set[str] = set()
    for n in sample:
        keys.update(k for k in G.nodes[n].keys() if k != cfg.node_type_attr)

    profile = f"\n  [Detected Node Group]: '{target_type}' ({count} instances)\n"

    for key in sorted(keys):
        vals = []
        for n in sample:
            v = G.nodes[n].get(key)
            if v is not None:
                vals.append(v)

        if not vals:
            continue

        density = (len(vals) / len(sample)) * 100.0

        norm_vals = [_stable(v) for v in vals]
        unique_ratio = len(set(norm_vals)) / len(norm_vals) if norm_vals else 0.0

        type_counts = Counter(type(v).__name__ for v in vals).most_common(3)
        types_str = ", ".join(f"{t}({c})" for t, c in type_counts)

        # neutral cardinality language (no "Unique ID")
        if unique_ratio > 0.98 and len(vals) >= 50:
            nature = "High-cardinality"
        elif unique_ratio < 0.10:
            nature = "Low-cardinality (enum-like)"
        else:
            nature = "Mixed/continuous"

        profile += (
            f"    - Property '{key}': {density:.1f}% fill. "
            f"Cardinality: {nature}. Types: {types_str}\n"
        )

    return profile


# ============================================================
# Stats-mode profiler
# ============================================================

def _best_kind(prop_kind_counter: Counter, prop: str) -> str:
    """
    Infer the most likely data type for a node property.

    This function selects the data type with the highest observed
    frequency for a given property based on collected statistics.

    Args:
        prop_kind_counter (Counter): Counter mapping (property, kind)
            pairs to observed counts.
        prop (str): Property name.

    Returns:
        str: Inferred canonical property type.
    """
    votes = Counter()
    for (p, k), c in prop_kind_counter.items():
        if p == prop:
            votes[k] += c
    if not votes:
        return "String"
    return votes.most_common(1)[0][0]


def profile_node_type_from_stats(ts, target_type: str, top_props: int = 60) -> str:
    """
    Profile a node type using streaming TypeStats.

    This function generates a node profile directly from aggregated
    statistics without requiring graph materialization. It reports
    property fill rates, inferred data types, and cardinality patterns.

    Args:
        ts: TypeStats object containing node statistics.
        target_type (str): Node type to profile.
        top_props (int): Maximum number of properties to report.

    Returns:
        str: Human-readable node profile summary.
    """
    ns = ts.node_types.get(str(target_type))
    if not ns or ns.count == 0:
        return ""

    profile = f"\n  [Detected Node Group]: '{ns.name}' ({ns.count} instances)\n"

    # Properties sorted by fill count (most informative first)
    items = list(ns.prop_fill.items())
    items.sort(key=lambda x: x[1], reverse=True)

    shown = 0
    for prop, filled in items:
        if shown >= top_props:
            break

        density = (filled / ns.count) * 100.0 if ns.count else 0.0
        kind = _best_kind(ns.prop_kind, prop)

        samples = list(ns.prop_samples.get(prop, [])) if hasattr(ns, "prop_samples") else []
        uniq = len(set(samples)) if samples else 0

        if not samples:
            nature = "Unknown-cardinality"
        else:
            if uniq == 1:
                nature = "Low-cardinality (enum-like)"
            elif uniq == len(samples):
                nature = "High-cardinality"
            else:
                nature = "Mixed"

        profile += (
            f"    - Property '{prop}': {density:.1f}% fill. "
            f"Cardinality: {nature}. Type: {kind}\n"
        )
        shown += 1

    return profile
