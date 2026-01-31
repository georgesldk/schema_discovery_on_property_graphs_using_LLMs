from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx


# ============================================================
# Config
# ============================================================

@dataclass
class EdgeProfileConfig:
    # graph-mode sampling
    sample_limit: int = 1_000_000          # hard cap on accepted edges (across this edge type)
    prop_sample_limit: int = 500          # how many edges to inspect for property keys
    overshoot_factor: float = 2.0         # safety valve: break if processed > sample_limit*overshoot_factor
    seed: Optional[int] = None            # deterministic runs if set

    # reporting
    top_k_topology: int = 15
    top_props: int = 100


# ============================================================
# Graph-mode helpers
# ============================================================

def build_edge_type_index(G: nx.MultiDiGraph) -> Dict[str, List[Tuple[str, str, int]]]:
    """
    Optional accelerator:
    Build an index: edge_type -> list of (u, v, key)

    Use it if you profile many edge types on a big graph.
    Without this, profiling each edge type requires scanning all edges.
    """
    idx: Dict[str, List[Tuple[str, str, int]]] = {}
    for u, v, k, attr in G.edges(keys=True, data=True):
        et = attr.get("type")
        if not et:
            continue
        idx.setdefault(str(et), []).append((u, v, k))
    return idx


def _iter_edges_of_type(
    G: nx.MultiDiGraph,
    target_type: str,
    edge_index: Optional[Dict[str, List[Tuple[str, str, int]]]] = None,
) -> Iterable[Tuple[str, str, int, dict]]:
    """
    Yield (u, v, key, attr) for edges of the requested type.
    If edge_index is supplied, only iterate those edges (fast).
    Otherwise scan all edges (slow).
    """
    if edge_index is not None:
        for (u, v, k) in edge_index.get(str(target_type), []):
            attr = G.get_edge_data(u, v, key=k) or {}
            # NetworkX can return None if deleted; guard
            if isinstance(attr, dict) and attr.get("type") == target_type:
                yield u, v, k, attr
        return

    # fallback: full scan
    for u, v, k, attr in G.edges(keys=True, data=True):
        if attr.get("type") == target_type:
            yield u, v, k, attr


# ============================================================
# Graph-mode profiler
# ============================================================

def profile_edge_type(
    G: nx.MultiDiGraph,
    target_type: str,
    *,
    edge_index: Optional[Dict[str, List[Tuple[str, str, int]]]] = None,
    config: Optional[EdgeProfileConfig] = None,
) -> str:
    """
    Graph-based edge profiler.

    Improvements vs your version:
    - Can use a prebuilt edge_index to avoid scanning all edges per type.
    - Deterministic sampling if seed is set.
    - Always preserves "rare" topology pairs via safety net.
    - Property sampling is bounded and cheap.
    """
    cfg = config or EdgeProfileConfig()

    if cfg.seed is not None:
        random.seed(cfg.seed)

    # If we have an index, we can compute a per-type acceptance probability.
    if edge_index is not None:
        total_edges_this_type = len(edge_index.get(str(target_type), []))
        if total_edges_this_type == 0:
            return ""
        # take all if below limit; else probabilistic acceptance
        acceptance_prob = min(1.0, (cfg.sample_limit * 1.2) / total_edges_this_type)
    else:
        # Without index we don't know per-type size; approximate using whole-graph edges
        total_graph_edges = G.number_of_edges()
        if total_graph_edges == 0:
            return ""
        acceptance_prob = min(1.0, (cfg.sample_limit * 1.5) / total_graph_edges)

    topology_counts: Dict[Tuple[str, str], int] = {}
    prop_key_samples: List[Sequence[str]] = []
    processed = 0

    for u, v, k, attr in _iter_edges_of_type(G, target_type, edge_index=edge_index):
        # topology (cheap)
        src_type = G.nodes[u].get("node_type", "Unknown")
        dst_type = G.nodes[v].get("node_type", "Unknown")
        pair = (src_type, dst_type)

        # safety net: always keep first time we see a topology pair
        is_new_topology = pair not in topology_counts
        is_selected = random.random() < acceptance_prob

        if not (is_new_topology or is_selected):
            continue

        topology_counts[pair] = topology_counts.get(pair, 0) + 1
        processed += 1

        # property sampling (bounded)
        if len(prop_key_samples) < cfg.prop_sample_limit:
            prop_key_samples.append(tuple(attr.keys()))

        # safety valve
        if processed > int(cfg.sample_limit * cfg.overshoot_factor):
            break

    if processed == 0:
        return ""

    # analyze properties
    prop_keys = set()
    for keys_list in prop_key_samples:
        for kk in keys_list:
            if kk != "type":
                prop_keys.add(kk)

    # format profile
    profile = f"\n  [Detected Edge Group]: '{target_type}'\n"
    profile += f"    - Scanned Sample Size: ~{processed} edges (Probabilistic Sample)\n"
    profile += "    - Observed Connection Patterns (Source -> Target):\n"

    sorted_topo = sorted(topology_counts.items(), key=lambda x: x[1], reverse=True)
    top_sorted = sorted_topo[: cfg.top_k_topology]

    for (src, dst), freq in top_sorted:
        pct = (freq / processed) * 100
        profile += f"      * ({src}) -> ({dst}) [sample_freq: {freq}, {pct:.1f}%]\n"

    if len(sorted_topo) > cfg.top_k_topology:
        profile += f"      * ... +{len(sorted_topo) - cfg.top_k_topology} more\n"

    if prop_keys:
        # keep output stable
        props = sorted(prop_keys)
        if len(props) > cfg.top_props:
            shown = props[: cfg.top_props]
            profile += f"    - Edge Properties: {', '.join(shown)}\n"
            profile += f"      * ... +{len(props) - cfg.top_props} more\n"
        else:
            profile += f"    - Edge Properties: {', '.join(props)}\n"
    else:
        profile += "    - Edge Properties: [None]\n"

    return profile


# ============================================================
# Stats-mode profiler
# ============================================================

def profile_edge_type_from_stats(ts, target_type: str, top_k_topology: int = 12, top_props: int = 80) -> str:
    """
    Stats-based edge profiler (no NetworkX).
    Expects ts: TypeStats.
    """
    es = ts.edge_types.get(str(target_type))
    if not es or es.count == 0:
        return ""

    profile = f"\n  [Detected Edge Group]: '{es.name}' ({es.count} instances)\n"

    # Topology patterns
    profile += "    - Observed Connection Patterns (Source -> Target):\n"
    for (src, dst), cnt in es.topology.most_common(top_k_topology):
        pct = (cnt / es.count) * 100 if es.count else 0.0
        profile += f"      * ({src}) -> ({dst}) [freq: {cnt}, {pct:.1f}%]\n"

    # Properties
    if es.prop_fill:
        props_sorted = sorted(es.prop_fill.items(), key=lambda x: x[1], reverse=True)
        prop_names = [p for p, _ in props_sorted[:top_props]]
        profile += f"    - Edge Properties: {', '.join(prop_names)}\n"
        if len(props_sorted) > top_props:
            profile += f"      * ... +{len(props_sorted) - top_props} more\n"
    elif es.prop_keys:
        props = sorted(list(es.prop_keys))
        profile += f"    - Edge Properties: {', '.join(props[:top_props])}\n"
        if len(props) > top_props:
            profile += f"      * ... +{len(props) - top_props} more\n"
    else:
        profile += "    - Edge Properties: [None]\n"

    return profile
