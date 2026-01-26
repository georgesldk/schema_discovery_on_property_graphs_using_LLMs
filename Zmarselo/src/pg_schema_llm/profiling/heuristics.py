import networkx as nx
from collections import Counter

def iter_edge_attrdicts(G, u, v):
    data = G.get_edge_data(u, v)
    if not data:
        return []
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        return list(data.values())
    return [data]

def suggest_edge_label_from_data(G, source_type, target_type, edge_type_key="type", max_pairs=2000):
    """
    Domain-agnostic:
    - If edges have an explicit label/type attribute, return the most frequent observed value.
    - Otherwise return None and rely on structural relationship discovery only.
    """
    source_nodes = [n for n, a in G.nodes(data=True) if a.get("node_type") == source_type]
    target_nodes = [n for n, a in G.nodes(data=True) if a.get("node_type") == target_type]
    if not source_nodes or not target_nodes:
        return None, {"reason": "no_nodes"}

    # Count observed edge labels on direct edges only (safe + fast)
    counts = Counter()
    checked = 0
    for u in source_nodes:
        for v in G.successors(u):
            if G.nodes[v].get("node_type") != target_type:
                continue
            for attrs in iter_edge_attrdicts(G, u, v):
                t = attrs.get(edge_type_key)
                if t:
                    counts[str(t)] += 1
            checked += 1
            if checked >= max_pairs:
                break
        if checked >= max_pairs:
            break

    if not counts:
        return None, {"reason": "no_edge_labels", "direct_edges_checked": checked}

    label, label_count = counts.most_common(1)[0]
    total = sum(counts.values())
    return label, {"votes": label_count, "total_labeled_edges": total, "distribution_top3": counts.most_common(3)}