import random
import json
from collections import Counter

def _labels_of(node_attr, node_type_attr):
    v = node_attr.get(node_type_attr)
    if v is None:
        return []
    if isinstance(v, (list, set, tuple)):
        return [str(x) for x in v]
    return [str(v)]

def _stable(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return v

def profile_node_type(G, target_type, node_type_attr="node_type", sample_size=500, seed=42):
    target_type = str(target_type)

    nodes = []
    for n, attr in G.nodes(data=True):
        if target_type in _labels_of(attr, node_type_attr):
            nodes.append(n)

    count = len(nodes)
    if count == 0:
        return ""

    rng = random.Random(seed)
    sample = rng.sample(nodes, k=min(count, sample_size))

    # all property keys except the label field
    keys = {k for n in sample for k in G.nodes[n].keys() if k != node_type_attr}

    profile = f"\n  [Detected Node Group]: '{target_type}' ({count} instances)\n"
    for key in sorted(keys):
        vals = [G.nodes[n].get(key) for n in sample if G.nodes[n].get(key) is not None]
        if not vals:
            continue

        density = (len(vals) / len(sample)) * 100

        norm_vals = [_stable(v) for v in vals]
        unique_ratio = len(set(norm_vals)) / len(norm_vals)

        type_counts = Counter(type(v).__name__ for v in vals).most_common(3)
        types_str = ", ".join(f"{t}({c})" for t, c in type_counts)

        # Keep nature neutral: avoid "Unique ID"
        if unique_ratio > 0.98 and len(vals) >= 50:
            nature = "High-cardinality"
        elif unique_ratio < 0.1:
            nature = "Low-cardinality (enum-like)"
        else:
            nature = "Mixed/continuous"

        profile += (f"    - Property '{key}': {density:.1f}% fill. "
                    f"Cardinality: {nature}. Types: {types_str}\n")

    return profile