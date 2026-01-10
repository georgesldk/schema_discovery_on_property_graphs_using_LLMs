import random


def profile_node_type(G, target_type):
    nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == target_type]
    count = len(nodes)
    if count == 0:
        return ""

    sample = random.sample(nodes, min(count, 500))
    keys = {k for n in sample for k in G.nodes[n].keys() if k != 'node_type'}

    profile = f"\n  [Detected Node Group]: '{target_type}' ({count} instances)\n"
    for key in sorted(keys):
        vals = [G.nodes[n].get(key) for n in sample if G.nodes[n].get(key) is not None]
        if not vals:
            continue

        density = (len(vals) / len(sample)) * 100
        unique_ratio = len(set(str(v) for v in vals)) / len(vals)
        nature = "Unique ID" if unique_ratio > 0.9 else "Category/Enum" if unique_ratio < 0.1 else "Value"

        profile += (f"    - Property '{key}': {density:.1f}% fill. "
                    f"Nature: {nature}. Type: {type(vals[0]).__name__}\n")
    return profile
