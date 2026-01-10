from collections import Counter


def profile_edge_type(G, target_type):
    edges = [(u, v, attr) for u, v, attr in G.edges(data=True) if attr.get('type') == target_type]
    if not edges:
        return ""

    conns = [
        f"({G.nodes[u].get('node_type', 'Unknown')})->({G.nodes[v].get('node_type', 'Unknown')})"
        for u, v, _ in edges[:100]
    ]
    top_conns = Counter(conns).most_common(2)

    # Analyze edge properties for semantic hints
    sample_edges = edges[:min(500, len(edges))]
    all_properties = {}
    for u, v, attr in sample_edges:
        for key, value in attr.items():
            if key != 'type':
                if key not in all_properties:
                    all_properties[key] = {'count': 0, 'sample_values': []}
                all_properties[key]['count'] += 1
                if len(all_properties[key]['sample_values']) < 3 and value is not None:
                    all_properties[key]['sample_values'].append(str(value)[:50])

    profile = f"\n  [Detected Edge Group]: '{target_type}' ({len(edges)} instances)\n"
    profile += f"    - Observed Connections: {', '.join([c[0] for c in top_conns])}\n"

    if all_properties:
        profile += f"    - Edge Properties: {', '.join(sorted(all_properties.keys()))}\n"

    return profile
