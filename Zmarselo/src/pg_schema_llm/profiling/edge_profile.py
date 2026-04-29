from __future__ import annotations


def profile_edge_type_from_stats(ts, target_type: str, top_k_topology: int = 12, top_props: int = 80) -> str:
    """
    Profile an edge type using streaming TypeStats.

    This function generates an edge profile directly from aggregated
    statistics without requiring graph materialization. It reports
    observed topology patterns and property presence frequencies.

    Args:
        ts: TypeStats object containing edge statistics.
        target_type (str): Edge type to profile.
        top_k_topology (int): Maximum number of topology patterns to report.
        top_props (int): Maximum number of properties to list.

    Returns:
        str: Human-readable edge profile summary.
    """
    es = ts.edge_types.get(str(target_type))
    if not es or es.count == 0:
        return ""

    profile = f"\n  [Detected Edge Group]: '{es.name}' ({es.count} instances)\n"
    profile += (
        f"    - Estimated Cardinality: {getattr(es, 'estimated_cardinality', 'UNKNOWN')} "
        f"(distinct_sources={getattr(es, 'source_distinct', 0)}, "
        f"distinct_targets={getattr(es, 'target_distinct', 0)})\n"
    )
    profile += f"    - Observed Labels: {es.name}\n"

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
