from __future__ import annotations

from collections import Counter


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
    label_counts = getattr(ns, "label_counts", None)
    if label_counts:
        labs = [lab for lab, _ in Counter(label_counts).most_common(20)]
        profile += f"    - Observed Labels: {', '.join(labs)}\n"
    else:
        profile += "    - Observed Labels: [None]\n"

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
