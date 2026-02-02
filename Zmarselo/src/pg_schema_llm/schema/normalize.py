def _norm_label(x: str) -> str:
    """
    Normalize a node label to its base form.

    This helper strips secondary qualifiers from labels (e.g., converting
    'Label:Something' to 'Label') to ensure consistent topology matching
    and comparison.

    Args:
        x (str): Raw label string.

    Returns:
        str: Normalized base label.
    """

    return x.split(":")[0].strip() if x else ""

def normalize_topology(edge_map, label_to_primary_labels=None):
    """
    Normalize edge topology definitions without altering semantics.

    This function cleans and deduplicates topology constraints by:
    - Normalizing label formatting
    - Optionally mapping labels to canonical primary labels
    - Preserving each topology rule independently (no union or collapse)

    The procedure ensures stable, comparable topology representations
    while maintaining the original structural intent.

    Args:
        edge_map (dict): Mapping of edge definitions containing topology rules.
        label_to_primary_labels (Optional[dict]): Optional mapping from
            secondary labels to primary canonical labels.

    Returns:
        None
    """

    for edge_def in edge_map.values():
        new_topology = []
        seen = set()

        for topo in edge_def.get("topology", []):
            srcs = []
            tgts = []

            for s in topo.get("allowed_sources", []):
                s2 = _norm_label(s)
                if label_to_primary_labels:
                    s2 = label_to_primary_labels.get(s2, s2)
                if s2:
                    srcs.append(s2)

            for t in topo.get("allowed_targets", []):
                t2 = _norm_label(t)
                if label_to_primary_labels:
                    t2 = label_to_primary_labels.get(t2, t2)
                if t2:
                    tgts.append(t2)

            srcs = tuple(sorted(set(srcs)))
            tgts = tuple(sorted(set(tgts)))
            if not srcs or not tgts:
                continue

            key = (srcs, tgts)
            if key in seen:
                continue
            seen.add(key)

            new_topology.append({
                "allowed_sources": list(srcs),
                "allowed_targets": list(tgts),
            })

        edge_def["topology"] = new_topology