def _norm_label(x: str) -> str:
    # keep only the base label if you have "Label:Something" formats
    return x.split(":")[0].strip() if x else ""

def normalize_topology(edge_map, label_to_primary_labels=None):
    """
    Normalizes topology without changing semantics:
    - keeps each topology rule separate (NO union/collapse)
    - normalizes label formatting
    - optionally maps labels -> primary labels
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