def normalize_topology(edge_map, label_to_primary_labels):
    for edge_def in edge_map.values():
        sources, targets = set(), set()

        for topo in edge_def["topology"]:
            for s in topo.get("allowed_sources", []):
                sources.add(s.split(":")[0])
            for t in topo.get("allowed_targets", []):
                targets.add(t.split(":")[0])

        if sources and targets:
            edge_def["topology"] = [{
                "allowed_sources": sorted(sources),
                "allowed_targets": sorted(targets)
            }]
