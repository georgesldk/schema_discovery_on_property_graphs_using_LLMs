from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Any, Iterable

def compare_properties(gt_props, inf_props):
    gt_set = set(p["name"] for p in gt_props)
    inf_set = set(p["name"] for p in inf_props)

    matches = len(gt_set & inf_set)
    total = len(gt_set)

    return matches, total


@dataclass
class NodeTypeStats:
    name: str
    count: int = 0

    # property -> non-null occurrences
    prop_fill: Counter = field(default_factory=Counter)

    # (property, kind) -> votes (kind: "String"/"Long"/"Double"/"Boolean"...)
    prop_kind: Counter = field(default_factory=Counter)

    # property -> small list of sample strings
    prop_samples: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class EdgeTypeStats:
    name: str
    count: int = 0

    prop_fill: Counter = field(default_factory=Counter)
    prop_kind: Counter = field(default_factory=Counter)

    # set/list of property keys seen on edges (excluding start/end)
    prop_keys: set = field(default_factory=set)

    # (src_type, dst_type) -> occurrences
    topology: Counter = field(default_factory=Counter)


@dataclass
class TypeStats:
    node_types: Dict[str, NodeTypeStats] = field(default_factory=dict)
    edge_types: Dict[str, EdgeTypeStats] = field(default_factory=dict)

    def sorted_node_types(self) -> List[str]:
        return sorted(self.node_types.keys())

    def sorted_edge_types(self) -> List[str]:
        return sorted(self.edge_types.keys())

    def topologies_for_edge(self, edge_type: str, top_k: int = 10) -> List[Tuple[Tuple[str, str], int]]:
        es = self.edge_types.get(edge_type)
        if not es:
            return []
        return es.topology.most_common(top_k)


def typestats_from_dict(d: dict) -> TypeStats:
    """
    Convert the raw dict returned by io.graph_builder.build_typestats(...)
    into TypeStats dataclasses.

    This keeps Phase 1 output stable while letting later phases use typed objects.
    """
    out = TypeStats()

    node_d = d.get("node_types") or {}
    for name, st in node_d.items():
        ns = NodeTypeStats(name=name)
        ns.count = int(st.get("count", 0))
        ns.prop_fill = Counter(st.get("prop_fill", {}))
        ns.prop_kind = Counter(st.get("prop_kind", {}))

        # prop_samples might be defaultdict(list) or dict
        raw_samples = st.get("prop_samples", {}) or {}
        # normalize to dict[str, list[str]]
        ns.prop_samples = defaultdict(list)
        for k, vals in raw_samples.items():
            if vals:
                ns.prop_samples[str(k)] = [str(v) for v in list(vals)]
        out.node_types[name] = ns

    edge_d = d.get("edge_types") or {}
    for name, st in edge_d.items():
        es = EdgeTypeStats(name=name)
        es.count = int(st.get("count", 0))
        es.prop_fill = Counter(st.get("prop_fill", {}))
        es.prop_kind = Counter(st.get("prop_kind", {}))
        es.prop_keys = set(st.get("prop_keys", set()) or [])
        es.topology = Counter(st.get("topology", {}))
        out.edge_types[name] = es

    return out
