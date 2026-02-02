from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Any, Iterable

def compare_properties(gt_props, inf_props):
    """
    Compare ground-truth and inferred property name sets.

    This function computes how many property names from the ground-truth
    schema are present in the inferred schema, ignoring types and
    cardinality.

    Args:
        gt_props (Iterable[dict]): Ground-truth property definitions.
        inf_props (Iterable[dict]): Inferred property definitions.

    Returns:
        Tuple[int, int]:
            - Number of matched properties
            - Total number of ground-truth properties
    """

    gt_set = set(p["name"] for p in gt_props)
    inf_set = set(p["name"] for p in inf_props)

    matches = len(gt_set & inf_set)
    total = len(gt_set)

    return matches, total


@dataclass
class NodeTypeStats:
    """
    Statistics container for a node type.

    This dataclass aggregates occurrence counts and property-level
    statistics observed during streaming schema profiling.
    """

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
    """
    Statistics container for an edge type.

    This dataclass aggregates property usage and topology statistics
    observed across all occurrences of a given edge type.
    """
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
    """
    Container for all node and edge type statistics.

    This dataclass groups NodeTypeStats and EdgeTypeStats objects and
    provides helper methods for accessing sorted types and common
    topology patterns.
    """
    node_types: Dict[str, NodeTypeStats] = field(default_factory=dict)
    edge_types: Dict[str, EdgeTypeStats] = field(default_factory=dict)

    def sorted_node_types(self) -> List[str]:
        """
        Return node type names in deterministic sorted order.

        Returns:
            List[str]: Sorted node type names.
        """
        return sorted(self.node_types.keys())

    def sorted_edge_types(self) -> List[str]:
        """
        Return edge type names in deterministic sorted order.

        Returns:
            List[str]: Sorted edge type names.
        """
        return sorted(self.edge_types.keys())

    def topologies_for_edge(self, edge_type: str, top_k: int = 10) -> List[Tuple[Tuple[str, str], int]]:
        """
        Return the most frequent topology patterns for an edge type.

        Args:
            edge_type (str): Edge type name.
            top_k (int): Maximum number of topology patterns to return.

        Returns:
            List[Tuple[Tuple[str, str], int]]: Topology pairs with occurrence counts.
        """
        es = self.edge_types.get(edge_type)
        if not es:
            return []
        return es.topology.most_common(top_k)


def typestats_from_dict(d: dict) -> TypeStats:
    """
    Convert a raw TypeStats dictionary into typed dataclass objects.

    This function transforms the untyped dictionary produced by the
    streaming statistics builder into structured TypeStats, NodeTypeStats,
    and EdgeTypeStats objects.

    Args:
        d (dict): Raw statistics dictionary.

    Returns:
        TypeStats: Structured statistics container.
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
