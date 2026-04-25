"""
Neo4j-based pattern mining for PG schema discovery.

Implements Definitions 3.5 (Node Pattern) and 3.6 (Edge Pattern) from:
  Sideri et al., "PG-HIVE: Hybrid Incremental Schema Discovery
  for Property Graphs", EDBT 2026.

Node Pattern  TNp = (L, K)
  L ⊆ Labels — the full label set of the node
  K ⊆ Keys   — the property-key set

Edge Pattern  TEp = (L, K, R)
  L ⊆ Labels — the edge label set (singleton in Neo4j)
  K ⊆ Keys   — the edge property-key set
  R = (Ls, Lt) — source and target label sets

Full-parse: every node and edge is inspected for pattern discovery.
Sampling is used only for property data-type inference.

Usage
-----
    from neo4j import GraphDatabase
    from pg_schema_llm.io.neo4j_io import mine_patterns

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "pw"))
    result = mine_patterns(driver)
    driver.close()
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Data-type inference  (paper §4.4 hierarchy)
# ============================================================

_DATE_RE = [
    re.compile(r"^\d{4}-\d{2}-\d{2}"),          # ISO  2024-01-15
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),   # 1/15/2024
    re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}$"),   # 15-01-2024
]

# rank: higher number = more general
_TYPE_RANK = {
    "BOOLEAN": 0,
    "INTEGER": 1,
    "DOUBLE": 2,
    "DATE": 3,
    "STRING": 4,
    "LIST": 5,
}


def _infer_value_type(v: Any) -> Optional[str]:
    """Infer the data type of a single property value."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "BOOLEAN"
    if isinstance(v, int):
        return "INTEGER"
    if isinstance(v, float):
        return "DOUBLE"
    if isinstance(v, list):
        return "LIST"
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s.lower() in ("true", "false"):
            return "BOOLEAN"
        try:
            int(s)
            return "INTEGER"
        except ValueError:
            pass
        try:
            float(s)
            return "DOUBLE"
        except ValueError:
            pass
        for pat in _DATE_RE:
            if pat.match(s):
                return "DATE"
        return "STRING"
    return "STRING"


def _resolve_property_type(type_counts: Counter) -> str:
    """
    Pick the most general compatible type from observed values.

    Hierarchy (paper §4.4): BOOLEAN < INTEGER < DOUBLE < DATE < STRING.
    """
    if not type_counts:
        return "STRING"
    return max(type_counts, key=lambda t: _TYPE_RANK.get(t, 99))


# ============================================================
# Helpers
# ============================================================

def _label_key(label_list: list) -> Tuple[str, ...]:
    """Canonical key for a label set (sorted tuple)."""
    return tuple(sorted(label_list))


def _props_key(props_list: list) -> Tuple[str, ...]:
    """Canonical key for a property-key set (sorted tuple)."""
    return tuple(sorted(props_list))


def _escape_label(label: str) -> str:
    """Backtick-escape a Neo4j label for use in Cypher."""
    return f"`{label.replace('`', '``')}`"


def _build_label_match(var: str, label_set: Tuple[str, ...]) -> str:
    """
    Build a WHERE clause for exact label-set matching.

    Example::

        _build_label_match("n", ("Actor", "Person"))
        # => "size(labels(n)) = 2 AND 'Actor' IN labels(n)
        #     AND 'Person' IN labels(n)"
    """
    parts = [f"size(labels({var})) = {len(label_set)}"]
    for lbl in label_set:
        escaped = lbl.replace("'", "\\'")
        parts.append(f"'{escaped}' IN labels({var})")
    return " AND ".join(parts)


# ============================================================
# mine_patterns — main entry point
# ============================================================

def mine_patterns(
    driver,
    *,
    type_sample_limit: int = 500,
    database: Optional[str] = None,
) -> dict:
    """
    Mine node and edge patterns from a Neo4j database (full parse).

    Implements Def 3.5 (Node Pattern) and Def 3.6 (Edge Pattern) from
    the PG-HIVE paper.

    For each **node type** (= unique label set) the function discovers:

    - All distinct patterns ``(L, K)`` with instance counts.
    - Property constraints: ``MANDATORY`` (appears in *every* instance of
      the type) or ``OPTIONAL``.
    - Property data types inferred from sampled values using the priority
      hierarchy BOOLEAN < INTEGER < DOUBLE < DATE < STRING.

    For each **edge type** (= unique ``(rel_type, src_labels, tgt_labels)``)
    it additionally computes **cardinality** from the observed maximum
    in-degree and out-degree (paper §4.4):

    - ``(max_out ≤ 1, max_in ≤ 1)`` → ``1:1``
    - ``(max_out > 1, max_in ≤ 1)`` → ``N:1``
    - ``(max_out ≤ 1, max_in > 1)`` → ``1:N``
    - ``(max_out > 1, max_in > 1)`` → ``M:N``

    Args:
        driver: Open ``neo4j.Driver``.
        type_sample_limit: Nodes/edges to sample per type for data-type
            inference.  Pattern discovery always uses a full scan.
        database: Neo4j database name (``None`` → default).

    Returns:
        ``dict`` with keys ``"node_types"`` (list of node-type dicts)
        and ``"edge_types"`` (list of edge-type dicts).
    """
    session_kw: dict = {}
    if database:
        session_kw["database"] = database

    # accumulators keyed by canonical label tuples
    node_acc: Dict[Tuple, dict] = {}
    edge_acc: Dict[Tuple, dict] = {}

    with driver.session(**session_kw) as session:

        # ==============================================================
        # 1.  NODE PATTERNS  (full scan)
        # ==============================================================
        print(">>> Mining node patterns (full scan) ...")
        rows = list(session.run(
            "MATCH (n) "
            "RETURN labels(n) AS labels, keys(n) AS props, count(*) AS cnt"
        ))

        for r in rows:
            lk = _label_key(r["labels"])
            pk = _props_key(r["props"])
            cnt = r["cnt"]

            if lk not in node_acc:
                node_acc[lk] = {
                    "count": 0,
                    "patterns_map": Counter(),   # props_key → count
                    "all_props": set(),
                }
            acc = node_acc[lk]
            acc["count"] += cnt
            acc["patterns_map"][pk] += cnt
            acc["all_props"].update(r["props"])

        # ---- property constraints + data types per node type ----------
        print(f">>> Computing node property constraints & types "
              f"({len(node_acc)} label sets) ...")
        node_types_out: List[dict] = []

        for lk, acc in sorted(node_acc.items()):
            total = acc["count"]
            all_props = acc["all_props"]

            # fill count: how many instances carry each property
            prop_fill: Counter = Counter()
            for pk, cnt in acc["patterns_map"].items():
                for p in pk:
                    prop_fill[p] += cnt

            # data-type inference (sampled)
            prop_types: Dict[str, str] = {}
            if all_props:
                lf = _build_label_match("n", lk)
                try:
                    srows = list(session.run(
                        f"MATCH (n) WHERE {lf} "
                        f"WITH n LIMIT $lim "
                        f"UNWIND keys(n) AS k "
                        f"WITH k, n[k] AS val "
                        f"RETURN k AS prop, collect(val) AS vals",
                        lim=type_sample_limit,
                    ))
                    for sr in srows:
                        tc = Counter(
                            _infer_value_type(v) for v in sr["vals"]
                        )
                        tc.pop(None, None)
                        prop_types[sr["prop"]] = _resolve_property_type(tc)
                except Exception as e:
                    print(f"    Warning: type sampling failed "
                          f"for {list(lk)}: {e}")

            # assemble properties
            properties: Dict[str, dict] = {}
            for p in sorted(all_props):
                fill = prop_fill.get(p, 0)
                properties[p] = {
                    "data_type": prop_types.get(p, "STRING"),
                    "constraint": (
                        "MANDATORY" if fill == total else "OPTIONAL"
                    ),
                    "fill_ratio": (
                        round(fill / total, 4) if total > 0 else 0.0
                    ),
                }

            # assemble patterns (sorted by count descending)
            patterns = [
                {"property_keys": list(pk), "count": cnt}
                for pk, cnt in sorted(
                    acc["patterns_map"].items(),
                    key=lambda x: -x[1],
                )
            ]

            node_types_out.append({
                "labels": list(lk),
                "count": total,
                "patterns": patterns,
                "properties": properties,
            })
            print(
                f"   Node {str(list(lk)):40s}  count={total:>8,}  "
                f"patterns={len(patterns)}  props={len(properties)}"
            )

        # ==============================================================
        # 2.  EDGE PATTERNS  (full scan)
        # ==============================================================
        print(">>> Mining edge patterns (full scan) ...")
        rows = list(session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN type(r) AS rt, labels(a) AS sl, labels(b) AS tl, "
            "       keys(r) AS props, count(*) AS cnt"
        ))

        for r in rows:
            rt = r["rt"]
            sk = _label_key(r["sl"])
            tk = _label_key(r["tl"])
            ekey = (rt, sk, tk)
            pk = _props_key(r["props"])
            cnt = r["cnt"]

            if ekey not in edge_acc:
                edge_acc[ekey] = {
                    "count": 0,
                    "patterns_map": Counter(),
                    "all_props": set(),
                }
            acc = edge_acc[ekey]
            acc["count"] += cnt
            acc["patterns_map"][pk] += cnt
            acc["all_props"].update(r["props"])

        # ---- cardinality  (paper §4.4) --------------------------------
        print(">>> Computing edge cardinality ...")

        # max out-degree per (rt, sorted_src, sorted_tgt)
        card_out: Dict[Tuple, int] = defaultdict(int)
        for r in session.run(
            "MATCH (a)-[r]->(b) "
            "WITH type(r) AS rt, labels(a) AS sl, labels(b) AS tl, "
            "     a, count(b) AS od "
            "RETURN rt, sl, tl, max(od) AS mx"
        ):
            ek = (r["rt"], _label_key(r["sl"]), _label_key(r["tl"]))
            card_out[ek] = max(card_out[ek], r["mx"])

        # max in-degree
        card_in: Dict[Tuple, int] = defaultdict(int)
        for r in session.run(
            "MATCH (a)-[r]->(b) "
            "WITH type(r) AS rt, labels(a) AS sl, labels(b) AS tl, "
            "     b, count(a) AS id "
            "RETURN rt, sl, tl, max(id) AS mx"
        ):
            ek = (r["rt"], _label_key(r["sl"]), _label_key(r["tl"]))
            card_in[ek] = max(card_in[ek], r["mx"])

        # ---- property constraints + types + cardinality ---------------
        print(f">>> Computing edge property constraints & types "
              f"({len(edge_acc)} edge types) ...")
        edge_types_out: List[dict] = []

        for ekey, acc in sorted(edge_acc.items()):
            rt, sk, tk = ekey
            total = acc["count"]
            all_props = acc["all_props"]

            # fill counts
            prop_fill: Counter = Counter()
            for pk, cnt in acc["patterns_map"].items():
                for p in pk:
                    prop_fill[p] += cnt

            # data-type inference (sampled)
            prop_types: Dict[str, str] = {}
            if all_props:
                esc_rt = _escape_label(rt)
                sf = _build_label_match("a", sk)
                tf = _build_label_match("b", tk)
                try:
                    srows = list(session.run(
                        f"MATCH (a)-[r:{esc_rt}]->(b) "
                        f"WHERE {sf} AND {tf} "
                        f"WITH r LIMIT $lim "
                        f"UNWIND keys(r) AS k "
                        f"WITH k, r[k] AS val "
                        f"RETURN k AS prop, collect(val) AS vals",
                        lim=type_sample_limit,
                    ))
                    for sr in srows:
                        tc = Counter(
                            _infer_value_type(v) for v in sr["vals"]
                        )
                        tc.pop(None, None)
                        prop_types[sr["prop"]] = _resolve_property_type(tc)
                except Exception as e:
                    print(f"    Warning: type sampling failed "
                          f"for {rt}: {e}")

            # properties
            properties: Dict[str, dict] = {}
            for p in sorted(all_props):
                fill = prop_fill.get(p, 0)
                properties[p] = {
                    "data_type": prop_types.get(p, "STRING"),
                    "constraint": (
                        "MANDATORY" if fill == total else "OPTIONAL"
                    ),
                    "fill_ratio": (
                        round(fill / total, 4) if total > 0 else 0.0
                    ),
                }

            # cardinality
            mx_out = card_out.get(ekey, 1)
            mx_in = card_in.get(ekey, 1)
            if mx_out <= 1 and mx_in <= 1:
                cardinality = "1:1"
            elif mx_out > 1 and mx_in <= 1:
                cardinality = "N:1"
            elif mx_out <= 1 and mx_in > 1:
                cardinality = "1:N"
            else:
                cardinality = "M:N"

            # patterns (sorted by count descending)
            patterns = [
                {"property_keys": list(pk), "count": cnt}
                for pk, cnt in sorted(
                    acc["patterns_map"].items(),
                    key=lambda x: -x[1],
                )
            ]

            edge_types_out.append({
                "labels": [rt],
                "source_labels": list(sk),
                "target_labels": list(tk),
                "count": total,
                "cardinality": cardinality,
                "max_out_degree": mx_out,
                "max_in_degree": mx_in,
                "patterns": patterns,
                "properties": properties,
            })
            print(
                f"   Edge {list(sk)}-[:{rt}]->{list(tk)}  "
                f"count={total:>8,}  card={cardinality}  "
                f"patterns={len(patterns)}  props={len(properties)}"
            )

    # ------------------------------------------------------------------
    # final result
    # ------------------------------------------------------------------
    result = {
        "node_types": node_types_out,
        "edge_types": edge_types_out,
    }
    n_np = sum(len(nt["patterns"]) for nt in node_types_out)
    n_ep = sum(len(et["patterns"]) for et in edge_types_out)
    print(f"\n Pattern mining complete.")
    print(f"   Node types: {len(node_types_out)}  "
          f"({n_np} distinct node patterns)")
    print(f"   Edge types: {len(edge_types_out)}  "
          f"({n_ep} distinct edge patterns)")
    return result