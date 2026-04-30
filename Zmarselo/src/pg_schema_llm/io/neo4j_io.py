"""
Neo4j-based pattern mining for PG schema discovery.

Implements Definitions 3.5 (Node Pattern) and 3.6 (Edge Pattern) from
Sideri et al., "PG-HIVE", EDBT 2026.

This version

* does a **full parse** of every property value for type inference (no
  sampling), in chunks to keep memory/CPU bounded;
* supports **0-cardinality** (the relationship is optional on either side):
  cardinality is reported as ``"{src_min}..{src_max} : {tgt_min}..{tgt_max}"``
  with each side ∈ {``0..1``, ``0..N``, ``1..1``, ``1..N``};
* expands every multi-label edge into **all label-subset permutations**
  so that downstream code sees, for an edge between ``{A,B}`` and
  ``{C,D}``, all of:
  ``[A]→[C], [A]→[D], [A]→{C,D}, [B]→[C], …, {A,B}→{C,D}`` —
  with ``is_canonical`` marking the entry that matches the actual data.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple


# ============================================================
# Data-type inference
# ============================================================

_DATE_RE = [
    re.compile(r"^\d{4}-\d{2}-\d{2}"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}$"),
]

_TYPE_RANK = {
    "BOOLEAN": 0, "INTEGER": 1, "DOUBLE": 2,
    "DATE": 3, "STRING": 4, "LIST": 5,
}


def _infer_value_type(v: Any) -> Optional[str]:
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
            int(s); return "INTEGER"
        except ValueError:
            pass
        try:
            float(s); return "DOUBLE"
        except ValueError:
            pass
        for pat in _DATE_RE:
            if pat.match(s):
                return "DATE"
        return "STRING"
    return "STRING"


def _resolve_property_type(type_counts: Counter) -> str:
    """Most general compatible type: BOOLEAN < INTEGER < DOUBLE < DATE < STRING < LIST."""
    if not type_counts:
        return "STRING"
    return max(type_counts, key=lambda t: _TYPE_RANK.get(t, 99))


# ============================================================
# Helpers
# ============================================================

def _label_key(label_list) -> Tuple[str, ...]:
    return tuple(sorted(label_list))


def _props_key(props_list) -> Tuple[str, ...]:
    return tuple(sorted(props_list))


def _escape_label(label: str) -> str:
    return f"`{label.replace('`', '``')}`"


def _build_label_match_exact(var: str, label_set: Tuple[str, ...]) -> str:
    """WHERE clause: node has exactly this label set."""
    parts = [f"size(labels({var})) = {len(label_set)}"]
    for lbl in label_set:
        parts.append(f"'{lbl.replace(chr(39), chr(92)+chr(39))}' IN labels({var})")
    return " AND ".join(parts) if label_set else f"size(labels({var})) = 0"


def _build_label_match_contains(var: str, label_set: Tuple[str, ...]) -> str:
    """WHERE clause: node contains every label in this set (may have more)."""
    if not label_set:
        return "true"
    parts = [
        f"'{lbl.replace(chr(39), chr(92)+chr(39))}' IN labels({var})"
        for lbl in label_set
    ]
    return " AND ".join(parts)


def _all_nonempty_subsets(label_set: Tuple[str, ...]) -> List[Tuple[str, ...]]:
    """All non-empty subsets of a label tuple, each sorted."""
    if not label_set:
        return [()]
    out: List[Tuple[str, ...]] = []
    n = len(label_set)
    for r in range(1, n + 1):
        for combo in combinations(label_set, r):
            out.append(tuple(sorted(combo)))
    return out


def _format_cardinality(min_out: int, max_out: int,
                        min_in: int, max_in: int) -> str:
    """Render '{a..b} : {c..d}' where each side ∈ {0..1, 0..N, 1..1, 1..N}."""
    def side(lo: int, hi: int) -> str:
        lo_s = "0" if lo == 0 else "1"
        hi_s = "1" if hi <= 1 else "N"
        return f"{lo_s}..{hi_s}"
    return f"{side(min_out, max_out)} : {side(min_in, max_in)}"


# ============================================================
# Full-scan property type inference (chunked)
# ============================================================

def _full_scan_node_property_types(
    session,
    label_set: Tuple[str, ...],
    total: int,
    chunk_size: int,
) -> Dict[str, Counter]:
    """
    Full scan of property values for nodes with EXACTLY this label set,
    chunked to bound memory.  Returns Counter of observed types per prop.
    """
    type_counts: Dict[str, Counter] = defaultdict(Counter)
    if total == 0:
        return type_counts

    where = _build_label_match_exact("n", label_set)
    skip = 0
    while skip < total:
        rows = list(session.run(
            f"MATCH (n) WHERE {where} "
            f"WITH n SKIP $skip LIMIT $chunk "
            f"UNWIND keys(n) AS k "
            f"WITH k, n[k] AS val "
            f"RETURN k AS prop, collect(val) AS vals",
            skip=skip, chunk=chunk_size,
        ))
        for r in rows:
            prop = r["prop"]
            for v in r["vals"]:
                t = _infer_value_type(v)
                if t:
                    type_counts[prop][t] += 1
        skip += chunk_size
    return type_counts


def _full_scan_edge_property_types(
    session,
    rel_type: str,
    src_lk: Tuple[str, ...],
    tgt_lk: Tuple[str, ...],
    total: int,
    chunk_size: int,
) -> Dict[str, Counter]:
    """Full chunked scan of edge property values for canonical (rt, src, tgt)."""
    type_counts: Dict[str, Counter] = defaultdict(Counter)
    if total == 0:
        return type_counts

    esc = _escape_label(rel_type)
    sf = _build_label_match_exact("a", src_lk)
    tf = _build_label_match_exact("b", tgt_lk)

    skip = 0
    while skip < total:
        rows = list(session.run(
            f"MATCH (a)-[r:{esc}]->(b) "
            f"WHERE {sf} AND {tf} "
            f"WITH r SKIP $skip LIMIT $chunk "
            f"UNWIND keys(r) AS k "
            f"WITH k, r[k] AS val "
            f"RETURN k AS prop, collect(val) AS vals",
            skip=skip, chunk=chunk_size,
        ))
        for r in rows:
            prop = r["prop"]
            for v in r["vals"]:
                t = _infer_value_type(v)
                if t:
                    type_counts[prop][t] += 1
        skip += chunk_size
    return type_counts


# ============================================================
# Cardinality with min participation (0-cardinality support)
# ============================================================

def _compute_edge_cardinality_exact(
    session,
    rel_type: str,
    src_lk: Tuple[str, ...],
    tgt_lk: Tuple[str, ...],
) -> Tuple[int, int, int, int]:
    """
    Min/max in/out degree for canonical (rt, src_lk, tgt_lk).

    Uses EXACT label-set semantics: src-eligible = nodes with labels = src_lk;
    likewise for tgt.  OPTIONAL MATCH ensures non-participating nodes are
    counted with degree 0.

    Returns (min_out, max_out, min_in, max_in).
    """
    esc = _escape_label(rel_type)
    sf = _build_label_match_exact("a", src_lk)
    tf = _build_label_match_exact("b", tgt_lk)

    # source side: each src-eligible node `a`, degree = outgoing rel of type rt to tgt-eligible
    out_rec = session.run(
        f"MATCH (a) WHERE {sf} "
        f"OPTIONAL MATCH (a)-[r:{esc}]->(b) WHERE {tf} "
        f"WITH a, count(r) AS od "
        f"RETURN min(od) AS mn, max(od) AS mx, count(a) AS total"
    ).single()
    min_out = int(out_rec["mn"] or 0)
    max_out = int(out_rec["mx"] or 0)

    # target side
    in_rec = session.run(
        f"MATCH (b) WHERE {tf} "
        f"OPTIONAL MATCH (a)-[r:{esc}]->(b) WHERE {sf} "
        f"WITH b, count(r) AS id "
        f"RETURN min(id) AS mn, max(id) AS mx, count(b) AS total"
    ).single()
    min_in = int(in_rec["mn"] or 0)
    max_in = int(in_rec["mx"] or 0)

    return min_out, max_out, min_in, max_in


# ============================================================
# mine_patterns
# ============================================================

def mine_patterns(
    driver,
    *,
    chunk_size: int = 5_000,
    expand_edge_subsets: bool = True,
    database: Optional[str] = None,
) -> dict:
    """
    Mine node and edge patterns from a Neo4j database (full parse).

    Args:
        driver: Open ``neo4j.Driver``.
        chunk_size: Rows per chunk for full-scan property type inference.
        expand_edge_subsets: When True, every multi-label edge is expanded
            into all non-empty source × target label-subset combinations.
        database: Neo4j database name (None → default).

    Returns:
        Dict with keys ``"node_types"`` and ``"edge_types"``.

    Output node entry::

        {
          "labels": ["Person"],
          "count": 1234,
          "patterns": [{"property_keys": [...], "count": N}, ...],
          "properties": {
            "name": {"data_type": "STRING", "constraint": "MANDATORY", "fill_ratio": 1.0},
            ...
          }
        }

    Output edge entry::

        {
          "labels": ["KNOWS"],
          "source_labels": ["Person"],
          "target_labels": ["Person"],
          "is_canonical": true,
          "count": 567,
          "cardinality": "1..1 : 0..N",
          "min_out_degree": 1, "max_out_degree": 1,
          "min_in_degree":  0, "max_in_degree":  12,
          "patterns": [...],
          "properties": {...}
        }
    """
    session_kw: dict = {}
    if database:
        session_kw["database"] = database

    node_acc: Dict[Tuple, dict] = {}
    edge_acc: Dict[Tuple, dict] = {}    # canonical only, by (rt, src_lk, tgt_lk)

    with driver.session(**session_kw) as session:

        # ==============================================================
        # 1.  NODE PATTERNS (full scan, single Cypher query)
        # ==============================================================
        print(">>> Mining node patterns ...")
        for r in session.run(
            "MATCH (n) "
            "RETURN labels(n) AS labels, keys(n) AS props, count(*) AS cnt"
        ):
            lk = _label_key(r["labels"])
            pk = _props_key(r["props"])
            cnt = r["cnt"]

            acc = node_acc.setdefault(lk, {
                "count": 0,
                "patterns_map": Counter(),
                "all_props": set(),
            })
            acc["count"] += cnt
            acc["patterns_map"][pk] += cnt
            acc["all_props"].update(r["props"])

        # ---- node property types: FULL parse, chunked --------------
        print(f">>> Full-scan node property types "
              f"({len(node_acc)} label sets, chunk={chunk_size}) ...")
        node_types_out: List[dict] = []

        for lk, acc in sorted(node_acc.items()):
            total = acc["count"]
            type_counts = _full_scan_node_property_types(
                session, lk, total, chunk_size,
            )

            prop_fill: Counter = Counter()
            for pk, cnt in acc["patterns_map"].items():
                for prop in pk:
                    prop_fill[prop] += cnt

            properties: Dict[str, dict] = {}
            for prop in sorted(acc["all_props"]):
                fill = prop_fill.get(prop, 0)
                properties[prop] = {
                    "data_type": _resolve_property_type(type_counts.get(prop, Counter())),
                    "constraint": "MANDATORY" if fill == total else "OPTIONAL",
                    "fill_ratio": round(fill / total, 4) if total > 0 else 0.0,
                }

            patterns = [
                {"property_keys": list(pk), "count": cnt}
                for pk, cnt in sorted(acc["patterns_map"].items(), key=lambda x: -x[1])
            ]

            node_types_out.append({
                "labels": list(lk),
                "count": total,
                "patterns": patterns,
                "properties": properties,
            })
            print(f"   Node {str(list(lk)):40s}  count={total:>8,}  "
                  f"patterns={len(patterns)}  props={len(properties)}")

        # ==============================================================
        # 2.  EDGE PATTERNS  (full scan, single Cypher query)
        # ==============================================================
        print(">>> Mining edge patterns ...")
        for r in session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN type(r) AS rt, labels(a) AS sl, labels(b) AS tl, "
            "       keys(r) AS props, count(*) AS cnt"
        ):
            rt = r["rt"]
            sk = _label_key(r["sl"])
            tk = _label_key(r["tl"])
            ekey = (rt, sk, tk)
            pk = _props_key(r["props"])
            cnt = r["cnt"]

            acc = edge_acc.setdefault(ekey, {
                "count": 0,
                "patterns_map": Counter(),
                "all_props": set(),
            })
            acc["count"] += cnt
            acc["patterns_map"][pk] += cnt
            acc["all_props"].update(r["props"])

        # ---- compute canonical cardinality + property types --------
        print(f">>> Full-scan edge property types & cardinality "
              f"({len(edge_acc)} canonical edge types) ...")

        canonical_entries: List[dict] = []
        # remember canonical props/types so we can derive subset entries
        canonical_meta: Dict[Tuple, dict] = {}

        for ekey, acc in sorted(edge_acc.items()):
            rt, sk, tk = ekey
            total = acc["count"]

            type_counts = _full_scan_edge_property_types(
                session, rt, sk, tk, total, chunk_size,
            )

            prop_fill: Counter = Counter()
            for pk, cnt in acc["patterns_map"].items():
                for prop in pk:
                    prop_fill[prop] += cnt

            properties: Dict[str, dict] = {}
            for prop in sorted(acc["all_props"]):
                fill = prop_fill.get(prop, 0)
                properties[prop] = {
                    "data_type": _resolve_property_type(type_counts.get(prop, Counter())),
                    "constraint": "MANDATORY" if fill == total else "OPTIONAL",
                    "fill_ratio": round(fill / total, 4) if total > 0 else 0.0,
                }

            min_out, max_out, min_in, max_in = _compute_edge_cardinality_exact(
                session, rt, sk, tk,
            )
            cardinality = _format_cardinality(min_out, max_out, min_in, max_in)

            patterns = [
                {"property_keys": list(pk), "count": cnt}
                for pk, cnt in sorted(acc["patterns_map"].items(), key=lambda x: -x[1])
            ]

            entry = {
                "labels": [rt],
                "source_labels": list(sk),
                "target_labels": list(tk),
                "is_canonical": True,
                "count": total,
                "cardinality": cardinality,
                "min_out_degree": min_out,
                "max_out_degree": max_out,
                "min_in_degree": min_in,
                "max_in_degree": max_in,
                "patterns": patterns,
                "properties": properties,
            }
            canonical_entries.append(entry)
            canonical_meta[ekey] = {
                "count": total,
                "patterns_map": acc["patterns_map"],
                "all_props": set(acc["all_props"]),
                "type_counts": type_counts,
                "prop_fill": prop_fill,
                "properties": properties,
            }
            print(f"   Edge {list(sk)}-[:{rt}]->{list(tk)}  "
                  f"count={total:>8,}  card={cardinality}  "
                  f"patterns={len(patterns)}  props={len(properties)}")

    # closed session — subset expansion is purely Python, no DB calls
    edge_types_out: List[dict] = list(canonical_entries)

    if expand_edge_subsets:
        print(">>> Expanding edge label-subset permutations ...")
        # bucket = (rt, src_sub, tgt_sub) -> aggregated meta from contributing canonicals
        sub_acc: Dict[Tuple, dict] = {}

        for ekey, meta in canonical_meta.items():
            rt, sk, tk = ekey
            src_subs = _all_nonempty_subsets(sk)
            tgt_subs = _all_nonempty_subsets(tk)
            for ss in src_subs:
                for ts in tgt_subs:
                    if ss == sk and ts == tk:
                        continue   # canonical already emitted
                    sub_key = (rt, ss, ts)
                    bucket = sub_acc.setdefault(sub_key, {
                        "count": 0,
                        "patterns_map": Counter(),
                        "all_props": set(),
                        "type_counts": defaultdict(Counter),
                        "prop_fill": Counter(),
                    })
                    bucket["count"] += meta["count"]
                    for pk, cnt in meta["patterns_map"].items():
                        bucket["patterns_map"][pk] += cnt
                    bucket["all_props"].update(meta["all_props"])
                    for prop, ctr in meta["type_counts"].items():
                        for t, n in ctr.items():
                            bucket["type_counts"][prop][t] += n
                    for prop, n in meta["prop_fill"].items():
                        bucket["prop_fill"][prop] += n

        for (rt, ss, ts), bucket in sorted(sub_acc.items()):
            total = bucket["count"]
            properties: Dict[str, dict] = {}
            for prop in sorted(bucket["all_props"]):
                fill = bucket["prop_fill"].get(prop, 0)
                properties[prop] = {
                    "data_type": _resolve_property_type(
                        bucket["type_counts"].get(prop, Counter())
                    ),
                    "constraint": "MANDATORY" if fill == total else "OPTIONAL",
                    "fill_ratio": round(fill / total, 4) if total > 0 else 0.0,
                }
            patterns = [
                {"property_keys": list(pk), "count": cnt}
                for pk, cnt in sorted(bucket["patterns_map"].items(), key=lambda x: -x[1])
            ]
            edge_types_out.append({
                "labels": [rt],
                "source_labels": list(ss),
                "target_labels": list(ts),
                "is_canonical": False,
                "count": total,
                "cardinality": None,    # not meaningful for derived subsets
                "patterns": patterns,
                "properties": properties,
            })

    # sort emitted edges for stable output
    edge_types_out.sort(key=lambda e: (
        e["labels"][0], e["source_labels"], e["target_labels"], not e.get("is_canonical")
    ))

    n_np = sum(len(nt["patterns"]) for nt in node_types_out)
    n_ep = sum(len(et["patterns"]) for et in edge_types_out)
    n_canon = sum(1 for e in edge_types_out if e.get("is_canonical"))
    print(f"\n Pattern mining complete.")
    print(f"   Node types: {len(node_types_out)}  "
          f"({n_np} distinct node patterns)")
    print(f"   Edge types: {len(edge_types_out)}  "
          f"({n_canon} canonical, {len(edge_types_out) - n_canon} subset; "
          f"{n_ep} distinct edge patterns)")
    return {"node_types": node_types_out, "edge_types": edge_types_out}