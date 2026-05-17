"""
Schema comparison aligned with PG-HIVE (EDBT 2026) evaluation.

Supports both:
  * the new mined-pattern format produced by mine_patterns() and the LLM,
  * the legacy ground-truth format that uses
       - "type"        instead of "data_type"
       - "mandatory"   instead of "constraint"
       - "topology": [{"allowed_sources": [...], "allowed_targets": [...]}]
         instead of per-edge "source_labels"/"target_labels"
       - Neo4j Java types ("Long", "Double", "String", "Point", "StringArray")
         instead of generic ones (INTEGER / DOUBLE / STRING / LIST).

Metrics  (all instance-weighted, following PG-HIVE F1* definition)
-------
Node level:
  - Node Type   P* / R* / F1*  — instance-weighted, exact label-set match.
  - Node Pattern P* / R* / F1* — instance-weighted (L, K) match per Def. 3.5.
  - Node Property Constraint Accuracy  (MANDATORY / OPTIONAL).
  - Node Property Data-Type Accuracy.

Edge level:
  - Edge Type   P* / R* / F1*  — instance-weighted (rel, src, tgt) match.
  - Edge Pattern P* / R* / F1* — instance-weighted per Def. 3.6.
  - Edge Property Constraint Accuracy.
  - Edge Property Data-Type Accuracy.
  - Cardinality Accuracy.

Macro F1* = mean of the four instance-weighted pattern-level F1* scores.

Instance counts come from _mined_patterns in the inferred JSON (the actual
graph counts).  For GT types with no mined counterpart, count defaults to 0.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ============================================================
# Type aliases
# ============================================================

LabelKey = FrozenSet[str]
PropKey = FrozenSet[str]
NodePatternKey = Tuple[LabelKey, PropKey]
EdgeTypeKey = Tuple[str, LabelKey, LabelKey]
EdgePatternKey = Tuple[str, LabelKey, LabelKey, PropKey]


# ============================================================
# Property type normalisation  (Java → generic)
# ============================================================

_TYPE_ALIASES = {
    "LONG":        "INTEGER",
    "INT":         "INTEGER",
    "INTEGER":     "INTEGER",
    "DOUBLE":      "DOUBLE",
    "FLOAT":       "DOUBLE",
    "STRING":      "STRING",
    "BOOLEAN":     "BOOLEAN",
    "BOOL":        "BOOLEAN",
    "DATE":        "DATE",
    "DATETIME":    "DATE",
    "POINT":       "STRING",      # spatial → STRING for cross-format matching
    "STRINGARRAY": "LIST",
    "LIST":        "LIST",
    "ARRAY":       "LIST",
}


def _norm_type(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t2 = str(t).strip().upper()
    return _TYPE_ALIASES.get(t2, t2)


# ============================================================
# Property descriptor extraction (legacy + new)
# ============================================================

def _lk(labels) -> LabelKey:
    # Normalize to lowercase so "Person" and "person" collapse to the same key.
    return frozenset(str(l).lower() for l in (labels or []))


def _pk(props) -> PropKey:
    if isinstance(props, dict):
        return frozenset(k for k in props if k)
    return frozenset(
        p["name"] for p in (props or [])
        if isinstance(p, dict) and p.get("name")
    )


def _prop_info(props, key: str) -> dict:
    """
    Return a normalised property descriptor for *key*.

    Output keys: data_type (UPPER, generic), constraint (MANDATORY/OPTIONAL).
    Handles both list-of-dicts and dict-of-dicts, both legacy and new fields.
    """
    if isinstance(props, dict):
        info = props.get(key, {}) or {}
        if not isinstance(info, dict):
            return {}
        out = dict(info)
    else:
        out = {}
        for p in (props or []):
            if isinstance(p, dict) and p.get("name") == key:
                out = dict(p)
                break
        if not out:
            return {}

    # constraint
    c = out.get("constraint")
    if c:
        out["constraint"] = c.upper()
    elif "mandatory" in out:
        out["constraint"] = "MANDATORY" if out["mandatory"] else "OPTIONAL"

    # data type
    raw_t = out.get("data_type") or out.get("type")
    nt = _norm_type(raw_t)
    if nt:
        out["data_type"] = nt

    return out


# ============================================================
# Cardinality normalisation
# ============================================================

_CARD_NEW = re.compile(
    r"\s*([01])\.\.([1N])\s*:\s*([01])\.\.([1N])\s*"
)


def _normalize_cardinality(s: Optional[str]) -> Optional[str]:
    """Return canonical "X..Y:A..B" or None."""
    if not s:
        return None
    s2 = s.strip()
    m = _CARD_NEW.fullmatch(s2)
    if m:
        return f"{m.group(1)}..{m.group(2)}:{m.group(3)}..{m.group(4)}"
    m = re.fullmatch(r"\s*([1MN])\s*:\s*([1MN])\s*", s2, re.IGNORECASE)
    if m:
        side = lambda c: "1..1" if c == "1" else "1..N"
        return f"{side(m.group(1).upper())}:{side(m.group(2).upper())}"
    return None


# ============================================================
# Schema normalisation
# ============================================================

@dataclass
class NormNodeType:
    label_key: LabelKey
    prop_key: PropKey
    props_raw: object
    patterns: List[PropKey] = field(default_factory=list)


@dataclass
class NormEdgeType:
    edge_key: EdgeTypeKey
    prop_key: PropKey
    props_raw: object
    cardinality: Optional[str]
    is_canonical: bool = True
    patterns: List[PropKey] = field(default_factory=list)


def _node_label_key_for_name(
    name: str,
    node_map: Dict[LabelKey, NormNodeType],
) -> LabelKey:
    """
    Resolve a node-type *name* (e.g. "Neuron") to its label key.

    The legacy GT topology refers to nodes by their declared "name", which
    is one of the labels.  We find the label-key whose label set contains
    this name.
    """
    for lk in node_map:
        if name in lk:
            return lk
    return frozenset([name]) if name else frozenset()


def _normalise_schema(schema: dict) -> Tuple[
    Dict[LabelKey, NormNodeType],
    Dict[EdgeTypeKey, NormEdgeType],
]:
    node_map: Dict[LabelKey, NormNodeType] = {}
    edge_map: Dict[EdgeTypeKey, NormEdgeType] = {}

    # ---- nodes ----
    for nt in schema.get("node_types", []):
        labels = nt.get("labels") or ([nt["name"]] if nt.get("name") else [])
        lk = _lk(labels)
        props_raw = nt.get("properties", {})
        pk = _pk(props_raw)

        sub_patterns: List[PropKey] = []
        for pat in nt.get("patterns", []):
            if isinstance(pat, dict):
                sub_patterns.append(frozenset(pat.get("property_keys", [])))
        if not sub_patterns:
            sub_patterns = [pk]

        node_map[lk] = NormNodeType(
            label_key=lk, prop_key=pk,
            props_raw=props_raw, patterns=sub_patterns,
        )

    # ---- edges ----
    # NEW format: edge_types is a list of {name, source_labels, target_labels, ...}
    # LEGACY format: edge_types is a list of
    #   {name, properties, topology:[{allowed_sources, allowed_targets}]}
    for et in schema.get("edge_types", []):
        rel = et.get("name") or (et.get("labels") or [""])[0]
        if not rel:
            continue                          # never accept unlabeled edges

        props_raw = et.get("properties", {})
        pk = _pk(props_raw)

        sub_patterns: List[PropKey] = []
        for pat in et.get("patterns", []):
            if isinstance(pat, dict):
                sub_patterns.append(frozenset(pat.get("property_keys", [])))
        if not sub_patterns:
            sub_patterns = [pk]

        cardinality = _normalize_cardinality(et.get("cardinality"))
        is_canonical = bool(et.get("is_canonical", True))

        # ---- legacy topology block: expand cross-product into edge entries
        topology = et.get("topology")
        if topology:
            for topo in topology:
                src_names = topo.get("allowed_sources", []) or []
                tgt_names = topo.get("allowed_targets", []) or []
                for s_name, t_name in product(src_names, tgt_names):
                    src_lk = _node_label_key_for_name(s_name, node_map)
                    tgt_lk = _node_label_key_for_name(t_name, node_map)
                    ek: EdgeTypeKey = (rel, src_lk, tgt_lk)
                    edge_map[ek] = NormEdgeType(
                        edge_key=ek,
                        prop_key=pk,
                        props_raw=props_raw,
                        cardinality=cardinality,
                        is_canonical=is_canonical,
                        patterns=sub_patterns,
                    )
            continue

        # ---- new format
        if et.get("source_labels") is not None or et.get("target_labels") is not None:
            src_lk = _lk(et.get("source_labels", []))
            tgt_lk = _lk(et.get("target_labels", []))
        else:
            # fallback: legacy start_node/end_node strings
            src_name = et.get("source") or et.get("start_node") or ""
            tgt_name = et.get("target") or et.get("end_node") or ""
            src_lk = _node_label_key_for_name(src_name, node_map)
            tgt_lk = _node_label_key_for_name(tgt_name, node_map)

        ek = (rel, src_lk, tgt_lk)
        edge_map[ek] = NormEdgeType(
            edge_key=ek,
            prop_key=pk,
            props_raw=props_raw,
            cardinality=cardinality,
            is_canonical=is_canonical,
            patterns=sub_patterns,
        )

    return node_map, edge_map


# ============================================================
# Pattern-set helpers
# ============================================================

def _node_pattern_keys(m: Dict[LabelKey, NormNodeType]) -> Set[NodePatternKey]:
    out: Set[NodePatternKey] = set()
    for lk, nt in m.items():
        for pk in nt.patterns:
            out.add((lk, pk))
    return out


def _edge_pattern_keys(m: Dict[EdgeTypeKey, NormEdgeType]) -> Set[EdgePatternKey]:
    out: Set[EdgePatternKey] = set()
    for (rel, src, tgt), et in m.items():
        for pk in et.patterns:
            out.add((rel, src, tgt, pk))
    return out


def _prf(tp: int, n_pred: int, n_gt: int) -> Tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ============================================================
# Configuration / Result
# ============================================================

@dataclass
class CompareConfig:
    verbose: bool = True
    list_limit: int = 20
    canonical_only: bool = True


@dataclass
class CompareResult:
    # Instance-weighted (F1* per PG-HIVE)
    node_type_precision: float
    node_type_recall: float
    node_type_f1: float
    node_pattern_precision: float
    node_pattern_recall: float
    node_pattern_f1: float
    node_constraint_accuracy: float
    node_data_type_accuracy: float

    edge_type_precision: float
    edge_type_recall: float
    edge_type_f1: float
    edge_pattern_precision: float
    edge_pattern_recall: float
    edge_pattern_f1: float
    edge_constraint_accuracy: float
    edge_data_type_accuracy: float
    cardinality_accuracy: float

    macro_f1: float   # mean of the 4 instance-weighted pattern F1* scores


# ============================================================
# Print helpers
# ============================================================

def _h1(p, t):
    p("\n" + "=" * 68); p(f"  {t}"); p("=" * 68)


def _h2(p, t):
    p("\n" + "-" * 68); p(f"  {t}"); p("-" * 68)


def _kv(p, k, v):
    p(f"  {k:<42s} {v}")


def _block(p, title, rows: List[str], limit: int):
    p(f"\n  [{title}]")
    if not rows:
        p("    (none)")
        return
    for r in rows[:limit]:
        p(f"    • {r}")
    if len(rows) > limit:
        p(f"    … +{len(rows) - limit} more")


def _pct(f: float) -> str:
    return f"{f * 100:.2f}%"


def _fmt_lk(lk: LabelKey) -> str:
    return "{" + ", ".join(sorted(lk)) + "}" if lk else "{}"


def _fmt_npk(npk: NodePatternKey) -> str:
    lk, pk = npk
    return f"{_fmt_lk(lk)}  K={{{', '.join(sorted(pk))}}}"


def _fmt_ek(ek: EdgeTypeKey) -> str:
    rel, src, tgt = ek
    return f"({_fmt_lk(src)})-[:{rel}]->({_fmt_lk(tgt)})"


def _fmt_epk(epk: EdgePatternKey) -> str:
    rel, src, tgt, pk = epk
    return (
        f"({_fmt_lk(src)})-[:{rel}]->({_fmt_lk(tgt)})"
        f"  K={{{', '.join(sorted(pk))}}}"
    )


# ============================================================
# Main comparison
# ============================================================

def run_compare(
    gt_file: str,
    inferred_file: str,
    config: Optional[CompareConfig] = None,
) -> Optional[CompareResult]:
    """
    Compare an inferred schema against a ground-truth schema at the
    pattern level per PG-HIVE Definitions 3.5 and 3.6.

    Both legacy GT format (topology blocks, "Long"/"Double" types,
    "mandatory" booleans) and new mined-pattern format are supported.
    """
    cfg = config or CompareConfig()
    p = print if cfg.verbose else (lambda *a, **k: None)

    _h1(p, "PG-HIVE PATTERN-LEVEL SCHEMA COMPARISON")

    for path in (gt_file, inferred_file):
        if not os.path.exists(path):
            p(f"  [ERROR] File not found: {path}")
            return None

    with open(gt_file, encoding="utf-8-sig") as f:
        gt_raw = json.load(f)
    with open(inferred_file, encoding="utf-8-sig") as f:
        inf_raw = json.load(f)

    if "node_types" not in gt_raw and "_mined_patterns" in gt_raw:
        gt_raw = gt_raw["_mined_patterns"]

    # Build a lookup of ground-truth cardinalities from the mined patterns
    # attached to the inferred file.  When a GT edge has cardinality=null
    # (common for .pgs-derived GTs), we fall back to the mined value, which
    # is computed directly from the graph and is the true reference.
    mined_card: Dict[EdgeTypeKey, str] = {}
    for et in inf_raw.get("_mined_patterns", {}).get("edge_types", []):
        if et.get("is_canonical") and et.get("cardinality"):
            rel = (et.get("labels") or [""])[0]
            if not rel:
                continue
            src_lk = _lk(et.get("source_labels", []))
            tgt_lk = _lk(et.get("target_labels", []))
            nc = _normalize_cardinality(et["cardinality"])
            if nc:
                mined_card[(rel, src_lk, tgt_lk)] = nc

    _kv(p, "Ground truth:", os.path.basename(gt_file))
    _kv(p, "Inferred:",      os.path.basename(inferred_file))

    gt_nodes,  gt_edges  = _normalise_schema(gt_raw)
    inf_nodes, inf_edges = _normalise_schema(inf_raw)

    if cfg.canonical_only:
        gt_edges  = {k: v for k, v in gt_edges.items()  if v.is_canonical}
        inf_edges = {k: v for k, v in inf_edges.items() if v.is_canonical}
        _kv(p, "Mode:", "canonical edges only")

    # ------------------------------------------------------------------
    # Instance counts from _mined_patterns (used for F1* weighting).
    # The mined patterns reflect the actual graph counts, so they serve as
    # ground truth for both GT and inferred types.
    # ------------------------------------------------------------------
    mined_node_cnt:    Dict[LabelKey, int]        = {}
    mined_node_pat_cnt: Dict[NodePatternKey, int] = {}
    mined_edge_cnt:    Dict[EdgeTypeKey, int]      = {}
    mined_edge_pat_cnt: Dict[EdgePatternKey, int] = {}

    for nt in inf_raw.get("_mined_patterns", {}).get("node_types", []):
        lk = _lk(nt.get("labels", []))
        mined_node_cnt[lk] = nt.get("count", 0)
        for pat in nt.get("patterns", []):
            pk: PropKey = frozenset(pat.get("property_keys", []))
            key = (lk, pk)
            mined_node_pat_cnt[key] = mined_node_pat_cnt.get(key, 0) + pat.get("count", 0)

    for et in inf_raw.get("_mined_patterns", {}).get("edge_types", []):
        if not et.get("is_canonical", True):
            continue
        rel = (et.get("labels") or [""])[0]
        if not rel:
            continue
        src_lk = _lk(et.get("source_labels", []))
        tgt_lk = _lk(et.get("target_labels", []))
        ek_key: EdgeTypeKey = (rel, src_lk, tgt_lk)
        mined_edge_cnt[ek_key] = et.get("count", 0)
        for pat in et.get("patterns", []):
            pk = frozenset(pat.get("property_keys", []))
            epk: EdgePatternKey = (rel, src_lk, tgt_lk, pk)
            mined_edge_pat_cnt[epk] = mined_edge_pat_cnt.get(epk, 0) + pat.get("count", 0)

    def _node_inst(lk: LabelKey) -> int:
        return mined_node_cnt.get(lk, 1)      # fallback 1 avoids zero-weight orphans

    def _node_pat_inst(key: NodePatternKey) -> int:
        return mined_node_pat_cnt.get(key, 1)

    def _edge_inst(ek: EdgeTypeKey) -> int:
        return mined_edge_cnt.get(ek, 1)

    def _edge_pat_inst(key: EdgePatternKey) -> int:
        return mined_edge_pat_cnt.get(key, 1)

    def _prf_inst(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        return pr, rc, f1

    # ====================================================
    # 1. NODE TYPES  (instance-weighted F1*)
    # ====================================================
    _h2(p, "1.  NODE TYPES  (instance-weighted F1*, label-set match)")
    gt_node_lks  = set(gt_nodes)
    inf_node_lks = set(inf_nodes)
    matched_nodes = gt_node_lks & inf_node_lks
    only_gt  = gt_node_lks  - inf_node_lks
    only_inf = inf_node_lks - gt_node_lks

    nt_tp = sum(_node_inst(lk) for lk in matched_nodes)
    nt_fp = sum(_node_inst(lk) for lk in only_inf)
    nt_fn = sum(_node_inst(lk) for lk in only_gt)
    nt_p, nt_r, nt_f1 = _prf_inst(nt_tp, nt_fp, nt_fn)

    _kv(p, "GT node types:",             len(gt_node_lks))
    _kv(p, "Inferred node types:",        len(inf_node_lks))
    _kv(p, "Matched types:",              len(matched_nodes))
    _kv(p, "TP instances:",               nt_tp)
    _kv(p, "FP instances (only inferred):", nt_fp)
    _kv(p, "FN instances (only GT):",     nt_fn)
    _kv(p, "Precision*:", _pct(nt_p))
    _kv(p, "Recall*:",    _pct(nt_r))
    _kv(p, "F1*:",        _pct(nt_f1))

    _block(p, "MATCHED",     [_fmt_lk(lk) for lk in sorted(matched_nodes)], cfg.list_limit)
    _block(p, "ONLY IN GT",  [_fmt_lk(lk) for lk in sorted(only_gt)],       cfg.list_limit)
    _block(p, "ONLY IN INF", [_fmt_lk(lk) for lk in sorted(only_inf)],      cfg.list_limit)

    # ====================================================
    # 2. NODE PATTERNS  (instance-weighted F1*)
    # ====================================================
    _h2(p, "2.  NODE PATTERNS  (instance-weighted F1*, (L,K) per Def. 3.5)")

    gt_np  = _node_pattern_keys(gt_nodes)
    inf_np = _node_pattern_keys(inf_nodes)

    def _np_covered(gp: NodePatternKey) -> bool:
        lk, pk = gp
        n = inf_nodes.get(lk)
        return n is not None and pk <= n.prop_key

    def _np_valid(ip: NodePatternKey) -> bool:
        lk, pk = ip
        g = gt_nodes.get(lk)
        return g is not None and pk <= g.prop_key

    np_tp = sum(_node_pat_inst(g) for g in gt_np if _np_covered(g))
    np_fn = sum(_node_pat_inst(g) for g in gt_np if not _np_covered(g))
    np_fp = sum(_node_pat_inst(i) for i in inf_np if not _np_valid(i))
    np_p, np_r, np_f1 = _prf_inst(np_tp, np_fp, np_fn)

    _kv(p, "GT node patterns:",       len(gt_np))
    _kv(p, "Inferred node patterns:", len(inf_np))
    _kv(p, "GT covered (instances):", np_tp)
    _kv(p, "GT missed  (instances):", np_fn)
    _kv(p, "Inf invalid (instances):", np_fp)
    _kv(p, "Precision*:", _pct(np_p))
    _kv(p, "Recall*:",    _pct(np_r))
    _kv(p, "F1*:",        _pct(np_f1))

    missed = [g for g in gt_np if not _np_covered(g)]
    _block(p, "MISSED GT NODE PATTERNS", [_fmt_npk(g) for g in sorted(missed)], cfg.list_limit)

    # ====================================================
    # 3. NODE PROPERTY ACCURACY
    # ====================================================
    _h2(p, "3.  NODE PROPERTY ACCURACY  (constraint + data_type)")

    nc_correct = nc_total = 0
    nd_correct = nd_total = 0
    rows: List[str] = []
    for lk in matched_nodes:
        gt_nt = gt_nodes[lk]; inf_nt = inf_nodes[lk]
        shared = gt_nt.prop_key & inf_nt.prop_key
        c_ok = c_tot = d_ok = d_tot = 0
        for key in shared:
            gi = _prop_info(gt_nt.props_raw, key)
            ii = _prop_info(inf_nt.props_raw, key)
            if gi.get("constraint") and ii.get("constraint"):
                c_tot += 1
                if gi["constraint"] == ii["constraint"]:
                    c_ok += 1
            if gi.get("data_type") and ii.get("data_type"):
                d_tot += 1
                if gi["data_type"] == ii["data_type"]:
                    d_ok += 1
        nc_correct += c_ok; nc_total += c_tot
        nd_correct += d_ok; nd_total += d_tot
        if c_tot or d_tot:
            rows.append(f"{_fmt_lk(lk)}  constraint={c_ok}/{c_tot}  type={d_ok}/{d_tot}")

    nc_acc = nc_correct / nc_total if nc_total else 0.0
    nd_acc = nd_correct / nd_total if nd_total else 0.0
    _kv(p, "Properties (constraint):", f"{nc_correct} / {nc_total}")
    _kv(p, "Constraint accuracy:",     _pct(nc_acc))
    _kv(p, "Properties (data_type):",  f"{nd_correct} / {nd_total}")
    _kv(p, "Data type accuracy:",      _pct(nd_acc))
    _block(p, "PER NODE TYPE", rows, cfg.list_limit)

    # ====================================================
    # 4. EDGE TYPES  (instance-weighted F1*)
    # ====================================================
    _h2(p, "4.  EDGE TYPES  (instance-weighted F1*, rel+src+tgt match)")
    gt_eks  = set(gt_edges)
    inf_eks = set(inf_edges)
    matched_eks = gt_eks & inf_eks
    only_gt_e   = gt_eks  - inf_eks
    only_inf_e  = inf_eks - gt_eks

    et_tp = sum(_edge_inst(ek) for ek in matched_eks)
    et_fp = sum(_edge_inst(ek) for ek in only_inf_e)
    et_fn = sum(_edge_inst(ek) for ek in only_gt_e)
    et_p, et_r, et_f1 = _prf_inst(et_tp, et_fp, et_fn)

    _kv(p, "GT edge types:",               len(gt_eks))
    _kv(p, "Inferred edge types:",          len(inf_eks))
    _kv(p, "Matched types:",                len(matched_eks))
    _kv(p, "TP instances:",                 et_tp)
    _kv(p, "FP instances (only inferred):", et_fp)
    _kv(p, "FN instances (only GT):",       et_fn)
    _kv(p, "Precision*:", _pct(et_p))
    _kv(p, "Recall*:",    _pct(et_r))
    _kv(p, "F1*:",        _pct(et_f1))

    _block(p, "MATCHED",     [_fmt_ek(ek) for ek in sorted(matched_eks)], cfg.list_limit)
    _block(p, "ONLY IN GT",  [_fmt_ek(ek) for ek in sorted(only_gt_e)],   cfg.list_limit)
    _block(p, "ONLY IN INF", [_fmt_ek(ek) for ek in sorted(only_inf_e)],  cfg.list_limit)

    # ====================================================
    # 5. EDGE PATTERNS  (instance-weighted F1*)
    # ====================================================
    _h2(p, "5.  EDGE PATTERNS  (instance-weighted F1*, (L,K,R) per Def. 3.6)")
    gt_ep  = _edge_pattern_keys(gt_edges)
    inf_ep = _edge_pattern_keys(inf_edges)

    def _ep_covered(gp: EdgePatternKey) -> bool:
        rel, src, tgt, pk = gp
        e = inf_edges.get((rel, src, tgt))
        return e is not None and pk <= e.prop_key

    def _ep_valid(ip: EdgePatternKey) -> bool:
        rel, src, tgt, pk = ip
        g = gt_edges.get((rel, src, tgt))
        return g is not None and pk <= g.prop_key

    ep_tp = sum(_edge_pat_inst(g) for g in gt_ep if _ep_covered(g))
    ep_fn = sum(_edge_pat_inst(g) for g in gt_ep if not _ep_covered(g))
    ep_fp = sum(_edge_pat_inst(i) for i in inf_ep if not _ep_valid(i))
    ep_p, ep_r, ep_f1 = _prf_inst(ep_tp, ep_fp, ep_fn)

    _kv(p, "GT edge patterns:",            len(gt_ep))
    _kv(p, "Inferred edge patterns:",       len(inf_ep))
    _kv(p, "GT covered (instances):",       ep_tp)
    _kv(p, "GT missed  (instances):",       ep_fn)
    _kv(p, "Inf invalid (instances):",      ep_fp)
    _kv(p, "Precision*:", _pct(ep_p))
    _kv(p, "Recall*:",    _pct(ep_r))
    _kv(p, "F1*:",        _pct(ep_f1))

    missed_ep = [g for g in gt_ep if not _ep_covered(g)]
    _block(p, "MISSED GT EDGE PATTERNS",
           [_fmt_epk(g) for g in sorted(missed_ep)], cfg.list_limit)

    # ====================================================
    # 6. EDGE PROPERTY ACCURACY
    # ====================================================
    _h2(p, "6.  EDGE PROPERTY ACCURACY  (constraint + data_type)")
    ec_correct = ec_total = 0
    ed_correct = ed_total = 0
    rows = []
    for ek in matched_eks:
        gt_et = gt_edges[ek]; inf_et = inf_edges[ek]
        shared = gt_et.prop_key & inf_et.prop_key
        c_ok = c_tot = d_ok = d_tot = 0
        for key in shared:
            gi = _prop_info(gt_et.props_raw, key)
            ii = _prop_info(inf_et.props_raw, key)
            if gi.get("constraint") and ii.get("constraint"):
                c_tot += 1
                if gi["constraint"] == ii["constraint"]:
                    c_ok += 1
            if gi.get("data_type") and ii.get("data_type"):
                d_tot += 1
                if gi["data_type"] == ii["data_type"]:
                    d_ok += 1
        ec_correct += c_ok; ec_total += c_tot
        ed_correct += d_ok; ed_total += d_tot
        if c_tot or d_tot:
            rows.append(f"{_fmt_ek(ek)}  constraint={c_ok}/{c_tot}  type={d_ok}/{d_tot}")

    ec_acc = ec_correct / ec_total if ec_total else 0.0
    ed_acc = ed_correct / ed_total if ed_total else 0.0
    _kv(p, "Properties (constraint):", f"{ec_correct} / {ec_total}")
    _kv(p, "Constraint accuracy:",     _pct(ec_acc))
    _kv(p, "Properties (data_type):",  f"{ed_correct} / {ed_total}")
    _kv(p, "Data type accuracy:",      _pct(ed_acc))
    _block(p, "PER EDGE TYPE", rows, cfg.list_limit)

    # ====================================================
    # 7. CARDINALITY
    # ====================================================
    _h2(p, "7.  CARDINALITY ACCURACY")
    _kv(p, "Reference:", "GT cardinality when explicit; mined (graph-derived) otherwise")
    card_ok = card_tot = 0
    rows = []
    for ek in matched_eks:
        gt_c  = gt_edges[ek].cardinality          # None when GT has no cardinality
        inf_c = inf_edges[ek].cardinality
        # Prefer explicit GT; fall back to mined (the true graph cardinality)
        ref_c = gt_c if gt_c is not None else mined_card.get(ek)
        if ref_c:
            card_tot += 1
            if ref_c == inf_c:
                card_ok += 1
                rows.append(f"OK  {_fmt_ek(ek)}  {ref_c}")
            else:
                src = "GT" if gt_c is not None else "mined"
                rows.append(f"X   {_fmt_ek(ek)}  {src}={ref_c}  INF={inf_c}")
    card_acc = card_ok / card_tot if card_tot else 0.0
    _kv(p, "Edge types with reference cardinality:", card_tot)
    _kv(p, "Correct:",                               card_ok)
    _kv(p, "Cardinality accuracy:",                  _pct(card_acc) if card_tot else "N/A")
    _block(p, "PER EDGE TYPE", rows, cfg.list_limit)

    # ====================================================
    # FINAL SCORES
    # ====================================================
    _h1(p, "FINAL SCORES  (PG-HIVE F1* — instance-weighted)")
    macro = (nt_f1 + np_f1 + et_f1 + ep_f1) / 4

    _kv(p, "Node Type F1*:",         _pct(nt_f1))
    _kv(p, "Node Pattern F1*:",      _pct(np_f1))
    _kv(p, "Edge Type F1*:",         _pct(et_f1))
    _kv(p, "Edge Pattern F1*:",      _pct(ep_f1))
    _kv(p, "Node Constraint Acc:",   _pct(nc_acc))
    _kv(p, "Node Data-Type Acc:",    _pct(nd_acc))
    _kv(p, "Edge Constraint Acc:",   _pct(ec_acc))
    _kv(p, "Edge Data-Type Acc:",    _pct(ed_acc))
    _kv(p, "Cardinality Acc:",       _pct(card_acc) if card_tot else "N/A")
    p("-" * 68)
    _kv(p, "MACRO F1 (4 pattern-level F1s):", _pct(macro))
    p("=" * 68)

    return CompareResult(
        node_type_precision=nt_p, node_type_recall=nt_r, node_type_f1=nt_f1,
        node_pattern_precision=np_p, node_pattern_recall=np_r, node_pattern_f1=np_f1,
        node_constraint_accuracy=nc_acc, node_data_type_accuracy=nd_acc,
        edge_type_precision=et_p, edge_type_recall=et_r, edge_type_f1=et_f1,
        edge_pattern_precision=ep_p, edge_pattern_recall=ep_r, edge_pattern_f1=ep_f1,
        edge_constraint_accuracy=ec_acc, edge_data_type_accuracy=ed_acc,
        cardinality_accuracy=card_acc,
        macro_f1=macro,
    )