from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ============================================================
# Config
# ============================================================

RESERVED_PROP_PREFIXES = (":",)  # anything starting with ":" like :ID, :LABEL
RESERVED_PROP_NAMES = {
    "id", "label", "labels", "type",
}

# Semantic matching model (optional)
DEFAULT_SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# Helpers: JSON load
# ============================================================

def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}") from e


# ============================================================
# Helpers: property normalization + filtering
# ============================================================

def normalize_prop_name(name: str) -> str:
    if not name:
        return ""
    n = name.strip().lower()
    n = n.replace("`", "")
    return n

def is_reserved_prop(name: str) -> bool:
    if not name:
        return True
    raw = name.strip()

    if raw.startswith(RESERVED_PROP_PREFIXES):
        return True

    n = normalize_prop_name(raw)

    if n in RESERVED_PROP_NAMES:
        return True

    # patterns like "id(...)" if they ever show up
    if re.match(r"^id\s*\(.*\)$", n):
        return True
    if re.match(r"^label\s*\(.*\)$", n):
        return True

    return False

def prop_set(props: Sequence[dict]) -> Set[str]:
    out: Set[str] = set()
    for p in props or []:
        nm = p.get("name", "")
        if not nm or is_reserved_prop(nm):
            continue
        out.add(normalize_prop_name(nm))
    return out

def compare_properties(gt_props: Sequence[dict], inf_props: Sequence[dict], verbose: bool = False) -> Tuple[int, int, int]:
    gt_set = prop_set(gt_props)
    inf_set = prop_set(inf_props)

    matches = len(gt_set & inf_set)
    total = len(gt_set)
    extra = len(inf_set - gt_set)

    if verbose and total > 0 and matches == 0:
        print("   [PROP DEBUG] Mismatch!")
        print(f"     GT Properties: {list(sorted(gt_set))[:8]}")
        print(f"     Inf Properties: {list(sorted(inf_set))[:8]}")

    return matches, total, extra


# ============================================================
# Helpers: string normalization + similarity
# ============================================================

def _norm_label(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ").replace(".", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def similar_string(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _norm_label(a), _norm_label(b)).ratio()

def find_best_string_match(target: str, candidates: Sequence[str], threshold: float) -> Optional[str]:
    best = None
    best_score = 0.0
    for c in candidates:
        s = similar_string(target, c)
        if s > best_score:
            best_score = s
            best = c
    return best if best and best_score >= threshold else None


# ============================================================
# Semantic matcher (lazy, cached)
# ============================================================

class SemanticEdgeMatcher:
    """
    Lazy-load sentence transformer and cache embeddings.
    Only used if enabled in CompareConfig.
    """
    def __init__(self, model_name: str = DEFAULT_SEMANTIC_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._util = None
        self._cache: Dict[str, Any] = {}

    def _load(self):
        if self._model is None:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            from sentence_transformers import SentenceTransformer, util  # type: ignore
            self._util = util
            self._model = SentenceTransformer(self.model_name)

    @staticmethod
    def _clean(label: str) -> str:
        return (label or "").replace("_", " ").replace(".", " ").lower().strip()

    def _embed(self, text: str):
        if text not in self._cache:
            self._cache[text] = self._model.encode(text, convert_to_tensor=True)
        return self._cache[text]

    def find_match(
        self,
        target: str,
        candidates: Sequence[str],
        threshold: float = 0.75,
        margin: float = 0.05,
        verbose: bool = False,
    ) -> Optional[str]:
        if not candidates:
            return None
        self._load()

        import torch  # type: ignore

        target_clean = self._clean(target)
        cand_clean = [self._clean(c) for c in candidates]

        target_emb = self._embed(target_clean)
        cand_embs = torch.stack([self._embed(c) for c in cand_clean], dim=0)

        scores = self._util.cos_sim(target_emb, cand_embs)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        best_match = candidates[best_idx]

        topk = scores.topk(k=min(2, scores.numel())).values
        second = float(topk[1]) if topk.numel() > 1 else -1.0

        if best_score >= threshold and (best_score - second) >= margin:
            if verbose:
                print(f"   [Edge Semantic Match] ACCEPTED: '{target}' -> '{best_match}' (Score: {best_score:.3f}, 2nd: {second:.3f})")
            return best_match

        if verbose:
            print(f"   [Edge Semantic Match] REJECTED: '{target}' best='{best_match}' (Score: {best_score:.3f}, 2nd: {second:.3f})")
        return None


# ============================================================
# Compare configuration + result
# ============================================================

@dataclass
class CompareConfig:
    verbose: bool = True

    # node matching
    node_string_threshold: float = 0.80

    # edge label mapping
    edge_string_threshold: float = 0.78
    use_semantic_edge_match: bool = True
    semantic_threshold: float = 0.75
    semantic_margin: float = 0.05

    # reporting
    list_limit: int = 15


@dataclass
class CompareResult:
    real_node_accuracy: float
    real_edge_accuracy: float
    real_node_property_accuracy: float
    real_edge_property_accuracy: float
    topology_score_f1: float
    overall_performance: float

    node_matches: Dict[str, str]
    extra_nodes: List[str]
    extra_edges: int


# ============================================================
# Scoring helpers
# ============================================================

def calculate_real_score(matches: int, total_gt: int, total_extra: int) -> float:
    """
    Real score over (GT + Extra).

    Important: when there is nothing to score (GT=0 and Extra=0), the metric is
    mathematically undefined (0/0). For reporting, treat it as 100% (perfect),
    because there were no required properties and no hallucinated extras.
    """
    denom = total_gt + total_extra
    if denom == 0:
        return 100.0
    return (matches / denom) * 100.0



# ============================================================
# Reporting helpers (same style, cleaner)
# ============================================================

def _make_printer(verbose: bool):
    if not verbose:
        def _p(*a, **k):  # noqa
            return
        return _p
    return print

def _header(p, title: str):
    p("\n" + "=" * 60)
    p(f"[ {title} ]")
    p("=" * 60)

def _section(p, title: str):
    p("\n" + "-" * 60)
    p(f"[ {title} ]")
    p("-" * 60)

def _kv(p, key: str, value: Any):
    p(f"  [ {key} ] {value}")

def _item(p, tag: str, text: str):
    p(f"  [ {tag} ] {text}")

def _list_block(p, title: str, tag: str, rows: Sequence[str], limit: int, suffix_note: Optional[str] = None):
    p(f"\n[ {title} ]")
    if suffix_note:
        p(f"  [ NOTE ] {suffix_note}")
    if not rows:
        _item(p, "INFO", "None")
        return
    for r in rows[:limit]:
        _item(p, tag, r)
    if len(rows) > limit:
        _item(p, "INFO", f"... +{len(rows)-limit} more")


# ============================================================
# Core compare
# ============================================================

def run_compare(gt_file: str, inferred_file: str, config: Optional[CompareConfig] = None) -> Optional[CompareResult]:
    cfg = config or CompareConfig()
    p = _make_printer(cfg.verbose)

    _header(p, "SCHEMA COMPARISON REPORT")
    _section(p, "INPUT")

    if not os.path.exists(gt_file) or not os.path.exists(inferred_file):
        _item(p, "ERROR", "One or both input files do not exist.")
        _kv(p, "GT", gt_file)
        _kv(p, "INFERRED", inferred_file)
        return None

    gt = load_json(gt_file)
    inf = load_json(inferred_file)

    _kv(p, "GT", os.path.basename(gt_file))
    _kv(p, "INFERRED", os.path.basename(inferred_file))

    gt_nodes = gt.get("node_types", [])
    inf_nodes = inf.get("node_types", [])

    gt_node_names = [n.get("name") or (n.get("labels") or [""])[0] for n in gt_nodes]
    inf_node_names = [n.get("name") or (n.get("labels") or [""])[0] for n in inf_nodes]

    # sanity: inferred edges referencing unknown node-types
    inf_node_set = set(inf_node_names)
    bad_edges = []
    for e in inf.get("edge_types", []):
        s, t = e.get("start_node"), e.get("end_node")
        if s and t and (s not in inf_node_set or t not in inf_node_set):
            bad_edges.append(f"{e.get('name')}: {s} -> {t}")

    _section(p, "NODE MATCHING")
    _kv(p, "GT Nodes", len(gt_node_names))
    _kv(p, "Inferred Nodes", len(inf_node_names))

    if cfg.verbose:
        _list_block(p, "RAW GT NODES", "NODE", gt_node_names, limit=min(15, cfg.list_limit))
        _list_block(p, "RAW INFERRED NODES", "NODE", inf_node_names, limit=min(15, cfg.list_limit))

    if bad_edges:
        _list_block(
            p,
            "INFERRED EDGES REFERENCING UNKNOWN NODES",
            "EDGE",
            bad_edges,
            limit=10,
            suffix_note="These can invalidate topology checks."
        )

    # Node matching (string)
    node_matches_map: Dict[str, str] = {}
    used_inf = set()
    for gt_name in gt_node_names:
        candidates = [n for n in inf_node_names if n not in used_inf]
        match = find_best_string_match(gt_name, candidates, threshold=cfg.node_string_threshold)
        if match:
            node_matches_map[gt_name] = match
            used_inf.add(match)

    node_matches = len(node_matches_map)
    extra_nodes = [n for n in inf_node_names if n not in used_inf]

    _section(p, "NODE TYPE MATCHING RESULTS")
    _kv(p, "Matched GT Nodes", f"{node_matches} / {len(gt_node_names)}")
    _kv(p, "Extra Inferred Nodes", len(extra_nodes))

    if node_matches_map:
        rows = [f"{gt_n} -> {inf_n}" for gt_n, inf_n in node_matches_map.items()]
        _list_block(p, "NODE MATCHES", "MAP", rows, limit=20)
    if extra_nodes:
        _list_block(p, "EXTRA INFERRED NODES", "NODE", extra_nodes, limit=min(15, cfg.list_limit))

    # --- EDGE MATCHING ---
    gt_edges = gt.get("edge_types", [])
    inf_edges = inf.get("edge_types", [])

    # GT allowed combos in GT label space: (edge_label, gt_src, gt_tgt)
    gt_allowed_combos: Set[Tuple[str, str, str]] = set()
    for edge in gt_edges:
        e_name = edge.get("type") or edge.get("name")
        for topo in edge.get("topology", []) or []:
            for s in topo.get("allowed_sources", []) or []:
                for t in topo.get("allowed_targets", []) or []:
                    gt_allowed_combos.add((e_name, s, t))

    # Inferred combos: (edge_label, inf_src, inf_tgt)
    inf_combo_set: Set[Tuple[str, str, str]] = set()
    inf_edges_by_combo: Dict[Tuple[str, str, str], dict] = {}
    for e in inf_edges:
        name = e.get("name")
        s = e.get("start_node")
        t = e.get("end_node")
        if not (name and s and t):
            continue
        combo = (name, s, t)
        inf_combo_set.add(combo)
        inf_edges_by_combo[combo] = e

    inf_edge_names = sorted({c[0] for c in inf_combo_set})
    gt_edge_names = sorted({c[0] for c in gt_allowed_combos})

    _section(p, "EDGE LABEL MAPPING")
    _kv(p, "GT Edge Types", len(gt_edge_names))
    _kv(p, "Inferred Edge Types", len(inf_edge_names))
    if cfg.verbose:
        _list_block(p, "RAW GT EDGE TYPES", "EDGE", gt_edge_names, limit=20)
        _list_block(p, "RAW INFERRED EDGE TYPES", "EDGE", inf_edge_names, limit=20)

    # Edge label mapping: exact -> string -> semantic (optional)
    edge_label_map: Dict[str, str] = {}
    missing_edge_labels: List[str] = []

    # exact
    for gt_name in gt_edge_names:
        if gt_name in inf_edge_names:
            edge_label_map[gt_name] = gt_name

    # lightweight normalizer for edge label matching
    def _edge_norm(x: str) -> str:
        if not x:
            return ""
        x = x.strip().upper()
        # conservative: only strip IS_ prefix (HAS_ can change meaning)
        if x.startswith("IS_"):
            x = x[3:]
        if x.endswith("_OF"):
            x = x[:-3]
        x = x.replace("_", "").replace(" ", "").replace("-", "")
        return x

    # string similarity fallback
    for gt_name in gt_edge_names:
        if gt_name in edge_label_map:
            continue
        gt_norm = _edge_norm(gt_name)
        best = None
        best_score = 0.0
        for cand in inf_edge_names:
            cand_norm = _edge_norm(cand)
            score = similar_string(gt_norm, cand_norm)
            if score > best_score:
                best_score = score
                best = cand
        if best and best_score >= cfg.edge_string_threshold:
            if cfg.verbose:
                p(f"   [Edge String Match] ACCEPTED: '{gt_name}' -> '{best}' (Score: {best_score:.3f})")
            edge_label_map[gt_name] = best

    # semantic fallback
    matcher = SemanticEdgeMatcher() if cfg.use_semantic_edge_match else None
    for gt_name in gt_edge_names:
        if gt_name in edge_label_map:
            continue
        if matcher is None:
            missing_edge_labels.append(gt_name)
            continue
        mapped = matcher.find_match(
            gt_name,
            inf_edge_names,
            threshold=cfg.semantic_threshold,
            margin=cfg.semantic_margin,
            verbose=cfg.verbose,
        )
        if mapped:
            edge_label_map[gt_name] = mapped
        else:
            missing_edge_labels.append(gt_name)

    if edge_label_map:
        rows = [f"{gt_l} -> {inf_l}" for gt_l, inf_l in edge_label_map.items()]
        _list_block(p, "EDGE LABEL MAP", "MAP", rows, limit=30)

    if missing_edge_labels:
        _list_block(
            p,
            "GT EDGE TYPES WITH NO MATCH",
            "MISS",
            missing_edge_labels,
            limit=20,
            suffix_note="This is name-level. Structural validity is handled by topology."
        )

    # Map GT allowed combos into inferred label/node space
    mapped_gt_allowed: Set[Tuple[str, str, str]] = set()
    for gt_e, gt_s, gt_t in gt_allowed_combos:
        mapped_edge = edge_label_map.get(gt_e)
        mapped_s = node_matches_map.get(gt_s)
        mapped_t = node_matches_map.get(gt_t)
        if mapped_edge and mapped_s and mapped_t:
            mapped_gt_allowed.add((mapped_edge, mapped_s, mapped_t))

    valid_inf = inf_combo_set & mapped_gt_allowed
    invalid_inf = inf_combo_set - mapped_gt_allowed
    missing_allowed = mapped_gt_allowed - inf_combo_set

    def _fmt_combo(c: Tuple[str, str, str]) -> str:
        return f"{c[0]}: {c[1]} -> {c[2]}"

    _section(p, "TOPOLOGY SUMMARY")
    _kv(p, "GT Allowed Combos (mapped)", len(mapped_gt_allowed))
    _kv(p, "Inferred Combos", len(inf_combo_set))
    _kv(p, "Valid Inferred", len(valid_inf))
    _kv(p, "Invalid Inferred", len(invalid_inf))
    _kv(p, "Missing Allowed (coverage gap)", len(missing_allowed))

    _section(p, "TOPOLOGY DETAILS")
    _list_block(p, "VALID (COUNTED AS CORRECT)", "EDGE", [_fmt_combo(c) for c in sorted(valid_inf)], limit=cfg.list_limit)
    _list_block(p, "INVALID (COUNTED AS WRONG)", "EDGE", [_fmt_combo(c) for c in sorted(invalid_inf)], limit=cfg.list_limit)
    _list_block(
        p,
        "MISSING ALLOWED (COVERAGE GAP)",
        "EDGE",
        [_fmt_combo(c) for c in sorted(missing_allowed)],
        limit=cfg.list_limit,
        suffix_note="Not necessarily an error: GT lists allowed possibilities."
    )

    # Scores
    inf_total = len(inf_combo_set)
    valid_count = len(valid_inf)
    invalid_count = len(invalid_inf)

    topo_precision = (valid_count / inf_total) if inf_total else 0.0
    topo_recall = (valid_count / len(mapped_gt_allowed)) if mapped_gt_allowed else 0.0
    topo_f1 = (2 * topo_precision * topo_recall / (topo_precision + topo_recall)) if (topo_precision + topo_recall) else 0.0

    _section(p, "TOPOLOGY SCORE")
    _kv(p, "Precision (Correctness)", f"{topo_precision*100:.2f}%")
    _kv(p, "Recall (Coverage)", f"{topo_recall*100:.2f}%")
    _kv(p, "F1", f"{topo_f1*100:.2f}%")

    matched_combo_set = valid_inf  # for edge prop scoring

    # --- PROPERTY MATCHING ---
    _section(p, "PROPERTY MATCHING")

    # Node properties
    prop_matches = 0
    total_props = 0
    total_extra_props = 0
    node_prop_rows = []

    # build quick lookup for inferred nodes by name
    inf_node_by_name = { (n.get("name") or (n.get("labels") or [""])[0]) : n for n in inf_nodes }

    for gt_node in gt_nodes:
        gt_name = gt_node.get("name") or (gt_node.get("labels") or [""])[0]
        inf_name = node_matches_map.get(gt_name)
        if not inf_name:
            continue
        inf_node = inf_node_by_name.get(inf_name)
        if not inf_node:
            continue

        m, t, e = compare_properties(gt_node.get("properties", []), inf_node.get("properties", []), verbose=False)
        prop_matches += m
        total_props += t
        total_extra_props += e

        if cfg.verbose:
            gt_props_list = sorted(list(prop_set(gt_node.get("properties", []))))
            inf_props_list = sorted(list(prop_set(inf_node.get("properties", []))))
            node_prop_rows.append(f"{gt_name}: {m}/{t} (extra={e}) | GT={len(gt_props_list)} INF={len(inf_props_list)}")
        else:
            node_prop_rows.append(f"{gt_name}: {m}/{t} (extra={e})")

    _list_block(p, "NODE PROPERTY SUMMARY", "NODE", node_prop_rows, limit=30)

    # Edge properties
    edge_prop_matches = 0
    total_edge_props = 0
    total_extra_edge_props = 0

    gt_edge_props_by_name = {}
    for gt_edge in gt_edges:
        gn = gt_edge.get("type") or gt_edge.get("name")
        gt_edge_props_by_name[gn] = gt_edge.get("properties", [])

    edge_prop_rows = []
    for combo in matched_combo_set:
        inf_edge_obj = inf_edges_by_combo.get(combo)
        if not inf_edge_obj:
            continue

        inf_name = combo[0]
        gt_prop_candidates = []
        for gt_label, mapped_label in edge_label_map.items():
            if mapped_label == inf_name:
                gt_prop_candidates.extend(gt_edge_props_by_name.get(gt_label, []))

        # de-dup
        seen = set()
        merged_gt_props = []
        for p0 in gt_prop_candidates:
            nm = p0.get("name")
            if nm and nm not in seen:
                seen.add(nm)
                merged_gt_props.append(p0)

        m, t, e = compare_properties(merged_gt_props, inf_edge_obj.get("properties", []), verbose=False)
        edge_prop_matches += m
        total_edge_props += t
        total_extra_edge_props += e

        edge_prop_rows.append(f"{inf_name} ({combo[1]}->{combo[2]}): {m}/{t} (extra={e})")

    _list_block(p, "EDGE PROPERTY SUMMARY (VALID TOPOLOGY ONLY)", "EDGE", sorted(edge_prop_rows), limit=30)

    # Final real scores
    real_node_score = calculate_real_score(node_matches, len(gt_nodes), len(extra_nodes))
    real_edge_score = topo_precision * 100.0
    real_prop_score = calculate_real_score(prop_matches, total_props, total_extra_props)
    real_edge_prop_score = calculate_real_score(edge_prop_matches, total_edge_props, total_extra_edge_props)
    topology_score = topo_f1 * 100.0

    overall = (real_node_score + real_edge_score + real_prop_score + real_edge_prop_score + topology_score) / 5.0
    
    node_prop_label = f"{real_prop_score:.2f}%" if (total_props + total_extra_props) > 0 else "N/A"
    edge_prop_label = f"{real_edge_prop_score:.2f}%" if (total_edge_props + total_extra_edge_props) > 0 else "N/A"
    
    _header(p, "FINAL SCORES")
    _kv(p, "NODE ACCURACY", f"{real_node_score:.2f}%")
    _kv(p, "EDGE CORRECTNESS", f"{real_edge_score:.2f}%")
    _kv(p, "NODE PROPERTY ACC", node_prop_label)
    _kv(p, "EDGE PROPERTY ACC", edge_prop_label)
    _kv(p, "TOPOLOGY SCORE (F1)", f"{topology_score:.2f}%")
    p("-" * 60)
    _kv(p, "OVERALL PERFORMANCE", f"{overall:.2f}%")

    return CompareResult(
        real_node_accuracy=real_node_score,
        real_edge_accuracy=real_edge_score,
        real_node_property_accuracy=real_prop_score,
        real_edge_property_accuracy=real_edge_prop_score,
        topology_score_f1=topology_score,
        overall_performance=overall,
        node_matches=node_matches_map,
        extra_nodes=extra_nodes,
        extra_edges=invalid_count,
    )
