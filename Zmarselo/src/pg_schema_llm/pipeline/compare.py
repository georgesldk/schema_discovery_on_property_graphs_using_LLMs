from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ============================================================
# Config
# ============================================================

RESERVED_PROP_PREFIXES = (":",) # any property starting with : is treated technical/import metadata.
RESERVED_PROP_NAMES = {"id", "label", "labels", "type"}
DEFAULT_SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # embedding model used for semantic edge matching. 


# ============================================================
# Helpers
# ============================================================

# Load JSON with UTF-8-SIG encoding to handle potential BOM issues, and provide clear error messages if loading fails.
def load_json(path: str) -> dict: 
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}") from e

# normalize the property names before comparison (e.g. "Name", " name ", "`name`" should all be treated as the same property "name")
def normalize_prop_name(name: str) -> str:
    if not name: return ""
    return name.strip().lower().replace("`", "")

# decides whether a property should beignored during evaluation  (like id label etc)
def is_reserved_prop(name: str) -> bool:
    if not name: return True
    raw = name.strip()
    if raw.startswith(RESERVED_PROP_PREFIXES): return True
    n = normalize_prop_name(raw)
    if n in RESERVED_PROP_NAMES: return True
    if re.match(r"^id\s*\(.*\)$", n) or re.match(r"^label\s*\(.*\)$", n): return True
    return False

# normalize datatype names before comparing gt vs inferred. 
def norm_data_type(t: str) -> str:
    if not t: return "STRING"
    t = t.upper()
    if t in ("LONG", "INT", "INTEGER"): return "INTEGER"
    if t in ("DOUBLE", "FLOAT", "NUMBER"): return "DOUBLE"
    if t in ("BOOLEAN", "BOOL"): return "BOOLEAN"
    if "DATE" in t or "TIME" in t: return "DATE"
    if "ARRAY" in t or "LIST" in t: return "LIST"
    if t == "POINT": return "POINT"
    return "STRING"

# precision = matches / pred_total
# recall = matches / gt_total
# F1 = 2 * (precision * recall) / (precision + recall)
# PG-hive's evaluation, uses comparing discovered schema elements against expected ones. 
def calculate_f1(matches: int, pred_total: int, gt_total: int) -> Tuple[float, float, float]:
    precision = (matches / pred_total) if pred_total > 0 else 0.0 
    recall = (matches / gt_total) if gt_total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# CO-OCCURENCE
def build_gt_edge_cooccurrence(gt_edges: List[dict]) -> Counter:
    """
    Build edge-label co-occurrence counts from GT topology definitions.

    Key shape:
      (edge_name, source_node_name, target_node_name)
    """
    co = Counter()
    for e in gt_edges:
        ename = e.get("name") or e.get("type") or ""
        for topo in e.get("topology", []) or []:
            for s in topo.get("allowed_sources", []) or []:
                for t in topo.get("allowed_targets", []) or []:
                    co[(ename, s, t)] += 1
    return co


def build_inf_edge_cooccurrence(inf_edges: List[dict], node_map_inf_to_gt: Dict[str, str], edge_label_map: Dict[str, str]) -> Counter:
    """
    Build inferred edge-label co-occurrence counts mapped into GT node/edge space.

    Key shape:
      (mapped_edge_name, mapped_source_node_name, mapped_target_node_name)
    """
    co = Counter()
    for e in inf_edges:
        inf_ename = e.get("name") or ""
        mapped_ename = edge_label_map.get(inf_ename, inf_ename)

        s_inf = e.get("start_node") or e.get("source")
        t_inf = e.get("end_node") or e.get("target")
        s_gt = node_map_inf_to_gt.get(s_inf)
        t_gt = node_map_inf_to_gt.get(t_inf)
        if not s_gt or not t_gt:
            continue
        co[(mapped_ename, s_gt, t_gt)] += 1
    return co


def _infer_mandatory_bool(prop: dict) -> bool:
    """
    Read mandatory/optional from the required schema format.

    Required:
      - mandatory: true|false
    """
    return bool(prop.get("mandatory", False))


def _mandatory_label(prop: dict) -> str:
    return "MANDATORY" if _infer_mandatory_bool(prop) else "OPTIONAL"

# normalize labels/edge names for string comparison
def _norm_label(s: str) -> str:
    if not s: return ""
    s = s.strip().lower().replace("_", " ").replace("-", " ").replace(".", " ")
    return re.sub(r"\s+", " ", s)

# similarity of strings (like WORKS_AT, 'works at')
def similar_string(a: str, b: str) -> float:
    if not a or not b: return 0.0
    return SequenceMatcher(None, _norm_label(a), _norm_label(b)).ratio()


# ============================================================
# Semantic Edge Matcher
# ============================================================

# This class is used when simple string similarity is not enough.
#  Example:
# GT edge: WORKS_AT
# Inferred edge: EMPLOYED_BY
# String similarity may be low, but semantically they are related.
class SemanticEdgeMatcher:
    
    
    def __init__(self, model_name: str = DEFAULT_SEMANTIC_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._util = None
        self._cache: Dict[str, Any] = {}

    def _load(self):
        if self._model is None:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            from sentence_transformers import SentenceTransformer, util # type: ignore
            self._util = util
            self._model = SentenceTransformer(self.model_name) # all-MiniLM-L6-v2

    @staticmethod
    def _clean(label: str) -> str: # prepare for semantic comparison
        return (label or "").replace("_", " ").replace(".", " ").lower().strip()

    # cache to avoid recompuitng embeddings for the same label 
    def _embed(self, text: str):
        if text not in self._cache:
            self._cache[text] = self._model.encode(text, convert_to_tensor=True)
        return self._cache[text]

    # Finds the best semantic match for one GT edge name among inferred edge candidates.
    def find_match(self, target: str, candidates: Sequence[str], threshold: float = 0.75, margin: float = 0.05) -> Optional[str]:
        if not candidates: return None
        self._load()
        import torch # type: ignore

        target_clean = self._clean(target)
        cand_clean = [self._clean(c) for c in candidates]

        target_emb = self._embed(target_clean)
        cand_embs = torch.stack([self._embed(c) for c in cand_clean], dim=0)

        scores = self._util.cos_sim(target_emb, cand_embs)[0] # cosine similarity scores between target and candidates
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        best_match = candidates[best_idx]

        topk = scores.topk(k=min(2, scores.numel())).values
        second = float(topk[1]) if topk.numel() > 1 else -1.0

        if best_score >= threshold and (best_score - second) >= margin:
            return best_match
        return None


# ============================================================
# Config & Data Classes
# ============================================================

@dataclass
class CompareConfig:
    verbose: bool = True
    node_jaccard_threshold: float = 0.50 # Minimum Jaccard similarity of node labels to consider a match
    edge_string_threshold: float = 0.78 # Minimum edge similarity needed to map edge labels
    use_semantic_edge_match: bool = True 
    semantic_threshold: float = 0.75 # Minimum cosine similarity for semantic edge matching
    semantic_margin: float = 0.05 # Minimum margin between best and second-best match for semantic edge matching


# ============================================================
# Core logic
# ============================================================

# GT schema
# vs
# inferred schema
def run_compare(gt_file: str, inferred_file: str, config: Optional[CompareConfig] = None) -> Optional[dict]:
    cfg = config or CompareConfig()
    
    if not cfg.verbose:
        p = lambda *a, **k: None
    else:
        p = print

    if not os.path.exists(gt_file) or not os.path.exists(inferred_file):
        p("Error: Missing input files.")
        return None

    gt = load_json(gt_file)
    inf = load_json(inferred_file)

    p(f"\n========================================================")
    p(f"                 DEEP SCHEMA EVALUATION                 ")
    p(f"========================================================")
    p(f" GT:  {os.path.basename(gt_file)}")
    p(f" INF: {os.path.basename(inferred_file)}\n")

    # ---- 1. NODE MATCHING ----
    gt_nodes = {n.get("name") or (n.get("labels") or [""])[0]: n for n in gt.get("node_types", [])}
    inf_nodes = {n.get("name") or (n.get("labels") or [""])[0]: n for n in inf.get("node_types", [])}

    node_map_inf_to_gt = {}
    matched_gt_nodes = set()

    for inf_name, inf_n in inf_nodes.items():
        inf_labels = set(inf_n.get("labels", []))
        best_gt = None
        best_score = 0.0

        for gt_name, gt_n in gt_nodes.items():
            if gt_name in matched_gt_nodes: continue
            gt_labels = set(gt_n.get("labels", []))
            score = len(inf_labels & gt_labels) / len(inf_labels | gt_labels) if (inf_labels or gt_labels) else 1.0
            
            if score > best_score:
                best_score = score
                best_gt = gt_name
        
        if best_score >= cfg.node_jaccard_threshold and best_gt:
            node_map_inf_to_gt[inf_name] = best_gt
            matched_gt_nodes.add(best_gt)

    n_prec, n_rec, n_f1 = calculate_f1(len(node_map_inf_to_gt), len(inf_nodes), len(gt_nodes))

    p(f"--- 1. EXHAUSTIVE NODE & LABEL LIST ---")
    p("  GROUND TRUTH NODES (Expected):")
    for gt_name in sorted(gt_nodes.keys()):
        if gt_name in matched_gt_nodes:
            inf_mapped = [k for k, v in node_map_inf_to_gt.items() if v == gt_name][0]
            p(f"    [✓] Node: {gt_name} (Found as '{inf_mapped}')")
            
            # --- LABEL COMPARISON FOR MATCHED NODES ---
            gt_labels = set(gt_nodes[gt_name].get("labels", []))
            inf_labels = set(inf_nodes[inf_mapped].get("labels", []))
            
            if gt_labels or inf_labels:
                p(f"        Labels:")
                for l in sorted(gt_labels & inf_labels):
                    p(f"          [✓] {l}")
                for l in sorted(gt_labels - inf_labels):
                    p(f"          [X] {l} (MISSING IN INF)")
                for l in sorted(inf_labels - gt_labels):
                    p(f"          [+] {l} (EXTRA IN INF)")
        else:
            p(f"    [X] Node: {gt_name} (MISSING)")
            gt_labels = set(gt_nodes[gt_name].get("labels", []))
            if gt_labels:
                p(f"        Expected Labels: {', '.join(sorted(gt_labels))}")

    unmatched_inf = [n for n in inf_nodes if n not in node_map_inf_to_gt]
    if unmatched_inf:
        p("\n  EXTRA / HALLUCINATED NODES:")
        for inf_name in sorted(unmatched_inf):
            p(f"    [+] Node: {inf_name}")
            inf_labels = set(inf_nodes[inf_name].get("labels", []))
            if inf_labels:
                p(f"        Labels Found: {', '.join(sorted(inf_labels))}")


    # ---- 2. EDGE MAPPING & TOPOLOGY ----
    gt_edges = gt.get("edge_types", [])
    inf_edges = inf.get("edge_types", [])

    gt_combos = set()
    gt_edge_dict = {}
    for e in gt_edges:
        ename = e.get("name") or e.get("type")
        gt_edge_dict[ename] = e
        for t in e.get("topology", []):
            for s in t.get("allowed_sources", []):
                for tgt in t.get("allowed_targets", []):
                    gt_combos.add((ename, s, tgt))

    gt_edge_names = list(set(c[0] for c in gt_combos))
    inf_edge_names = list(set(e.get("name") for e in inf_edges if e.get("name")))

    edge_label_map = {}
    for gt_name in gt_edge_names:
        if gt_name in inf_edge_names:
            edge_label_map[gt_name] = gt_name 
            continue
        best, best_score = None, 0.0
        for cand in inf_edge_names:
            score = similar_string(gt_name, cand)
            if score > best_score:
                best_score = score
                best = cand
        if best and best_score >= cfg.edge_string_threshold:
            edge_label_map[best] = gt_name

    if cfg.use_semantic_edge_match:
        matcher = SemanticEdgeMatcher() 
        unmapped_gt = [g for g in gt_edge_names if g not in edge_label_map.values()]
        for gt_name in unmapped_gt:
            match = matcher.find_match(gt_name, inf_edge_names, threshold=cfg.semantic_threshold, margin=cfg.semantic_margin)
            if match: edge_label_map[match] = gt_name

    inf_combos_mapped = set()
    inf_edges_by_combo = {}

    for e in inf_edges:
        inf_ename = e.get("name")
        gt_ename = edge_label_map.get(inf_ename, inf_ename) 
        s_inf = e.get("source") or e.get("start_node")
        t_inf = e.get("target") or e.get("end_node")
        
        s_gt = node_map_inf_to_gt.get(s_inf)
        t_gt = node_map_inf_to_gt.get(t_inf)

        if s_gt and t_gt:
            combo = (gt_ename, s_gt, t_gt)
            inf_combos_mapped.add(combo)
            inf_edges_by_combo[combo] = e

    valid_edges = inf_combos_mapped & gt_combos
    invalid_edges = inf_combos_mapped - gt_combos
    missing_edges = gt_combos - inf_combos_mapped

    p(f"\n--- 2. EXHAUSTIVE EDGE TOPOLOGY & LABEL LIST ---")
    p("  GROUND TRUTH EDGES (Expected Allowed Combos):")
    for combo in sorted(gt_combos):
        if combo in valid_edges:
            p(f"    [✓] Topology: {combo[0]} ({combo[1]} -> {combo[2]})")
            
            # --- LABEL COMPARISON FOR MATCHED EDGES ---
            gt_e_data = gt_edge_dict.get(combo[0], {})
            inf_e_data = inf_edges_by_combo.get(combo, {})
            
            gt_labels = set(gt_e_data.get("labels", [combo[0]]))
            inf_labels = set(inf_e_data.get("labels", [inf_e_data.get("name")]))
            
            if gt_labels != {combo[0]} or inf_labels != {inf_e_data.get("name")}:
                p(f"        Labels:")
                for l in sorted(gt_labels & inf_labels):
                    p(f"          [✓] {l}")
                for l in sorted(gt_labels - inf_labels):
                    p(f"          [X] {l} (MISSING IN INF)")
                for l in sorted(inf_labels - gt_labels):
                    p(f"          [+] {l} (EXTRA IN INF)")
                    
        else:
            p(f"    [X] Topology: {combo[0]} ({combo[1]} -> {combo[2]}) (MISSING IN INF)")

    if invalid_edges:
        p("\n  EXTRA / INVALID INFERRED EDGES:")
        for c in sorted(invalid_edges):
            p(f"    [+] Topology: {c[0]} ({c[1]} -> {c[2]})")

    # ---- 2b. EDGE LABEL CO-OCCURRENCE ----
    gt_co = build_gt_edge_cooccurrence(gt_edges)
    inf_co = build_inf_edge_cooccurrence(inf_edges, node_map_inf_to_gt, edge_label_map)

    p(f"\n--- 2b. EDGE LABEL CO-OCCURRENCE (GT vs INF) ---")
    if not gt_co and not inf_co:
        p("  (No edge co-occurrence entries in GT or INF)")
    else:
        all_keys = sorted(set(gt_co.keys()) | set(inf_co.keys()))
        for k in all_keys:
            gt_n = gt_co.get(k, 0)
            inf_n = inf_co.get(k, 0)
            tag = "[✓]" if gt_n == inf_n else "[!]"
            p(f"  {tag} {k[0]} ({k[1]} -> {k[2]}): GT={gt_n} INF={inf_n}")

    e_prec, e_rec, e_f1 = calculate_f1(len(valid_edges), len(inf_combos_mapped), len(gt_combos))

    # ---- 3. PROPERTIES & CONSTRAINTS ----
    p(f"\n--- 3. EXHAUSTIVE PROPERTY LIST ---")
    total_prop_matches = 0
    total_inf_props = 0
    total_gt_props = 0
    type_matches = 0
    constraint_matches = 0

    def eval_props(inf_prop_list, gt_prop_list, context_name):
        nonlocal total_prop_matches, total_inf_props, total_gt_props, type_matches, constraint_matches
        
        inf_dict = {normalize_prop_name(p.get("name")): p for p in inf_prop_list if not is_reserved_prop(p.get("name"))}
        gt_dict = {normalize_prop_name(p.get("name")): p for p in gt_prop_list if not is_reserved_prop(p.get("name"))}
        
        total_inf_props += len(inf_dict)
        total_gt_props += len(gt_dict)
        
        p(f"\n  [{context_name}]")
        if not gt_dict and not inf_dict:
            p("    (No properties defined in GT or INF)")
            return

        for k, gt_p in gt_dict.items():
            gt_type = norm_data_type(gt_p.get("type"))
            gt_mand = gt_p.get("mandatory", False)
            gt_mand_str = "MANDATORY" if gt_mand else "OPTIONAL"

            if k in inf_dict:
                total_prop_matches += 1
                inf_p = inf_dict[k]
                
                inf_type = norm_data_type(inf_p.get("type"))
                inf_mand = _infer_mandatory_bool(inf_p)
                inf_mand_str = _mandatory_label(inf_p)
                
                type_ok = gt_type == inf_type
                mand_ok = gt_mand == inf_mand

                if type_ok: type_matches += 1
                if mand_ok: constraint_matches += 1
                
                if type_ok and mand_ok:
                    p(f"    [✓] {k} ({gt_type}, {gt_mand_str})")
                else:
                    errs = []
                    if not type_ok: errs.append(f"Type: GT=[{gt_type}] INF=[{inf_type}]")
                    if not mand_ok: errs.append(f"Constraint: GT=[{gt_mand_str}] INF=[{inf_mand_str}]")
                    p(f"    [!] {k} -> " + " | ".join(errs))
            else:
                p(f"    [X] {k} ({gt_type}, {gt_mand_str}) -> MISSING IN INF")

        for k, inf_p in inf_dict.items():
            if k not in gt_dict:
                inf_type = norm_data_type(inf_p.get("type"))
                inf_mand_str = _mandatory_label(inf_p)
                p(f"    [+] {k} ({inf_type}, {inf_mand_str}) -> EXTRA IN INF (Hallucinated)")

    # Node Props
    for inf_name, gt_name in node_map_inf_to_gt.items():
        eval_props(inf_nodes[inf_name].get("properties", []), gt_nodes[gt_name].get("properties", []), f"Node: {gt_name}")

    # Edge Props
    for combo in valid_edges:
        inf_e = inf_edges_by_combo[combo]
        gt_props = gt_edge_dict[combo[0]].get("properties", [])
        eval_props(inf_e.get("properties", []), gt_props, f"Edge Topology: {combo[0]} ({combo[1]} -> {combo[2]})")

    p_prec, p_rec, p_f1 = calculate_f1(total_prop_matches, total_inf_props, total_gt_props)
    type_acc = (type_matches / total_prop_matches) if total_prop_matches else 0.0
    const_acc = (constraint_matches / total_prop_matches) if total_prop_matches else 0.0

    p(f"\n========================================================")
    p(f"         EVALUATION METRICS (Same as pg-hive but ask Sophia if they are okay xo)       ")
    p(f"========================================================")
    
    # Paper Section 5: Nodes & Edges (F1*)
    p(f"1. TYPE DISCOVERY (CLUSTERING QUALITY)")
    p(f"   Node Types F1*-Score: {n_f1:.2%}  (Precision: {n_prec:.2%} | Recall: {n_rec:.2%})")
    p(f"   Edge Types F1*-Score: {e_f1:.2%}  (Precision: {e_prec:.2%} | Recall: {e_rec:.2%})")
    
    # Paper Section 5: Constraints & Datatypes
    p(f"\n2. SCHEMA CONSTRAINTS")
    p(f"   Property Completeness: {p_f1:.2%}  (Found {total_prop_matches} of {total_gt_props} expected)")
    p(f"   Data Type Accuracy:    {type_acc:.2%}  (Error: {1 - type_acc:.2%})")
    p(f"   Constraint Accuracy:   {const_acc:.2%}  (Mandatory/Optional)")
    p(f"========================================================\n")

    return {
        "nodes": {"precision": n_prec, "recall": n_rec, "f1": n_f1},
        "edges": {"precision": e_prec, "recall": e_rec, "f1": e_f1},
        "props": {
            "precision": p_prec, 
            "recall": p_rec, 
            "f1": p_f1, 
            "type_accuracy": type_acc, 
            "constraint_accuracy": const_acc
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        run_compare(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python compare.py <gt_json> <inferred_json>")
