import json
import os
from difflib import SequenceMatcher
import re

RESERVED_PROP_PREFIXES = (":",)  # anything starting with ":" like :ID, :LABEL
RESERVED_PROP_NAMES = {
    "id", "label", "labels", "type",  # if they appear without ":" somehow
}

def normalize_prop_name(name: str) -> str:
    if not name:
        return ""
    n = name.strip()

    # normalize case for everything
    n = n.lower()

    # normalize common neo4j-ish variants (optional)
    n = n.replace("`", "")

    return n

def is_reserved_prop(name: str) -> bool:
    if not name:
        return True
    raw = name.strip()

    # ignore neo4j import style props: :ID, :LABEL, :ID(Body-ID) etc.
    if raw.startswith(RESERVED_PROP_PREFIXES):
        return True

    n = normalize_prop_name(raw)

    # ignore bare reserved words
    if n in RESERVED_PROP_NAMES:
        return True

    # ignore patterns like "id(...)" if you ever get them without ':'
    if re.match(r"^id\s*\(.*\)$", n):
        return True
    if re.match(r"^label\s*\(.*\)$", n):
        return True

    return False

def prop_set(props):
    out = set()
    for p in props:
        nm = p.get("name", "")
        if not nm or is_reserved_prop(nm):
            continue
        out.add(normalize_prop_name(nm))
    return out

#################new method
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_semantic_model = None
_util = None

def get_semantic_model():
    """
    Lazy-load the semantic model so imports don't have side effects.
    Keeps behavior identical once run_compare() is called.
    """
    global _semantic_model, _util
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer, util
        _util = util
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model, _util



# --- Semantic edge matcher (cached + lazy model load) ---
# FOR EDGESSSSSSSSSSSSSSSSSSSS
_EDGE_EMB_CACHE = {}  # text -> embedding tensor
_semantic_model = None
_util = None

def get_semantic_model():
    global _semantic_model, _util
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer, util
        _util = util
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model, _util

def clean_label_for_embedding(label):
    if not label:
        return ""
    return label.replace("_", " ").replace(".", " ").lower().strip()

def _embed(text, model):
    if text not in _EDGE_EMB_CACHE:
        _EDGE_EMB_CACHE[text] = model.encode(text, convert_to_tensor=True)
    return _EDGE_EMB_CACHE[text]

def find_edge_match_semantic(target, candidates, threshold=0.75, verbose=False):
    if not candidates:
        return None

    model, util = get_semantic_model()

    target_clean = clean_label_for_embedding(target)
    candidates_clean = [clean_label_for_embedding(c) for c in candidates]

    target_emb = _embed(target_clean, model)

    # IMPORTANT: cos_sim expects a 2D tensor for candidates, not a list of tensors
    import torch
    cand_embs = torch.stack([_embed(c, model) for c in candidates_clean], dim=0)

    cosine_scores = util.cos_sim(target_emb, cand_embs)[0]
    best_idx = int(cosine_scores.argmax())
    best_score = float(cosine_scores[best_idx])
    best_match = candidates[best_idx]

    if best_score >= threshold:
        if verbose:
            print(f"   [Edge Semantic Match] ACCEPTED: '{target}' -> '{best_match}' (Score: {best_score:.3f})")
        return best_match

    return None


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        raise

#For nodes
def similar_string(a, b):
    if not a or not b:
        return 0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_node_match_string(target, candidates, threshold=0.8):
    best = None
    best_score = 0
    for c in candidates:
        s = similar_string(target, c)
        if s > best_score:
            best_score = s
            best = c
    return best if best_score >= threshold else None


def compare_properties(gt_props, inf_props, verbose=False):
    gt_set = prop_set(gt_props)
    inf_set = prop_set(inf_props)

    matches = len(gt_set & inf_set)
    total = len(gt_set)
    extra = len(inf_set - gt_set)

    if verbose and total > 0 and matches == 0:
        print(f"   [PROP DEBUG] Mismatch!")
        print(f"     GT Properties: {list(gt_set)[:5]}")
        print(f"     Inf Properties: {list(inf_set)[:5]}")

    return matches, total, extra

def check_edge_topology(gt_edge, inf_edge, node_matches):
    """
    Checks if inferred edge start/end match the ground truth topology.
    GT edges have many allowed source/target combos.
    Inferred edges have a single start_node/end_node.
    """
    gt_name = gt_edge.get("type") or gt_edge.get("name")
    inf_name = inf_edge.get("name")

    inf_src = inf_edge.get("start_node")
    inf_tgt = inf_edge.get("end_node")

    # Expand GT allowed source/target combos
    combos = []
    for topo in gt_edge.get("topology", []):
        for s in topo.get("allowed_sources", []):
            for t in topo.get("allowed_targets", []):
                combos.append((s, t))

    # Check if inferred pair matches any GT pair after node name matching
    for gt_s, gt_t in combos:
        mapped_s = node_matches.get(gt_s)
        mapped_t = node_matches.get(gt_t)
        if mapped_s == inf_src and mapped_t == inf_tgt:
            return True

    return False


def calculate_real_score(matches, total_gt, total_extra):
    """
    Calculates accuracy by penalizing for over-inference (extra items).
    True Score = Matches / (Actual GT Items + Extra False Positives)
    """
    denominator = total_gt + total_extra
    if denominator == 0:
        return 0
    return (matches / denominator) * 100


def run_compare(gt_file, inferred_file, verbose=True):
    # allow silent runs for experiments/tests, without changing default behavior
    if not verbose:
        def _p(*a, **k):
            return
        print_fn = _p
    else:
        print_fn = print

    if not os.path.exists(gt_file) or not os.path.exists(inferred_file):
        print_fn(" Error: One or both input files do not exist.")
        return None

    gt = load_json(gt_file)
    inf = load_json(inferred_file)

    print_fn(f"\n==== REAL SCHEMA COMPARISON REPORT ====")
    print_fn(f"GT: {os.path.basename(gt_file)}")
    print_fn(f"Inferred: {os.path.basename(inferred_file)}")

    gt_nodes = gt.get("node_types", [])
    inf_nodes = inf.get("node_types", [])

    # --- NODE MATCHING ---
    gt_node_names = [n.get("name") or n.get("labels", [""])[0] for n in gt_nodes]
    inf_node_names = [n.get("name") or n.get("labels", [""])[0] for n in inf_nodes]
    
    # === NEW: PRINT RAW NODE LISTS ===
    print_fn("\n--- DEBUG: RAW NODE LISTS ---")
    print_fn(f"GT Nodes:       {gt_node_names}")
    print_fn(f"Inferred Nodes: {inf_node_names}")
    # =================================

    node_matches_map = {}
    used_inf_nodes = set()

    for gt_name in gt_node_names:
        match = find_node_match_string(gt_name, [n for n in inf_node_names if n not in used_inf_nodes])
        if match:
            node_matches_map[gt_name] = match
            used_inf_nodes.add(match)

    node_matches = len(node_matches_map)
    extra_nodes = [n for n in inf_node_names if n not in used_inf_nodes]

    print_fn("\n--- NODE TYPE MATCHING ---")
    print_fn(f"Matched Node Types: {node_matches} / {len(gt_node_names)}")
    if node_matches_map:
        print_fn("  Matches (String Similarity):")
        for gt_n, inf_n in node_matches_map.items():
            print_fn(f"    '{gt_n}' -> '{inf_n}'")

    print_fn(f"Extra Inferred Node Types: {len(extra_nodes)}")
    if extra_nodes:
        print_fn(f"    Extra: {extra_nodes[:5]}")

    # --- EDGE MATCHING ---
    gt_edges = gt.get("edge_types", [])
    inf_edges = inf.get("edge_types", [])

    gt_edge_combos = []
    for edge in gt_edges:
        e_name = edge.get("type") or edge.get("name")
        for topo in edge.get("topology", []):
            for s in topo.get("allowed_sources", []):
                for t in topo.get("allowed_targets", []):
                    gt_edge_combos.append((e_name, s, t))

    inf_combo_set = set()
    inf_edges_by_combo = {}
    for e in inf_edges:
        name = e.get("name")
        s = e.get("start_node")
        t = e.get("end_node")
        if not (name and s and t): continue
        combo = (name, s, t)
        inf_combo_set.add(combo)
        inf_edges_by_combo.setdefault(combo, e)

    # --- EDGE LABEL MAPPING (SEMANTIC BASED) ---
    inf_edge_names = sorted({c[0] for c in inf_combo_set})
    gt_edge_names = sorted({c[0] for c in gt_edge_combos})

    # === NEW: PRINT RAW EDGE TYPES ===
    print_fn("\n--- DEBUG: RAW EDGE TYPES ---")
    print_fn(f"GT Edge Types:       {gt_edge_names}")
    print_fn(f"Inferred Edge Types: {inf_edge_names}")
    # =================================
    

    edge_label_map = {}
    missing_edge_labels = []
    
    for gt_name in gt_edge_names:
        if gt_name in inf_edge_names:
            edge_label_map[gt_name] = gt_name
        else:
            # USE SEMANTIC MATCHER FOR EDGES
            mapped = find_edge_match_semantic(gt_name, inf_edge_names, threshold=0.75, verbose=verbose)
            if mapped:
                edge_label_map[gt_name] = mapped
            else:
                missing_edge_labels.append(gt_name)

    print_fn("\n--- EDGE LABEL MAPPING (Semantic) ---")
    if edge_label_map:
        for gt_l, inf_l in edge_label_map.items():
            print_fn(f"  '{gt_l}' -> '{inf_l}'")
    else:
        print_fn("  No edge labels matched.")

    # Build Mapped GT Combo Set
    mapped_gt_combo_set = set()

    for gt_name, gt_s, gt_t in gt_edge_combos:
        mapped_s = node_matches_map.get(gt_s)
        mapped_t = node_matches_map.get(gt_t)
        mapped_edge_name = edge_label_map.get(gt_name)

        if mapped_s and mapped_t and mapped_edge_name:
            mapped_gt_combo_set.add((mapped_edge_name, mapped_s, mapped_t))

    matched_combo_set = mapped_gt_combo_set & inf_combo_set
    missing_combo_set = mapped_gt_combo_set - inf_combo_set
    extra_combo_set = inf_combo_set - mapped_gt_combo_set

    edge_matches = len(matched_combo_set)
    total_gt = len(mapped_gt_combo_set)
    total_extra = len(extra_combo_set)

    edge_precision = (edge_matches / (edge_matches + total_extra)) if (edge_matches + total_extra) > 0 else 0.0
    edge_recall = (edge_matches / total_gt) if total_gt > 0 else 0.0
    edge_f1 = (2 * edge_precision * edge_recall / (edge_precision + edge_recall)) if (edge_precision + edge_recall) > 0 else 0.0

    print_fn("\n--- EDGE TOPOLOGY STATS ---")
    print_fn(f"Matched Edge Topology Combos: {edge_matches} / {total_gt}")
    print_fn(f"GT Total Topology Combos: {total_gt}")
    print_fn(f"Extra Inferred Topology Combos: {total_extra}")
    print_fn(f"Edge Precision: {edge_precision*100:.2f}% | Recall: {edge_recall*100:.2f}% | F1: {edge_f1*100:.2f}%")

    if missing_edge_labels:
        print_fn("\nMissing inferred edge labels (no semantic match found):")
        for lbl in missing_edge_labels[:5]: print_fn(f"  - {lbl}")

    if extra_combo_set:
        print_fn("\nExtra topology combos (produced but not in GT):")
        for name, s, t in list(sorted(extra_combo_set))[:5]:
            print_fn(f"  - {name}: {s} -> {t}")

    if missing_combo_set:
        print_fn("\nMissing topology combos (expected but not produced):")
        for name, s, t in list(sorted(missing_combo_set))[:5]:
            print_fn(f"  - {name}: {s} -> {t}")

    # --- PROPERTY MATCHING ---
    print_fn("\n--- PROPERTY MATCHING DETAIL ---")
    prop_matches = 0
    total_props = 0
    total_extra_props = 0  # <--- NEW: Track extra properties

    for gt_node in gt_nodes:
        gt_name = gt_node.get("name") or gt_node.get("labels", [""])[0]
        inf_name = node_matches_map.get(gt_name)
        if not inf_name: continue

        inf_node = next((n for n in inf_nodes if (n.get("name") or n.get("labels", [""])[0]) == inf_name), None)
        if not inf_node: continue

        # === PRINT NODE PROPERTIES ===
        gt_props_list = sorted(list(prop_set(gt_node.get("properties", []))))
        inf_props_list = sorted(list(prop_set(inf_node.get("properties", []))))
       
        print_fn(f"\n[Node: {gt_name}]")
        print_fn(f"  GT Props : {gt_props_list}")
        print_fn(f"  Inf Props: {inf_props_list}")
        # =============================

        m, t, e = compare_properties(gt_node.get("properties", []), inf_node.get("properties", []))
        prop_matches += m
        total_props += t
        total_extra_props += e  # <--- NEW

    # Edge Properties
    edge_prop_matches = 0
    total_edge_props = 0
    total_extra_edge_props = 0  # <--- NEW
    
    gt_edge_props_by_name = {}
    for gt_edge in gt_edges:
        gn = gt_edge.get("type") or gt_edge.get("name")
        gt_edge_props_by_name[gn] = gt_edge.get("properties", [])

    for combo in matched_combo_set:
        inf_edge_obj = inf_edges_by_combo.get(combo)
        if not inf_edge_obj: continue

        inf_name = combo[0]
        gt_prop_candidates = []
        for gt_label, mapped_label in edge_label_map.items():
            if mapped_label == inf_name:
                gt_prop_candidates.extend(gt_edge_props_by_name.get(gt_label, []))

        # Deduplicate
        if gt_prop_candidates:
            seen = set()
            merged_gt_props = []
            for p in gt_prop_candidates:
                n = p.get("name")
                if n and n not in seen:
                    seen.add(n)
                    merged_gt_props.append(p)
        else:
            merged_gt_props = []

        # === NEW: PRINT EDGE PROPERTIES ===
        gt_props_list = sorted([p["name"] for p in merged_gt_props])
        inf_props_list = sorted([p["name"] for p in inf_edge_obj.get("properties", [])])
        print_fn(f"\n[Edge: {inf_name} (Combo: {combo[1]}->{combo[2]})]")
        print_fn(f"  GT Props : {gt_props_list}")
        print_fn(f"  Inf Props: {inf_props_list}")
        # ==================================

        m, t, e = compare_properties(merged_gt_props, inf_edge_obj.get("properties", []))
        edge_prop_matches += m
        total_edge_props += t
        total_extra_edge_props += e  

    # FINAL REAL SCORES

    real_node_score = calculate_real_score(node_matches, len(gt_nodes), len(extra_nodes))
    real_edge_score = calculate_real_score(edge_matches, total_gt, total_extra)
    real_prop_score = calculate_real_score(prop_matches, total_props, total_extra_props)
    real_edge_prop_score = calculate_real_score(edge_prop_matches, total_edge_props, total_extra_edge_props)

    print_fn("\n" + "=" * 30)
    print_fn(f"REAL NODE ACCURACY: {real_node_score:.2f}%")
    print_fn(f"REAL EDGE ACCURACY: {real_edge_score:.2f}%")
    print_fn(f"REAL NODE PROPERTY ACCURACY: {real_prop_score:.2f}%")
    print_fn(f"REAL EDGE PROPERTY ACCURACY: {real_edge_prop_score:.2f}%")
    print_fn(f"OVERALL PERFORMANCE: {(real_node_score + real_edge_score + real_prop_score + real_edge_prop_score) / 4:.2f}%")
    print_fn("=" * 30)

    return {
        "real_node_accuracy": real_node_score,
        "real_edge_accuracy": real_edge_score,
        "real_node_property_accuracy": real_prop_score,
        "real_edge_property_accuracy": real_edge_prop_score,
        "overall_performance": (real_node_score + real_edge_score + real_prop_score + real_edge_prop_score) / 4,
        "node_matches": node_matches_map,
        "extra_nodes": extra_nodes,
        "extra_edges": total_extra
        
    }