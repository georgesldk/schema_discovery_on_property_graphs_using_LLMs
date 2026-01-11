import json
import os
from difflib import SequenceMatcher


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        raise


def similar(a, b):
    if not a or not b:
        return 0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_match(target, candidates, threshold=0.8):
    best = None
    best_score = 0
    for c in candidates:
        s = similar(target, c)
        if s > best_score:
            best_score = s
            best = c
    return best if best_score >= threshold else None


def compare_properties(gt_props, inf_props):
    gt_set = set(p["name"] for p in gt_props)
    inf_set = set(p["name"] for p in inf_props)

    matches = len(gt_set & inf_set)
    total = len(gt_set)
    return matches, total


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

    node_matches_map = {}
    used_inf_nodes = set()

    for gt_name in gt_node_names:
        match = find_best_match(gt_name, [n for n in inf_node_names if n not in used_inf_nodes])
        if match:
            node_matches_map[gt_name] = match
            used_inf_nodes.add(match)

    node_matches = len(node_matches_map)
    extra_nodes = [n for n in inf_node_names if n not in used_inf_nodes]

    print_fn("\n--- NODE TYPE MATCHING ---")
    print_fn(f"Matched Node Types: {node_matches} / {len(gt_node_names)}")
    print_fn(f"Extra Inferred Node Types: {len(extra_nodes)}")
    if extra_nodes:
        print_fn(f"    Extra: {extra_nodes[:5]}" + (f" ... and {len(extra_nodes) - 5} more" if len(extra_nodes) > 5 else ""))

    # --- EDGE MATCHING ---
    gt_edges = gt.get("edge_types", [])
    inf_edges = inf.get("edge_types", [])

    # Expand GT edges into (edge_type, source, target) combos
    gt_edge_combos = []
    for edge in gt_edges:
        e_name = edge.get("type") or edge.get("name")
        for topo in edge.get("topology", []):
            for s in topo.get("allowed_sources", []):
                for t in topo.get("allowed_targets", []):
                    gt_edge_combos.append((e_name, s, t))

    # Index inferred edges by name
    inf_edge_by_name = {}
    for e in inf_edges:
        inf_edge_by_name.setdefault(e.get("name"), []).append(e)

    # edge_matches = 0
    # used_inf_edge_names = set()
    # topology_misses = []

    # for gt_edge in gt_edges:
    #     gt_edge_name = gt_edge.get("type") or gt_edge.get("name")

    #     # find inferred edge label match (exact or fuzzy)
    #     inf_edge_name = gt_edge_name if gt_edge_name in inf_edge_by_name else find_best_match(gt_edge_name, list(inf_edge_by_name.keys()))
    #     if not inf_edge_name:
    #         continue

    #     # for each inferred edge with that name, check topology
    #     found = False
    #     for inf_edge in inf_edge_by_name.get(inf_edge_name, []):
    #         if check_edge_topology(gt_edge, inf_edge, node_matches_map):
    #             edge_matches += 1
    #             used_inf_edge_names.add(inf_edge_name)
    #             found = True
    #             break

    #     if not found:
    #         topology_misses.append(gt_edge_name)

    edge_matches = 0
    used_inf_edge_names = set()
    topology_misses = []

    # CHANGE: Iterate over every unique topology combo instead of just edge names (3 total)
    for gt_name, gt_s, gt_t in gt_edge_combos:
        
        # find inferred edge label match (exact or fuzzy)
        inf_edge_name = gt_name if gt_name in inf_edge_by_name else find_best_match(gt_name, list(inf_edge_by_name.keys()))
        
        if not inf_edge_name:
            topology_misses.append(f"{gt_name} (Label Missing)")
            continue

        # For this specific GT pair (e.g., Neuron -> Neuron), check if LLM produced a match
        found = False
        mapped_gt_s = node_matches_map.get(gt_s)
        mapped_gt_t = node_matches_map.get(gt_t)

        for inf_edge in inf_edge_by_name.get(inf_edge_name, []):
            inf_src = inf_edge.get("start_node")
            inf_tgt = inf_edge.get("end_node")

            if inf_src == mapped_gt_s and inf_tgt == mapped_gt_t:
                edge_matches += 1
                used_inf_edge_names.add(inf_edge_name)
                found = True
                break

        if not found:
            topology_misses.append(f"{gt_name}: {gt_s} -> {gt_t}")

    extra_edges = [e for e in inf_edge_by_name.keys() if e not in used_inf_edge_names]

    # total GT combos (for real score denominator)
    total_gt = len(gt_edge_combos)
    total_extra = len(extra_edges)

    print_fn("\n--- EDGE TYPE MATCHING ---")
    print_fn(f"Matched Edge Types: {edge_matches} / {len(gt_edges)}")
    print_fn(f"GT Total Topology Combos: {total_gt}")
    print_fn(f"Extra Inferred Edge Types: {len(extra_edges)}")
    if extra_edges:
        print_fn(f"    Extra: {extra_edges[:5]}" + (f" ... and {len(extra_edges) - 5} more" if len(extra_edges) > 5 else ""))

    if topology_misses:
        print_fn("\nTopology mismatches (edge name matched but start/end wrong):")
        for miss in topology_misses[:5]:
            print_fn(f"  - {miss}")
        if len(topology_misses) > 5:
            print_fn(f"    ... and {len(topology_misses) - 5} more")

    # --- PROPERTY MATCHING ---
    prop_matches = 0
    total_props = 0

    for gt_node in gt_nodes:
        gt_name = gt_node.get("name") or gt_node.get("labels", [""])[0]
        inf_name = node_matches_map.get(gt_name)
        if not inf_name:
            continue

        inf_node = next((n for n in inf_nodes if (n.get("name") or n.get("labels", [""])[0]) == inf_name), None)
        if not inf_node:
            continue

        m, t = compare_properties(gt_node.get("properties", []), inf_node.get("properties", []))
        prop_matches += m
        total_props += t

    # Edge properties: count only for matched edge names
    edge_prop_matches = 0
    total_edge_props = 0

    for gt_edge in gt_edges:
        gt_edge_name = gt_edge.get("type") or gt_edge.get("name")
        if gt_edge_name not in used_inf_edge_names:
            continue

        inf_edge = None
        # find inferred edge that matched by name
        for e in inf_edges:
            if e.get("name") == gt_edge_name:
                inf_edge = e
                break

        if not inf_edge:
            continue

        m, t = compare_properties(gt_edge.get("properties", []), inf_edge.get("properties", []))
        edge_prop_matches += m
        total_edge_props += t

    # FINAL REAL SCORES
    real_node_score = calculate_real_score(node_matches, len(gt_nodes), len(extra_nodes))
    real_edge_score = calculate_real_score(edge_matches, total_gt, total_extra)
    real_prop_score = (prop_matches / total_props * 100) if total_props > 0 else 0
    real_edge_prop_score = (edge_prop_matches / total_edge_props * 100) if total_edge_props > 0 else 0

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
        "edge_matches": list(used_inf_edge_names),
        "extra_nodes": extra_nodes,
        "extra_edges": extra_edges
    }
