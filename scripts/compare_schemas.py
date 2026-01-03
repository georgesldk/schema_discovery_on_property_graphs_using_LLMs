import json
import sys
import os
import argparse
from difflib import SequenceMatcher

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

def similar(a, b):
    if not a or not b: return 0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_match(name, target_list):
    best_score = 0
    best_match = None
    for target in target_list:
        score = similar(name, target)
        if score > 0.8:  # Fuzzy threshold
            if score > best_score:
                best_score = score
                best_match = target
    return best_match

def compare_properties(gt_props, inf_props):
    gt_names = {str(p['name']) for p in gt_props if p.get('name')}
    inf_names = {str(p['name']) for p in inf_props if p.get('name')}
    return gt_names.intersection(inf_names), gt_names - inf_names, inf_names - gt_names

def check_edge_topology(gt_edge, inf_edge, node_name_mapping):
    """
    Check if the inferred edge connects the correct nodes according to GT topology.
    
    Args:
        gt_edge: Ground truth edge with 'topology' field
        inf_edge: Inferred edge with 'start_node' and 'end_node' fields
        node_name_mapping: Dict mapping GT node names to inferred node names
    
    Returns:
        bool: True if topology matches, False otherwise
    """
    # Get inferred edge source and target nodes
    inf_start = inf_edge.get('start_node')
    inf_end = inf_edge.get('end_node')
    
    if not inf_start or not inf_end:
        return False
    
    # Get GT topology (can have multiple topology rules)
    gt_topology = gt_edge.get('topology', [])
    if not gt_topology:
        # If no topology specified in GT, assume it's valid
        return True
    
    # Reverse mapping: inferred node name -> GT node name
    reverse_mapping = {inf: gt for gt, inf in node_name_mapping.items()}
    
    # Get candidate GT node names for start and end
    # Try mapped name first, then direct match
    gt_start_candidates = []
    if inf_start in reverse_mapping:
        gt_start_candidates.append(reverse_mapping[inf_start])
    gt_start_candidates.append(inf_start)  # Also try direct match
    
    gt_end_candidates = []
    if inf_end in reverse_mapping:
        gt_end_candidates.append(reverse_mapping[inf_end])
    gt_end_candidates.append(inf_end)  # Also try direct match
    
    # Check if any topology rule allows this connection
    for topology_rule in gt_topology:
        allowed_sources = topology_rule.get('allowed_sources', [])
        allowed_targets = topology_rule.get('allowed_targets', [])
        
        # Check if any candidate start node matches allowed sources
        source_match = any(
            any(gt_start == allowed or similar(gt_start, allowed) > 0.8 for allowed in allowed_sources)
            for gt_start in gt_start_candidates if gt_start
        )
        
        # Check if any candidate end node matches allowed targets
        target_match = any(
            any(gt_end == allowed or similar(gt_end, allowed) > 0.8 for allowed in allowed_targets)
            for gt_end in gt_end_candidates if gt_end
        )
        
        if source_match and target_match:
            return True
    
    return False

def calculate_real_score(matches, total_gt, total_extra):
    """
    Calculates accuracy by penalizing for over-inference (extra items).
    True Score = Matches / (Actual GT Items + Extra False Positives)
    """
    denominator = total_gt + total_extra
    if denominator == 0: return 0
    return (matches / denominator) * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True, help="Path to Golden Truth JSON")
    parser.add_argument("--inferred_file", required=True, help="Path to Inferred Schema JSON")
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_file) or not os.path.exists(args.inferred_file):
        print(" Error: One or both input files do not exist.")
        return

    gt = load_json(args.gt_file)
    inf = load_json(args.inferred_file)
    
    print(f"\n==== REAL SCHEMA COMPARISON REPORT ====")
    print(f"GT: {os.path.basename(args.gt_file)}")
    print(f"Inferred: {os.path.basename(args.inferred_file)}")
    
    # 1. Node Types
    print("\n--- 1. Node Types ---")
    gt_nodes = {n.get('name') or n.get('labels')[0]: n for n in gt.get('node_types', [])}
    inf_nodes = {n.get('name') or n.get('labels', [''])[0]: n for n in inf.get('node_types', [])}
    
    node_matches = 0
    matches_map = []
    for gt_name in gt_nodes:
        match_name = find_best_match(gt_name, inf_nodes.keys())
        if match_name:
            print(f" Match: GT '{gt_name}' <--> Inferred '{match_name}'")
            node_matches += 1
            matches_map.append((gt_name, match_name))
        else:
            print(f" Missed: GT '{gt_name}' not found.")
            
    extra_nodes = [n for n in inf_nodes if not find_best_match(n, gt_nodes.keys())]
    if extra_nodes:
        print(f" Extra Inferred Nodes (Penalty): {', '.join(extra_nodes)}")

    # Create node name mapping dict for topology checking
    node_name_mapping = dict(matches_map)

    # 2. Edges - Match by topology (node type pairs), not just names
    print("\n--- 2. Edge Types (Topology-Based Matching) ---")
    gt_edges = {e.get('type') or e.get('name'): e for e in gt.get('edge_types', [])}
    inf_edges_list = inf.get('edge_types', [])
    
    # Build inferred edges index: (edge_name, start_node, end_node) -> edge
    inf_edge_map = {}
    for inf_edge in inf_edges_list:
        edge_name = inf_edge.get('type') or inf_edge.get('name')
        start_node = inf_edge.get('start_node')
        end_node = inf_edge.get('end_node')
        if edge_name and start_node and end_node:
            key = (edge_name, start_node, end_node)
            if key not in inf_edge_map:
                inf_edge_map[key] = []
            inf_edge_map[key].append(inf_edge)
    
    # Expand GT edges into topology combinations (source, target, edge_name)
    gt_topology_combinations = []
    gt_edge_name_to_combinations = {}
    
    for gt_edge_name, gt_edge in gt_edges.items():
        topology_rules = gt_edge.get('topology', [])
        combinations = []
        
        if topology_rules:
            for rule in topology_rules:
                sources = rule.get('allowed_sources', [])
                targets = rule.get('allowed_targets', [])
                for source in sources:
                    for target in targets:
                        combo = (source, target, gt_edge_name)
                        combinations.append(combo)
                        gt_topology_combinations.append(combo)
        else:
            # If no topology, we can't match by topology - skip for now
            pass
        
        gt_edge_name_to_combinations[gt_edge_name] = combinations
    
    # Match each GT topology combination to inferred edges
    topology_matches = 0
    topology_misses = []
    
    for gt_source, gt_target, gt_edge_name in gt_topology_combinations:
        # Map GT node names to inferred node names
        inf_source_candidates = []
        inf_target_candidates = []
        
        if gt_source in node_name_mapping:
            inf_source_candidates.append(node_name_mapping[gt_source])
        inf_source_candidates.append(gt_source)  # Also try direct match
        
        if gt_target in node_name_mapping:
            inf_target_candidates.append(node_name_mapping[gt_target])
        inf_target_candidates.append(gt_target)  # Also try direct match
        
        # Try to find matching inferred edge
        found_match = False
        for inf_source in inf_source_candidates:
            for inf_target in inf_target_candidates:
                # Try exact edge name match first
                if (gt_edge_name, inf_source, inf_target) in inf_edge_map:
                    found_match = True
                    break
                # Try fuzzy edge name match
                for inf_edge_name in set(key[0] for key in inf_edge_map.keys()):
                    if similar(gt_edge_name, inf_edge_name) > 0.8:
                        if (inf_edge_name, inf_source, inf_target) in inf_edge_map:
                            found_match = True
                            break
                if found_match:
                    break
            if found_match:
                break
        
        if found_match:
            topology_matches += 1
        else:
            topology_misses.append((gt_source, gt_target, gt_edge_name))
    
    # Count unique edge names in inferred that don't match GT
    matched_inf_edge_names = set()
    for gt_source, gt_target, gt_edge_name in gt_topology_combinations:
        for inf_source_cand in [node_name_mapping.get(gt_source, gt_source), gt_source]:
            for inf_target_cand in [node_name_mapping.get(gt_target, gt_target), gt_target]:
                for inf_edge_name in set(key[0] for key in inf_edge_map.keys()):
                    if similar(gt_edge_name, inf_edge_name) > 0.8:
                        if (inf_edge_name, inf_source_cand, inf_target_cand) in inf_edge_map:
                            matched_inf_edge_names.add(inf_edge_name)
    
    all_inf_edge_names = {e.get('type') or e.get('name') for e in inf_edges_list}
    extra_edge_names = all_inf_edge_names - matched_inf_edge_names
    
    total_gt_combinations = len(gt_topology_combinations)
    edge_match_rate = (topology_matches / total_gt_combinations * 100) if total_gt_combinations > 0 else 0
    
    print(f"  GT Topology Combinations: {total_gt_combinations}")
    print(f"  Matched Combinations: {topology_matches}")
    print(f"  Missed Combinations: {len(topology_misses)}")
    if topology_misses:
        for source, target, edge_name in topology_misses[:5]:  # Show first 5
            print(f"    Missing: {source} --[{edge_name}]--> {target}")
        if len(topology_misses) > 5:
            print(f"    ... and {len(topology_misses) - 5} more")
    if extra_edge_names:
        print(f"  Extra Inferred Edge Names: {', '.join(sorted(extra_edge_names))}")
    
    # Calculate edge score (penalize for missing combinations and extra edges)
    edge_matches = topology_matches
    total_gt = total_gt_combinations
    total_extra = len(extra_edge_names)

    # 2b. Edge Properties (only count for matched topology combinations)
    print("\n--- 2b. Edge Property Accuracy ---")
    edge_prop_matches = 0
    total_edge_props = 0
    
    # For each matched topology combination, compare properties
    matched_combinations = []
    for gt_source, gt_target, gt_edge_name in gt_topology_combinations:
        inf_source_candidates = [node_name_mapping.get(gt_source, gt_source), gt_source]
        inf_target_candidates = [node_name_mapping.get(gt_target, gt_target), gt_target]
        
        for inf_source in inf_source_candidates:
            for inf_target in inf_target_candidates:
                for inf_edge_name in set(key[0] for key in inf_edge_map.keys()):
                    if similar(gt_edge_name, inf_edge_name) > 0.8:
                        if (inf_edge_name, inf_source, inf_target) in inf_edge_map:
                            matched_combinations.append((gt_source, gt_target, gt_edge_name, inf_edge_name, inf_source, inf_target))
                            break
                if matched_combinations and matched_combinations[-1][:3] == (gt_source, gt_target, gt_edge_name):
                    break
            if matched_combinations and matched_combinations[-1][:3] == (gt_source, gt_target, gt_edge_name):
                break
    
    # Get properties for each matched combination
    for gt_source, gt_target, gt_edge_name, inf_edge_name, inf_source, inf_target in matched_combinations:
        gt_edge = gt_edges[gt_edge_name]
        inf_edge = inf_edge_map[(inf_edge_name, inf_source, inf_target)][0]
        
        gt_edge_props = gt_edge.get('properties', [])
        inf_edge_props = inf_edge.get('properties', [])
        tp, fn, fp = compare_properties(gt_edge_props, inf_edge_props)
        total_edge_props += len(gt_edge_props)
        edge_prop_matches += len(tp)
    
    if total_edge_props > 0:
        print(f"  Properties matched: {edge_prop_matches}/{total_edge_props} across all matched edge combinations")

    # 3. Node Property Accuracy
    print("\n--- 3. Node Property Accuracy ---")
    total_props, prop_matches = 0, 0
    for gt_name, inf_name in matches_map:
        tp, fn, fp = compare_properties(gt_nodes[gt_name].get('properties', []), inf_nodes[inf_name].get('properties', []))
        total_props += len(gt_nodes[gt_name].get('properties', []))
        prop_matches += len(tp)

    # FINAL REAL SCORES
    real_node_score = calculate_real_score(node_matches, len(gt_nodes), len(extra_nodes))
    real_edge_score = calculate_real_score(edge_matches, total_gt, total_extra)
    real_prop_score = (prop_matches / total_props * 100) if total_props > 0 else 0
    real_edge_prop_score = (edge_prop_matches / total_edge_props * 100) if total_edge_props > 0 else 0

    print("\n" + "="*30)
    print(f"REAL NODE ACCURACY: {real_node_score:.2f}%")
    print(f"REAL EDGE ACCURACY: {real_edge_score:.2f}%")
    print(f"REAL NODE PROPERTY ACCURACY: {real_prop_score:.2f}%")
    print(f"REAL EDGE PROPERTY ACCURACY: {real_edge_prop_score:.2f}%")
    print(f"OVERALL PERFORMANCE: {(real_node_score + real_edge_score + real_prop_score + real_edge_prop_score)/4:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()