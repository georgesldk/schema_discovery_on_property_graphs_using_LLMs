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

    # 2. Edges
    print("\n--- 2. Edge Types ---")
    gt_edges = {e.get('type') or e.get('name'): e for e in gt.get('edge_types', [])}
    inf_edges = {e.get('type') or e.get('name'): e for e in inf.get('edge_types', [])}
    
    edge_matches = 0
    for gt_name in gt_edges:
        match_name = find_best_match(gt_name, inf_edges.keys())
        if match_name:
            print(f" Match: Edge '{gt_name}' <--> '{match_name}'")
            edge_matches += 1
        else:
            print(f" Missed Edge: '{gt_name}'")

    extra_edges = [e for e in inf_edges if not find_best_match(e, gt_edges.keys())]
    if extra_edges:
        print(f" Extra Inferred Edges (Penalty): {', '.join(extra_edges)}")

    # 3. Property Accuracy
    print("\n--- 3. Property Accuracy ---")
    total_props, prop_matches = 0, 0
    for gt_name, inf_name in matches_map:
        tp, fn, fp = compare_properties(gt_nodes[gt_name].get('properties', []), inf_nodes[inf_name].get('properties', []))
        total_props += len(gt_nodes[gt_name].get('properties', []))
        prop_matches += len(tp)

    # FINAL REAL SCORES
    real_node_score = calculate_real_score(node_matches, len(gt_nodes), len(extra_nodes))
    real_edge_score = calculate_real_score(edge_matches, len(gt_edges), len(extra_edges))
    real_prop_score = (prop_matches / total_props * 100) if total_props > 0 else 0

    print("\n" + "="*30)
    print(f"REAL NODE ACCURACY: {real_node_score:.2f}%")
    print(f"REAL EDGE ACCURACY: {real_edge_score:.2f}%")
    print(f"REAL PROPERTY ACCURACY: {real_prop_score:.2f}%")
    print(f"OVERALL PERFORMANCE: {(real_node_score + real_edge_score + real_prop_score)/3:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()