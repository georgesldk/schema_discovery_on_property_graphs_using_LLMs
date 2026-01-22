import json
import sys

def inspect(path):
    print(f"--- INSPECTING: {path} ---")
    with open(path, 'r') as f:
        data = json.load(f)
    
    # 1. Print Top-Level Keys
    print(f"Top-Level Keys: {list(data.keys())}")
    
    # 2. Search for Edges
    edge_key_candidates = ['edge_types', 'relationships', 'edges', 'relationship_types']
    found_key = None
    for k in edge_key_candidates:
        if k in data:
            found_key = k
            break
            
    if found_key:
        print(f"FOUND EDGE LIST UNDER KEY: '{found_key}'")
        edges = data[found_key]
        print(f"Total Edges Found: {len(edges)}")
        if len(edges) > 0:
            print("SAMPLE EDGE OBJECT:")
            print(json.dumps(edges[0], indent=2))
    else:
        print("!! CRITICAL ERROR: Could not find any key containing edge definitions !!")
        print("Please check if the file is empty or formatted correctly.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_gt.py <path_to_gt.json>")
    else:
        inspect(sys.argv[1])