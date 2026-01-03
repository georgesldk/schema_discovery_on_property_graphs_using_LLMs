import json
import os
import glob
import argparse
import re
import csv

def parse_pgs_file(file_path):
    """Parses .pgs to extract Node and Edge definitions with standardized types."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    node_pattern = re.compile(r"NODE\s+([a-zA-Z0-9_]+)\s*\{([^}]*)\}", re.DOTALL)
    edge_pattern = re.compile(r"EDGE\s+([a-zA-Z0-9_]+)\s*\{([^}]*)\}", re.DOTALL)

    node_types = []
    edge_map = {}

    def standardize_type(ptype):
        ptype = ptype.lower().strip().strip(',')
        if "int" in ptype: return "Long"
        if "float" in ptype or "double" in ptype: return "Double"
        if "bool" in ptype: return "Boolean"
        if "datetime" in ptype or "string" in ptype: return "String"
        return "String"

    def parse_props(prop_block):
        props = []
        for line in prop_block.strip().split('\n'):
            if ':' in line:
                name, ptype = line.split(':', 1)
                props.append({
                    "name": name.strip().replace('`', ''),
                    "type": standardize_type(ptype),
                    "mandatory": True
                })
        return props

    for match in node_pattern.finditer(content):
        name, props_raw = match.groups()
        node_types.append({
            "name": name,
            "labels": [name],
            "properties": parse_props(props_raw)
        })

    for match in edge_pattern.finditer(content):
        name, props_raw = match.groups()
        edge_map[name] = {
            "type": name,
            "properties": parse_props(props_raw),
            "topology": []
        }

    return node_types, edge_map

def add_topology_from_csv(csv_path, edge_map):
    """Enriches edge definitions with allowed source/target connections from CSV."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_type = row.get('relType', '').strip()
            if rel_type in edge_map:
                sources = [s.strip() for s in row.get('sources', '').strip("[]").split(',') if s.strip()]
                targets = [t.strip() for t in row.get('targets', '').strip("[]").split(',') if t.strip()]
                edge_map[rel_type]["topology"].append({
                    "allowed_sources": sources,
                    "allowed_targets": targets
                })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing .pgs and edge_types.csv")
    parser.add_argument("--output_dir", required=True, help="Folder to save the result")
    args = parser.parse_args()
    
    dataset_name = os.path.basename(os.path.normpath(args.input_dir))
    output_file = os.path.join(args.output_dir, f"golden_truth_{dataset_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)

    pgs_files = glob.glob(os.path.join(args.input_dir, "*.pgs"))
    csv_topology = glob.glob(os.path.join(args.input_dir, "*edge_types.csv"))

    if not pgs_files:
        print(f"Error: No .pgs file found in {args.input_dir}")
        return

    print(f"Parsing {os.path.basename(pgs_files[0])}...")
    nodes, edge_map = parse_pgs_file(pgs_files[0])

    if csv_topology:
        print(f"Adding topology from {os.path.basename(csv_topology[0])}...")
        add_topology_from_csv(csv_topology[0], edge_map)

    final_schema = {
        "dataset_name": dataset_name,
        "node_types": nodes,
        "edge_types": list(edge_map.values())
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_schema, f, indent=4)
    print(f"Success! Golden Truth saved to: {output_file}")

if __name__ == "__main__":
    main()