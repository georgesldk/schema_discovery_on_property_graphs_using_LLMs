import json
import os
import glob
import re
import csv


# ==========================================
# PASTE THIS BLOCK TO REPLACE THE PARSING LOGIC
# ==========================================

def clean_comments(content):
    """Removes //, --, and /* */ comments."""
    content = re.sub(r'--.*?$', '', content, flags=re.MULTILINE)
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    return content

def standardize_type(ptype):
    ptype = ptype.lower().strip().strip(',')
    if "point" in ptype: return "Point"
    if "int" in ptype or "long" in ptype: return "Long"
    if "float" in ptype or "double" in ptype: return "Double"
    if "bool" in ptype: return "Boolean"
    if "array" in ptype: return "StringArray" if "string" in ptype else "Array"
    return "String"

def parse_props_block(prop_str):
    """Parses properties inside a { ... } block."""
    props = []
    if not prop_str or not prop_str.strip(): return props
    
    for line in prop_str.split('\n'):
        line = line.strip().rstrip(',')
        if not line: continue
        
        is_optional = 'OPTIONAL' in line.upper()
        line = re.sub(r'\bOPTIONAL\b', '', line, flags=re.IGNORECASE).strip()
        
        parts = line.split()
        if len(parts) >= 2:
            ptype = parts[-1]
            name = ' '.join(parts[:-1])
            if name:
                props.append({
                    "name": name,
                    "type": standardize_type(ptype),
                    "mandatory": not is_optional
                })
    return props

def parse_pgs_file(file_path):
    """Robust Parser: Splits by Semicolon ';' to handle all formatting variations."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = clean_comments(content)
    
    node_types = []
    edge_map = {}
    label_to_primary_labels = {}

    # 1. Split by Statement Separator (;) - This fixes the formatting issues
    statements = content.split(';')

    for stmt in statements:
        stmt = stmt.strip()
        if not stmt: continue

        # === NODE PARSING ===
        if "CREATE NODE TYPE" in stmt.upper():
            # Extract content between outer parens: ( Name : Label { ... } )
            start = stmt.find('(')
            end = stmt.rfind(')')
            if start == -1 or end == -1: continue
            
            body = stmt[start+1 : end].strip()
            
            # Extract Property Block { ... }
            prop_str = ""
            p_start = body.find('{')
            p_end = body.rfind('}')
            
            if p_start != -1 and p_end != -1:
                prop_str = body[p_start+1 : p_end]
                def_str = body[:p_start].strip()
            else:
                def_str = body.strip()

            # Parse "Name : Label" or "Name : (Label)"
            if ':' in def_str:
                parts = def_str.split(':', 1)
                type_name = parts[0].strip()
                label_part = parts[1].strip()
                
                # Clean Labels (Remove parens, split by &)
                label_part = label_part.replace('(', '').replace(')', '')
                labels = [l.strip() for l in label_part.split('&') if l.strip()]
                
                node_types.append({
                    "type_name": type_name,
                    "labels": sorted(labels),
                    "properties": parse_props_block(prop_str)
                })

                if labels:
                    primary = labels[0]
                    for l in labels:
                        label_to_primary_labels.setdefault(l, set()).add(primary)

        # === EDGE PARSING ===
        elif "CREATE EDGE TYPE" in stmt.upper():
            try:
                # Regex to find the Arrow Block: -[ ... ]->
                arrow_match = re.search(r'-\s*\[(.*?)\]\s*->', stmt, re.DOTALL)
                if not arrow_match: continue
                
                definition = arrow_match.group(1).strip()
                
                # Handle Props in Edge Definition
                e_props = []
                if '{' in definition:
                    p_start = definition.find('{')
                    p_end = definition.rfind('}')
                    e_props = parse_props_block(definition[p_start+1 : p_end])
                    definition = definition[:p_start].strip() # Clean def
                
                # Parse Name/Type (Handle "Name : Type" OR ": Type")
                e_name = "UNKNOWN"
                if ':' in definition:
                    parts = definition.split(':')
                    e_type = parts[-1].strip()
                    e_name = parts[0].strip() if parts[0].strip() else e_type
                else:
                    continue 

                # Extract Source/Target (Pre and Post Arrow)
                arrow_start = arrow_match.start()
                arrow_end = arrow_match.end()
                
                # Source: After "TYPE (" and before Arrow
                src_block = stmt[:arrow_start].split('(', 1)[1].strip()
                
                # Target: After Arrow, remove trailing ')'
                tgt_block = stmt[arrow_end:].strip()
                if tgt_block.endswith(')'): tgt_block = tgt_block[:-1]
                if tgt_block.startswith('('): tgt_block = tgt_block[1:]

                def extract_labels(expr):
                    results = []
                    for opt in expr.split('|'):
                        clean = opt.strip().replace('(', '').replace(')', '')
                        if ':' in clean: clean = clean.split(':', 1)[1]
                        lbs = [l.strip() for l in clean.split('&') if l.strip()]
                        if lbs: results.append(f"{lbs[0]}:{':'.join(lbs)}")
                    return results

                if e_name not in edge_map:
                    edge_map[e_name] = {"name": e_name, "type": e_type, "properties": e_props, "topology": []}
                
                edge_map[e_name]["topology"].append({
                    "allowed_sources": extract_labels(src_block),
                    "allowed_targets": extract_labels(tgt_block)
                })
            except: continue

    return node_types, edge_map, label_to_primary_labels

def normalize_topology(edge_map, label_to_primary_labels):
    for edge_def in edge_map.values():
        sources, targets = set(), set()

        for topo in edge_def["topology"]:
            for s in topo.get("allowed_sources", []):
                sources.add(s.split(":")[0])
            for t in topo.get("allowed_targets", []):
                targets.add(t.split(":")[0])

        if sources and targets:
            edge_def["topology"] = [{
                "allowed_sources": sorted(sources),
                "allowed_targets": sorted(targets)
            }]


def add_topology_from_csv(csv_path, edge_map):
    if not os.path.exists(csv_path):
        return

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_type = row.get('relType', '').strip()
            if rel_type not in edge_map:
                continue
            if edge_map[rel_type].get("topology"):
                continue

            sources = re.findall(r'([^,\[\]]+)', row.get('sources', ''))
            targets = re.findall(r'([^,\[\]]+)', row.get('targets', ''))

            if sources and targets:
                edge_map[rel_type]["topology"].append({
                    "allowed_sources": [s.strip() for s in sources],
                    "allowed_targets": [t.strip() for t in targets]
                })


def extract_golden_truth(input_dir):
    pgs_files = glob.glob(os.path.join(input_dir, "*.pgs"))
    csv_topology = glob.glob(os.path.join(input_dir, "*edge_types.csv"))

    if not pgs_files:
        raise FileNotFoundError("No .pgs file found")

    nodes, edge_map, label_map = parse_pgs_file(pgs_files[0])
    normalize_topology(edge_map, label_map)

    if csv_topology:
        add_topology_from_csv(csv_topology[0], edge_map)

    return {
        "dataset_name": os.path.basename(os.path.normpath(input_dir)),
        "node_types": nodes,
        "edge_types": list(edge_map.values())
    }


def run_extract_gt(input_dir, output_file):
    schema = extract_golden_truth(input_dir)
    
    # FIX: Get the folder from the file path, don't use the filename as a folder
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # FIX: Use the 'output_file' passed from the pipeline directly
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)

    return output_file

