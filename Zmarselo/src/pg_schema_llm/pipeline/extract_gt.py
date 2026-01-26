import json
import os
import glob
import re
import csv

# ==========================================
# 1. CLEANING & STANDARDIZATION
# ==========================================

def clean_comments(content):
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

# ==========================================
# 2. INTELLIGENT NAMING
# ==========================================

def derive_node_name(type_name, labels):
    tn = type_name.lower()
    
    # Priority 1: Check Variable Name for distinct keywords
    if "segment" in tn and "neuron" in tn:
        if "segmenttype" in tn or "_segment" in tn: return "Segment"
        if "neurontype" in tn or "_neuron" in tn: return "Neuron"
            
    # Priority 2: Single keywords
    if "segment" in tn: return "Segment"
    if "neuron" in tn: return "Neuron"
    if "synapseset" in tn: return "SynapseSet"
    if "synapse" in tn: return "Synapse"
    if "meta" in tn: return "Meta"

    # Priority 3: First label
    if labels: return labels[0]
    return type_name

# ==========================================
# 3. ROBUST PARSING LOGIC
# ==========================================

def parse_pgs_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = clean_comments(content)
    
    node_types = []
    # Store raw definitions to help edge matching later
    # Format: { "NodeName": set([label1, label2, ...]) }
    node_label_map = {} 

    edge_definitions = [] # Temporary storage for edges before resolving

    statements = content.split(';')

    for stmt in statements:
        stmt = stmt.strip()
        if not stmt: continue

        # --- NODE PARSING ---
        if "CREATE NODE TYPE" in stmt.upper():
            start = stmt.find('(')
            end = stmt.rfind(')')
            if start == -1 or end == -1: continue
            
            body = stmt[start+1 : end].strip()
            
            # Extract Props
            prop_str = ""
            p_start = body.find('{')
            p_end = body.rfind('}')
            if p_start != -1 and p_end != -1:
                prop_str = body[p_start+1 : p_end]
                def_str = body[:p_start].strip()
            else:
                def_str = body.strip()

            if ':' in def_str:
                parts = def_str.split(':', 1)
                type_name = parts[0].strip()
                label_part = parts[1].strip()
                
                label_part = label_part.replace('(', '').replace(')', '')
                labels = [l.strip() for l in label_part.split('&') if l.strip()]
                
                clean_name = derive_node_name(type_name, labels)

                node_types.append({
                    "name": clean_name,
                    "type_name": type_name,
                    "labels": sorted(labels),
                    "properties": parse_props_block(prop_str)
                })
                
                # Register labels for this node type
                node_label_map[clean_name] = set(labels)

        # --- EDGE PARSING ---
        elif "CREATE EDGE TYPE" in stmt.upper():
            try:
                arrow_match = re.search(r'-\s*\[(.*?)\]\s*->', stmt, re.DOTALL)
                if not arrow_match: continue
                
                definition = arrow_match.group(1).strip()
                
                e_props = []
                if '{' in definition:
                    p_start = definition.find('{')
                    p_end = definition.rfind('}')
                    e_props = parse_props_block(definition[p_start+1 : p_end])
                    definition = definition[:p_start].strip()
                
                e_name = "UNKNOWN"
                if ':' in definition:
                    parts = definition.split(':')
                    e_name = parts[0].strip() if parts[0].strip() else parts[-1].strip()

                arrow_start = arrow_match.start()
                arrow_end = arrow_match.end()
                
                src_block = stmt[:arrow_start].split('(', 1)[1].strip()
                tgt_block = stmt[arrow_end:].strip()
                if tgt_block.endswith(')'): tgt_block = tgt_block[:-1]
                if tgt_block.startswith('('): tgt_block = tgt_block[1:]

                edge_definitions.append({
                    "name": e_name,
                    "properties": e_props,
                    "raw_source": src_block,
                    "raw_target": tgt_block
                })
            except: continue

    return node_types, edge_definitions, node_label_map

# ==========================================
# 4. TOPOLOGY RESOLUTION (THE FIX)
# ==========================================

def resolve_node_type(label_expression, node_label_map):
    """
    Matches an edge's label expression (e.g., 'Neuron & Segment & medulla7column_Segment')
    to the correct Node Type Name (e.g., 'Segment').
    """
    clean_expr = label_expression.replace('(', '').replace(')', '')
    if ':' in clean_expr: clean_expr = clean_expr.split(':', 1)[1]
    
    required_labels = set([l.strip() for l in clean_expr.split('&') if l.strip()])
    
    best_match = None
    best_overlap = 0

    for node_name, node_labels in node_label_map.items():
        # Check if the Node satisfies ALL required labels in the expression
        if required_labels.issubset(node_labels):
            # Prefer the most specific match (highest number of shared labels)
            if len(required_labels) > best_overlap:
                best_match = node_name
                best_overlap = len(required_labels)
    
    return best_match

def build_edge_map(edge_definitions, node_label_map):
    edge_map = {}

    for e_def in edge_definitions:
        name = e_def["name"]
        if name not in edge_map:
            edge_map[name] = {
                "name": name,
                "properties": e_def["properties"],
                "topology": []
            }

        # Resolve Sources
        sources = set()
        for opt in e_def["raw_source"].split('|'):
            resolved = resolve_node_type(opt, node_label_map)
            if resolved: sources.add(resolved)
        
        # Resolve Targets
        targets = set()
        for opt in e_def["raw_target"].split('|'):
            resolved = resolve_node_type(opt, node_label_map)
            if resolved: targets.add(resolved)
            
        if sources and targets:
            edge_map[name]["topology"].append({
                "allowed_sources": sorted(list(sources)),
                "allowed_targets": sorted(list(targets))
            })

    # Merge Topologies
    for e_name, e_data in edge_map.items():
        all_s = set()
        all_t = set()
        for topo in e_data["topology"]:
            all_s.update(topo["allowed_sources"])
            all_t.update(topo["allowed_targets"])
        
        e_data["topology"] = [{
            "allowed_sources": sorted(list(all_s)),
            "allowed_targets": sorted(list(all_t))
        }]
        
    return list(edge_map.values())

def add_topology_from_csv(csv_path, edge_types):
    if not os.path.exists(csv_path): return

    # Convert list to dict for easier update
    edge_map = {e["name"]: e for e in edge_types}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_type = row.get('relType', '').strip()
            if rel_type not in edge_map: continue
            
            # If topology exists from PGS, skip CSV (trust PGS more)
            if edge_map[rel_type]["topology"][0]["allowed_sources"]: continue

            sources = re.findall(r'([^,\[\]]+)', row.get('sources', ''))
            targets = re.findall(r'([^,\[\]]+)', row.get('targets', ''))

            if sources and targets:
                edge_map[rel_type]["topology"] = [{
                    "allowed_sources": [s.strip() for s in sources],
                    "allowed_targets": [t.strip() for t in targets]
                }]

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def extract_ground_truth(input_dir):
    pgs_files = glob.glob(os.path.join(input_dir, "*.pgs"))
    csv_topology = glob.glob(os.path.join(input_dir, "*edge_types.csv"))

    if not pgs_files:
        raise FileNotFoundError("No .pgs file found")

    node_types, edge_defs, node_label_map = parse_pgs_file(pgs_files[0])
    edge_types = build_edge_map(edge_defs, node_label_map)

    if csv_topology:
        add_topology_from_csv(csv_topology[0], edge_types)

    return {
        "dataset_name": os.path.basename(os.path.normpath(input_dir)),
        "node_types": node_types,
        "edge_types": edge_types
    }

def run_extract_gt(input_dir, output_file):
    schema = extract_ground_truth(input_dir)
    
    parent_dir = os.path.dirname(output_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)

    return output_file