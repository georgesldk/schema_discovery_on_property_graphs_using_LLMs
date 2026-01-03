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

    # Remove comments and normalize whitespace
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    node_types = []
    edge_map = {}
    # Maps label sets (raw_type_string) to node types for edge topology normalization
    node_type_by_label_set = {}
    # Maps any label to its possible primary labels (for normalization)
    label_to_primary_labels = {}

    def standardize_type(ptype):
        ptype = ptype.lower().strip().strip(',')
        if "point" in ptype: return "Point"
        if "int" in ptype or "long" in ptype: return "Long"
        if "float" in ptype or "double" in ptype: return "Double"
        if "bool" in ptype: return "Boolean"
        if "array" in ptype:
            if "string" in ptype: return "StringArray"
            return "Array"
        if "datetime" in ptype or "date" in ptype: return "String"
        if "string" in ptype: return "String"
        return "String"

    def parse_labels(label_expr):
        """Parse label expression like '(Meta & medulla7column_Meta)' into list of labels."""
        # Remove outer parentheses and split by &
        label_expr = label_expr.strip().strip('()')
        labels = [l.strip() for l in label_expr.split('&') if l.strip()]
        return labels

    def get_primary_label(labels):
        """Get the primary (non-namespaced) label from a label set."""
        # Primary label is typically the first one that doesn't look like a namespace
        # Namespace labels often have underscores or lowercase prefixes
        for label in labels:
            # Skip labels that look like namespaces (e.g., medulla7column_Meta)
            if not re.match(r'^[a-z]', label) and '_' not in label:
                return label
        # If all look namespaced, return the first one
        return labels[0] if labels else "Unknown"

    def parse_props(prop_block):
        """Parse property block, handling OPTIONAL keywords.
        
        Format: propertyName Type, or OPTIONAL propertyName Type,
        """
        props = []
        if not prop_block or not prop_block.strip():
            return props
        
        for line in prop_block.strip().split('\n'):
            line = line.strip().rstrip(',')
            if not line:
                continue
            
            # Check for OPTIONAL keyword
            is_optional = 'OPTIONAL' in line.upper()
            line = re.sub(r'\bOPTIONAL\b', '', line, flags=re.IGNORECASE).strip()
            
            # Parse format: name Type (space-separated, type is last word)
            # Handle types like "String", "StringArray", "Point", "Long", etc.
            parts = line.split()
            if len(parts) >= 2:
                # Last part is the type, everything before is the name
                ptype = parts[-1]
                name = ' '.join(parts[:-1])
                
                # Skip empty names
                if name:
                    props.append({
                        "name": name,
                        "type": standardize_type(ptype),
                        "mandatory": not is_optional
                    })
        return props

    # Parse NODE TYPE definitions
    # Pattern: CREATE NODE TYPE ( TypeName :(Label1 & Label2 & ...) { properties } );
    # Or: CREATE NODE TYPE ( TypeName :(Label1 & Label2 & ...) );
    node_pattern = re.compile(
        r'CREATE\s+NODE\s+TYPE\s*\(\s*([a-zA-Z0-9_]+)\s*:\(([^)]+)\)\s*(?:\{([^}]*)\})?\s*\);',
        re.DOTALL | re.IGNORECASE
    )
    
    for match in node_pattern.finditer(content):
        type_name, label_expr, props_block = match.groups()
        labels = parse_labels(label_expr)
        primary_label = get_primary_label(labels)
        properties = parse_props(props_block) if props_block else []
        
        # Create raw_type_string as unique identifier (sorted labels)
        raw_type_string = ":".join(sorted(labels))
        
        # Keep ALL distinct node types - don't merge by primary label
        # Each node type definition in the .pgs file is preserved as a separate entry
        node_type = {
            "type_name": type_name,  # Store the type name from .pgs file
            "labels": sorted(labels),
            "raw_type_string": raw_type_string,
            "properties": properties
        }
        node_types.append(node_type)
        node_type_by_label_set[raw_type_string] = node_type
        
        # Build mapping for edge topology normalization
        # Map each individual label to the primary label(s) it can represent
        for label in labels:
            if label not in label_to_primary_labels:
                label_to_primary_labels[label] = set()
            label_to_primary_labels[label].add(primary_label)
        # Also map the full label string to primary label
        label_to_primary_labels[raw_type_string] = {primary_label}

    # Parse EDGE TYPE definitions
    # Pattern: CREATE EDGE TYPE ( :(Label1 & Label2) | ... ) - [ EDGE_NAME :EDGE_NAME { props } ] -> ( :(Label1) | ... );
    # Or: CREATE EDGE TYPE ( :(Label1) ) - [ EDGE_NAME :EDGE_NAME ] -> ( :(Label2) );
    edge_pattern = re.compile(
        r'CREATE\s+EDGE\s+TYPE\s*\((.*?)\)\s*-\s*\[\s*([a-zA-Z0-9_]+)\s*:([a-zA-Z0-9_]+)(?:\s*\{([^}]*)\})?\s*\]\s*->\s*\((.*?)\)\s*;',
        re.DOTALL | re.IGNORECASE
    )
    
    for match in edge_pattern.finditer(content):
        source_expr, edge_type_name1, edge_type_name2, props_block, target_expr = match.groups()
        edge_type_name = edge_type_name1  # Use the first occurrence
        
        # Parse source label sets
        source_labels = []
        for label_block in re.findall(r':\(([^)]+)\)', source_expr):
            labels = parse_labels(f'({label_block})')
            source_labels.extend(labels)
        
        # Parse target label sets  
        target_labels = []
        for label_block in re.findall(r':\(([^)]+)\)', target_expr):
            labels = parse_labels(f'({label_block})')
            target_labels.extend(labels)
        
        properties = parse_props(props_block) if props_block else []
        
        # Create normalized source/target label strings for topology
        # Handle both :(Label & ...) and (Label & ...) patterns
        source_label_strings = []
        # Find all label blocks (with or without leading colon)
        for match in re.finditer(r'(?::)?\(([^)]+)\)', source_expr):
            label_block = match.group(1)
            labels = parse_labels(f'({label_block})')
            primary = get_primary_label(labels)
            # Create label string in format: Primary:all:labels
            label_str = f"{primary}:{':'.join(labels)}"
            source_label_strings.append(label_str)
        
        target_label_strings = []
        for match in re.finditer(r'(?::)?\(([^)]+)\)', target_expr):
            label_block = match.group(1)
            labels = parse_labels(f'({label_block})')
            primary = get_primary_label(labels)
            label_str = f"{primary}:{':'.join(labels)}"
            target_label_strings.append(label_str)
        
        if edge_type_name not in edge_map:
            edge_map[edge_type_name] = {
                "type": edge_type_name,
                "properties": properties,
                "topology": []
            }
        
        edge_map[edge_type_name]["topology"].append({
            "allowed_sources": source_label_strings,
            "allowed_targets": target_label_strings
        })

    return node_types, edge_map, label_to_primary_labels

def normalize_topology(edge_map, label_to_primary_labels):
    """Normalize edge topology to use primary labels only.
    
    Since multiple node types can share the same primary label (e.g., both 
    NeuronSegmentMedullaNeuronType and NeuronSegmentMedullaSegmentType have 
    'Neuron' as primary), we normalize edges to primary labels for visualization,
    while preserving all distinct node types in the schema.
    """
    # Normalize each edge's topology
    for edge_type_name, edge_def in edge_map.items():
        normalized_sources = set()
        normalized_targets = set()
        
        for topology_entry in edge_def["topology"]:
            for source_label_str in topology_entry.get("allowed_sources", []):
                # Extract primary label from label string (format: Primary:all:labels)
                if ':' in source_label_str:
                    # Format is "Primary:label1:label2:..."
                    primary = source_label_str.split(':')[0]
                    normalized_sources.add(primary)
                else:
                    # Try to find primary label(s) from mapping
                    primary_labels = label_to_primary_labels.get(source_label_str, {source_label_str})
                    normalized_sources.update(primary_labels)
            
            for target_label_str in topology_entry.get("allowed_targets", []):
                if ':' in target_label_str:
                    # Format is "Primary:label1:label2:..."
                    primary = target_label_str.split(':')[0]
                    normalized_targets.add(primary)
                else:
                    # Try to find primary label(s) from mapping
                    primary_labels = label_to_primary_labels.get(target_label_str, {target_label_str})
                    normalized_targets.update(primary_labels)
        
        # Replace topology with normalized version (deduplicated)
        if normalized_sources and normalized_targets:
            edge_def["topology"] = [{
                "allowed_sources": sorted(list(normalized_sources)),
                "allowed_targets": sorted(list(normalized_targets))
            }]

def add_topology_from_csv(csv_path, edge_map):
    """Enriches edge definitions with allowed source/target connections from CSV (if needed)."""
    if not os.path.exists(csv_path):
        return
    
    # If topology is already populated from PGS, CSV is supplementary
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_type = row.get('relType', '').strip()
            if rel_type not in edge_map:
                continue
                
            # Only add from CSV if topology is empty
            if not edge_map[rel_type].get("topology"):
                sources_str = row.get('sources', '').strip("[]\"'")
                targets_str = row.get('targets', '').strip("[]\"'")
                
                # Parse CSV format: "[Label1:ns1, Label2:ns2]"
                sources = []
                for s in re.findall(r'([^,\[\]]+)', sources_str):
                    s = s.strip().strip('"\'')
                    if s:
                        sources.append(s)
                
                targets = []
                for t in re.findall(r'([^,\[\]]+)', targets_str):
                    t = t.strip().strip('"\'')
                    if t:
                        targets.append(t)
                
                if sources and targets:
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
    nodes, edge_map, label_to_primary_labels = parse_pgs_file(pgs_files[0])

    # Normalize topology to use primary labels (for visualization)
    # This preserves all distinct node types while simplifying edge topology
    normalize_topology(edge_map, label_to_primary_labels)

    if csv_topology:
        print(f"Adding topology from {os.path.basename(csv_topology[0])} (if needed)...")
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