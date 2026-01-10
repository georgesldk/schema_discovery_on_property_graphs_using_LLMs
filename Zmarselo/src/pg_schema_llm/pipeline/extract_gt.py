import json
import os
import glob
import re
import csv


def parse_pgs_file(file_path):
    """Parses .pgs to extract Node and Edge definitions with standardized types."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove comments
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    node_types = []
    edge_map = {}
    node_type_by_label_set = {}
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
        label_expr = label_expr.strip().strip('()')
        return [l.strip() for l in label_expr.split('&') if l.strip()]

    def get_primary_label(labels):
        for label in labels:
            if not re.match(r'^[a-z]', label) and '_' not in label:
                return label
        return labels[0] if labels else "Unknown"

    def parse_props(prop_block):
        props = []
        if not prop_block:
            return props

        for line in prop_block.strip().split('\n'):
            line = line.strip().rstrip(',')
            if not line:
                continue

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

    node_pattern = re.compile(
        r'CREATE\s+NODE\s+TYPE\s*\(\s*([a-zA-Z0-9_]+)\s*:\(([^)]+)\)\s*(?:\{([^}]*)\})?\s*\);',
        re.DOTALL | re.IGNORECASE
    )

    for match in node_pattern.finditer(content):
        type_name, label_expr, props_block = match.groups()
        labels = parse_labels(label_expr)
        primary_label = get_primary_label(labels)
        properties = parse_props(props_block) if props_block else []

        raw_type_string = ":".join(sorted(labels))
        node_type = {
            "type_name": type_name,
            "labels": sorted(labels),
            "raw_type_string": raw_type_string,
            "properties": properties
        }
        node_types.append(node_type)
        node_type_by_label_set[raw_type_string] = node_type

        for label in labels:
            label_to_primary_labels.setdefault(label, set()).add(primary_label)
        label_to_primary_labels[raw_type_string] = {primary_label}

    edge_pattern = re.compile(
        r'CREATE\s+EDGE\s+TYPE\s*\((.*?)\)\s*-\s*\[\s*([a-zA-Z0-9_]+)\s*:([a-zA-Z0-9_]+)(?:\s*\{([^}]*)\})?\s*\]\s*->\s*\((.*?)\)\s*;',
        re.DOTALL | re.IGNORECASE
    )

    for match in edge_pattern.finditer(content):
        source_expr, edge_type_name1, _, props_block, target_expr = match.groups()
        edge_type_name = edge_type_name1

        def extract_label_strings(expr):
            results = []
            for m in re.finditer(r'(?::)?\(([^)]+)\)', expr):
                labels = parse_labels(m.group(1))
                primary = get_primary_label(labels)
                results.append(f"{primary}:{':'.join(labels)}")
            return results

        source_labels = extract_label_strings(source_expr)
        target_labels = extract_label_strings(target_expr)
        properties = parse_props(props_block) if props_block else []

        if edge_type_name not in edge_map:
            edge_map[edge_type_name] = {
                "type": edge_type_name,
                "properties": properties,
                "topology": []
            }

        edge_map[edge_type_name]["topology"].append({
            "allowed_sources": source_labels,
            "allowed_targets": target_labels
        })

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


def run_extract_gt(input_dir, output_dir):
    schema = extract_golden_truth(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    dataset = schema["dataset_name"]
    out_path = os.path.join(output_dir, f"golden_truth_{dataset}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)

    return out_path
