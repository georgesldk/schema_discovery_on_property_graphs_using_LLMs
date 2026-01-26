import os
import json
import random
import networkx as nx
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter
from pg_schema_llm.llm import build_inference_prompt

# Keep compatibility during restructure:
# - after refactor you will import from pg_schema_llm.io.graph_builder
# - for now it can still work if old build_graph.py exists on PYTHONPATH
try:
    from pg_schema_llm.io.graph_builder import build_graph
except Exception:
    from build_graph import build_graph


def infer_schema_from_folder(data_dir):
    G = build_graph(data_dir)
    node_types = sorted(list(set(nx.get_node_attributes(G, 'node_type').values())))
    edge_types = sorted(list(set(nx.get_edge_attributes(G, 'type').values())))

    # 1. Standard Profiling
    profile_text = "".join([profile_node_type(G, nt) for nt in node_types])
    profile_text += "".join([profile_edge_type(G, et) for et in edge_types])

    # 2. Structural Fingerprinting (The Agnostic Logic)
    topology_summary = "\n  [STRUCTURAL FINGERPRINTS (Connectivity Analysis)]\n"
    for nt in node_types:
        nodes = [n for n, a in G.nodes(data=True) if a['node_type'] == nt]
        if not nodes: continue
        
        sample = nodes[:50]
        outgoing = set()
        incoming = set()
        
        for n in sample:
            for succ in G.successors(n):
                outgoing.add(G.nodes[succ].get('node_type', 'UNKNOWN'))
            for pred in G.predecessors(n):
                incoming.add(G.nodes[pred].get('node_type', 'UNKNOWN'))
        
        # Calculate Density: How many properties does this node actually have?
        # A "Tag" node usually has 0-1 meaningful properties (excluding ID).
        avg_prop_count = sum(len([k for k in G.nodes[n].keys() if k != 'node_type']) for n in sample) / len(sample)
        
        topology_summary += f"    - Node '{nt}':\n"
        topology_summary += f"      * Avg Properties per Node: {avg_prop_count:.1f}\n"
        topology_summary += f"      * Connection Role: {'Source/Hub' if len(outgoing) > len(incoming) else 'Sink/Leaf'}\n"

    profile_text += topology_summary

    # 3. Logical Summary
    logical_summary = generate_logical_relationship_summary(G)
    if logical_summary:
        profile_text += logical_summary

    # 4. Generate Prompt
    prompt = build_inference_prompt(profile_text)
    
    print("--- Asking Gemini for Logical Architect Schema ---")
    raw_res = call_gemini_api(prompt)
    if raw_res:
        return extract_json(raw_res)
    return None


# --- Enhanced Profiling ---

def node_pattern_signature(G, n):
    """
    Pattern = (set of property keys, set of outgoing edge types, set of incoming edge types)
    """
    props = tuple(sorted(k for k in G.nodes[n].keys() if k != "node_type"))
    out_edges = tuple(sorted(
        G[n][v].get("type", "") for v in G.successors(n)
    ))
    in_edges = tuple(sorted(
        G[u][n].get("type", "") for u in G.predecessors(n)
    ))
    return (props, out_edges, in_edges)


def profile_node_type(G, target_type):

    nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == target_type]
    count = len(nodes)
    if count == 0: 
        return ""

    # Build pattern ? instances map
    pattern_map = {}
    for n in nodes:
        sig = node_pattern_signature(G, n)
        pattern_map.setdefault(sig, []).append(n)

    # Take top-K most frequent patterns
    top_patterns = sorted(
        pattern_map.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]

    # Representative nodes for property inspection
    sample = [nodes[0] for _, nodes in top_patterns]
    keys = {k for n in sample for k in G.nodes[n].keys() if k != "node_type"}


    profile = f"\n  [Detected Node Group]: '{target_type}' ({count} instances)\n"
    profile += f"    - Distinct structural patterns: {len(pattern_map)}\n"
    profile += "    - Top patterns (frequency, outgoing edges, incoming edges):\n"

    for (props, out_e, in_e), nodes_in_pattern in top_patterns:
        profile += (
            f"      * freq={len(nodes_in_pattern)}, "
            f"out={set(out_e)}, in={set(in_e)}\n"
        )


    for key in sorted(keys):
        vals = [G.nodes[n].get(key) for n in sample if G.nodes[n].get(key) is not None]
        if not vals: 
            continue

        density = (len(vals) / len(sample)) * 100
        unique_ratio = len(set(str(v) for v in vals)) / len(vals)
        nature = "Unique ID" if unique_ratio > 0.9 else "Category/Enum" if unique_ratio < 0.1 else "Value"

        profile += (f"    - Property '{key}': {density:.1f}% fill. "
                    f"Nature: {nature}. Type: {type(vals[0]).__name__}\n")
    return profile

# CHANGED BY MARSELO
def profile_edge_type(G, target_type):
    edges = [(u, v, attr) for u, v, attr in G.edges(data=True) if attr.get('type') == target_type]
    if not edges:
        return ""

    # Build edge topology patterns
    pattern_map = {}
    for u, v, attr in edges:
        src = G.nodes[u].get("node_type", "Unknown")
        dst = G.nodes[v].get("node_type", "Unknown")
        props = tuple(sorted(k for k in attr.keys() if k != "type"))
        sig = (src, dst, props)
        pattern_map.setdefault(sig, 0)
        pattern_map[sig] += 1

    top_patterns = sorted(
        pattern_map.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]



    # Analyze edge properties (keeps current logic)
    sample_edges = edges[:min(500, len(edges))]
    all_properties = {}
    for u, v, attr in sample_edges:
        for key, value in attr.items():
            if key != 'type':
                if key not in all_properties:
                    all_properties[key] = {'count': 0, 'sample_values': []}
                all_properties[key]['count'] += 1
                if len(all_properties[key]['sample_values']) < 3 and value is not None:
                    all_properties[key]['sample_values'].append(str(value)[:50])

    profile = f"\n  [Detected Edge Group]: '{target_type}' ({len(edges)} instances)\n"
    # Include counts so the model sees multiple valid pairs (not just names)
    profile += "    - Observed Edge Topology Patterns:\n"
    for (src, dst, props), cnt in top_patterns:
        profile += (
            f"      * ({src})->({dst}), "
            f"edge_props={set(props)}, "
            f"count={cnt}\n"
        )


    if all_properties:
        profile += f"    - Edge Properties: {', '.join(sorted(all_properties.keys()))}\n"

    return profile



def identify_technical_containers(G, node_types):
    """
    Algorithmically identify technical container nodes based on:
    - Property count (typically < 3 meaningful properties)
    - Structural role (Hubs/Bridges)
    - Generic naming patterns (Sets/Collections)
    
    100% AGNOSTIC: Relies on graph structure and generic data modeling terms.
    """
    tech_containers = set()

    for node_type in node_types:
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == node_type]
        if not nodes:
            continue

        sample = nodes[:min(100, len(nodes))]

        # Count meaningful properties (exclude node_type and ID-like keys)
        all_keys = {k for n in sample for k in G.nodes[n].keys() if k != 'node_type'}
        id_like_keys = {k for k in all_keys if any(pattern in k.lower() for pattern in ['id', 'key', 'uuid', 'hash', 'guid'])}
        meaningful_props = len(all_keys - id_like_keys)

        # Generic Data Modeling Terms (Set Theory)
        # These are standard Computer Science terms, not domain-specific.
        name_patterns = ['set', 'collection', 'group', 'container', 'link', 'join', 'mapping', 'association', 'batch', 'list']
        name_match = any(pattern in node_type.lower() for pattern in name_patterns)

        # Classify as technical container if:
        # (1) Name matches generic pattern AND has few properties (structural hub), OR
        # (2) Has very few properties (< 2) regardless of name (pure join node)
        if (name_match and meaningful_props < 3) or meaningful_props < 2:
            tech_containers.add(node_type)

    return tech_containers


def analyze_edge_semantics(G, source_type, target_type):
    """
    Analyze edge patterns to determine the STRUCTURAL CATEGORY of the relationship.
    Does NOT suggest specific labels. Returns abstract categories only.
    """
    # Find edges between these node types
    source_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == source_type]
    target_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == target_type]

    if not source_nodes or not target_nodes:
        return "UNKNOWN_STRUCTURE", "No instances found"

    # Sample edges
    sample_size = min(100, len(source_nodes))
    source_sample = source_nodes[:sample_size]

    edge_properties = set()
    path_found = False

    for source_node in source_sample:
        for target_node in target_nodes[:min(50, len(target_nodes))]:
            try:
                # Direct check
                if G.has_edge(source_node, target_node):
                    edge_attrs = G[source_node][target_node]
                    edge_properties.update(edge_attrs.keys())
                    path_found = True
                # Indirect check (shortest path)
                elif nx.has_path(G, source_node, target_node):
                    path = nx.shortest_path(G, source_node, target_node)
                    if len(path) > 1:
                        path_found = True
                        # Collect properties from the path
                        for i in range(len(path) - 1):
                            edge_attrs = G[path[i]][path[i+1]]
                            edge_properties.update(edge_attrs.keys())
            except (KeyError, nx.NetworkXError):
                pass

    if not path_found:
        return "DISCONNECTED", "No path found"

    # --- PURELY ABSTRACT CATEGORIZATION ---
    # We classify based on the *nature* of the properties found, not specific words.
    
    prop_str = ' '.join(str(p).lower() for p in edge_properties)
    
    # 1. Quantitative / Weighted (Has numbers)
    has_metrics = any(k in prop_str for k in ['weight', 'score', 'cost', 'dist', 'count', 'strength', 'val'])
    
    # 2. Compositional / Hierarchical (Has part/whole terms)
    has_composition = any(k in prop_str for k in ['part', 'parent', 'child', 'member', 'element', 'index'])
    
    # 3. Temporal (Has time)
    has_time = any(k in prop_str for k in ['time', 'date', 'created', 'updated'])

    if has_composition:
        category = "HIERARCHICAL_STRUCTURE"
    elif has_metrics:
        category = "WEIGHTED_CONNECTION"
    elif has_time:
        category = "TEMPORAL_FLOW"
    else:
        category = "FUNCTIONAL_ASSOCIATION"

    rationale_props = sorted(list(edge_properties))[:3]
    return category, f"Evidence: Properties {rationale_props} suggest {category.lower().replace('_', ' ')}"


def analyze_logical_paths(G, tech_containers):
    """
    Find logical bypass paths: Entity A -> TechnicalContainer -> Entity C
    Returns abstract path definitions.
    """
    logical_paths = []
    node_types = set(nx.get_node_attributes(G, 'node_type').values())
    entity_types = node_types - tech_containers

    for container_type in tech_containers:
        container_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == container_type]
        sample = container_nodes[:min(500, len(container_nodes))]

        source_target_pairs = {}

        for container_node in sample:
            for pred in G.predecessors(container_node):
                pred_type = G.nodes[pred].get('node_type')
                if pred_type in entity_types:
                    for succ in G.successors(container_node):
                        succ_type = G.nodes[succ].get('node_type')
                        if succ_type in entity_types:
                            key = (pred_type, succ_type)
                            if key not in source_target_pairs:
                                source_target_pairs[key] = {
                                    'in_edge_types': set(),
                                    'out_edge_types': set()
                                }
                            if G.has_edge(pred, container_node):
                                source_target_pairs[key]['in_edge_types'].add(G[pred][container_node].get('type', ''))
                            if G.has_edge(container_node, succ):
                                source_target_pairs[key]['out_edge_types'].add(G[container_node][succ].get('type', ''))

        for (source, target), edge_info in source_target_pairs.items():
            category, rationale = analyze_edge_semantics(G, source, target)
            logical_paths.append((source, container_type, target, category, rationale))

    return logical_paths


def analyze_bidirectional_patterns(G):
    """
    Detect bidirectional relationship patterns in the graph.
    Returns analysis of symmetric edge patterns.
    """
    node_types = set(nx.get_node_attributes(G, 'node_type').values())
    bidirectional_patterns = []

    for node_type_a in node_types:
        nodes_a = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == node_type_a]
        if not nodes_a: continue

        sample_a = nodes_a[:min(100, len(nodes_a))]

        for node_type_b in node_types:
            if node_type_a >= node_type_b: continue # Avoid duplicates

            nodes_b = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == node_type_b]
            if not nodes_b: continue

            sample_b = nodes_b[:min(100, len(nodes_b))]

            forward_count = 0
            backward_count = 0

            for node_a in sample_a[:50]:
                for node_b in sample_b[:50]:
                    if G.has_edge(node_a, node_b): forward_count += 1
                    if G.has_edge(node_b, node_a): backward_count += 1

            if forward_count > 5 and backward_count > 5:
                bidirectional_patterns.append((node_type_a, node_type_b))

    return bidirectional_patterns


def generate_logical_relationship_summary(G):
    """
    Generate a summary of logical relationships that should exist.
    """
    node_types = sorted(list(set(nx.get_node_attributes(G, 'node_type').values())))
    tech_containers = identify_technical_containers(G, node_types)
    logical_paths = analyze_logical_paths(G, tech_containers)
    bidirectional_patterns = analyze_bidirectional_patterns(G)

    if not logical_paths and not bidirectional_patterns:
        return ""

    summary = "\n  [STRUCTURAL RELATIONSHIP ANALYSIS]\n"
    summary += f"    - Identified Intermediate/Join Nodes: {', '.join(sorted(tech_containers))}\n"

    if logical_paths:
        summary += "    - Suggested Direct Logical Relationships (bypassing intermediates):\n"

        direct_rels = {}
        for path_info in logical_paths:
            if len(path_info) == 5:
                source, container, target, category, rationale = path_info
            else:
                source, container, target = path_info[:3]
                category, rationale = analyze_edge_semantics(G, source, target)

            key = (source, target)
            if key not in direct_rels:
                direct_rels[key] = {'containers': [], 'categories': [], 'rationales': []}
            direct_rels[key]['containers'].append(container)
            direct_rels[key]['categories'].append(category)
            if rationale not in direct_rels[key]['rationales']:
                direct_rels[key]['rationales'].append(rationale)

        for (source, target), info in sorted(direct_rels.items()):
            containers_str = ', '.join(set(info['containers']))
            # Most common category
            cat_counts = {}
            for c in info['categories']: cat_counts[c] = cat_counts.get(c, 0) + 1
            best_cat = max(cat_counts.items(), key=lambda x: x[1])[0] if cat_counts else "UNKNOWN"

            summary += f"      * {source} -> {target} (via: {containers_str})\n"
            summary += f"        Relationship Category: {best_cat}\n"

    if bidirectional_patterns:
        summary += "    - Detected Bidirectional/Symmetric Patterns:\n"
        for type_a, type_b in bidirectional_patterns:
            summary += f"      * {type_a} <-> {type_b}\n"

    summary += "    - CRITICAL: Your schema MUST include these direct relationships.\n"
    summary += "      Determine the specific semantic verb (e.g., OWNS, LINKS, INCLUDES) based on the Relationship Category.\n"

    return summary


# --- LLM Helpers ---
def call_gemini_api(prompt):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return response.text
    except Exception as e:
        print(f"API Error: {e}")
        return None


def extract_json(text):
    try:
        return json.loads(text.strip().replace("```json", "").replace("```", ""))
    except:
        return None


def run_infer_schema(data_dir, output_path):
    # 1. Generate Schema using your existing logic
    schema = infer_schema_from_folder(data_dir)
    
    # 2. FIX: Create the PARENT directory, not the file itself as a folder
    # 'output_path' is the full path including filename (e.g., .../inferred_schema.json)
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # 3. Save the file
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=4)

    return output_path
