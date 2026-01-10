import os
import json
import random
import networkx as nx
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter

# Keep compatibility during restructure:
# - after refactor you will import from pg_schema_llm.io.graph_builder
# - for now it can still work if old build_graph.py exists on PYTHONPATH
try:
    from pg_schema_llm.io.graph_builder import build_graph
except Exception:
    from build_graph import build_graph



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
    - Naming patterns (Set, Collection, Group, Container, Link, Join, Mapping, Association)
    - High connectivity relative to property richness
    """
    tech_containers = set()

    for node_type in node_types:
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == node_type]
        if not nodes: 
            continue

        sample = nodes[:min(100, len(nodes))]

        # Count meaningful properties (exclude node_type, exclude ID-like properties)
        all_keys = {k for n in sample for k in G.nodes[n].keys() if k != 'node_type'}
        id_like_keys = {k for k in all_keys if any(pattern in k.lower() for pattern in ['id', 'key', 'uuid', 'hash'])}
        meaningful_props = len(all_keys - id_like_keys)

        # Check naming patterns
        name_patterns = ['set', 'collection', 'group', 'container', 'link', 'join', 'mapping', 'association']
        name_match = any(pattern in node_type.lower() for pattern in name_patterns)

        # High out-degree relative to properties suggests container role
        out_degrees = [G.out_degree(n) for n in sample]
        avg_out_degree = sum(out_degrees) / len(sample) if sample else 0
        _ = avg_out_degree  # keep as-is (original computed, not used)

        # Classify as technical container if:
        # (1) Name matches pattern AND has few properties, OR
        # (2) Has very few properties (< 2) regardless of name
        if (name_match and meaningful_props < 3) or meaningful_props < 2:
            tech_containers.add(node_type)

    return tech_containers


def analyze_edge_semantics(G, source_type, target_type):
    """
    Analyze edge patterns to suggest appropriate edge label semantics.
    Returns a suggested edge label and rationale.
    """
    source_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == source_type]
    target_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == target_type]

    if not source_nodes or not target_nodes:
        return "CONNECTS_TO", "default"

    sample_size = min(100, len(source_nodes))
    source_sample = source_nodes[:sample_size]

    edge_properties = set()
    edge_labels = []
    path_lengths = []

    for source_node in source_sample:
        for target_node in target_nodes[:min(50, len(target_nodes))]:
            try:
                if G.has_edge(source_node, target_node):
                    edge_attrs = G[source_node][target_node]
                    edge_labels.append(edge_attrs.get('type', ''))
                    edge_properties.update(edge_attrs.keys())
                    path_lengths.append(1)
                else:
                    try:
                        path = nx.shortest_path(G, source_node, target_node)
                        if len(path) > 1:
                            path_lengths.append(len(path) - 1)
                            for i in range(len(path) - 1):
                                edge_attrs = G[path[i]][path[i+1]]
                                edge_labels.append(edge_attrs.get('type', ''))
                                edge_properties.update(edge_attrs.keys())
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
            except (KeyError, nx.NetworkXError):
                pass

    property_hints = {
        'CONTAINS': ['contains', 'has', 'includes', 'parent', 'child', 'member', 'element'],
        'CONNECTS_TO': ['weight', 'strength', 'distance', 'connection', 'link', 'connects'],
        'SYNAPSES_TO': ['synapse', 'junction', 'interface', 'contact']
    }

    prop_str = ' '.join(str(p).lower() for p in edge_properties)
    label_str = ' '.join(str(l).lower() for l in edge_labels if l)
    _ = label_str  # keep as-is (computed, used indirectly)

    scores = {}
    for edge_type, hints in property_hints.items():
        score = sum(1 for hint in hints if hint in prop_str or hint in label_str)
        scores[edge_type] = score

    if scores['CONTAINS'] > 0:
        suggested = 'CONTAINS'
    elif scores['SYNAPSES_TO'] > scores['CONNECTS_TO']:
        suggested = 'SYNAPSES_TO'
    else:
        suggested = 'CONNECTS_TO'

    return suggested, f"based on properties: {', '.join(sorted(edge_properties)[:3])}" if edge_properties else "default"


def analyze_logical_paths(G, tech_containers):
    """
    Find logical bypass paths: Entity A -> TechnicalContainer -> Entity C
    Returns a list of suggested direct relationships with semantic hints.
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
                                source_target_pairs[key]['in_edge_types'].add(
                                    G[pred][container_node].get('type', '')
                                )
                            if G.has_edge(container_node, succ):
                                source_target_pairs[key]['out_edge_types'].add(
                                    G[container_node][succ].get('type', '')
                                )

        for (source, target), edge_info in source_target_pairs.items():
            _ = edge_info  # keep as-is (collected, not used further)
            suggested_label, rationale = analyze_edge_semantics(G, source, target)
            logical_paths.append((source, container_type, target, suggested_label, rationale))

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
        if not nodes_a:
            continue

        sample_a = nodes_a[:min(100, len(nodes_a))]

        for node_type_b in node_types:
            if node_type_a >= node_type_b:
                continue

            nodes_b = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == node_type_b]
            if not nodes_b:
                continue

            sample_b = nodes_b[:min(100, len(nodes_b))]

            forward_count = 0
            backward_count = 0

            for node_a in sample_a[:50]:
                for node_b in sample_b[:50]:
                    if G.has_edge(node_a, node_b):
                        forward_count += 1
                    if G.has_edge(node_b, node_a):
                        backward_count += 1

            if forward_count > 5 and backward_count > 5:
                bidirectional_patterns.append((node_type_a, node_type_b))

    return bidirectional_patterns


def generate_logical_relationship_summary(G):
    """
    Generate a summary of logical relationships that should exist
    by analyzing technical container bypass patterns.
    """
    node_types = sorted(list(set(nx.get_node_attributes(G, 'node_type').values())))
    tech_containers = identify_technical_containers(G, node_types)
    logical_paths = analyze_logical_paths(G, tech_containers)
    bidirectional_patterns = analyze_bidirectional_patterns(G)

    if not logical_paths and not bidirectional_patterns:
        return ""

    summary = "\n  [LOGICAL RELATIONSHIP ANALYSIS - Technical Container Bypass]\n"
    summary += f"    - Identified Technical Container Nodes: {', '.join(sorted(tech_containers))}\n"

    if logical_paths:
        summary += "    - Suggested Direct Logical Relationships (bypassing containers):\n"

        direct_rels = {}
        for path_info in logical_paths:
            if len(path_info) == 5:
                source, container, target, suggested_label, rationale = path_info
            else:
                source, container, target = path_info[:3]
                suggested_label, rationale = analyze_edge_semantics(G, source, target)

            key = (source, target)
            if key not in direct_rels:
                direct_rels[key] = {
                    'containers': [],
                    'suggested_labels': [],
                    'rationales': []
                }
            direct_rels[key]['containers'].append(container)
            direct_rels[key]['suggested_labels'].append(suggested_label)
            if rationale not in direct_rels[key]['rationales']:
                direct_rels[key]['rationales'].append(rationale)

        for (source, target), info in sorted(direct_rels.items()):
            containers_str = ', '.join(set(info['containers']))
            label_counts = {}
            for label in info['suggested_labels']:
                label_counts[label] = label_counts.get(label, 0) + 1
            most_common_label = max(label_counts.items(), key=lambda x: x[1])[0] if label_counts else "CONNECTS_TO"

            summary += f"      * {source} -> {target} (currently via: {containers_str})\n"
            summary += f"        Suggested edge label: {most_common_label}\n"

    if bidirectional_patterns:
        summary += "    - Detected Bidirectional Relationship Patterns:\n"
        for type_a, type_b in bidirectional_patterns:
            summary += f"      * {type_a} <-> {type_b} (consider bidirectional edge definition)\n"

    summary += "    - CRITICAL: Your schema MUST include these direct relationships.\n"
    summary += "      Do NOT include the technical container nodes in edge definitions for these relationships.\n"
    summary += "      Use the suggested edge labels (CONNECTS_TO, CONTAINS, or SYNAPSES_TO) based on semantic meaning.\n"

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


def infer_schema_from_folder(data_dir):
    G = build_graph(data_dir)
    node_types = sorted(list(set(nx.get_node_attributes(G, 'node_type').values())))
    edge_types = sorted(list(set(nx.get_edge_attributes(G, 'type').values())))

    profile_text = "".join([profile_node_type(G, nt) for nt in node_types])
    profile_text += "".join([profile_edge_type(G, et) for et in edge_types])

#    logical_summary = generate_logical_relationship_summary(G)
#    if logical_summary:
#        profile_text += logical_summary

    prompt = f"""
    You are a Senior Property Graph Schema Architect. Your mission is to infer a logical Property Graph schema from raw physical data structures, following industry-standard Property Graph principles.

    DATA PROFILE:
    {profile_text}

    STRICT DATASET-AGNOSTIC PROPERTY GRAPH HEURISTICS:

    1. PHYSICAL SCHEMA RULE (Match Observed Graph Structure) - HIGHEST PRIORITY:
        - Do NOT bypass or remove intermediary/container node types.
        - If edges exist in the observed data profile, represent them explicitly in the schema.
        - Your goal is to match the physical structure implied by the DATA PROFILE.
       - Property Graphs prioritize direct semantic connections over intermediate technical structures.

    2. ENTITY CONSOLIDATION (Deduplicate Semantic Equivalents):
        - If multiple node types represent the same logical entity with different attribute sets, merge them into a single node type.
        - Properties from all variants should be merged, with "mandatory" set based on > 98% fill density.
        - Only keep truly distinct entity types that represent different concepts.
        - IMPORTANT: Do NOT invent or rename node types.
        - Use ONLY the node type names that appear in the DATA PROFILE "[Detected Node Group]" headers.
        - Keep names EXACTLY as shown (no ROI/RegionOfInterest renaming, no abbreviations).


    3. STANDARDIZED EDGE LABELS (Property Graph Convention):
       - MANDATORY: Use ONLY these three edge labels - no exceptions:
         * CONNECTS_TO: For high-level structural/functional connections between major entities. Use when edges have properties like weight, strength, distance, or represent general connectivity.
         * CONTAINS: For parent-child or containment relationships (hierarchical). Use when one entity logically contains or groups another, or when relationships suggest membership/hierarchy.
         * SYNAPSES_TO: For fine-grained functional/operational links. Use sparingly, only when CONNECTS_TO is too coarse and the relationship represents a specific operational/junctional connection.
       - If the profile suggests an edge label, use that suggestion (it's based on semantic analysis of properties and patterns).
       - DO NOT use: technical names (HAS_SET, LINKS_TO), action verbs (CREATES, DELETES), file-based names (FROM_CSV, TO_TABLE), or generic verbs (ASSOCIATED_WITH, DEPENDS_ON).
        - Do NOT collapse distinct observed (start_node -> end_node) patterns into one generic edge.
        - If multiple (start_node -> end_node) pairs exist for the same label, include each pair explicitly in topology.
       - Bidirectional relationships: If the profile indicates bidirectional patterns between two node types, you may need to create edges in both directions using the same edge label, OR create a single edge type that can be traversed in both directions (depending on the semantic meaning).

    4. SCHEMA NORMALIZATION:
       - NODE NAMING: Singular PascalCase (e.g., "Entity", "Item", "Category" - NOT "Entities", "EntityType").
       - PROPERTY TYPES: Use "String", "Long", "Double", "Boolean", "StringArray", "Point" (for spatial data).
       - MANDATORY FLAGS: Set "mandatory: true" ONLY for properties with > 98% fill density across all instances.
       - EDGE PROPERTIES: Include edge properties (e.g., weight, confidence, count) when they carry semantic meaning.

    5. TOPOLOGY REQUIREMENTS:
       - Each edge_type MUST specify "start_node" and "end_node" fields with the exact node type names.
       - Self-loops (same node type as source and target) are allowed and valid.
       - The schema should represent a logical graph, not a physical data model.

    CRITICAL: This is a Property Graph schema, not a relational model. Focus on:
    - Direct entity-to-entity relationships
    - Logical, not physical, structure
    - Semantic clarity over technical accuracy
    - Standard naming conventions

    OUTPUT JSON FORMAT:
    {{
      "node_types": [
        {{
          "name": "NodeLabel",
          "properties": [
            {{"name": "propertyName", "type": "String|Long|Double|Boolean|StringArray|Point", "mandatory": true|false}}
          ]
        }}
      ],
      "edge_types": [
        {{
          "name": "CONNECTS_TO|CONTAINS|SYNAPSES_TO",
          "start_node": "SourceNodeLabel",
          "end_node": "TargetNodeLabel",
          "properties": [
            {{"name": "propertyName", "type": "String|Long|Double|Boolean", "mandatory": true|false}}
          ]
        }}
      ]
    }}
    """

    print("--- Asking Gemini for Logical Architect Schema ---")
    raw_res = call_gemini_api(prompt)
    if raw_res:
        return extract_json(raw_res)
    return None


def run_infer_schema(data_dir, output_dir):
    schema = infer_schema_from_folder(data_dir)
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "inferred_schema.json")
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=4)

    return out_path
