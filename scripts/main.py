import os
import json
import random
import argparse
import networkx as nx
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter
from build_graph import build_graph

# --- Enhanced Profiling ---
def profile_node_type(G, target_type):
    nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == target_type]
    count = len(nodes)
    if count == 0: return ""
    
    sample = random.sample(nodes, min(count, 500))
    keys = {k for n in sample for k in G.nodes[n].keys() if k != 'node_type'}

    profile = f"\n  [Detected Node Group]: '{target_type}' ({count} instances)\n"
    for key in sorted(keys):
        vals = [G.nodes[n].get(key) for n in sample if G.nodes[n].get(key) is not None]
        if not vals: continue
        
        density = (len(vals) / len(sample)) * 100
        unique_ratio = len(set(str(v) for v in vals)) / len(vals)
        nature = "Unique ID" if unique_ratio > 0.9 else "Category/Enum" if unique_ratio < 0.1 else "Value"
        
        profile += (f"    - Property '{key}': {density:.1f}% fill. "
                    f"Nature: {nature}. Type: {type(vals[0]).__name__}\n")
    return profile

def profile_edge_type(G, target_type):
    edges = [(u, v, attr) for u, v, attr in G.edges(data=True) if attr.get('type') == target_type]
    if not edges: return ""
    
    conns = [f"({G.nodes[u].get('node_type', 'Unknown')})->({G.nodes[v].get('node_type', 'Unknown')})" 
             for u, v, _ in edges[:100]]
    top_conns = Counter(conns).most_common(2)
    
    # Analyze edge properties for semantic hints
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
    profile += f"    - Observed Connections: {', '.join([c[0] for c in top_conns])}\n"
    
    # Add property information if available
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
        if not nodes: continue
        
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
    # Find edges between these node types (could be through containers)
    source_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == source_type]
    target_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == target_type]
    
    if not source_nodes or not target_nodes:
        return "CONNECTS_TO", "default"
    
    # Sample edges
    sample_size = min(100, len(source_nodes))
    source_sample = source_nodes[:sample_size]
    
    # Analyze edge properties and patterns
    edge_properties = set()
    edge_labels = []
    path_lengths = []
    
    for source_node in source_sample:
        # Find paths to target nodes (direct or through containers)
        for target_node in target_nodes[:min(50, len(target_nodes))]:
            try:
                if G.has_edge(source_node, target_node):
                    # Direct edge
                    edge_attrs = G[source_node][target_node]
                    edge_labels.append(edge_attrs.get('type', ''))
                    edge_properties.update(edge_attrs.keys())
                    path_lengths.append(1)
                else:
                    # Try to find shortest path
                    try:
                        path = nx.shortest_path(G, source_node, target_node)
                        if len(path) > 1:
                            path_lengths.append(len(path) - 1)
                            # Collect edge types along path
                            for i in range(len(path) - 1):
                                edge_attrs = G[path[i]][path[i+1]]
                                edge_labels.append(edge_attrs.get('type', ''))
                                edge_properties.update(edge_attrs.keys())
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
            except (KeyError, nx.NetworkXError):
                pass
    
    # Analyze properties to infer semantics
    property_hints = {
        'CONTAINS': ['contains', 'has', 'includes', 'parent', 'child', 'member', 'element'],
        'CONNECTS_TO': ['weight', 'strength', 'distance', 'connection', 'link', 'connects'],
        'SYNAPSES_TO': ['synapse', 'junction', 'interface', 'contact']
    }
    
    prop_str = ' '.join(str(p).lower() for p in edge_properties)
    label_str = ' '.join(str(l).lower() for l in edge_labels if l)
    
    scores = {}
    for edge_type, hints in property_hints.items():
        score = sum(1 for hint in hints if hint in prop_str or hint in label_str)
        scores[edge_type] = score
    
    # Default to CONTAINS if path suggests containment, CONNECTS_TO otherwise
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
    
    # For each technical container, find paths through it
    for container_type in tech_containers:
        container_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == container_type]
        
        # Sample to avoid performance issues
        sample = container_nodes[:min(500, len(container_nodes))]
        
        # Find entities that connect through this container (track edge types)
        source_target_pairs = {}
        
        for container_node in sample:
            # Entities that connect TO the container
            for pred in G.predecessors(container_node):
                pred_type = G.nodes[pred].get('node_type')
                if pred_type in entity_types:
                    # Entities that the container connects TO
                    for succ in G.successors(container_node):
                        succ_type = G.nodes[succ].get('node_type')
                        if succ_type in entity_types:
                            key = (pred_type, succ_type)
                            if key not in source_target_pairs:
                                source_target_pairs[key] = {
                                    'in_edge_types': set(),
                                    'out_edge_types': set()
                                }
                            # Get edge types
                            if G.has_edge(pred, container_node):
                                source_target_pairs[key]['in_edge_types'].add(
                                    G[pred][container_node].get('type', '')
                                )
                            if G.has_edge(container_node, succ):
                                source_target_pairs[key]['out_edge_types'].add(
                                    G[container_node][succ].get('type', '')
                                )
        
        # Suggest direct relationships with semantic analysis
        for (source, target), edge_info in source_target_pairs.items():
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
            if node_type_a >= node_type_b:  # Avoid duplicates
                continue
                
            nodes_b = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == node_type_b]
            if not nodes_b:
                continue
                
            sample_b = nodes_b[:min(100, len(nodes_b))]
            
            # Check for bidirectional edges
            forward_count = 0
            backward_count = 0
            
            for node_a in sample_a[:50]:
                for node_b in sample_b[:50]:
                    if G.has_edge(node_a, node_b):
                        forward_count += 1
                    if G.has_edge(node_b, node_a):
                        backward_count += 1
            
            # If significant bidirectional pattern exists
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
        
        # Group by source->target pair, collect suggestions
        direct_rels = {}
        for path_info in logical_paths:
            if len(path_info) == 5:
                source, container, target, suggested_label, rationale = path_info
            else:
                # Backward compatibility
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
            # Use most common suggested label
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
    except: return None

# --- Main Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    G = build_graph(args.data_dir)
    node_types = sorted(list(set(nx.get_node_attributes(G, 'node_type').values())))
    edge_types = sorted(list(set(nx.get_edge_attributes(G, 'type').values())))
    
    profile_text = "".join([profile_node_type(G, nt) for nt in node_types])
    profile_text += "".join([profile_edge_type(G, et) for et in edge_types])
    
    # Add logical relationship analysis
    logical_summary = generate_logical_relationship_summary(G)
    if logical_summary:
        profile_text += logical_summary

    # THE IMPROVED PROMPT - DATASET-AGNOSTIC PROPERTY GRAPH STANDARDS
    prompt = f"""
    You are a Senior Property Graph Schema Architect. Your mission is to infer a logical Property Graph schema from raw physical data structures, following industry-standard Property Graph principles.
    
    DATA PROFILE:
    {profile_text}
    
    STRICT DATASET-AGNOSTIC PROPERTY GRAPH HEURISTICS:
    
    1. LOGICAL BYPASS RULE (Collapse Technical Intermediaries) - HIGHEST PRIORITY:
       - Technical Container nodes are grouping/collection mechanisms with minimal properties (typically < 3 meaningful properties beyond IDs).
       - Patterns: names containing "Set", "Collection", "Group", "Container", "Link", "Join", "Mapping", "Association".
       - MANDATORY ACTION: If the profile shows "Entity A -> TechnicalContainer -> Entity C" paths, you MUST:
         * Create a direct edge type: Entity A -> Entity C
         * Use the suggested edge label from the "[LOGICAL RELATIONSHIP ANALYSIS]" section if provided, OR determine appropriate label (CONNECTS_TO, CONTAINS, or SYNAPSES_TO) based on semantic meaning
         * Analyze edge properties to determine semantics: properties like "weight", "strength", "distance" suggest CONNECTS_TO; properties suggesting membership/hierarchy suggest CONTAINS
         * DO NOT create edges that go through the technical container in your schema
       - If the profile includes "[LOGICAL RELATIONSHIP ANALYSIS]", those direct relationships are REQUIRED in your output.
       - The logical schema represents functional relationships, not physical storage artifacts.
       - Property Graphs prioritize direct semantic connections over intermediate technical structures.
    
    2. ENTITY CONSOLIDATION (Deduplicate Semantic Equivalents):
       - If multiple node types represent the same logical entity with different attribute sets, merge them into a single node type.
       - Properties from all variants should be merged, with "mandatory" set based on > 98% fill density.
       - Only keep truly distinct entity types that represent different concepts.
    
    3. STANDARDIZED EDGE LABELS (Property Graph Convention):
       - MANDATORY: Use ONLY these three edge labels - no exceptions:
         * CONNECTS_TO: For high-level structural/functional connections between major entities. Use when edges have properties like weight, strength, distance, or represent general connectivity.
         * CONTAINS: For parent-child or containment relationships (hierarchical). Use when one entity logically contains or groups another, or when relationships suggest membership/hierarchy.
         * SYNAPSES_TO: For fine-grained functional/operational links. Use sparingly, only when CONNECTS_TO is too coarse and the relationship represents a specific operational/junctional connection.
       - If the profile suggests an edge label, use that suggestion (it's based on semantic analysis of properties and patterns).
       - DO NOT use: technical names (HAS_SET, LINKS_TO), action verbs (CREATES, DELETES), file-based names (FROM_CSV, TO_TABLE), or generic verbs (ASSOCIATED_WITH, DEPENDS_ON).
       - Consolidate all edges between the same two node types into ONE edge type using the appropriate standard label.
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
        schema = extract_json(raw_res)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "inferred_schema.json"), "w") as f:
            json.dump(schema, f, indent=4)
        print("✅ Refined Schema Generated.") 