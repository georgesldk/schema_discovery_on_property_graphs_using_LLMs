import os
import json
import random
import argparse
import networkx as nx
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter
from build_graph import build_graph

# --- Statistical Profiling ---

def profile_node_type(G, target_type):
    """Summarizes node properties with density and uniqueness hints."""
    nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == target_type]
    count = len(nodes)
    if count == 0: return ""
    
    sample = random.sample(nodes, min(count, 500))
    keys = {k for n in sample for k in G.nodes[n].keys() if k != 'node_type'}

    profile = f"\n  [Node Group]: '{target_type}' ({count} nodes)\n"
    for key in sorted(keys):
        vals = [G.nodes[n].get(key) for n in sample if G.nodes[n].get(key) is not None]
        if not vals: continue
        
        density = (len(vals) / len(sample)) * 100
        unique_ratio = len(set(str(v) for v in vals)) / len(vals)
        nature = "ID" if unique_ratio > 0.9 else "Category" if unique_ratio < 0.1 else "Value"
        
        profile += (f"    - '{key}': {density:.1f}% fill | {nature} | Type: {type(vals[0]).__name__}\n")
    return profile

def profile_edge_type(G, target_type):
    """Profiles connections and property density for edges."""
    edges = [(u, v, attr) for u, v, attr in G.edges(data=True) if attr.get('type') == target_type]
    if not edges: return ""
    
    # Topology: (SourceType)->(TargetType)
    conns = [f"({G.nodes[u].get('node_type')})->({G.nodes[v].get('node_type')})" for u, v, _ in edges[:100]]
    top_conns = Counter(conns).most_common(2)
    
    profile = f"\n  [Edge Group]: '{target_type}' ({len(edges)} edges)\n"
    profile += f"    - Connections: {', '.join([c[0] for c in top_conns])}\n"
    return profile

# --- LLM Integration ---

def call_gemini_api(prompt):
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return response.text
    except Exception as e:
        print(f"API Error: {e}")
        return None

def extract_json(text):
    """Cleans and parses JSON from API markdown response."""
    try:
        return json.loads(text.strip().replace("```json", "").replace("```", ""))
    except: return None

# --- Main Execution ---

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

    prompt = f"""
    You are a Senior Graph Data Architect. Output a clean, logical Property Graph Schema.
    
    INPUT DATA PROFILE:
    {profile_text}
    
    STRICT HIERARCHY RULES:
    1. NAMING: Use Singular PascalCase (e.g., 'Neuron', not 'Neurons').
    2. CONSOLIDATION: Ignore 'Technical' nodes (ending in 'Set', 'Meta', 'Info'). 
       - Fold their properties into the primary parent node. 
       - Example: If 'SynapseSet' properties belong to 'Synapse', merge them.
    3. RELATIONSHIPS: Only map core functional edges. Ignore technical links like 'HAS_SET'.
    4. MANDATORY: Set 'mandatory: true' only if property fill density is >98%.
    
    JSON OUTPUT FORMAT:
    {{
      "node_types": [{{ "name": "Label", "properties": [...] }}],
      "edge_types": [{{ "type": "REL", "from": "Src", "to": "Tgt", "properties": [...] }}]
    }}
    """
    
    print("--- Requesting Logical Schema ---")
    raw_res = call_gemini_api(prompt)
    if raw_res:
        schema = extract_json(raw_res)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "inferred_schema.json"), "w") as f:
            json.dump(schema, f, indent=4)
        print("✅ Schema Inference Complete.")