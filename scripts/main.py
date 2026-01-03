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
    
    profile = f"\n  [Detected Edge Group]: '{target_type}' ({len(edges)} instances)\n"
    profile += f"    - Observed Connections: {', '.join([c[0] for c in top_conns])}\n"
    return profile

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

    # THE IMPROVED PROMPT
    prompt = f"""
    You are a Senior Graph Data Architect tasked with reverse-engineering a high-level LOGICAL schema from a raw data profile.
    
    DATA PROFILE:
    {profile_text}
    
    STRICT DATASET-AGNOSTIC ARCHITECTURAL HEURISTICS:
    
    1. ENTITY VS. ATTRIBUTE (Low-Entropy Pruning):
       - Analyze each detected node type. If a node type contains ONLY 1 or 2 properties (e.g., just a 'name' or an 'ID'), it is likely a 'Metadata Tag' or a 'Join Table' rather than a primary entity.
       - RULE: Do NOT create a separate node for these technical artifacts. Instead, fold their identity (their name/label) into a property of the core entities they connect to.
    
    2. PATH COLLAPSING (Intermediary Removal):
       - Identify "Bridge" nodes that only exist to link two other entities.
       - RULE: If Node A connects to Technical Intermediary B, and B connects to Node C, replace this chain with a DIRECT semantic relationship between A and C.
    
    3. RELATIONSHIP GENERALIZATION (Semantic Cleanliness):
       - Avoid technical or file-based edge names (e.g., 'HAS_DATA', 'LINKS_TO_CSV').
       - RULE: Use high-level functional verbs (e.g., 'CONNECTS_TO', 'CONTAINS', 'DEPENDS_ON'). If multiple technical edges exist between the same two entities, consolidate them into a single logical relationship.
    
    4. SCHEMA NORMALIZATION:
       - Strictly use Singular PascalCase for labels (e.g., 'Neuron', not 'Neurons').
       - properties: Set 'mandatory: true' only if fill density > 98%.
    
    GOAL: Produce the most minimal, human-readable logical model that accurately describes the domain entities, ignoring the physical storage artifacts of the raw data.
    
    OUTPUT JSON FORMAT:
    {{
      "node_types": [{{ "name": "NodeLabel", "properties": [...] }}],
      "edge_types": [{{ "type": "REL_NAME", "from": "Source", "to": "Target", "properties": [...] }}]
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