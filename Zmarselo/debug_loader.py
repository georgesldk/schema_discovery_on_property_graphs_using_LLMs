from pg_schema_llm.io.graph_builder import build_graph
import json

# Load graph
G = build_graph("pg_data_mb6")

# Pick 1 random node
if len(G) > 0:
    sample_id = list(G.nodes())[0]
    data = G.nodes[sample_id]
    print(f"\n--- INSPECTING NODE '{sample_id}' ---")
    print(json.dumps(data, indent=2))
else:
    print("Graph is empty!")