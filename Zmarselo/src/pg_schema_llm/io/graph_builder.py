import pandas as pd
import networkx as nx
import os
import re
import glob

from pg_schema_llm.io.csv_detect import detect_file_role


def clean_type_name(filename):
    """Extracts node/edge type from filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    # Generic cleanup
    base = re.sub(r'Neuprint_', '', base, flags=re.IGNORECASE)
    base = re.sub(r'_fib25', '', base, flags=re.IGNORECASE)
    base = re.sub(r'_hemibrain', '', base, flags=re.IGNORECASE)
    base = re.sub(r'_strict', '', base, flags=re.IGNORECASE)
    return base


def build_graph(data_folder):
    """
    Builds a NetworkX graph from ANY folder of CSVs.
    """
    print(f"--- Building Graph from: {data_folder} ---")
    G = nx.DiGraph()

    if not os.path.isdir(data_folder):
        print(f" Error: Folder '{data_folder}' does not exist.")
        return G

    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

    if not csv_files:
        print(f" No CSV files found in {data_folder}")
        return G

    # Pass 1: Nodes
    print(">>> Scanning for Nodes...")
    for file_path in csv_files:
        try:
            df_preview = pd.read_csv(file_path, nrows=0)
            role, cols = detect_file_role(df_preview)

            if role == 'node':
                type_name = clean_type_name(file_path)
                print(f"   Processing Nodes: {os.path.basename(file_path)} -> '{type_name}'")
                df = pd.read_csv(file_path, low_memory=False)
                id_col = cols['id']
                df[id_col] = df[id_col].astype(str)

                for _, row in df.iterrows():
                    G.add_node(row[id_col], node_type=type_name, **row.to_dict())
        except Exception as e:
            print(f"    Error reading {file_path}: {e}")

    # Pass 2: Edges
    print("\n>>> Scanning for Edges...")
    for file_path in csv_files:
        try:
            df_preview = pd.read_csv(file_path, nrows=0)
            role, cols = detect_file_role(df_preview)

            if role == 'edge':
                type_name = clean_type_name(file_path)
                print(f"   Processing Edges: {os.path.basename(file_path)} -> '{type_name}'")
                df = pd.read_csv(file_path, low_memory=False)
                start_col, end_col = cols['start'], cols['end']
                df[start_col] = df[start_col].astype(str)
                df[end_col] = df[end_col].astype(str)

                for _, row in df.iterrows():
                    u, v = row[start_col], row[end_col]
                    if not G.has_node(u):
                        G.add_node(u, node_type="Inferred")
                    if not G.has_node(v):
                        G.add_node(v, node_type="Inferred")
                    G.add_edge(u, v, type=type_name, **row.to_dict())
        except Exception as e:
            print(f"    Error reading {file_path}: {e}")

    print(f"\n Graph Built. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G
