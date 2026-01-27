import pandas as pd
import networkx as nx
import os
import glob
from pg_schema_llm.io.csv_detect import detect_file_role

# --- NEW: Global Noise Detectors ---
def get_common_affixes(filenames):
    """
    Dataset-Agnostic Noise Detector.
    Finds the common prefix (e.g., "Neuprint_") and suffix (e.g., "_fib25.csv")
    shared by ALL files in the directory.
    """
    if not filenames:
        return "", ""
    
    # 1. Find Longest Common Prefix
    prefix = os.path.commonprefix(filenames)
    
    # 2. Find Longest Common Suffix (by reversing strings)
    # We must operate on the full filenames (including extension)
    reversed_names = [f[::-1] for f in filenames]
    common_rev = os.path.commonprefix(reversed_names)
    suffix = common_rev[::-1]
    
    return prefix, suffix

def clean_name_smart(filename, prefix, suffix):
    """
    Strips the detected common noise from a filename.
    """
    base = os.path.basename(filename)
    
    # Remove Prefix
    if prefix and base.startswith(prefix):
        base = base[len(prefix):]
        
    # Remove Suffix
    if suffix and base.endswith(suffix):
        base = base[:-len(suffix)]
        
    # Final cleanup: remove extension if suffix didn't cover it
    base = os.path.splitext(base)[0]
    
    # Cleanup trailing/leading underscores
    base = base.strip("_")
    
    return base

# --- LEGACY SUPPORT (Fixes ImportError) ---
def clean_type_name(filename):
    """
    Legacy wrapper. Used by __init__.py imports.
    Falls back to simple extension removal if called directly without global context.
    """
    return os.path.splitext(os.path.basename(filename))[0]

# --- MAIN GRAPH BUILDER ---
def build_graph(data_folder):
    print(f"--- Building Graph from: {data_folder} ---")
    G = nx.MultiDiGraph() 

    if not os.path.isdir(data_folder):
        print(f" Error: Folder '{data_folder}' does not exist.")
        return G

    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not csv_files:
        print(f" No CSV files found in {data_folder}")
        return G

    # STEP 1: CALCULATE GLOBAL NOISE
    all_filenames = [os.path.basename(f) for f in csv_files]
    common_prefix, common_suffix = get_common_affixes(all_filenames)
    
    print(f"   [Auto-Cleaner] Detected Common Prefix: '{common_prefix}'")
    print(f"   [Auto-Cleaner] Detected Common Suffix: '{common_suffix}'")

    # STEP 2: BUILD GRAPH WITH CLEAN NAMES
    print(">>> Scanning for Nodes & Edges...")
    
    for file_path in csv_files:
        try:
            # Use the smart cleaner
            clean_type = clean_name_smart(file_path, common_prefix, common_suffix)
            
            # Read minimal preview
            df_preview = pd.read_csv(file_path, nrows=0)
            role, cols = detect_file_role(df_preview)
            
            if role == 'node':
                print(f"   Processing Nodes: {os.path.basename(file_path)} -> '{clean_type}'")
                df = pd.read_csv(file_path, low_memory=False)
                id_col = cols['id']
                df[id_col] = df[id_col].astype(str)
                for _, row in df.iterrows():
                    G.add_node(row[id_col], node_type=clean_type, **row.to_dict())

            elif role == 'edge':
                print(f"   Processing Edges: {os.path.basename(file_path)} -> '{clean_type}'")
                df = pd.read_csv(file_path, low_memory=False)
                start_col, end_col = cols['start'], cols['end']
                df[start_col] = df[start_col].astype(str)
                df[end_col] = df[end_col].astype(str)
                for _, row in df.iterrows():
                    u, v = row[start_col], row[end_col]
                    if not G.has_node(u): G.add_node(u, node_type="Inferred")
                    if not G.has_node(v): G.add_node(v, node_type="Inferred")
                    
                    G.add_edge(u, v, type=clean_type, **row.to_dict())

        except Exception as e:
            print(f"    Error reading {file_path}: {e}")

    print(f"\n Graph Built. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G