import random

def profile_edge_type(G, target_type, sample_limit=1000000):
    """
    SMART RANDOM SAMPLING (Reservoir/Probabilistic)
    - Enforces a hard cap (~1M edges).
    - Scans the entire file distribution (start to end) to avoid sorting bias.
    - SAFETY NET: Always keeps 'rare' topology pairs even if the random sampler skips them.
    """
    
    # 1. Calculate Sampling Probability
    # Note: G.number_of_edges() is the total for the whole graph, not just this type.
    # We use it to set a conservative 'p' (this ensures we don't over-sample).
    total_graph_edges = G.number_of_edges()
    if total_graph_edges == 0: return ""
    
    # We aim for sample_limit. If graph is smaller, p = 1.0 (take all).
    acceptance_prob = min(1.0, (sample_limit * 1.5) / total_graph_edges)
    
    topology_counts = {}   # (SrcType, DstType) -> count
    sample_props = []      # Heavy property dictionaries
    processed_count = 0
    
    # 2. Iterate ALL edges (Fast Scan)
    # We use keys=True to handle MultiDiGraph safely
    for u, v, k, attr in G.edges(keys=True, data=True):
        if attr.get("type") != target_type:
            continue

        # --- TOPOLOGY LOOKUP (Cheap) ---
        src_type = G.nodes[u].get("node_type", "Unknown")
        dst_type = G.nodes[v].get("node_type", "Unknown")
        pair = (src_type, dst_type)

        # --- SMART SAMPLER LOGIC ---
        # Rule 1: Always keep if it's a NEW topology pattern (Safety Net)
        is_new_topology = pair not in topology_counts
        
        # Rule 2: Randomly keep based on probability
        is_selected = random.random() < acceptance_prob

        if is_new_topology or is_selected:
            # Record Topology
            topology_counts[pair] = topology_counts.get(pair, 0) + 1
            processed_count += 1
            
            # Record Properties (Cap this smaller, e.g., 500, for speed)
            if len(sample_props) < 500:
                sample_props.append(list(attr.keys()))
                
            # Hard Break if we somehow massively overshoot (safety valve)
            if processed_count > (sample_limit * 2):
                break

    if processed_count == 0:
        return ""

    # 3. Analyze Properties
    prop_keys = set()
    for keys_list in sample_props:
        prop_keys.update(k for k in keys_list if k != 'type')

    # 4. Generate Profile
    profile = f"\n  [Detected Edge Group]: '{target_type}'\n"
    # Note: Counts are estimated from the sample, but topology presence is exact
    profile += f"    - Scanned Sample Size: ~{processed_count} edges (Probabilistic Sample)\n"
    profile += "    - Observed Connection Patterns (Source -> Target):\n"
    
    sorted_topo = sorted(topology_counts.items(), key=lambda x: x[1], reverse=True)
    
    for (src, dst), freq in sorted_topo:
        # Scale freq back up to estimate real size? No, keep raw sample counts 
        # but emphasizing presence.
        pct = (freq / processed_count) * 100
        profile += f"      * ({src}) -> ({dst}) [sample_freq: {freq}, {pct:.1f}%]\n"

    if prop_keys:
        profile += f"    - Edge Properties: {', '.join(sorted(list(prop_keys)))}\n"
    else:
        profile += "    - Edge Properties: [None]\n"

    return profile