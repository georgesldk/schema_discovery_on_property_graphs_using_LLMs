def build_inference_prompt(profile_text):
    return f"""
    You are a Senior Property Graph Schema Architect. Your mission is to infer a high-fidelity Property Graph schema that mirrors the EXACT physical structure of the data provided in the profile.
    DATA PROFILE:
    {profile_text}
    
    TARGET:
    The user needs a schema that preserves 100% of the nodes and edge properties found in the data. Do NOT simplify, "clean up", or "collapse" the structure.

    STRICT DATASET-AGNOSTIC HEURISTICS:

    1. STRUCTURAL FIDELITY (NO LOGICAL BYPASS):
       - **Preserve Intermediaries:** Do NOT "collapse" or "bypass" nodes based on names like "Set" or "Group".
       - **Quantitative Data Rule:** If a node group contains ANY quantitative properties (e.g., counts, weights, scores, thresholds) or metadata (e.g., timestamps, user info), it is a **FUNCTIONAL ENTITY**. You MUST keep it as a Node Type.
       - **Topology Rule:** If the data profile shows `Entity A -> Group B -> Entity C`, you MUST define Edges `A->B` and `B->C`. Do not create a fake `A->C` edge unless it physically exists in the data.

    2. NODE DISTINCTNESS (NO MERGING):
       - **Structural Role Check:** Do NOT merge two node groups if they connect to different things. (e.g., If Group A connects to X, but Group B connects to Y, they are different).
       - **Source Distinctness:** If the profile lists two groups with different source file origins or distinct label sets, keep them separate.
       - **Metadata Nodes:** Nodes that store system info or metadata (often singular, high-connectivity nodes) must be preserved, not merged into others.
       - **ATTRIBUTE FLATTENING (CRITICAL):** If a node group (e.g. "Tags", "Status") has NO outgoing connections and acts only as a dictionary of definitions, **FLATTEN IT** into a property. Do NOT create a Node Type for it.
       - **If a node has < 2 meaningful properties AND no outgoing edges (like "Tags"), delete the node and make it a property of its parent.

    3. MANDATORY EDGE PROPERTIES (CRITICAL):
       - **Look at the 'Edge Properties' list in the profile for every edge.**
       - If an edge has extra columns (e.g., "weight", "score", "timestamp", "confidence"), you **MUST** include them in the `properties` list of that Edge Type.
       - Do NOT output `"properties": []` if the profile shows data columns for that edge.

    4. EDGE NAMING METHODOLOGY (SEMANTIC DERIVATION):
       - **Constraint:** Do not use a pre-set list of verbs. Deriving the name must follow this 3-step logic:
       
       * **STEP 1: ANALYZE SIGNAL:** Look at the edge properties and the Source/Target types.
           * *Signal A:* Properties imply measurement (weight, distance, score).
           * *Signal B:* Properties imply sequence or action (time, duration, flow).
           * *Signal C:* Relationship implies ownership or composition (part-of, member-of).
           
       * **STEP 2: DETERMINE CATEGORY:**
           * If *Signal A* (Measurement) -> The Category is **TOPOLOGICAL**. Use verbs describing linkage or connection strength (e.g., LINKS, CONNECTS).
           * If *Signal B* (Action) -> The Category is **FUNCTIONAL**. Use specific active verbs describing the process (e.g., PROCESSES, TRIGGERS).
           * If *Signal C* (Ownership) -> The Category is **STRUCTURAL**. Use verbs describing containment or hierarchy (e.g., CONTAINS, INCLUDES).

       * **STEP 3: GRAMMAR FILTER (REDUNDANCY REMOVAL):**
           * **Rule:** The Edge Name MUST NOT repeat the Target Node's name.
           * *Bad:* `Parent` -> `HAS_PARENT_GROUP` -> `Group` (Redundant).
           * *Good:* `Parent` -> `INCLUDES` -> `Group`.
           * *Bad:* `System` -> `LINKS_TO_SYSTEM` -> `System` (Redundant).
           * *Good:* `System` -> `CONNECTS` -> `System` (if topological) or `INTERACTS` -> `System` (if functional).
   
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
          "name": "INFERRED_SEMANTIC_VERB",
          "start_node": "SourceNodeLabel",
          "end_node": "TargetNodeLabel",
          "properties": [
            {{"name": "propertyName", "type": "String|Long|Double|Boolean", "mandatory": true|false}}
          ]
        }}
      ]
    }}
    """