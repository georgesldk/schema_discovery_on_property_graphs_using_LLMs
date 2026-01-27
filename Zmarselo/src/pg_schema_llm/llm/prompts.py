def build_inference_prompt(profile_text):
    return f"""
    You are a Senior Property Graph Schema Architect. Your mission is to infer a high-fidelity Property Graph schema that mirrors the EXACT physical structure of the data provided in the profile.
    DATA PROFILE:
    {profile_text}
    
    TARGET:
    The user needs a schema that preserves 100% of the nodes and edge properties found in the data. Do NOT simplify, "clean up", or "collapse" the structure.

    CRITICAL AGNOSTIC HEURISTICS (Apply strictly in this order):

    1. **NO "SMART" MERGING (Distinctness Rule) - PRIORITY #1:**
       - **Constraint:** If the Data Profile lists distinct node types (e.g., "EntityA" and "EntityB"), you **MUST** output them as separate Node Types in your JSON.
       - **Reasoning:** Even if they share properties, they represent different entities.
       - **Strict Instruction:** Do NOT merge nodes just because they look similar. Keep them separate.

    2. **NOISE FILTER (The "Fake Node" Check):**
       - **Goal:** Identify properties that masquerade as nodes (e.g., Tags, Categories, Labels).
       - **Detection Logic:** Look at the [STRUCTURAL FINGERPRINTS] section. A node is a "Fake Node" if it meets BOTH criteria:
         * **Criteria A:** Low Information Density (Avg Properties < 2, excluding purely internal IDs).
         * **Criteria B:** Passive Role (It is a "Leaf" or "Sink" node with 0 outgoing edges to other entity types).
       - **Action:** If detected, DELETE the Node Type and add its name as a property to the Source Node (e.g., `label: String`).
       - **EXCEPTION:** If the node has *outgoing edges* to other entities, it is a structural bridge. KEEP IT.

    3. **PROPERTY FORMATTING (Strict JSON Structure):**
       - **Constraint:** The 'name' field in your JSON must contain **ONLY the property name**.
       - **Forbidden:** Do NOT include the type in the name (e.g., "id:long" is WRONG).
       - **Correct:** "name": "id", "type": "Long".
       - **Forbidden:** Do NOT output internal Neo4j keys like ":START_ID" or ":END_ID".

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