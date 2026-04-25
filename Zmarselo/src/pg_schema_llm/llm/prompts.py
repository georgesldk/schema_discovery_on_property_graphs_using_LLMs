def build_inference_prompt(profile_text):
    return f"""
    You are a Senior Property Graph Schema Architect.  You will receive the
    results of an EXHAUSTIVE pattern mining pass over a Neo4j property graph.
    Your mission is to infer a high-fidelity PG-Schema that mirrors the EXACT
    physical structure of the data.

    DATA PROFILE:
    {profile_text}

    TARGET:
    The user needs a schema that preserves 100% of the node and edge properties
    found in the data.  Do NOT simplify, "clean up", or "collapse" the structure.

    ============================================================
    THEORETICAL FRAMEWORK 
    ============================================================

      Node Pattern  TNp = (L, K)
        L ⊆ Labels — the full label set on the node (may be empty or multi-label).
        K ⊆ Keys   — the set of property keys present on the node.

      Edge Pattern  TEp = (L, K, R)
        L ⊆ Labels — the edge label set (singleton in Neo4j).
        K ⊆ Keys   — the edge property-key set.
        R = (Ls, Lt) — source and target node label sets.

      Two patterns are DISTINCT when their label set, property-key set,
      or (for edges) endpoints differ.  Multiple patterns may belong to the
      same *type* when they share labels but differ in property-key sets.

      Node Type  Vs = (labels, properties)
        Each property: name, data_type, constraint ∈ {{MANDATORY, OPTIONAL}}.
        MANDATORY = fill_ratio = 1.0 (appears in every instance of the type).

      Edge Type  Es = (labels, properties, endpoints, cardinality)
        endpoints = (source_node_type, target_node_type).
        cardinality ∈ {{1:1, 1:N, N:1, M:N}}  — derived from max in/out-degree.

    ============================================================
    CRITICAL HEURISTICS  (apply strictly in this order)
    ============================================================

    1. **NO "SMART" MERGING  (Distinctness Rule) — PRIORITY #1:**
       - **Constraint:** If the Data Profile lists distinct node types (distinct label sets),
         you **MUST** output them as separate Node Types in your JSON.
       - **Reasoning:** Even if they share properties, they represent different entities.
       - **Strict Instruction:** Do NOT merge nodes just because they look similar. Keep them separate.

    2. **MULTI-LABEL AWARENESS:**
       - A node type is defined by its FULL label set.  {{Person}} and
         {{Person, Actor}} are two different types.
       - Preserve every label set exactly as mined.
       - An empty label set {{}} becomes an ABSTRACT type — suggest a descriptive name.

    3. **PATTERN PRESERVATION:**
       - Report the number of distinct patterns per type.
       - If a type has multiple patterns (e.g., some instances have property "bday" and
         others don't), every property that appears in ANY pattern must appear in the
         type's property list — marked MANDATORY or OPTIONAL per the mined fill_ratio.

    4. **NOISE FILTER  (The "Fake Node" Check):**
       - **Goal:** Identify properties that masquerade as nodes (e.g., Tags, Categories, Labels).
       - **Detection Logic:** A node is a "Fake Node" if it meets BOTH criteria:
         * **Criteria A:** Low Information Density (≤ 1 non-ID property).
         * **Criteria B:** Passive Role (0 outgoing edges to other entity types — pure leaf/sink).
       - **Action:** If detected, DELETE the Node Type and add its name as a property to the Source Node.
       - **EXCEPTION:** If the node has *outgoing edges* to other entities, it is a structural bridge. KEEP IT.

    5. **PROPERTY FORMATTING (Strict JSON Structure):**
       - **Constraint:** The 'name' field in your JSON must contain **ONLY the property name**.
       - **Forbidden:** Do NOT include the type in the name (e.g., "id:long" is WRONG).
       - **Correct:** "name": "id", "type": "INTEGER".
       - **Forbidden:** Do NOT output internal Neo4j keys like ":START_ID" or ":END_ID".
       - Respect the mined data type (STRING, INTEGER, DOUBLE, BOOLEAN, DATE, LIST).
       - Respect the mined constraint (MANDATORY / OPTIONAL).

    6. **EDGE NAMING METHODOLOGY (SEMANTIC DERIVATION):**
       - **Constraint:** Do not use a pre-set list of verbs. Deriving the name must follow this 3-step logic:

       * **STEP 1: ANALYZE SIGNAL:** Look at the edge properties and the Source/Target types.
           * *Signal A:* Properties imply measurement (weight, distance, score).
           * *Signal B:* Properties imply sequence or action (time, duration, flow).
           * *Signal C:* Relationship implies ownership or composition (part-of, member-of).

       * **STEP 2: DETERMINE CATEGORY:**
           * If *Signal A* (Measurement) → Category = **TOPOLOGICAL**. Use linkage verbs (LINKS, CONNECTS).
           * If *Signal B* (Action) → Category = **FUNCTIONAL**. Use active verbs (PROCESSES, TRIGGERS).
           * If *Signal C* (Ownership) → Category = **STRUCTURAL**. Use hierarchy verbs (CONTAINS, INCLUDES).

       * **STEP 3: GRAMMAR FILTER (REDUNDANCY REMOVAL):**
           * **Rule:** The Edge Name MUST NOT repeat the Target Node's name.
           * *Bad:* `Parent` → `HAS_PARENT_GROUP` → `Group` (Redundant).
           * *Good:* `Parent` → `INCLUDES` → `Group`.
           * *Bad:* `System` → `LINKS_TO_SYSTEM` → `System` (Redundant).
           * *Good:* `System` → `CONNECTS` → `System` (if topological) or `INTERACTS` → `System` (if functional).

    7. **CARDINALITY:**
       - Preserve the mined cardinality for every edge type.  It is derived
         from observed max in/out-degree:
           (max_out ≤ 1, max_in ≤ 1) → 1:1
           (max_out > 1, max_in ≤ 1) → N:1
           (max_out ≤ 1, max_in > 1) → 1:N
           (max_out > 1, max_in > 1) → M:N

    ============================================================
    OUTPUT JSON FORMAT
    ============================================================

    {{
      "node_types": [
        {{
          "name": "NodeLabel",
          "labels": ["Label1", "Label2"],
          "properties": [
            {{"name": "propertyName", "type": "STRING|INTEGER|DOUBLE|BOOLEAN|DATE|LIST", "constraint": "MANDATORY|OPTIONAL"}}
          ]
        }}
      ],
      "edge_types": [
        {{
          "name": "RELATIONSHIP_TYPE",
          "source": "SourceNodeName",
          "target": "TargetNodeName",
          "cardinality": "1:1|1:N|N:1|M:N",
          "properties": [
            {{"name": "propertyName", "type": "STRING|INTEGER|DOUBLE|BOOLEAN|DATE|LIST", "constraint": "MANDATORY|OPTIONAL"}}
          ]
        }}
      ],
      "notes": "Any observations about semantic overlap, unlabeled types, or anomalies."
    }}
    """