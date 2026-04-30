def build_inference_prompt(profile_text):
    return f"""
    You are a Senior Property Graph Schema Architect.  You will receive the
    results of an EXHAUSTIVE pattern mining pass over a Neo4j property graph.
    Every node and edge has been parsed; property data types come from a
    full scan of every value, not a sample.

    Your mission is to organize the mined patterns into a clean PG-Schema.
    The data is the source of truth — you are NOT cleaning it up, fixing it,
    or compensating for sampling error.  None of those exist here.

    DATA PROFILE FORMAT
    -------------------
    The profile uses a compact line-based format with a legend at the top.
    Each line begins with a one- or two-character marker:

      N:    a node type, followed by its label set in {{}} braces
      E:    a canonical edge type — labels(src)-[:REL]->labels(tgt) with
            a "card=..." cardinality string
      E*:   a derived subset edge (no cardinality)
      K:    one distinct (L, K) pattern of the preceding N or E entry —
            shows the property-key set in {{}} braces
      P:    the property list of the preceding N or E, comma-separated
            as "name:DATA_TYPE/M" (MANDATORY) or "name:DATA_TYPE/O" (OPTIONAL)

    Lines starting with "##" are section headers and "#" are comments —
    ignore them when constructing the schema, but treat all N/E/E*/K/P
    lines as authoritative.

    DATA PROFILE:
    {profile_text}

    ============================================================
    THEORETICAL FRAMEWORK  (PG-Schema / PG-HIVE, EDBT 2026)
    ============================================================

      Node Pattern  TNp = (L, K)
        L ⊆ Labels — the full label set on the node (may be empty or multi-label).
        K ⊆ Keys   — the set of property keys present on the node.

      Edge Pattern  TEp = (L, K, R)
        L ⊆ Labels — the edge label set (always non-empty in Neo4j).
        K ⊆ Keys   — the edge property-key set.
        R = (Ls, Lt) — source and target node label sets.

      Two patterns are DISTINCT when their label set, property-key set, or
      (for edges) endpoints differ.  Multiple patterns may belong to the
      same *type* when they share labels but differ in property-key sets.

      Node Type  Vs = (labels, properties)
        Each property: name, data_type, constraint ∈ {{MANDATORY, OPTIONAL}}.
        MANDATORY = fill_ratio = 1.0 (appears in every instance of the type).

      Edge Type  Es = (labels, properties, endpoints, cardinality)
        endpoints   = (source_node_labels, target_node_labels).
        cardinality = "{{src_min..src_max}} : {{tgt_min..tgt_max}}",
                      with each side ∈ {{0..1, 0..N, 1..1, 1..N}}.

    ============================================================
    HARD RULES  (NON-NEGOTIABLE)
    ============================================================

    R1. **PRESERVE MINED LABELS — DO NOT RENAME, DO NOT INVENT.**
       - Every label that appears in the profile MUST appear verbatim in the
         output (same casing, same spelling).
       - You may NOT change capitalization, abbreviate, expand, translate,
         or otherwise transform any label.
       - You may NOT introduce new labels for nodes that already have labels.
       - The "labels" array of every node type is COPIED from the profile.
       - The "name" field of every edge_type is COPIED from the profile's
         relationship type, verbatim.

    R2. **NAMING RULE.**
       - For node types WITH labels:
           "name" MUST be exactly one of the labels in the type's label set.
           Choose whichever single label is the most semantically meaningful
           identifier for the type.  Do NOT concatenate labels.  Do NOT
           invent a new name.  Do NOT modify a label to make a name.
       - For node types with EMPTY label set (labels = []):
           ONLY THEN may you invent a descriptive PascalCase name based on
           the type's properties.

    R3. **EDGES ARE NEVER UNLABELED.**
       - Every edge in the profile has a relationship type.  Use it verbatim.
       - You may NOT invent edge names, abbreviate them, derive them from
         endpoint types, or merge two distinct relationship types.

    R4. **PRESERVE SOURCE / TARGET LABEL SETS.**
       - Every edge_type in your output MUST include "source_labels" and
         "target_labels" arrays — copied verbatim from the profile.
       - If the profile lists multiple endpoint combinations for the same
         relationship type (canonical and subset permutations), output a
         SEPARATE edge_type entry for each combination.
       - Mark subset permutations with "is_canonical": false.

    R5. **NO PATTERN COLLAPSING, NO PROPERTY FILTERING.**
       - If a node/edge type has multiple distinct (L, K) patterns, the
         union of their property keys appears in the type's property list.
       - Each property's "constraint" is MANDATORY only when fill_ratio = 1.0
         in the profile, OPTIONAL otherwise.
       - Do NOT remove properties for being uncommon, technical-looking, or
         stylistically odd.  The data is exhaustive — every property is real.

    R6. **PROPERTY FORMATTING.**
       - "name" contains only the property name, copied verbatim from the
         profile.  Do NOT include the type in the name.
       - Do NOT emit Neo4j import keys (":START_ID", ":END_ID", ":TYPE",
         ":LABEL").  If the profile contains them, drop them.
       - "type" copies the data_type from the profile (STRING / INTEGER /
         DOUBLE / BOOLEAN / DATE / LIST).
       - "constraint" copies MANDATORY / OPTIONAL from the profile.

    R7. **CARDINALITY IS COPIED, NOT INFERRED.**
       - Copy each canonical edge's cardinality string verbatim from the
         profile.  Subset (is_canonical=false) edges have cardinality=null.

    ============================================================
    OUTPUT JSON FORMAT
    ============================================================

    {{
      "node_types": [
        {{
          "name": "<one of the labels>",
          "labels": ["<label_1>", "<label_2>", "..."],
          "properties": [
            {{"name": "<property_name>", "type": "STRING|INTEGER|DOUBLE|BOOLEAN|DATE|LIST", "constraint": "MANDATORY|OPTIONAL"}}
          ]
        }}
      ],
      "edge_types": [
        {{
          "name": "<relationship_type_verbatim>",
          "source_labels": ["<src_label_1>", "..."],
          "target_labels": ["<tgt_label_1>", "..."],
          "is_canonical": true,
          "cardinality": "<min..max : min..max>",
          "properties": [
            {{"name": "<property_name>", "type": "STRING|INTEGER|DOUBLE|BOOLEAN|DATE|LIST", "constraint": "MANDATORY|OPTIONAL"}}
          ]
        }}
      ],
      "notes": "Free-form observations, e.g. about unlabeled types."
    }}
    """