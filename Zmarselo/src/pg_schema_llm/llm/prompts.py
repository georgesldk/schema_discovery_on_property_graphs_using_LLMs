def build_inference_prompt(profile_text):
    """
    Build the Gemini prompt for logical Property Graph schema inference.
    
    UPDATED: Now strictly dataset-agnostic for Edge Types. 
    Removes hardcoded 'CONNECTS_TO/CONTAINS/SYNAPSES_TO' constraints 
    and enforces semantic verb inference.
    """

    prompt = f"""
    You are a Senior Property Graph Schema Architect. Your mission is to infer a logical Property Graph schema from raw physical data structures, following industry-standard Property Graph principles.
    
    DATA PROFILE:
    {profile_text}
    
    STRICT DATASET-AGNOSTIC PROPERTY GRAPH HEURISTICS:
    
    1. LOGICAL BYPASS RULE (Collapse Technical Intermediaries) - HIGHEST PRIORITY:
       - Technical Container nodes are grouping/collection mechanisms with minimal properties (typically < 3 meaningful properties beyond IDs).
       - Patterns: names containing "Set", "Collection", "Group", "Container", "Link", "Join", "Mapping", "Association".
       - MANDATORY ACTION: If the profile shows "Entity A -> TechnicalContainer -> Entity C" paths, you MUST:
         * Create a direct edge type: Entity A -> Entity C
         * Use the suggested edge label from the "[LOGICAL RELATIONSHIP ANALYSIS]" section if provided.
         * If not provided, determine the appropriate SEMANTIC VERB (e.g., MEMBERS, CONTAINS, LINKED_TO) based on the context.
         * Analyze edge properties to determine semantics: properties like "weight" suggest connectivity strength; properties suggesting hierarchy suggest containment.
         * DO NOT create edges that go through the technical container in your schema.
       - If the profile includes "[LOGICAL RELATIONSHIP ANALYSIS]", those direct relationships are REQUIRED in your output.
       - The logical schema represents functional relationships, not physical storage artifacts.
    
    2. ENTITY CONSOLIDATION (Deduplicate Semantic Equivalents):
       - If multiple node types represent the same logical entity with different attribute sets, merge them into a single node type.
       - Properties from all variants should be merged, with "mandatory" set based on > 98% fill density.
       - Only keep truly distinct entity types that represent different concepts.
    
    3. SEMANTIC EDGE NAMING (Dynamic Contextual Inference):
       - MANDATORY: You must infer specific, descriptive relationship verbs based on the Source and Target node context.
       - DO NOT use generic/lazy labels (e.g., "RELATED_TO", "HAS", "EDGE") unless the data is completely abstract.
       - INFERENCE LOGIC:
         * **Action-Based:** If Source is an Actor (User) and Target is an Action (Order), use the verb (e.g., "PLACED", "EXECUTED").
         * **Hierarchy-Based:** If Target is a sub-component of Source, use "CONTAINS" or "INCLUDES".
         * **Ownership-Based:** If Source possesses Target, use specific ownership verbs (e.g., "OWNS", "MAINTAINS").
         * **Spatial:** If properties involve coordinates/location, use "LOCATED_AT" or "NEAR".
       - Consolidate all edges between the same two node types into ONE edge type using the most descriptive label.
       - Bidirectional relationships: If appropriate, create a single edge type that implies the connection, or explicit directional edges if the semantics differ (e.g., "FOLLOWS" vs "FOLLOWED_BY").
    
    4. SCHEMA NORMALIZATION:
       - NODE NAMING: Singular PascalCase (e.g., "Entity", "Item", "Category" - NOT "Entities", "EntityType").
       - PROPERTY TYPES: Use "String", "Long", "Double", "Boolean", "StringArray", "Point" (for spatial data).
       - MANDATORY FLAGS: Set "mandatory: true" ONLY for properties with > 98% fill density across all instances.
       - EDGE PROPERTIES: Include edge properties (e.g., weight, confidence, count) when they carry semantic meaning.
    
    5. TOPOLOGY REQUIREMENTS:
       - Each edge_type MUST specify "start_node" and "end_node" fields with the exact node type names.
       - Self-loops (same node type as source and target) are allowed and valid.
       - The schema should represent a logical graph, not a physical data model.
    
    CRITICAL: This is a Property Graph schema, not a relational model. Focus on:
    - Direct entity-to-entity relationships
    - Logical, not physical, structure  
    - Semantic clarity (Specific Verbs) over technical accuracy
    - Standard naming conventions
    
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

    return prompt