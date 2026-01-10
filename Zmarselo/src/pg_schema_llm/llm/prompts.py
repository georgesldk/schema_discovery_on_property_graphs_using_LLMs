def build_inference_prompt(profile_text):
    """
    Build the Gemini prompt for logical Property Graph schema inference.

    This is a direct extraction of the prompt string from scripts/main.py.
    No logic or wording has been changed.
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
         * Use the suggested edge label from the "[LOGICAL RELATIONSHIP ANALYSIS]" section if provided, OR determine appropriate label (CONNECTS_TO, CONTAINS, or SYNAPSES_TO) based on semantic meaning
         * Analyze edge properties to determine semantics: properties like "weight", "strength", "distance" suggest CONNECTS_TO; properties suggesting membership/hierarchy suggest CONTAINS
         * DO NOT create edges that go through the technical container in your schema
       - If the profile includes "[LOGICAL RELATIONSHIP ANALYSIS]", those direct relationships are REQUIRED in your output.
       - The logical schema represents functional relationships, not physical storage artifacts.
       - Property Graphs prioritize direct semantic connections over intermediate technical structures.
    
    2. ENTITY CONSOLIDATION (Deduplicate Semantic Equivalents):
       - If multiple node types represent the same logical entity with different attribute sets, merge them into a single node type.
       - Properties from all variants should be merged, with "mandatory" set based on > 98% fill density.
       - Only keep truly distinct entity types that represent different concepts.
    
    3. STANDARDIZED EDGE LABELS (Property Graph Convention):
       - MANDATORY: Use ONLY these three edge labels - no exceptions:
         * CONNECTS_TO: For high-level structural/functional connections between major entities. Use when edges have properties like weight, strength, distance, or represent general connectivity.
         * CONTAINS: For parent-child or containment relationships (hierarchical). Use when one entity logically contains or groups another, or when relationships suggest membership/hierarchy.
         * SYNAPSES_TO: For fine-grained functional/operational links. Use sparingly, only when CONNECTS_TO is too coarse and the relationship represents a specific operational/junctional connection.
       - If the profile suggests an edge label, use that suggestion (it's based on semantic analysis of properties and patterns).
       - DO NOT use: technical names (HAS_SET, LINKS_TO), action verbs (CREATES, DELETES), file-based names (FROM_CSV, TO_TABLE), or generic verbs (ASSOCIATED_WITH, DEPENDS_ON).
       - Consolidate all edges between the same two node types into ONE edge type using the appropriate standard label.
       - Bidirectional relationships: If the profile indicates bidirectional patterns between two node types, you may need to create edges in both directions using the same edge label, OR create a single edge type that can be traversed in both directions (depending on the semantic meaning).
    
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
    - Semantic clarity over technical accuracy
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
          "name": "CONNECTS_TO|CONTAINS|SYNAPSES_TO",
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
