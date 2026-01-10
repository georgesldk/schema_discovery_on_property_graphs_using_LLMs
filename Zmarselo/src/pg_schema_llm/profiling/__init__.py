"""
Profiling layer.

This package contains the profiling + heuristic analysis used to build the
DATA PROFILE section of the LLM prompt.

All logic here is moved 1:1 from scripts/main.py.
"""

from pg_schema_llm.profiling.node_profile import profile_node_type
from pg_schema_llm.profiling.edge_profile import profile_edge_type
from pg_schema_llm.profiling.heuristics import (
    identify_technical_containers,
    analyze_edge_semantics,
    analyze_logical_paths,
    analyze_bidirectional_patterns,
    generate_logical_relationship_summary,
)

__all__ = [
    "profile_node_type",
    "profile_edge_type",
    "identify_technical_containers",
    "analyze_edge_semantics",
    "analyze_logical_paths",
    "analyze_bidirectional_patterns",
    "generate_logical_relationship_summary",
]
