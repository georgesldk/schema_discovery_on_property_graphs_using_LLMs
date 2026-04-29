"""
Profiling layer for property-graph schema inference.

This package provides profiling and heuristic analysis utilities used
to construct the DATA PROFILE section of the LLM inference prompt.

The public API exposes statistics-based profiling and structural
heuristics. Legacy graph-materialization profiling has been removed.
"""


# Stats-based profiling (scalable / large datasets)
from pg_schema_llm.profiling.node_profile import profile_node_type_from_stats
from pg_schema_llm.profiling.edge_profile import profile_edge_type_from_stats
from pg_schema_llm.profiling.heuristics import (
    identify_technical_containers_from_stats,
    analyze_logical_paths_from_stats,
    analyze_bidirectional_patterns_from_stats,
    generate_logical_relationship_summary_from_stats,
)

__all__ = [
    # stats-based
    "profile_node_type_from_stats",
    "profile_edge_type_from_stats",
    "identify_technical_containers_from_stats",
    "analyze_logical_paths_from_stats",
    "analyze_bidirectional_patterns_from_stats",
    "generate_logical_relationship_summary_from_stats",
]
