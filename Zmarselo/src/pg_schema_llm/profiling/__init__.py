"""
Profiling layer for property-graph schema inference.

This package provides profiling and heuristic analysis utilities used
to construct the DATA PROFILE section of the LLM inference prompt.
It supports both legacy graph-based profiling (for small datasets)
and scalable statistics-based profiling (for large datasets).

The module exposes a stable public API that prefers statistics-based
profiling functions, while conditionally supporting legacy graph-based
heuristics when available. This design allows the pipeline to scale
without forcing full graph materialization.
"""


# Graph-based profiling (legacy / small datasets)
from pg_schema_llm.profiling.node_profile import profile_node_type
from pg_schema_llm.profiling.edge_profile import profile_edge_type

# Try importing legacy graph-based heuristics if they exist.
# (In your current repo, profiling/heuristics.py may not define them.)
try:
    from pg_schema_llm.profiling.heuristics import (
        identify_technical_containers,
        analyze_edge_semantics,
        analyze_logical_paths,
        analyze_bidirectional_patterns,
        generate_logical_relationship_summary,
    )
    _HAS_LEGACY_HEURISTICS = True
except ImportError:
    _HAS_LEGACY_HEURISTICS = False

    # Optional: define stubs so imports don't break if someone references them.
    def identify_technical_containers(*args, **kwargs):
        raise NotImplementedError("Legacy graph-based heuristics not available in profiling/heuristics.py")

    def analyze_edge_semantics(*args, **kwargs):
        raise NotImplementedError("Legacy graph-based heuristics not available in profiling/heuristics.py")

    def analyze_logical_paths(*args, **kwargs):
        raise NotImplementedError("Legacy graph-based heuristics not available in profiling/heuristics.py")

    def analyze_bidirectional_patterns(*args, **kwargs):
        raise NotImplementedError("Legacy graph-based heuristics not available in profiling/heuristics.py")

    def generate_logical_relationship_summary(*args, **kwargs):
        raise NotImplementedError("Legacy graph-based heuristics not available in profiling/heuristics.py")


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
    # graph-based profiles always exist
    "profile_node_type",
    "profile_edge_type",

    # stats-based (what your new pipeline uses)
    "profile_node_type_from_stats",
    "profile_edge_type_from_stats",
    "identify_technical_containers_from_stats",
    "analyze_logical_paths_from_stats",
    "analyze_bidirectional_patterns_from_stats",
    "generate_logical_relationship_summary_from_stats",
]

# only export legacy heuristics if present
if _HAS_LEGACY_HEURISTICS:
    __all__ += [
        "identify_technical_containers",
        "analyze_edge_semantics",
        "analyze_logical_paths",
        "analyze_bidirectional_patterns",
        "generate_logical_relationship_summary",
    ]
