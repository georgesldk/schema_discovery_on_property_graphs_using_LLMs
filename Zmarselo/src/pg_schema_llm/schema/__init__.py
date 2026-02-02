"""
Schema-level utilities for property-graph analysis.

This package provides operations that act directly on schema objects
rather than raw data. It includes utilities for schema normalization,
type matching, property comparison, and evaluation metrics.

The functionality exposed here represents logic factored out of the
pipeline layer to enable reuse, clearer separation of concerns, and
more testable schema-centric operations.
"""


from pg_schema_llm.schema.normalize import normalize_topology
from pg_schema_llm.schema.matching import similar, find_best_match

from pg_schema_llm.schema.models import TypeStats, NodeTypeStats, EdgeTypeStats, typestats_from_dict

__all__ = [
    "normalize_topology",
    "similar",
    "find_best_match",
    "TypeStats",
    "NodeTypeStats",
    "EdgeTypeStats",
    "typestats_from_dict",
]
