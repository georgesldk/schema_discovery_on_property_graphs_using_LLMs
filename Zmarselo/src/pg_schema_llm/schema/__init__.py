"""
Schema-level utilities.

This package contains logic that operates on schema objects themselves:
- normalization
- matching
- property comparison
- scoring metrics

All functions are moved 1:1 from existing pipeline code.
"""

from pg_schema_llm.schema.normalize import normalize_topology
from pg_schema_llm.schema.matching import similar, find_best_match

__all__ = [
    "normalize_topology",
    "similar",
    "find_best_match",
]

