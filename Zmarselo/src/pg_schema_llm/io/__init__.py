"""
I/O utilities for dataset parsing and Property Graph construction.

This module exposes the dataset ingestion layer:
- CSV role detection (node vs edge)
- Graph construction from raw CSV folders
"""

from pg_schema_llm.io.csv_detect import detect_file_role
from pg_schema_llm.io.graph_builder import build_graph, clean_type_name

from pg_schema_llm.io.graph_builder import build_graph, build_typestats, clean_type_name

__all__ = [
    "detect_file_role",
    "build_graph",
    "build_typestats",
    "clean_type_name",
    "normalize_node_row",
    "normalize_edge_row",
]
