"""
I/O utilities for dataset parsing and Property Graph construction.

This package exposes the dataset ingestion layer:
- CSV role detection (node vs edge)
- Row normalization
- CSV helpers (sniff/read/chunks)
- Naming utilities
- Streaming typestats builder
- (Legacy) NetworkX graph builder
"""

from pg_schema_llm.io.detect import detect_file_role
from pg_schema_llm.io.normalize import normalize_node_row, normalize_edge_row
from pg_schema_llm.io.csv_tools import sniff_delimiter, read_preview, read_full_df, iter_chunks
from pg_schema_llm.io.naming import clean_name_smart, clean_type_name, get_common_affixes
from pg_schema_llm.io.typestats import build_typestats, build_graph

__all__ = [
    "detect_file_role",
    "normalize_node_row",
    "normalize_edge_row",
    "sniff_delimiter",
    "read_preview",
    "read_full_df",
    "iter_chunks",
    "clean_name_smart",
    "clean_type_name",
    "get_common_affixes",
    "build_typestats",
    "build_graph",
]
