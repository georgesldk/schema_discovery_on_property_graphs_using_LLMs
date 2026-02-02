"""
I/O utilities for dataset parsing and Property Graph construction.

This package provides the dataset ingestion layer and exposes a stable public API for:
- Detecting CSV file roles (node vs edge)
- Normalizing rows into a consistent internal representation
- Reading CSVs reliably (delimiter sniffing, previews, full reads, chunk iteration)
- Cleaning and standardizing names/types across heterogeneous datasets
- Building streaming type statistics for schema inference
- (Legacy) Constructing a NetworkX graph when needed

Notes:
- This module intentionally re-exports selected functions to keep imports concise
  (e.g., `from pg_schema_llm.io import detect_file_role`).
- `__all__` defines the supported public surface; anything not listed should be
  treated as internal and subject to change.
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
