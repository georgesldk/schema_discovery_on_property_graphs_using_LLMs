"""
Utility helpers.

This package contains small, reusable helpers extracted from existing code.
No new behavior is introduced here.
"""

from pg_schema_llm.utils.text import strip_comments
from pg_schema_llm.utils.paths import get_dataset_name, ensure_dir

__all__ = [
    "strip_comments",
    "get_dataset_name",
    "ensure_dir",
]
