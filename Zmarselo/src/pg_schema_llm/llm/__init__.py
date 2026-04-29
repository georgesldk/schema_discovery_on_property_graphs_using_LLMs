"""
Large Language Model (LLM) interface utilities for schema inference.

This package provides the abstraction layer between the schema
discovery pipeline and prompt construction helpers used by the
inference flow.

The public API re-exports a minimal interface for building
inference prompts.
"""
from pg_schema_llm.llm.prompts import build_inference_prompt

__all__ = [
    "build_inference_prompt",
]
