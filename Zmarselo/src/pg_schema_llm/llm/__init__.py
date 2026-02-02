"""
Large Language Model (LLM) interface utilities for schema inference.

This package provides the abstraction layer between the schema
discovery pipeline and the underlying LLM backend. It includes
helpers for prompt construction, model invocation, and structured
JSON extraction from model responses.

The public API re-exports a minimal, stable interface for:
- Building inference prompts
- Calling the Gemini LLM API
- Extracting structured JSON outputs
"""
from pg_schema_llm.llm.gemini_client import call_gemini_api, extract_json
from pg_schema_llm.llm.prompts import build_inference_prompt

__all__ = [
    "call_gemini_api",
    "extract_json",
    "build_inference_prompt",
]
