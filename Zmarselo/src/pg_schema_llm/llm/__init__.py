"""
LLM layer (Gemini) utilities.

This package contains:
- Gemini API call wrapper (expects JSON output)
- Prompt template assembly for schema inference
"""

from pg_schema_llm.llm.gemini_client import call_gemini_api, extract_json
from pg_schema_llm.llm.prompts import build_inference_prompt

__all__ = [
    "call_gemini_api",
    "extract_json",
    "build_inference_prompt",
]
