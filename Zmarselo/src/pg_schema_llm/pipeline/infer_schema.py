from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from neo4j import GraphDatabase

# UPDATED: import from new io layout
from pg_schema_llm.io.neo4j_io import mine_patterns

# prompt lives in pg_schema_llm/llm/prompts.py (unchanged location)
from pg_schema_llm.llm import build_inference_prompt


@dataclass
class InferConfig:
    """
    Configuration container for schema inference.

    Neo4j connection parameters fall back to environment variables
    when left empty.
    """

    # Neo4j connection
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    neo4j_database: Optional[str] = None

    # pattern mining
    type_sample_limit: int = 500

    # LLM
    gemini_model: str = "gemini-2.5-flash"
    response_mime_type: str = "application/json"

    # logging
    verbose: bool = True


# ============================================================
# Logging helper
# ============================================================

def _p(verbose: bool, *args, **kwargs):
    """
    Conditional print helper for verbose logging.

    Args:
        verbose (bool): Whether logging is enabled.
    """
    if verbose:
        print(*args, **kwargs)


# ============================================================
# Neo4j driver factory  (reads .env when fields are empty)
# ============================================================

def _make_driver(cfg: InferConfig):
    """
    Create a Neo4j driver from config, falling back to .env variables.

    Environment variables consulted:
        NEO4J_URI       →  cfg.neo4j_uri
        NEO4J_USER      →  cfg.neo4j_user
        NEO4J_PASSWORD  →  cfg.neo4j_password
    """
    load_dotenv()
    uri = cfg.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = cfg.neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = cfg.neo4j_password or os.getenv("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, password))


# ============================================================
# LLM helpers
# ============================================================

def call_gemini_api(
    prompt: str,
    *,
    model_name: str,
    response_mime_type: str,
    verbose: bool,
) -> Optional[str]:
    """
    Invoke the Gemini LLM API for schema inference.

    Args:
        prompt (str): Fully constructed inference prompt.
        model_name (str): Gemini model identifier.
        response_mime_type (str): Expected response MIME type.
        verbose (bool): Whether to print diagnostic messages.

    Returns:
        Optional[str]: Raw response text, or None if the request fails.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        _p(verbose, "API Error: GOOGLE_API_KEY not set.")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    try:
        res = model.generate_content(
            prompt,
            generation_config={"response_mime_type": response_mime_type},
        )
        return res.text
    except Exception as e:
        _p(verbose, f"API Error: {e}")
        return None


def extract_json(text: Optional[str]) -> Optional[dict]:
    """
    Extract and parse a JSON object from an LLM response.

    Args:
        text (Optional[str]): Raw LLM response text.

    Returns:
        Optional[dict]: Parsed JSON object, or None if parsing fails.
    """
    if not text:
        return None
    try:
        cleaned = text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned)
    except Exception:
        return None


# ============================================================
# Profile construction (from mined patterns)
# ============================================================

def build_profile_text_from_patterns(patterns: dict) -> str:
    """
    Convert mined patterns into structured text for LLM consumption.

    Presents node and edge types with their patterns, property
    constraints, data types, and (for edges) cardinalities.

    Args:
        patterns: Dict returned by mine_patterns().

    Returns:
        str: Profile text used to build the LLM prompt.
    """
    lines: List[str] = []

    # ---- Node Types ----
    lines.append("=" * 60)
    lines.append("NODE TYPES")
    lines.append("=" * 60)

    for nt in patterns["node_types"]:
        labels = nt["labels"]
        label_str = ", ".join(labels) if labels else "(unlabeled)"
        lines.append("")
        lines.append(f"Node Type: labels={{{label_str}}}")
        lines.append(f"  Total instances: {nt['count']}")

        # patterns
        lines.append(f"  Distinct patterns ({len(nt['patterns'])}):")
        for pat in nt["patterns"]:
            pk = ", ".join(pat["property_keys"]) if pat["property_keys"] else "(no properties)"
            pct = round(100 * pat["count"] / nt["count"], 1) if nt["count"] > 0 else 0
            lines.append(f"    - {{{pk}}}: {pat['count']} instances ({pct}%)")

        # properties
        if nt["properties"]:
            lines.append(f"  Properties ({len(nt['properties'])}):")
            for pname, pinfo in nt["properties"].items():
                fill_pct = round(100 * pinfo["fill_ratio"], 1)
                lines.append(
                    f"    - {pname}: {pinfo['data_type']}, "
                    f"{pinfo['constraint']} ({fill_pct}%)"
                )
        else:
            lines.append("  Properties: none")

    # ---- Edge Types ----
    lines.append("")
    lines.append("=" * 60)
    lines.append("EDGE TYPES")
    lines.append("=" * 60)

    for et in patterns["edge_types"]:
        rel = et["labels"][0] if et["labels"] else "?"
        src = ", ".join(et["source_labels"]) if et["source_labels"] else "(unlabeled)"
        tgt = ", ".join(et["target_labels"]) if et["target_labels"] else "(unlabeled)"
        lines.append("")
        lines.append(f"Edge Type: ({{{src}}})-[:{rel}]->({{{tgt}}})")
        lines.append(f"  Total instances: {et['count']}")
        lines.append(f"  Cardinality: {et['cardinality']}  "
                      f"(max_out={et['max_out_degree']}, max_in={et['max_in_degree']})")

        # patterns
        lines.append(f"  Distinct patterns ({len(et['patterns'])}):")
        for pat in et["patterns"]:
            pk = ", ".join(pat["property_keys"]) if pat["property_keys"] else "(no properties)"
            pct = round(100 * pat["count"] / et["count"], 1) if et["count"] > 0 else 0
            lines.append(f"    - {{{pk}}}: {pat['count']} instances ({pct}%)")

        # properties
        if et["properties"]:
            lines.append(f"  Properties ({len(et['properties'])}):")
            for pname, pinfo in et["properties"].items():
                fill_pct = round(100 * pinfo["fill_ratio"], 1)
                lines.append(
                    f"    - {pname}: {pinfo['data_type']}, "
                    f"{pinfo['constraint']} ({fill_pct}%)"
                )
        else:
            lines.append("  Properties: none")

    return "\n".join(lines)


# ============================================================
# Main inference
# ============================================================

def infer_schema_from_folder(
    data_dir: str,
    config: Optional[InferConfig] = None,
) -> Optional[dict]:
    """
    Infer a property-graph schema.

    ``data_dir`` is accepted for backward compatibility but is no
    longer used — the graph is read directly from Neo4j.  Connection
    details come from ``config`` or from the environment variables
    in ``.env``.

    Args:
        data_dir (str): Ignored; retained so existing call sites need no change.
        config (Optional[InferConfig]): Inference configuration.

    Returns:
        Optional[dict]: Inferred schema, or None if inference fails.
    """
    cfg = config or InferConfig()

    _p(cfg.verbose, f"--- Connecting to Neo4j (data_dir arg ignored) ---")

    driver = _make_driver(cfg)
    try:
        patterns = mine_patterns(
            driver,
            type_sample_limit=cfg.type_sample_limit,
            database=cfg.neo4j_database,
        )
    finally:
        driver.close()

    profile_text = build_profile_text_from_patterns(patterns)

    _p(cfg.verbose, "\n--- PROFILE TEXT SENT TO LLM ---")
    _p(cfg.verbose, profile_text)
    _p(cfg.verbose, "--- END PROFILE TEXT ---\n")

    prompt = build_inference_prompt(profile_text)

    _p(cfg.verbose, f"--- Asking Gemini for schema: {cfg.gemini_model} ---")
    raw_res = call_gemini_api(
        prompt,
        model_name=cfg.gemini_model,
        response_mime_type=cfg.response_mime_type,
        verbose=cfg.verbose,
    )
    schema = extract_json(raw_res)
    if not schema:
        _p(cfg.verbose, "LLM returned no JSON schema (parse failed).")
        return None

    # attach raw patterns for downstream use / comparison
    schema["_mined_patterns"] = patterns
    return schema


def run_infer_schema(
    data_dir: str,
    output_path: str,
    config: Optional[InferConfig] = None,
) -> str:
    """
    Run schema inference and write the result to disk.

    ``data_dir`` is accepted for backward compatibility but is no
    longer used — see ``infer_schema_from_folder``.

    Args:
        data_dir (str): Ignored; retained so existing call sites need no change.
        output_path (str): Path to the output JSON file.
        config (Optional[InferConfig]): Inference configuration.

    Returns:
        str: Path to the written schema file.
    """
    schema = infer_schema_from_folder(data_dir, config=config)
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema or {}, f, indent=4, default=str)
    return output_path