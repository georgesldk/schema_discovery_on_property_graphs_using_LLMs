from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import google.generativeai as genai
from dotenv import load_dotenv

# UPDATED: explicit import from new io layout
from pg_schema_llm.io.typestats import build_typestats

from pg_schema_llm.llm import build_inference_prompt
from pg_schema_llm.schema import typestats_from_dict
from pg_schema_llm.profiling import (
    profile_node_type_from_stats,
    profile_edge_type_from_stats,
    generate_logical_relationship_summary_from_stats,
)



@dataclass
class InferConfig:
    """
    Configuration container for schema inference.

    This dataclass defines all tunable parameters controlling statistics
    collection, LLM invocation, deterministic post-processing, and logging
    behavior for schema inference.
    """

    # stats building
    chunksize: int = 100_000
    sample_values_per_prop: int = 3

    # LLM
    gemini_model: str = "gemini-2.5-flash"
    response_mime_type: str = "application/json"

    # post-processing
    backfill_props_from_stats: bool = True
    backfill_missing_edge_props_by_label: bool = True

    # logging
    verbose: bool = True


# ============================================================
# Logging helper
# ============================================================

def _p(verbose: bool, *args, **kwargs):
    """
    Conditional print helper for verbose logging.

    This function centralizes verbosity checks to keep logging calls
    concise and consistent across the inference pipeline.

    Args:
        verbose (bool): Whether logging is enabled.
    """
    if verbose:
        print(*args, **kwargs)


# ============================================================
# LLM helpers
# ============================================================

def call_gemini_api(prompt: str, *, model_name: str, response_mime_type: str, verbose: bool) -> Optional[str]:
    """
    Invoke the Gemini LLM API for schema inference.

    This function configures the Gemini client, submits a schema inference
    prompt, and returns the raw textual response. API errors and missing
    credentials are handled defensively.

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
        res = model.generate_content(prompt, generation_config={"response_mime_type": response_mime_type})
        return res.text
    except Exception as e:
        _p(verbose, f"API Error: {e}")
        return None


def extract_json(text: Optional[str]) -> Optional[dict]:
    """
    Extract and parse a JSON object from an LLM response.

    This function removes common Markdown code-fence wrappers and attempts
    to deserialize the remaining content into JSON. Parsing failures are
    handled gracefully.

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
# Schema backfilling (stats-driven, deterministic)
# ============================================================

def _infer_best_kind(kind_counter, raw_prop: str) -> str:
    """
    Infer the most likely data type for a property from TypeStats.

    This function selects the property type with the highest observed
    frequency for a given property name, falling back to String when
    no evidence is available.

    Args:
        kind_counter (Counter): Counter mapping (property, kind) to counts.
        raw_prop (str): Raw property name.

    Returns:
        str: Inferred canonical property type.
    """

    # kind_counter: Counter((prop, kind) -> count)
    best_kind = None
    best_c = -1
    for (p, k), c in kind_counter.items():
        if p == raw_prop and c > best_c:
            best_kind, best_c = k, c
    return best_kind or "String"


def _clean_prop_name(raw: str) -> str:
    """
    Normalize a raw property name for schema output.

    This function removes Neo4j import artifacts, strips quoting, and
    filters out technical columns that should not appear in inferred
    schemas.

    Args:
        raw (str): Raw property name.

    Returns:
        str: Cleaned property name, or empty string if invalid.
    """

    if not raw:
        return ""
    p = str(raw).strip().strip('"').strip("'")

    # drop neo4j import columns
    if p.startswith(":"):
        return ""

    # common pattern: id:ID(Person) -> id
    if ":ID(" in p:
        p = p.split(":ID(", 1)[0].strip()

    # hard drop other import tokens if they appear
    if any(tok in p for tok in (":START_ID", ":END_ID", ":TYPE", ":LABEL")):
        return ""

    return p.strip()


def backfill_properties_from_typestats(schema: dict, ts) -> dict:
    """
    Backfill missing node and edge properties using TypeStats.

    This function deterministically populates empty property lists in
    the inferred schema using statistics derived directly from the data.
    It ensures completeness when the LLM omits properties.

    Args:
        schema (dict): Inferred schema returned by the LLM.
        ts: TypeStats object containing observed property statistics.

    Returns:
        dict: Updated schema with backfilled properties.
    """
    if not schema:
        return schema

    # nodes
    for n in schema.get("node_types", []) or []:
        nt = n.get("name")
        if not nt:
            continue
        ns = ts.node_types.get(nt)
        if not ns:
            continue

        props = n.get("properties") or []
        if props:
            continue

        out_props = []
        for raw_prop, filled in ns.prop_fill.items():
            prop = _clean_prop_name(raw_prop)
            if not prop:
                continue

            mandatory = (ns.count > 0) and (filled / ns.count >= 0.99)
            out_props.append({
                "name": prop,
                "type": _infer_best_kind(ns.prop_kind, raw_prop),
                "mandatory": bool(mandatory),
            })

        # stable order: most filled first
        out_props.sort(key=lambda x: (-ns.prop_fill.get(x["name"], 0), x["name"]))
        n["properties"] = out_props

    # edges (by label)
    for e in schema.get("edge_types", []) or []:
        et = e.get("name")
        if not et:
            continue
        es = ts.edge_types.get(et)
        if not es:
            continue

        props = e.get("properties") or []
        if props:
            continue

        out_props = []
        all_keys = set(es.prop_keys) | set(es.prop_fill.keys())
        for raw_prop in sorted(all_keys):
            prop = _clean_prop_name(raw_prop)
            if not prop:
                continue
            filled = es.prop_fill.get(raw_prop, 0)
            mandatory = (es.count > 0) and (filled / es.count >= 0.99)
            out_props.append({
                "name": prop,
                "type": _infer_best_kind(es.prop_kind, raw_prop),
                "mandatory": bool(mandatory),
            })
        e["properties"] = out_props

    return schema


def backfill_missing_edge_properties_by_label(schema: dict) -> dict:
    """
    Propagate edge properties across identical edge labels.

    If multiple edges share the same label and at least one occurrence
    defines properties, this function applies those properties to all
    other occurrences with empty property lists.

    Args:
        schema (dict): Inferred schema.

    Returns:
        dict: Updated schema with edge properties propagated by label.
    """
    edges = schema.get("edge_types") or []
    if not edges:
        return schema

    label_to_props: Dict[str, Dict[str, dict]] = {}

    for e in edges:
        name = e.get("name")
        if not name:
            continue
        for p in e.get("properties") or []:
            pn = (p.get("name") or "").strip()
            if not pn:
                continue
            label_to_props.setdefault(name, {})[pn] = p

    for e in edges:
        name = e.get("name")
        if not name or name not in label_to_props:
            continue
        if e.get("properties"):
            continue

        filled = []
        for pn, pobj in sorted(label_to_props[name].items(), key=lambda x: x[0].lower()):
            filled.append({
                "name": pobj.get("name", pn),
                "type": pobj.get("type", "String"),
                "mandatory": bool(pobj.get("mandatory", False)),
            })
        e["properties"] = filled

    return schema


# ============================================================
# Profile construction (stats path only)
# ============================================================

def build_profile_text_from_stats(ts) -> str:
    """
    Construct a textual data profile from TypeStats.

    This function converts node and edge statistics into a structured
    textual description suitable for LLM consumption, including logical
    relationship summaries.

    Args:
        ts: TypeStats object.

    Returns:
        str: Profile text used to build the LLM prompt.
    """
    node_types = ts.sorted_node_types()
    edge_types = ts.sorted_edge_types()

    profile = ""
    profile += "".join(profile_node_type_from_stats(ts, nt) for nt in node_types)
    profile += "".join(profile_edge_type_from_stats(ts, et) for et in edge_types)

    logical_summary = generate_logical_relationship_summary_from_stats(ts)
    if logical_summary:
        profile += logical_summary

    return profile


# ============================================================
# Main inference
# ============================================================

def infer_schema_from_folder(data_dir: str, config: Optional[InferConfig] = None) -> Optional[dict]:
    """
    Infer a property-graph schema from a dataset directory.

    This function performs streaming statistics collection, profile
    construction, LLM-based schema inference, and deterministic
    post-processing to produce a final schema.

    Args:
        data_dir (str): Path to the dataset directory.
        config (Optional[InferConfig]): Inference configuration.

    Returns:
        Optional[dict]: Inferred schema, or None if inference fails.
    """

    cfg = config or InferConfig()

    _p(cfg.verbose, f"--- Building TypeStats (streaming) for: {data_dir} ---")
    raw = build_typestats(
        data_dir,
        chunksize=cfg.chunksize,
        sample_values_per_prop=cfg.sample_values_per_prop,
    )
    ts = typestats_from_dict(raw)

    profile_text = build_profile_text_from_stats(ts)
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

    if cfg.backfill_props_from_stats:
        schema = backfill_properties_from_typestats(schema, ts)

    if cfg.backfill_missing_edge_props_by_label:
        schema = backfill_missing_edge_properties_by_label(schema)

    return schema


def run_infer_schema(data_dir: str, output_path: str, config: Optional[InferConfig] = None) -> str:
    """
    Run schema inference and write the result to disk.

    This function executes schema inference for a dataset directory and
    serializes the resulting schema to a JSON file.

    Args:
        data_dir (str): Path to the dataset directory.
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
        json.dump(schema or {}, f, indent=4)

    return output_path
