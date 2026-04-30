from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from neo4j import GraphDatabase

# UPDATED: Neo4j-based pattern mining
from pg_schema_llm.io.neo4j_io import mine_patterns

# prompt lives in pg_schema_llm/llm/prompts.py
from pg_schema_llm.llm import build_inference_prompt


@dataclass
class InferConfig:
    """
    Configuration container for schema inference.

    Neo4j connection parameters fall back to environment variables when
    left empty.
    """

    # Neo4j connection
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    neo4j_database: Optional[str] = None

    # pattern mining
    chunk_size: int = 5000
    expand_edge_subsets: bool = True

    # LLM
    gemini_model: str = "gemini-3-flash-preview"
    response_mime_type: str = "application/json"

    # logging
    verbose: bool = True


# ============================================================
# Logging helper
# ============================================================

def _p(verbose: bool, *args, **kwargs):
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
    max_output_tokens: int = 65536,
) -> Optional[str]:
    """
    Invoke the Gemini LLM API for schema inference.

    Returns the raw text response (or None on failure).  Logs the
    finish_reason and a preview of the response when verbose, so that
    truncation, safety filtering, or malformed JSON are easy to diagnose.
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
            generation_config={
                "response_mime_type": response_mime_type,
                "max_output_tokens": max_output_tokens,
            },
        )
    except Exception as e:
        _p(verbose, f"API Error: {e}")
        return None

    # ---- diagnostics -------------------------------------------------
    if verbose:
        try:
            cand = res.candidates[0] if getattr(res, "candidates", None) else None
            finish = getattr(cand, "finish_reason", None) if cand else None
            usage = getattr(res, "usage_metadata", None)
            _p(verbose, f"  finish_reason: {finish}")
            if usage:
                _p(verbose,
                   f"  tokens: prompt={getattr(usage, 'prompt_token_count', '?')} "
                   f"output={getattr(usage, 'candidates_token_count', '?')} "
                   f"total={getattr(usage, 'total_token_count', '?')}")
        except Exception as e:
            _p(verbose, f"  (could not read response metadata: {e})")

    # ---- extract text ------------------------------------------------
    try:
        text = res.text
    except Exception:
        # res.text raises when the candidate finished with a non-STOP reason
        # (truncation, safety, etc.).  Fall back to manually concatenating parts.
        text = None
        try:
            parts = res.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") for p in parts) or None
        except Exception:
            text = None

    if verbose and text:
        preview = text if len(text) <= 400 else text[:200] + "  …  " + text[-200:]
        _p(verbose, f"  response preview: {preview}")

    return text


def extract_json(text: Optional[str], *, verbose: bool = False) -> Optional[dict]:
    """
    Extract and parse a JSON object from an LLM response.

    Tries:
      1. Strip code-fences and parse.
      2. Find the largest balanced {...} substring and parse.
      3. Last-resort: trim trailing chars one at a time looking for
         a parseable prefix (rescues simple truncations).
    """
    if not text:
        return None

    cleaned = text.strip().replace("```json", "").replace("```", "").strip()

    # 1. straight parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. balanced-brace substring
    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        end = -1
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                return json.loads(cleaned[start:end])
            except Exception:
                pass

    # 3. truncation rescue: walk back from the end, closing braces as we go
    if start >= 0:
        body = cleaned[start:]
        for trim in range(0, min(2000, len(body))):
            candidate = body[: len(body) - trim]
            for closers in ("", "}", "]}", "]]}", "]}]}"):
                try:
                    return json.loads(candidate + closers)
                except Exception:
                    continue

    if verbose:
        _p(True, f"  JSON parse failed; first 300 chars of cleaned text:")
        _p(True, f"    {cleaned[:300]}")
    return None


# ============================================================
# Profile construction (from mined patterns)
# ============================================================

def build_profile_text_from_patterns(patterns: dict) -> str:
    """
    Convert mined patterns into structured text for LLM consumption.

    Includes node patterns, edge patterns, source/target label sets,
    cardinality, and is_canonical flag.

    Compact format — minimises tokens while preserving every field the
    LLM is required to copy.  Emitted shape::

        N: {Label1, Label2}
          K: {prop_a, prop_b}
          K: {prop_a}
          P: prop_a:STRING/M, prop_b:INTEGER/O

        E: {Src1}-[:REL]->{Tgt1}  card=1..1:0..N
          K: {weight}
          P: weight:INTEGER/O
        E*: {Src1}-[:REL]->{Tgt1,Tgt2}      (subset, no cardinality)
          K: {weight}
          P: weight:INTEGER/O

    Legend (placed once at the top):
        N = node type, E = canonical edge, E* = subset edge,
        K = pattern (property-key set), P = property list,
        /M = MANDATORY, /O = OPTIONAL.
    """
    lines: List[str] = []

    # ---- Legend (once) ----
    lines.append(
        "# Legend: N=node, E=canonical edge, E*=subset edge, "
        "K=pattern keys, P=properties; /M=MANDATORY, /O=OPTIONAL."
    )
    lines.append("")

    # ---- Nodes ----
    lines.append("## NODE TYPES")
    for nt in patterns["node_types"]:
        labels = nt["labels"]
        label_str = ", ".join(labels) if labels else ""
        lines.append(f"N: {{{label_str}}}")

        # patterns: one line per distinct (L, K) — only the K varies
        for pat in nt["patterns"]:
            keys = ", ".join(pat["property_keys"])
            lines.append(f"  K: {{{keys}}}")

        # properties: single dense line
        if nt["properties"]:
            parts = []
            for pname, pinfo in nt["properties"].items():
                flag = "M" if pinfo["constraint"] == "MANDATORY" else "O"
                parts.append(f"{pname}:{pinfo['data_type']}/{flag}")
            lines.append(f"  P: {', '.join(parts)}")

    # ---- Edges ----
    lines.append("")
    lines.append("## EDGE TYPES")

    canonicals = [e for e in patterns["edge_types"] if e.get("is_canonical", True)]
    subsets    = [e for e in patterns["edge_types"] if not e.get("is_canonical", True)]

    def _emit_edge(et, marker: str):
        rel = et["labels"][0] if et["labels"] else "?"
        src = ", ".join(et["source_labels"]) if et["source_labels"] else ""
        tgt = ", ".join(et["target_labels"]) if et["target_labels"] else ""
        head = f"{marker}: {{{src}}}-[:{rel}]->{{{tgt}}}"
        if et.get("cardinality"):
            head += f"  card={et['cardinality']}"
        lines.append(head)

        for pat in et["patterns"]:
            keys = ", ".join(pat["property_keys"])
            lines.append(f"  K: {{{keys}}}")

        if et["properties"]:
            parts = []
            for pname, pinfo in et["properties"].items():
                flag = "M" if pinfo["constraint"] == "MANDATORY" else "O"
                parts.append(f"{pname}:{pinfo['data_type']}/{flag}")
            lines.append(f"  P: {', '.join(parts)}")

    for et in canonicals:
        _emit_edge(et, "E")
    for et in subsets:
        _emit_edge(et, "E*")

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

    ``data_dir`` is accepted for backward compatibility but is no longer
    used — the graph is read directly from Neo4j.  Connection details
    come from ``config`` or from the environment variables in ``.env``.

    Args:
        data_dir (str): Ignored; retained so existing call sites need no change.
        config (Optional[InferConfig]): Inference configuration.

    Returns:
        Optional[dict]: Inferred schema, or None if inference fails.
    """
    cfg = config or InferConfig()

    _p(cfg.verbose, "--- Connecting to Neo4j (data_dir arg ignored) ---")

    driver = _make_driver(cfg)
    try:
        patterns = mine_patterns(
            driver,
            chunk_size=cfg.chunk_size,
            expand_edge_subsets=cfg.expand_edge_subsets,
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
    schema = extract_json(raw_res, verbose=cfg.verbose)
    if not schema:
        _p(cfg.verbose, "LLM returned no JSON schema (parse failed).")
        return None

    # attach raw patterns for downstream comparison
    schema["_mined_patterns"] = patterns
    return schema


def run_infer_schema(
    data_dir: str,
    output_path: str,
    config: Optional[InferConfig] = None,
) -> str:
    """
    Run schema inference and write the result to disk.

    Args:
        data_dir (str): Ignored; kept for backward compatibility.
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