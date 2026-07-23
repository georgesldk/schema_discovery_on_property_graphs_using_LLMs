from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import requests
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
    expand_edge_subsets: bool = True

    # LLM
    mistral_model: str = "mistral-large-latest"
    mistral_api_url: str = "https://api.mistral.ai/v1/chat/completions"
    response_format: str = "json_object"
    max_output_tokens: int = 65536
    temperature: float = 0.0

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
        NEO4J_URI       â†’  cfg.neo4j_uri
        NEO4J_USER      â†’  cfg.neo4j_user
        NEO4J_PASSWORD  â†’  cfg.neo4j_password
    """
    load_dotenv()
    uri = cfg.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = cfg.neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = cfg.neo4j_password or os.getenv("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, password))


# ============================================================
# LLM helpers
# ============================================================

def call_mistral_api(
    prompt: str,
    *,
    model_name: str,
    api_url: str,
    response_format: str,
    verbose: bool,
    max_output_tokens: int,
    temperature: float,
) -> Optional[str]:
    """Invoke Mistral chat completions and return the assistant text."""
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        _p(verbose, "API Error: MISTRAL_API_KEY not set.")
        return None

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = {"type": response_format}

    try:
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        detail = ""
        if getattr(e, "response", None) is not None:
            detail = f" Response body: {e.response.text[:500]}"
        _p(verbose, f"Mistral API Error: {e}.{detail}")
        return None
    except ValueError as e:
        _p(verbose, f"Mistral API Error: response was not valid JSON: {e}")
        return None

    choices = data.get("choices") or []
    if not choices:
        _p(verbose, "Mistral API Error: no choices returned.")
        return None

    message = choices[0].get("message") or {}
    text = message.get("content")
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in text
        )

    if verbose:
        finish = choices[0].get("finish_reason")
        usage = data.get("usage") or {}
        _p(verbose, f"  finish_reason: {finish}")
        if usage:
            _p(
                verbose,
                f"  tokens: prompt={usage.get('prompt_tokens', '?')} "
                f"output={usage.get('completion_tokens', '?')} "
                f"total={usage.get('total_tokens', '?')}",
            )
        if text:
            preview = text if len(text) <= 400 else text[:200] + " ... " + text[-200:]
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

    Compact format â€” minimises tokens while preserving every field the
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

        # patterns: one line per distinct (L, K) â€” only the K varies
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
    used â€” the graph is read directly from Neo4j.  Connection details
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

    model_name = os.getenv("MISTRAL_MODEL", cfg.mistral_model)
    api_url = os.getenv("MISTRAL_API_URL", cfg.mistral_api_url)
    _p(cfg.verbose, f"--- Asking Mistral for schema: {model_name} ---")
    raw_res = call_mistral_api(
        prompt,
        model_name=model_name,
        api_url=api_url,
        response_format=cfg.response_format,
        verbose=cfg.verbose,
        max_output_tokens=cfg.max_output_tokens,
        temperature=cfg.temperature,
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

