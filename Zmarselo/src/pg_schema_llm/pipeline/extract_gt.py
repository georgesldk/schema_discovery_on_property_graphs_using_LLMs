"""
Extract ground-truth schema from .pgs (PG-Schema) files.

Outputs the same format used by mine_patterns() and expected by
compare.py, so GT and inferred schemas are directly comparable.

Supported .pgs dialect
----------------------
    CREATE NODE TYPE
    ( TypeName : Label1 & Label2 {
        prop1 Long,
        OPTIONAL prop2 String
    } );

    CREATE EDGE TYPE
    ( : SrcLabel1 | : SrcLabel2 )-[:REL_TYPE {
        prop1 String
    }]->( : TgtLabel );

    CREATE GRAPH TYPE Name LOOSE { ... };

Output format
-------------
    {
      "node_types": [
        {
          "name": "Label1",
          "labels": ["Label1", "Label2"],
          "properties": [
            {"name": "prop1", "type": "INTEGER", "constraint": "MANDATORY"}
          ]
        }
      ],
      "edge_types": [
        {
          "name": "REL_TYPE",
          "source_labels": ["SrcLabel1"],
          "target_labels": ["TgtLabel"],
          "is_canonical": true,
          "cardinality": null,
          "properties": [...]
        }
      ]
    }
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# Type mapping  (PGS types → generic types used by compare.py)
# ============================================================

_TYPE_MAP = {
    "long":        "INTEGER",
    "int":         "INTEGER",
    "integer":     "INTEGER",
    "double":      "DOUBLE",
    "float":       "DOUBLE",
    "string":      "STRING",
    "boolean":     "BOOLEAN",
    "bool":        "BOOLEAN",
    "date":        "DATE",
    "datetime":    "DATE",
    "point":       "STRING",       # spatial → STRING
    "stringarray": "LIST",
    "array":       "LIST",
}


def _map_type(raw: str) -> str:
    """Map a PGS property type to a generic schema type."""
    return _TYPE_MAP.get(raw.lower().strip(), "STRING")


# ============================================================
# Comment stripping
# ============================================================

def _strip_comments(content: str) -> str:
    content = re.sub(r"--.*?$",       "", content, flags=re.MULTILINE)
    content = re.sub(r"//.*?$",       "", content, flags=re.MULTILINE)
    content = re.sub(r"/\*.*?\*/",    "", content, flags=re.DOTALL)
    return content


# ============================================================
# Property block parsing
# ============================================================

def _parse_props(block: str) -> List[dict]:
    """
    Parse a brace-delimited property block.

    Handles lines like:
        prop1 Long,
        OPTIONAL prop2 String
    """
    props: List[dict] = []
    if not block or not block.strip():
        return props

    for line in block.split("\n"):
        line = line.strip().rstrip(",")
        if not line:
            continue

        is_optional = "OPTIONAL" in line.upper()
        line = re.sub(r"\bOPTIONAL\b", "", line, flags=re.IGNORECASE).strip()

        parts = line.split()
        if len(parts) >= 2:
            pname = parts[0]
            ptype = parts[-1]
            props.append({
                "name": pname,
                "type": _map_type(ptype),
                "constraint": "OPTIONAL" if is_optional else "MANDATORY",
            })
    return props


# ============================================================
# Node type parsing
# ============================================================

def _parse_node_stmt(stmt: str) -> Optional[dict]:
    """
    Parse a CREATE NODE TYPE statement.

    Returns a node-type dict or None on parse failure.
    """
    start = stmt.find("(")
    end = stmt.rfind(")")
    if start < 0 or end < 0:
        return None

    body = stmt[start + 1 : end].strip()

    # separate property block (in braces) from definition
    prop_str = ""
    p_start = body.find("{")
    p_end = body.rfind("}")
    if p_start >= 0 and p_end >= 0:
        prop_str = body[p_start + 1 : p_end]
        def_str = body[:p_start].strip()
    else:
        def_str = body.strip()

    # def_str is like "TypeName : Label1 & Label2"
    if ":" not in def_str:
        return None

    type_name, label_part = def_str.split(":", 1)
    type_name = type_name.strip()
    label_part = label_part.strip().replace("(", "").replace(")", "")

    labels = sorted(l.strip() for l in label_part.split("&") if l.strip())
    if not labels:
        return None

    # name = the label itself for single-label types;
    # for multi-label, pick the shortest / most general label
    name = min(labels, key=len)

    properties = _parse_props(prop_str)

    return {
        "name": name,
        "labels": labels,
        "properties": properties,
    }


# ============================================================
# Edge type parsing
# ============================================================

def _parse_label_alternatives(block: str) -> List[List[str]]:
    """
    Parse a source or target block that may contain '|' alternatives.

    Each alternative may itself have '&' conjunctions.

    Example:
        ": Object | : Vehicle"      →  [["Object"], ["Vehicle"]]
        ": Person & Actor"           →  [["Actor", "Person"]]
    """
    alternatives: List[List[str]] = []
    for alt in block.split("|"):
        alt = alt.strip().replace("(", "").replace(")", "")
        # strip leading ':'
        alt = re.sub(r"^\s*:\s*", "", alt)
        labels = sorted(l.strip() for l in alt.split("&") if l.strip())
        if labels:
            alternatives.append(labels)
    return alternatives


def _parse_edge_stmt(stmt: str) -> List[dict]:
    """
    Parse a CREATE EDGE TYPE statement.

    Returns a list of edge-type dicts (one per source×target combination
    when '|' alternatives appear on either side).
    """
    # find the arrow pattern:  -[...]->
    arrow = re.search(r"-\s*\[(.*?)\]\s*->", stmt, re.DOTALL)
    if not arrow:
        return []

    definition = arrow.group(1).strip()

    # extract edge properties if present inside the bracket
    edge_props: List[dict] = []
    if "{" in definition:
        p_start = definition.find("{")
        p_end = definition.rfind("}")
        if p_end > p_start:
            edge_props = _parse_props(definition[p_start + 1 : p_end])
        definition = definition[:p_start].strip()

    # edge name: strip optional variable and leading ':'
    edge_name = definition.strip()
    if ":" in edge_name:
        edge_name = edge_name.split(":")[-1].strip()
    edge_name = re.sub(r"[^A-Za-z0-9_]", "", edge_name).strip()
    if not edge_name:
        return []

    # source block: everything before the arrow, inside the first ( )
    src_block = stmt[: arrow.start()]
    # find the outermost parenthesised section
    src_p = src_block.find("(")
    if src_p >= 0:
        src_block = src_block[src_p + 1 :]
    src_block = src_block.rstrip(")").strip()

    # target block: everything after the arrow
    tgt_block = stmt[arrow.end() :].strip().rstrip(";").strip()
    if tgt_block.startswith("("):
        tgt_block = tgt_block[1:]
    tgt_block = tgt_block.rstrip(")").strip()

    src_alternatives = _parse_label_alternatives(src_block)
    tgt_alternatives = _parse_label_alternatives(tgt_block)

    # produce one edge entry per source × target combination
    edges: List[dict] = []
    for src_labels in src_alternatives:
        for tgt_labels in tgt_alternatives:
            edges.append({
                "name": edge_name,
                "source_labels": src_labels,
                "target_labels": tgt_labels,
                "is_canonical": True,
                "cardinality": None,     # .pgs doesn't declare cardinality
                "properties": edge_props,
            })

    return edges


# ============================================================
# Main extraction
# ============================================================

def extract_ground_truth(input_dir: str) -> dict:
    """
    Extract a ground-truth schema from a .pgs file in the given directory.

    Args:
        input_dir (str): Dataset directory containing a .pgs file.

    Returns:
        dict: Schema in the format expected by compare.py.

    Raises:
        FileNotFoundError: If no .pgs file is found.
    """
    pgs_files = glob.glob(os.path.join(input_dir, "*.pgs"))
    if not pgs_files:
        raise FileNotFoundError(f"No .pgs file found in {input_dir}")

    with open(pgs_files[0], "r", encoding="utf-8") as f:
        content = f.read()

    content = _strip_comments(content)

    node_types: List[dict] = []
    edge_types: List[dict] = []

    # split on semicolons
    for stmt in content.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue

        upper = stmt.upper()

        if "CREATE NODE TYPE" in upper:
            nt = _parse_node_stmt(stmt)
            if nt:
                node_types.append(nt)

        elif "CREATE EDGE TYPE" in upper:
            edges = _parse_edge_stmt(stmt)
            edge_types.extend(edges)

        # CREATE GRAPH TYPE is ignored — purely declarative

    return {
        "dataset_name": os.path.basename(os.path.normpath(input_dir)),
        "node_types": node_types,
        "edge_types": edge_types,
    }


def run_extract_gt(input_dir: str, output_file: str) -> str:
    """
    Extract ground truth and write to a JSON file.

    Args:
        input_dir (str): Dataset directory.
        output_file (str): Output JSON path.

    Returns:
        str: Path to the written file.
    """
    schema = extract_ground_truth(input_dir)

    parent_dir = os.path.dirname(output_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)

    return output_file