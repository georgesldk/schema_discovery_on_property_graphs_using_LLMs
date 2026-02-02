from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple


def _is_blank(v: Any) -> bool:
    """
    Check whether a value should be treated as blank.

    This helper function identifies missing or empty values in a
    data-agnostic way, treating None and empty or whitespace-only
    strings as blank.

    Args:
        v (Any): Input value.

    Returns:
        bool: True if the value is considered blank, False otherwise.
    """

    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _maybe_expand_json_blob(v: Any) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse a JSON object embedded as a string value.

    This function detects whether a value appears to be a JSON object
    serialized as a string and, if so, parses and returns it as a
    dictionary. Non-dictionary JSON values and malformed inputs are
    safely ignored.

    Args:
        v (Any): Input value.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON object if successful,
        otherwise None.
    """
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def normalize_node_row(row: Dict[str, Any], id_col: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Normalize a raw node row into an identifier and property dictionary.

    This function extracts a stable node identifier and consolidates
    all remaining non-empty attributes into a property dictionary.
    Embedded JSON objects are expanded into individual properties to
    preserve structural information.

    Args:
        row (Dict[str, Any]): Raw row representing a node.
        id_col (str): Column name containing the node identifier.

    Returns:
        Tuple[Optional[str], Dict[str, Any]]:
            - Node identifier, or None if missing
            - Dictionary of normalized node properties
    """
    raw_id = row.get(id_col)
    if _is_blank(raw_id):
        return None, {}

    node_id = str(raw_id).strip()
    props: Dict[str, Any] = {}

    for k, v in row.items():
        if k == id_col:
            continue
        if _is_blank(v):
            continue

        obj = _maybe_expand_json_blob(v)
        if obj is not None:
            for jk, jv in obj.items():
                props[str(jk)] = jv
            continue

        props[str(k)] = v

    return node_id, props


def normalize_edge_row(
    row: Dict[str, Any],
    start_col: str,
    end_col: str,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Normalize a raw edge row into endpoints and property dictionary.

    This function extracts source and target identifiers for an edge
    and aggregates all remaining non-empty attributes as edge properties.
    Rows with missing or invalid endpoints are safely discarded.

    Args:
        row (Dict[str, Any]): Raw row representing an edge.
        start_col (str): Column name containing the source node identifier.
        end_col (str): Column name containing the target node identifier.

    Returns:
        Tuple[Optional[str], Optional[str], Dict[str, Any]]:
            - Source node identifier, or None if missing
            - Target node identifier, or None if missing
            - Dictionary of normalized edge properties
    """
    raw_src = row.get(start_col)
    raw_dst = row.get(end_col)

    if _is_blank(raw_src) or _is_blank(raw_dst):
        return None, None, {}

    src = str(raw_src).strip()
    dst = str(raw_dst).strip()

    if src == "" or dst == "":
        return None, None, {}

    props: Dict[str, Any] = {}

    for k, v in row.items():
        if k in (start_col, end_col):
            continue
        if _is_blank(v):
            continue

        obj = _maybe_expand_json_blob(v)
        if obj is not None:
            for jk, jv in obj.items():
                props[str(jk)] = jv
            continue

        props[str(k)] = v

    return src, dst, props
