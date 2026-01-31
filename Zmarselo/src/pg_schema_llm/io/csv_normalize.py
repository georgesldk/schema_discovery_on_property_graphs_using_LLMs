from __future__ import annotations

import json
from typing import Any, Dict, Tuple, Optional


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _maybe_expand_json_blob(v: Any) -> Optional[Dict[str, Any]]:
    """
    If v looks like a JSON object string, parse and return dict; else None.
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
    Returns:
      node_id: str | None
      properties: dict
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
    end_col: str
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Returns:
      source_id: str | None
      target_id: str | None
      properties: dict
    """
    raw_src = row.get(start_col)
    raw_dst = row.get(end_col)

    # SAFER than str(None) -> "None"
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
