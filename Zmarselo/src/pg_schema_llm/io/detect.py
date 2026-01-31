from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


_EDGE_START_ALIASES = {"source", "src", "from", "start", "start_id", "startid", "u"}
_EDGE_END_ALIASES = {"target", "dst", "to", "end", "end_id", "endid", "v"}

_NODE_ID_ALIASES = {"id", "node_id", "nodeid"}


def _clean_col(c: Any) -> str:
    # strip whitespace + surrounding quotes/backticks
    return str(c).strip().strip('"').strip("'").strip("`")


def detect_file_role(df) -> Tuple[str, Dict[str, str]]:
    """
    Detect whether a CSV file is a node file or edge file based on headers.

    Returns:
      ("edge", {"start": <col>, "end": <col>})
      ("node", {"id": <col>})
      ("unknown", {})
    """
    cols_raw: List[Any] = list(df.columns)
    cols: List[str] = [_clean_col(c) for c in cols_raw]
    cols_l: List[str] = [c.lower() for c in cols]

    # ---------- EDGE detection ----------
    # 1) Neo4j headers (:START_ID..., :END_ID...)
    start_col = next((c for c in cols if c.startswith(":START_ID")), None)
    end_col = next((c for c in cols if c.startswith(":END_ID")), None)

    # 2) common generic headers (source/target variants)
    if not start_col:
        start_col = next((cols[i] for i, cl in enumerate(cols_l) if cl in _EDGE_START_ALIASES), None)
    if not end_col:
        end_col = next((cols[i] for i, cl in enumerate(cols_l) if cl in _EDGE_END_ALIASES), None)

    if start_col and end_col:
        return "edge", {"start": start_col, "end": end_col}

    # ---------- NODE detection ----------
    # 1) Neo4j headers (:ID..., :ID(...))
    id_col = next((c for c in cols if c.startswith(":ID")), None)

    # 2) generic id headers
    if not id_col:
        id_col = next((cols[i] for i, cl in enumerate(cols_l) if cl in _NODE_ID_ALIASES), None)

    # 3) patterns like "Person.id:ID(Person)" or any header containing word 'id'
    if not id_col:
        id_col = next((c for c in cols if re.search(r"\bid\b", c.lower())), None)

    if id_col:
        return "node", {"id": id_col}

    return "unknown", {}
