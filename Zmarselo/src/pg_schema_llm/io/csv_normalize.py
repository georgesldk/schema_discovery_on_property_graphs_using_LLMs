import json

def normalize_node_row(row: dict, id_col: str):
    """
    Returns:
      node_id: str
      properties: dict
    """
    raw_id = row.get(id_col)
    if raw_id is None or str(raw_id).strip() == "":
        return None, {}
    node_id = str(raw_id).strip()


    props = {}

    for k, v in row.items():
        if k == id_col:
            continue
        if v is None or v == "":
            continue

        # JSON blob column (StarWars-style)
        if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
            try:
                obj = json.loads(v)
                if isinstance(obj, dict):
                    for jk, jv in obj.items():
                        props[str(jk)] = jv
                    continue
            except Exception:
                pass

        # normal column
        props[str(k)] = v

    return node_id, props

def normalize_edge_row(row: dict, start_col: str, end_col: str):
    """
    Returns:
      source_id: str
      target_id: str
      properties: dict
    """
    src = str(row.get(start_col))
    dst = str(row.get(end_col))

    props = {}

    for k, v in row.items():
        if k in (start_col, end_col):
            continue
        if v is None or v == "":
            continue

        # JSON blob support (rare but possible)
        if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
            try:
                obj = json.loads(v)
                if isinstance(obj, dict):
                    for jk, jv in obj.items():
                        props[str(jk)] = jv
                    continue
            except Exception:
                pass

        props[str(k)] = v

    return src, dst, props
