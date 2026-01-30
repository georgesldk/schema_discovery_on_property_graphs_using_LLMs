import re

def _clean_col(c: str) -> str:
    # strip whitespace + surrounding quotes/backticks
    return str(c).strip().strip('"').strip("'").strip("`")

def detect_file_role(df):
    cols_raw = df.columns.tolist()
    cols = [_clean_col(c) for c in cols_raw]

    # ---------- EDGE detection ----------
    # 1) Neo4j headers
    start_col = next((c for c in cols if c.startswith(":START_ID")), None)
    end_col   = next((c for c in cols if c.startswith(":END_ID")), None)

    # 2) common generic headers (source/target variants)
    if not start_col:
        start_col = next((c for c in cols if _clean_col(c).lower() in {
            "source","src","from","start","start_id","startid","u"
        }), None)
    if not end_col:
        end_col = next((c for c in cols if _clean_col(c).lower() in {
            "target","dst","to","end","end_id","endid","v"
        }), None)

    if start_col and end_col:
        return "edge", {"start": start_col, "end": end_col}

    # ---------- NODE detection ----------
    # 1) Neo4j headers
    id_col = next((c for c in cols if c.startswith(":ID")), None)

    # 2) generic id headers: id, ID, <type>.id, node_id, etc.
    if not id_col:
        id_col = next((c for c in cols if _clean_col(c).lower() in {
            "id","node_id","nodeid"
        }), None)

    # 3) patterns like "Person.id:ID(Person)" or "id:ID(Person)"
    if not id_col:
        id_col = next((c for c in cols if re.search(r"\bid\b", _clean_col(c).lower())), None)

    if id_col:
        return "node", {"id": id_col}

    return "unknown", {}
