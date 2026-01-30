from __future__ import annotations

import csv
import glob
import json
import os
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import networkx as nx

from pg_schema_llm.io.csv_detect import detect_file_role
from pg_schema_llm.io.csv_normalize import normalize_node_row, normalize_edge_row


# ----------------------------
# CSV parsing utilities
# ----------------------------

CSV_SAMPLE_BYTES = 4096


def sniff_delimiter(file_path: str, default: str = ",") -> str:
    """
    Detect delimiter robustly. Handles:
    - comma separated
    - pipe separated (common in LDBC exports)
    - tab / semicolon
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(CSV_SAMPLE_BYTES)
    except Exception:
        return default

    first_line = sample.splitlines()[0] if sample else ""
    if "|" in first_line and "," not in first_line:
        return "|"

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "|", "\t", ";"])
        return dialect.delimiter
    except Exception:
        return default


def normalize_header_columns(cols: Sequence[str]) -> List[str]:
    """
    Normalize column names so detect_file_role works across formats:
    - strip surrounding quotes
    - map ':ID' / 'ID' / 'id' -> 'id' (only for header convenience)
    NOTE: We DO NOT remove ':START_ID' or ':END_ID' because detect_file_role uses them.
    """
    out: List[str] = []
    for c in cols:
        c2 = str(c).strip().strip('"').strip("'")
        if c2 in (":ID", "ID", "id"):
            c2 = "id"
        out.append(c2)
    return out


def _read_preview(file_path: str, delim: str) -> pd.DataFrame:
    """
    Read only headers with consistent parsing options (works for quoted CSV too).
    """
    df = pd.read_csv(
        file_path,
        nrows=0,
        sep=delim,
        engine="python",
        quotechar='"',
        doublequote=True,
    )
    df.columns = normalize_header_columns(df.columns)
    return df


def _iter_chunks(
    file_path: str,
    delim: str,
    chunksize: int,
    *,
    dtype_str: bool = True,
    on_bad_lines: str = "skip",
) -> Iterable[pd.DataFrame]:
    """
    Chunk iterator with robust parsing options.
    dtype_str=True makes reading stable for mixed-type columns / messy files.
    """
    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        sep=delim,
        engine="python",
        quotechar='"',
        doublequote=True,
        dtype=str if dtype_str else None,
        keep_default_na=False,
        na_values=["", "null", "NULL"],
        on_bad_lines=on_bad_lines,
    ):
        chunk.columns = normalize_header_columns(chunk.columns)
        yield chunk


# ----------------------------
# Name cleaning utilities
# ----------------------------

def get_common_affixes(filenames: Sequence[str]) -> Tuple[str, str]:
    """
    Dataset-agnostic noise detector: common prefix/suffix across *all* filenames.
    """
    if not filenames:
        return "", ""

    prefix = os.path.commonprefix(list(filenames))

    rev = [f[::-1] for f in filenames]
    common_rev = os.path.commonprefix(rev)
    suffix = common_rev[::-1]
    return prefix, suffix


def clean_name_smart(filename: str, prefix: str, suffix: str) -> str:
    """
    Strip global common prefix/suffix and common dataset wrappers like:
      nodes_Character.csv -> Character
      rels_HOMEWORLD.csv  -> HOMEWORLD
    """
    base = os.path.basename(filename)

    if prefix and base.startswith(prefix):
        base = base[len(prefix):]
    if suffix and base.endswith(suffix):
        base = base[:-len(suffix)]

    base = os.path.splitext(base)[0]
    base = base.strip("_")

    # common structural prefixes (StarWars etc.)
    lower = base.lower()
    if lower.startswith("nodes_"):
        base = base[len("nodes_"):]
    elif lower.startswith("rels_"):
        base = base[len("rels_"):]

    return base


# --- LEGACY SUPPORT (Fixes ImportError in older modules) ---
def clean_type_name(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]


# ----------------------------
# JSON props helper (StarWars)
# ----------------------------

def extract_json_keys_sample(series: pd.Series, max_rows: int = 200) -> List[str]:
    """
    Extract JSON keys from a sample of values.
    StarWars stores all properties inside a JSON blob column called "props".
    """
    keys = set()
    vals = series.dropna().head(max_rows).tolist()
    for s in vals:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                keys.update(obj.keys())
        except Exception:
            continue
    return sorted(keys)


# ----------------------------
# Lightweight type inference for dtype=str
# ----------------------------

def _infer_simple_kind(v) -> Optional[str]:
    """
    For streaming reads we use dtype=str for robustness; infer types from strings.
    """
    if v is None:
        return None

    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        if s.lower() in ("true", "false"):
            return "Boolean"
        # int
        try:
            int(s)
            return "Long"
        except Exception:
            pass
        # float
        try:
            float(s)
            return "Double"
        except Exception:
            pass
        return "String"

    if isinstance(v, bool):
        return "Boolean"
    if isinstance(v, int):
        return "Long"
    if isinstance(v, float):
        return "Double"
    return "String"


def _is_reserved_property(col: str) -> bool:
    """
    Drop Neo4j import / technical columns from schema properties.
    """
    if not col:
        return True
    if col.startswith(":"):
        return True
    # Neo4j helper tokens
    if col in (":START_ID", ":END_ID", ":TYPE", ":LABEL"):
        return True
    return False

import re
from typing import Optional, Sequence

def resolve_col(actual_cols: Sequence[str], detected: str) -> Optional[str]:
    """
    Map a detected column name to an actual column name in the dataframe.

    Handles:
    - quoted headers
    - normalized headers (':ID' -> 'id')
    - neo4j token columns like ':ID(Body-ID)', ':START_ID(...)', ':END_ID(...)'
    - generic 'id/source/target' to neo4j forms
    """
    if not detected:
        return None

    # exact
    if detected in actual_cols:
        return detected

    # compare with stripped quotes/backticks and lower
    def norm(x: str) -> str:
        return str(x).strip().strip('"').strip("'").strip("`")

    det_n = norm(detected)
    det_l = det_n.lower()

    # 1) try exact after stripping
    for c in actual_cols:
        if norm(c) == det_n:
            return c

    # 2) case-insensitive match
    for c in actual_cols:
        if norm(c).lower() == det_l:
            return c

    # 3) special mapping for id / start / end
    if det_l in {"id", "node_id", "nodeid"}:
        # prefer neo4j :ID(...) or :ID, else a column literally named id
        for c in actual_cols:
            cn = norm(c)
            if cn.startswith(":ID(") or cn == ":ID":
                return c
        for c in actual_cols:
            if norm(c).lower() == "id":
                return c

    if det_l in {"source", "src", "from", "start", "start_id", "startid", "u"}:
        for c in actual_cols:
            cn = norm(c)
            if cn.startswith(":START_ID"):
                return c

    if det_l in {"target", "dst", "to", "end", "end_id", "endid", "v"}:
        for c in actual_cols:
            cn = norm(c)
            if cn.startswith(":END_ID"):
                return c

    # 4) last resort: if detected contains 'id', pick the best candidate
    if "id" in det_l:
        # prefer :ID(...) first
        for c in actual_cols:
            cn = norm(c)
            if cn.startswith(":ID("):
                return c

    return None


# ----------------------------
# Legacy graph builder (NetworkX) - for small datasets only
# ----------------------------

def build_graph(data_folder: str) -> nx.MultiDiGraph:
    """
    Legacy: materializes full NetworkX MultiDiGraph.
    Works but not scalable. Kept for compatibility and debugging.
    """
    print(f"--- Building Graph from: {data_folder} ---")
    G = nx.MultiDiGraph()

    if not os.path.isdir(data_folder):
        print(f" Error: Folder '{data_folder}' does not exist.")
        return G

    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not csv_files:
        print(f" No CSV files found in {data_folder}")
        return G

    all_filenames = [os.path.basename(f) for f in csv_files]
    common_prefix, common_suffix = get_common_affixes(all_filenames)

    print(f"   [Auto-Cleaner] Detected Common Prefix: '{common_prefix}'")
    print(f"   [Auto-Cleaner] Detected Common Suffix: '{common_suffix}'")
    print(">>> Scanning for Nodes & Edges...")

    for file_path in csv_files:
        try:
            clean_type = clean_name_smart(file_path, common_prefix, common_suffix)
            delim = sniff_delimiter(file_path)
            df_preview = _read_preview(file_path, delim)
            role, cols = detect_file_role(df_preview)
            id_col = resolve_col(df_preview.columns, cols["id"])
            if not id_col:
                print(f"    [SKIP] Could not resolve node id column for {os.path.basename(file_path)}. detected={cols['id']} cols={list(df_preview.columns)[:6]}")
                continue

            if role == "node":
                df = pd.read_csv(
                    file_path,
                    sep=delim,
                    engine="python",
                    quotechar='"',
                    doublequote=True,
                    dtype=str,
                    keep_default_na=False,
                    na_values=["", "null", "NULL"],
                    on_bad_lines="skip",
                )
                df.columns = normalize_header_columns(df.columns)

                id_col = resolve_col(df.columns, cols["id"])
                if not id_col:
                    print(f"    [SKIP] Node file {os.path.basename(file_path)}: can't resolve id col. detected={cols['id']} cols={list(df.columns)}")
                    continue

                print(f"   Processing Nodes: {os.path.basename(file_path)} -> '{clean_type}' (id={id_col})")

                for _, row in df.iterrows():
                    node_id, props = normalize_node_row(row, id_col=id_col_chunk)
                    if node_id is None:
                        continue
                    G.add_node(node_id, node_type=clean_type, **props)

            elif role == "edge":
                df = pd.read_csv(
                    file_path,
                    sep=delim,
                    engine="python",
                    quotechar='"',
                    doublequote=True,
                    dtype=str,
                    keep_default_na=False,
                    na_values=["", "null", "NULL"],
                    on_bad_lines="skip",
                )
                df.columns = normalize_header_columns(df.columns)

                start_col = resolve_col(df.columns, cols["start"])
                end_col   = resolve_col(df.columns, cols["end"])
                if not start_col or not end_col:
                    print(f"    [SKIP] Edge file {os.path.basename(file_path)}: can't resolve start/end. detected=({cols['start']},{cols['end']}) cols={list(df.columns)}")
                    continue

                print(f"   Processing Edges: {os.path.basename(file_path)} -> '{clean_type}' (start={start_col}, end={end_col})")

                for _, row in df.iterrows():
                    u, v, props = normalize_edge_row(row.to_dict(), start_col=start_col, end_col=end_col)
                    if u is None or v is None:
                        continue
                    if not G.has_node(u):
                        G.add_node(u, node_type="Inferred")
                    if not G.has_node(v):
                        G.add_node(v, node_type="Inferred")
                    G.add_edge(u, v, type=clean_type, **props)




        except Exception as e:
            print(f"    Error reading {file_path}: {e}")

    print(f"\n Graph Built. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


# ----------------------------
# SQLite KV store for id -> node_type
# ----------------------------

def _sqlite_kv_open(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS node_type_map (
            node_id TEXT PRIMARY KEY,
            node_type TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_node_type ON node_type_map(node_type)")
    return conn


def _sqlite_kv_put_many(conn: sqlite3.Connection, rows: List[Tuple[str, str]]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO node_type_map(node_id, node_type) VALUES (?, ?)",
        rows
    )


def _sqlite_kv_get_many(conn: sqlite3.Connection, ids: List[str]) -> Dict[str, str]:
    if not ids:
        return {}
    out: Dict[str, str] = {}
    CH = 900  # avoid SQLite parameter limit
    for i in range(0, len(ids), CH):
        batch = ids[i:i + CH]
        qs = ",".join(["?"] * len(batch))
        cur = conn.execute(
            f"SELECT node_id, node_type FROM node_type_map WHERE node_id IN ({qs})",
            batch
        )
        out.update({k: v for (k, v) in cur.fetchall()})
    return out


# ----------------------------
# Streaming TypeStats builder (scalable)
# ----------------------------

def build_typestats(
    data_folder: str,
    chunksize: int = 100_000,
    db_path: Optional[str] = None,
    sample_values_per_prop: int = 3,
) -> Dict[str, dict]:
    """
    Streaming stats builder for schema discovery.

    Output:
    {
      "node_types": { type: {count, prop_fill, prop_kind, prop_samples}, ... },
      "edge_types": { edge: {count, prop_fill, prop_kind, prop_keys, topology}, ... }
    }
    """
    print(f"--- Building TypeStats (streaming) from: {data_folder} ---")

    if not os.path.isdir(data_folder):
        print(f" Error: Folder '{data_folder}' does not exist.")
        return {"node_types": {}, "edge_types": {}}

    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not csv_files:
        print(f" No CSV files found in {data_folder}")
        return {"node_types": {}, "edge_types": {}}

    all_filenames = [os.path.basename(f) for f in csv_files]
    common_prefix, common_suffix = get_common_affixes(all_filenames)

    if db_path is None:
        db_path = os.path.join(data_folder, ".pg_schema_llm_node_types.sqlite")

    conn = _sqlite_kv_open(db_path)

    node_stats: Dict[str, dict] = {}
    edge_stats: Dict[str, dict] = {}

    # -------------------------
    # PASS 1: NODES
    # -------------------------
    print(">>> PASS 1: Nodes (build id -> type map + node stats)")
    for file_path in csv_files:
        clean_type = clean_name_smart(file_path, common_prefix, common_suffix)
        delim = sniff_delimiter(file_path)

        # preview header
        try:
            df_preview = _read_preview(file_path, delim)
        except Exception as e:
            print(f"    Error reading header {file_path}: {e}")
            continue

        role, cols = detect_file_role(df_preview)
        if role != "node":
            continue

        id_col = resolve_col(df_preview.columns, cols.get("id"))
        if not id_col:
            print(
                f"    [SKIP] Could not resolve node id column for {os.path.basename(file_path)}. "
                f"detected={cols.get('id')} cols={list(df_preview.columns)}"
            )
            continue



        # df_preview columns are already normalized; resolve against preview columns
        id_col = resolve_col(df_preview.columns, cols["id"])
        if not id_col:
            print(f"    [SKIP] Could not resolve node id column in preview for {os.path.basename(file_path)}. detected={cols['id']} cols={list(df_preview.columns)}")
            continue

        print(f"   Nodes: {os.path.basename(file_path)} -> '{clean_type}' (id_col={id_col})")

        st = node_stats.setdefault(clean_type, {
            "count": 0,
            "prop_fill": Counter(),
            "prop_kind": Counter(),
            "prop_samples": defaultdict(list),
        })

        try:
            for chunk in _iter_chunks(file_path, delim, chunksize):
                id_col_chunk = resolve_col(chunk.columns, id_col)
                if not id_col_chunk:
                    continue

                chunk[id_col_chunk] = chunk[id_col_chunk].astype(str)
                rows = list(zip(chunk[id_col_chunk].tolist(), [clean_type] * len(chunk)))
                
            
                _sqlite_kv_put_many(conn, rows)

                st["count"] += len(chunk)

                for row in chunk.to_dict(orient="records"):
                    node_id, props = normalize_node_row(row, id_col=id_col_chunk)
                    if node_id is None:
                        continue

                    for prop, value in props.items():
                    

                        if not prop:
                            continue
                        if prop.lower() in ("id",):
                            continue
                        if _is_reserved_property(prop):
                            continue

                        st["prop_fill"][prop] += 1
                        kind = _infer_simple_kind(value)
                        if kind:
                            st["prop_kind"][(prop, kind)] += 1


                        # samples (optional)
                        if sample_values_per_prop > 0:
                            lst = st["prop_samples"][prop]
                            if len(lst) < sample_values_per_prop:
                                s = str(value)
                                if s:
                                    lst.append(s[:80])

                # StarWars: expand JSON blob column "props" into pseudo-properties
                if "props" in chunk.columns:
                    keys = extract_json_keys_sample(chunk["props"])
                    if keys:
                        # presence signal only (cheap)
                        for k in keys:
                            st["prop_fill"][k] += 1
                            st["prop_kind"][(k, "String")] += 1

            conn.commit()
        except Exception as e:
            print(f"    Error streaming nodes {file_path}: {e}")

    # -------------------------
    # PASS 2: EDGES
    # -------------------------
    print(">>> PASS 2: Edges (topology + edge stats using SQLite id->type)")
    for file_path in csv_files:
        clean_type = clean_name_smart(file_path, common_prefix, common_suffix)
        delim = sniff_delimiter(file_path)

        try:
            df_preview = _read_preview(file_path, delim)
        except Exception as e:
            print(f"    Error reading header {file_path}: {e}")
            continue

        role, cols = detect_file_role(df_preview)
        if role != "edge":
            continue

        start_col = resolve_col(df_preview.columns, cols.get("start"))
        end_col   = resolve_col(df_preview.columns, cols.get("end"))
        if not start_col or not end_col:
            print(
                f"    [SKIP] Could not resolve edge columns for {os.path.basename(file_path)}. "
                f"detected=({cols.get('start')},{cols.get('end')}) cols={list(df_preview.columns)}"
            )
            continue



        start_col = resolve_col(df_preview.columns, cols["start"])
        end_col   = resolve_col(df_preview.columns, cols["end"])
        if not start_col or not end_col:
            print(f"    [SKIP] Could not resolve edge columns in preview for {os.path.basename(file_path)}. detected=({cols['start']},{cols['end']}) cols={list(df_preview.columns)}")
            continue
        print(f"   Edges: {os.path.basename(file_path)} -> '{clean_type}' (start={start_col}, end={end_col})")

        st = edge_stats.setdefault(clean_type, {
            "count": 0,
            "prop_fill": Counter(),
            "prop_kind": Counter(),
            "prop_keys": set(),
            "topology": Counter(),
        })

        try:
            for chunk in _iter_chunks(file_path, delim, chunksize):
                start_c = resolve_col(chunk.columns, start_col)
                end_c   = resolve_col(chunk.columns, end_col)
                if not start_c or not end_c:
                    continue

                chunk[start_c] = chunk[start_c].astype(str)
                chunk[end_c]   = chunk[end_c].astype(str)

                st["count"] += len(chunk)

                # edge props: skip start_c/end_c
                for col in chunk.columns:
                    if col in (start_c, end_c):
                        continue
                    if _is_reserved_property(col):
                        continue

                    st["prop_keys"].add(col)
                    nonnull = chunk[col].notna()
                    nn = int(nonnull.sum())
                    if nn <= 0:
                        continue

                    st["prop_fill"][col] += nn

                    sample_vals = chunk.loc[nonnull, col].head(20).tolist()
                    for v in sample_vals[:10]:
                        kind = _infer_simple_kind(v)
                        if kind:
                            st["prop_kind"][(col, kind)] += 1

                # topology mapping using SQLite
                ids = list(set(chunk[start_c].tolist() + chunk[end_c].tolist()))
                id2type = _sqlite_kv_get_many(conn, ids)

                for u, v in zip(chunk[start_c].tolist(), chunk[end_c].tolist()):
                    su = id2type.get(u, "Inferred")
                    sv = id2type.get(v, "Inferred")
                    st["topology"][(su, sv)] += 1

        except Exception as e:
            print(f"    Error streaming edges {file_path}: {e}")

    print("\n TypeStats built.")
    print(f"   Node types: {len(node_stats)}")
    print(f"   Edge types: {len(edge_stats)}")

    conn.close()
    return {"node_types": node_stats, "edge_types": edge_stats}
