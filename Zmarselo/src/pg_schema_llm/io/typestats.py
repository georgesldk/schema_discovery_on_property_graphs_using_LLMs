from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd

from pg_schema_llm.io.detect import detect_file_role
from pg_schema_llm.io.normalize import normalize_edge_row, normalize_node_row
from pg_schema_llm.io.csv_tools import iter_chunks, read_full_df, read_preview, sniff_delimiter
from pg_schema_llm.io.naming import clean_name_smart, get_common_affixes
from pg_schema_llm.io.kv_store import sqlite_kv_get_many, sqlite_kv_open, sqlite_kv_put_many


# ============================================================
# Column resolution helper
# ============================================================

def resolve_col(actual_cols: Sequence[str], detected: Optional[str]) -> Optional[str]:
    """
    Resolve a detected column name to an actual dataframe column.

    This function maps a column name inferred during header inspection
    to the corresponding concrete column name in a dataframe. It handles
    Neo4j-style headers, common aliases, quoting artifacts, and
    case-insensitive matches.

    Args:
        actual_cols (Sequence[str]): Column names present in the dataframe.
        detected (Optional[str]): Detected column name or alias.

    Returns:
        Optional[str]: Resolved column name if found, otherwise None.
    """
    if not detected:
        return None

    if detected in actual_cols:
        return detected

    def norm(x: str) -> str:
        return str(x).strip().strip('"').strip("'").strip("`")

    det_n = norm(detected)
    det_l = det_n.lower()

    # exact after stripping
    for c in actual_cols:
        if norm(c) == det_n:
            return c

    # case-insensitive
    for c in actual_cols:
        if norm(c).lower() == det_l:
            return c

    # id special
    if det_l in {"id", "node_id", "nodeid"}:
        for c in actual_cols:
            cn = norm(c)
            if cn.startswith(":ID(") or cn == ":ID":
                return c
        for c in actual_cols:
            if norm(c).lower() == "id":
                return c

    # start/end special
    if det_l in {"source", "src", "from", "start", "start_id", "startid", "u"}:
        for c in actual_cols:
            if norm(c).startswith(":START_ID"):
                return c
    if det_l in {"target", "dst", "to", "end", "end_id", "endid", "v"}:
        for c in actual_cols:
            if norm(c).startswith(":END_ID"):
                return c

    # last resort: if contains id, prefer :ID(...)
    if "id" in det_l:
        for c in actual_cols:
            if norm(c).startswith(":ID("):
                return c

    return None


# ============================================================
# JSON props helper (StarWars)
# ============================================================

def extract_json_keys_sample(series: pd.Series, max_rows: int = 200) -> List[str]:
    """
    Extract property keys from JSON blobs stored as string values.

    This function samples values from a dataframe column and attempts
    to parse them as JSON objects. It is primarily used for datasets
    where properties are embedded inside a single JSON column.

    Args:
        series (pd.Series): Column containing JSON-encoded strings.
        max_rows (int): Maximum number of rows to sample.

    Returns:
        List[str]: Sorted list of discovered JSON object keys.
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


# ============================================================
# Lightweight type inference (works with dtype=str)
# ============================================================

def _infer_simple_kind(v: Any) -> Optional[str]:
    """
    Infer a coarse-grained data type from a value.

    This helper function performs lightweight type inference suitable
    for schema discovery, distinguishing between booleans, integers,
    floats, and strings using conservative parsing rules.

    Args:
        v (Any): Input value.

    Returns:
        Optional[str]: Inferred type name, or None if the value is blank.
    """

    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        if s.lower() in ("true", "false"):
            return "Boolean"
        try:
            int(s)
            return "Long"
        except Exception:
            pass
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
    Determine whether a column represents a reserved or technical property.

    This function filters out Neo4j import columns and other technical
    fields that should not be included in inferred schema properties.

    Args:
        col (str): Column name.

    Returns:
        bool: True if the column should be excluded from schema inference.
    """
    if not col:
        return True
    if col.startswith(":"):
        return True
    if col in (":START_ID", ":END_ID", ":TYPE", ":LABEL"):
        return True
    return False


# ============================================================
# Legacy graph builder (NetworkX) - small datasets only
# ============================================================

def _list_csv_files(data_folder: str) -> List[str]:
    """
    List all CSV files in a dataset directory.

    Args:
        data_folder (str): Path to the dataset directory.

    Returns:
        List[str]: Absolute paths to CSV files found in the directory.
    """
    
    return glob.glob(os.path.join(data_folder, "*.csv"))


def build_graph(data_folder: str) -> nx.MultiDiGraph:
    """
    Construct a full in-memory property graph using NetworkX.

    This legacy function materializes all nodes and edges into a
    NetworkX MultiDiGraph. It is intended only for small datasets
    and debugging purposes, as it does not scale to large graphs.

    Args:
        data_folder (str): Path to the dataset directory.

    Returns:
        nx.MultiDiGraph: Constructed property graph.
    """
    print(f"--- Building Graph from: {data_folder} ---")
    G = nx.MultiDiGraph()

    if not os.path.isdir(data_folder):
        print(f" Error: Folder '{data_folder}' does not exist.")
        return G

    csv_files = _list_csv_files(data_folder)
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

            df_preview = read_preview(file_path, delim)
            role, cols = detect_file_role(df_preview)

            if role == "node":
                id_col_prev = resolve_col(df_preview.columns, cols.get("id"))
                if not id_col_prev:
                    print(
                        f"    [SKIP] Node {os.path.basename(file_path)}: cannot resolve id. "
                        f"detected={cols.get('id')} cols={list(df_preview.columns)[:8]}"
                    )
                    continue

                df = read_full_df(file_path, delim)
                id_col = resolve_col(df.columns, id_col_prev)
                if not id_col:
                    print(f"    [SKIP] Node {os.path.basename(file_path)}: cannot resolve id in full df.")
                    continue

                print(f"   Processing Nodes: {os.path.basename(file_path)} -> '{clean_type}' (id={id_col})")

                for _, row in df.iterrows():
                    node_id, props = normalize_node_row(row.to_dict(), id_col=id_col)
                    if node_id is None:
                        continue
                    G.add_node(node_id, node_type=clean_type, **props)

            elif role == "edge":
                start_prev = resolve_col(df_preview.columns, cols.get("start"))
                end_prev = resolve_col(df_preview.columns, cols.get("end"))
                if not start_prev or not end_prev:
                    print(
                        f"    [SKIP] Edge {os.path.basename(file_path)}: cannot resolve start/end. "
                        f"detected=({cols.get('start')},{cols.get('end')}) cols={list(df_preview.columns)[:8]}"
                    )
                    continue

                df = read_full_df(file_path, delim)
                start_col = resolve_col(df.columns, start_prev)
                end_col = resolve_col(df.columns, end_prev)
                if not start_col or not end_col:
                    print(f"    [SKIP] Edge {os.path.basename(file_path)}: cannot resolve start/end in full df.")
                    continue

                print(
                    f"   Processing Edges: {os.path.basename(file_path)} -> '{clean_type}' "
                    f"(start={start_col}, end={end_col})"
                )

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


# ============================================================
# Streaming TypeStats builder (scalable)
# ============================================================

def build_typestats(
    data_folder: str,
    chunksize: int = 100_000,
    db_path: Optional[str] = None,
    sample_values_per_prop: int = 3,
) -> Dict[str, dict]:
    """
    Build streaming type statistics for property-graph schema discovery.

    This function processes CSV-based graph datasets in a streaming
    fashion to infer node types, edge types, property presence,
    property kinds, and topology patterns. It avoids materializing
    the full graph in memory and uses a lightweight SQLite store to
    track node identifier to type mappings across passes.

    Args:
        data_folder (str): Path to the dataset directory.
        chunksize (int): Number of rows per CSV chunk.
        db_path (Optional[str]): Path to the SQLite key?value store.
        sample_values_per_prop (int): Maximum number of sample values
            retained per property.

    Returns:
        Dict[str, dict]: Dictionary containing node and edge type statistics.
    """
    print(f"--- Building TypeStats (streaming) from: {data_folder} ---")

    if not os.path.isdir(data_folder):
        print(f" Error: Folder '{data_folder}' does not exist.")
        return {"node_types": {}, "edge_types": {}}

    csv_files = _list_csv_files(data_folder)
    if not csv_files:
        print(f" No CSV files found in {data_folder}")
        return {"node_types": {}, "edge_types": {}}

    all_filenames = [os.path.basename(f) for f in csv_files]
    common_prefix, common_suffix = get_common_affixes(all_filenames)

    if db_path is None:
        db_path = os.path.join(data_folder, ".pg_schema_llm_node_types.sqlite")

    conn = sqlite_kv_open(db_path)

    node_stats: Dict[str, dict] = {}
    edge_stats: Dict[str, dict] = {}

    # -------------------------
    # PASS 1: NODES
    # -------------------------
    print(">>> PASS 1: Nodes (build id -> type map + node stats)")

    for file_path in csv_files:
        clean_type = clean_name_smart(file_path, common_prefix, common_suffix)
        delim = sniff_delimiter(file_path)

        try:
            df_preview = read_preview(file_path, delim)
        except Exception as e:
            print(f"    Error reading header {file_path}: {e}")
            continue

        role, cols = detect_file_role(df_preview)
        if role != "node":
            continue

        id_col_preview = resolve_col(df_preview.columns, cols.get("id"))
        if not id_col_preview:
            print(
                f"    [SKIP] Node {os.path.basename(file_path)}: cannot resolve id. "
                f"detected={cols.get('id')} cols={list(df_preview.columns)}"
            )
            continue

        print(f"   Nodes: {os.path.basename(file_path)} -> '{clean_type}' (id_col={id_col_preview})")

        st = node_stats.setdefault(
            clean_type,
            {
                "count": 0,
                "prop_fill": Counter(),
                "prop_kind": Counter(),
                "prop_samples": defaultdict(list),
            },
        )

        try:
            total_rows_this_file = 0

            for chunk in iter_chunks(file_path, delim, chunksize):
                id_col = resolve_col(chunk.columns, id_col_preview)
                if not id_col:
                    continue

                total_rows_this_file += len(chunk)

                # Update id -> type map in SQLite
                chunk[id_col] = chunk[id_col].astype(str)
                rows = list(zip(chunk[id_col].tolist(), [clean_type] * len(chunk)))
                sqlite_kv_put_many(conn, rows)

                # ------------------------------------------------------------
                # PATH A: Fast column-based stats (no per-row normalize)
                # Fallback to per-row only when JSON blob column exists.
                # ------------------------------------------------------------

                # If StarWars-like JSON blob exists, we must fall back to per-row normalize
                if "props" in chunk.columns:
                    for row in chunk.to_dict(orient="records"):
                        node_id, props = normalize_node_row(row, id_col=id_col)
                        if node_id is None:
                            continue

                        for prop, value in props.items():
                            if not prop:
                                continue
                            if prop.lower() == "id":
                                continue
                            if _is_reserved_property(prop):
                                continue

                            st["prop_fill"][prop] += 1
                            kind = _infer_simple_kind(value)
                            if kind:
                                st["prop_kind"][(prop, kind)] += 1

                            if sample_values_per_prop > 0:
                                lst = st["prop_samples"][prop]
                                if len(lst) < sample_values_per_prop:
                                    s = str(value).strip()
                                    if s:
                                        lst.append(s[:80])

                    # presence-only extra keys
                    keys = extract_json_keys_sample(chunk["props"])
                    for k in keys:
                        st["prop_fill"][k] += 1
                        st["prop_kind"][(k, "String")] += 1

                else:
                    # -------- PATH A (fast) --------
                    for col in chunk.columns:
                        if col == id_col:
                            continue
                        if not col:
                            continue
                        if col.lower() == "id":
                            continue
                        if _is_reserved_property(col):
                            continue

                        s = chunk[col]
                        nonblank = s.astype(str).str.strip()
                        mask = nonblank.ne("")

                        nn = int(mask.sum())
                        if nn <= 0:
                            continue

                        st["prop_fill"][col] += nn

                        sample_vals = nonblank[mask].head(200).tolist()
                        for v in sample_vals:
                            kind = _infer_simple_kind(v)
                            if kind:
                                st["prop_kind"][(col, kind)] += 1

                        if sample_values_per_prop > 0:
                            lst = st["prop_samples"][col]
                            if len(lst) < sample_values_per_prop:
                                for v in nonblank[mask].head(sample_values_per_prop).tolist():
                                    if len(lst) >= sample_values_per_prop:
                                        break
                                    vv = str(v).strip()
                                    if vv:
                                        lst.append(vv[:80])

            st["count"] += total_rows_this_file
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
            df_preview = read_preview(file_path, delim)
        except Exception as e:
            print(f"    Error reading header {file_path}: {e}")
            continue

        role, cols = detect_file_role(df_preview)
        if role != "edge":
            continue

        start_preview = resolve_col(df_preview.columns, cols.get("start"))
        end_preview = resolve_col(df_preview.columns, cols.get("end"))
        if not start_preview or not end_preview:
            print(
                f"    [SKIP] Edge {os.path.basename(file_path)}: cannot resolve start/end. "
                f"detected=({cols.get('start')},{cols.get('end')}) cols={list(df_preview.columns)}"
            )
            continue

        print(f"   Edges: {os.path.basename(file_path)} -> '{clean_type}' (start={start_preview}, end={end_preview})")

        st = edge_stats.setdefault(
            clean_type,
            {
                "count": 0,
                "prop_fill": Counter(),
                "prop_kind": Counter(),
                "prop_keys": set(),
                "topology": Counter(),
            },
        )

        try:
            for chunk in iter_chunks(file_path, delim, chunksize):
                start_col = resolve_col(chunk.columns, start_preview)
                end_col = resolve_col(chunk.columns, end_preview)
                if not start_col or not end_col:
                    continue

                st["count"] += len(chunk)

                # edge properties per column
                for col in chunk.columns:
                    if col in (start_col, end_col):
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

###################################################################################
                if "props" in chunk.columns:
        
                    for val in chunk["props"].dropna():
                        if not isinstance(val, str) or not val.strip():
                            continue
                        try:
                            # Fast parse
                            obj = json.loads(val)
                            if isinstance(obj, dict):
                                for k, v in obj.items():
                                    if not k or _is_reserved_property(k): 
                                        continue
                                    
                                    st["prop_keys"].add(k)
                                    st["prop_fill"][k] += 1
                                    
                                    k_type = _infer_simple_kind(v)
                                    if k_type:
                                        st["prop_kind"][(k, k_type)] += 1
                        except Exception:
                            continue

###############################################################################

                # topology via sqlite id->type
                start_vals = chunk[start_col].astype(str).tolist()
                end_vals = chunk[end_col].astype(str).tolist()
                ids = list(set(start_vals + end_vals))
                id2type = sqlite_kv_get_many(conn, ids)

                for u, v in zip(start_vals, end_vals):
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
