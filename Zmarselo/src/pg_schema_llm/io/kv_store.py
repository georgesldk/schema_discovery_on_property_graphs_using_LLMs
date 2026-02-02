from __future__ import annotations

import sqlite3
from typing import Dict, List, Tuple


def sqlite_kv_open(db_path: str) -> sqlite3.Connection:
    """
    Open and initialize a lightweight SQLite-backed key?value store.

    This function creates (or opens) a SQLite database used to persist
    node identifier to node-type mappings. The database is configured
    with performance-oriented pragmas and ensures that the required
    table and index exist.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        sqlite3.Connection: Open SQLite connection ready for use.
    """

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS node_type_map (
            node_id TEXT PRIMARY KEY,
            node_type TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_node_type ON node_type_map(node_type)")
    return conn


def sqlite_kv_put_many(conn: sqlite3.Connection, rows: List[Tuple[str, str]]) -> None:
    """
    Insert or update multiple key?value entries in the SQLite store.

    This function performs a batched upsert operation that maps node
    identifiers to their corresponding node types. Existing entries
    are replaced to keep the store consistent with the latest inference.

    Args:
        conn (sqlite3.Connection): Open SQLite connection.
        rows (List[Tuple[str, str]]): List of (node_id, node_type) pairs.

    Returns:
        None
    """

    conn.executemany(
        "INSERT OR REPLACE INTO node_type_map(node_id, node_type) VALUES (?, ?)",
        rows,
    )


def sqlite_kv_get_many(conn: sqlite3.Connection, ids: List[str]) -> Dict[str, str]:
    """
    Retrieve multiple node-type mappings from the SQLite store.

    This function fetches node-type assignments for a given list of
    node identifiers. Queries are executed in chunks to avoid SQLite
    parameter limits and to ensure stable performance.

    Args:
        conn (sqlite3.Connection): Open SQLite connection.
        ids (List[str]): List of node identifiers to query.

    Returns:
        Dict[str, str]: Mapping from node identifiers to node types.
    """

    if not ids:
        return {}
    out: Dict[str, str] = {}
    CH = 900  # avoid SQLite parameter limit
    for i in range(0, len(ids), CH):
        batch = ids[i : i + CH]
        qs = ",".join(["?"] * len(batch))
        cur = conn.execute(
            f"SELECT node_id, node_type FROM node_type_map WHERE node_id IN ({qs})",
            batch,
        )
        out.update({k: v for (k, v) in cur.fetchall()})
    return out
