from __future__ import annotations

import sqlite3
from typing import Dict, List, Tuple


def sqlite_kv_open(db_path: str) -> sqlite3.Connection:
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
    conn.executemany(
        "INSERT OR REPLACE INTO node_type_map(node_id, node_type) VALUES (?, ?)",
        rows,
    )


def sqlite_kv_get_many(conn: sqlite3.Connection, ids: List[str]) -> Dict[str, str]:
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
