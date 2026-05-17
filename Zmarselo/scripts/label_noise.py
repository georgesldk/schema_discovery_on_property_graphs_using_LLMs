#!/usr/bin/env python3
"""
Unlabeled-data evaluation support for PG-HIVE experiments.

The core problem: schema discovery on unlabeled (or partially-labeled) graphs
produces clusters that have no type name.  Following Sideri et al., "PG-HIVE",
EDBT 2026 (Sec 5.2), each cluster is assigned the majority original label(s)
of the nodes it contains — letting us compute F1 against a labeled GT.

Sub-commands
------------
  backup          Snapshot every node's current labels into the _orig_labels
                  property and _orig_label_concat string, and add a Neo4j
                  marker label `OriginalLabel`.  Idempotent — safe to re-run.

  majority-assign <inferred_json> [--output <path>]
                  For each node type in the inferred schema, query Neo4j for
                  the nodes that belong to that type (by label-set match, or
                  by OriginalLabel + property-key match for unlabeled types),
                  tally their _orig_labels, and rewrite the schema with the
                  majority-voted labels.  Outputs a GT-comparable JSON.

  apply <pct>     Randomly remove all original labels from <pct>% of nodes
                  (keeps the OriginalLabel marker).  Use this to simulate
                  partial- or zero-label scenarios before running infer.py.

  restore         Re-add _orig_labels to every stripped node.

  status          Print a summary of the current graph state.

  clean           Remove all backup metadata and the OriginalLabel label.

Typical workflow
----------------
  # --- baseline (100 % labels) ---
  python scripts/label_noise.py backup
  python scripts/infer.py <dataset>
  python scripts/label_noise.py majority-assign \\
      03_outputs/schemas/inferred/<dataset>/inf_<dataset>.json \\
      --output 03_outputs/schemas/inferred/<dataset>/inf_<dataset>_majority.json
  python scripts/compare.py <dataset> --inferred-suffix _majority

  # --- 50 % label availability ---
  python scripts/label_noise.py apply 50
  python scripts/infer.py <dataset>
  python scripts/label_noise.py majority-assign ...
  python scripts/label_noise.py restore          # ready for next scenario

  python scripts/label_noise.py clean            # done; graph back to normal

Environment variables (or .env in Zmarselo/):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKER_LABEL     = "OriginalLabel"
PROP_ORIG_LABELS = "_orig_labels"
PROP_ORIG_CONCAT = "_orig_label_concat"
PROP_STRIPPED    = "_label_stripped"

BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Cypher helpers
# ---------------------------------------------------------------------------

def _escape(label: str) -> str:
    return f"`{label.replace('`', '``')}`"


def _label_match_exact(var: str, label_set: Tuple[str, ...]) -> str:
    """WHERE clause: node has EXACTLY these labels (plus OriginalLabel is ignored)."""
    real = [l for l in label_set if l != MARKER_LABEL]
    parts = [f"size([l IN labels({var}) WHERE l <> '{MARKER_LABEL}' | l]) = {len(real)}"]
    for lbl in real:
        parts.append(f"'{lbl}' IN labels({var})")
    return " AND ".join(parts) if parts else "true"


def _run_batched(session, cypher_template: str, ids: List[int], **extra):
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        session.run(cypher_template, {"ids": batch, **extra})


# ---------------------------------------------------------------------------
# backup
# ---------------------------------------------------------------------------

def cmd_backup(session, verbose: bool) -> None:
    print(">>> Backing up labels …")

    result = session.run(
        "MATCH (n) WHERE n._orig_labels IS NULL "
        "RETURN id(n) AS nid, labels(n) AS lbls"
    )
    rows = [(r["nid"], list(r["lbls"])) for r in result]

    if not rows:
        print("   All nodes already backed up — nothing to do.")
    else:
        print(f"   Storing _orig_labels on {len(rows):,} nodes …")
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            data = [
                {
                    "nid": nid,
                    "orig": lbls,
                    "concat": "_".join(sorted(lbls)),
                }
                for nid, lbls in batch
            ]
            session.run(
                "UNWIND $data AS d "
                "MATCH (n) WHERE id(n) = d.nid "
                "SET n._orig_labels = d.orig, "
                "    n._orig_label_concat = d.concat",
                {"data": data},
            )

    esc = _escape(MARKER_LABEL)
    cnt = session.run(
        f"MATCH (n) WHERE NOT n:{esc} RETURN count(n) AS c"
    ).single()["c"]
    if cnt:
        print(f"   Adding :{MARKER_LABEL} to {cnt:,} nodes …")
        session.run(f"MATCH (n) WHERE NOT n:{esc} SET n:{esc}")
    else:
        print(f"   :{MARKER_LABEL} already present on all nodes.")

    if verbose and rows:
        combo_counts: Counter = Counter()
        for _, lbls in rows:
            combo_counts["_".join(sorted(lbls)) or "<empty>"] += 1
        print(f"\n   Distinct original label sets ({len(combo_counts)}):")
        for combo, n in combo_counts.most_common(20):
            print(f"     {{{combo}}}  →  {n:,} nodes")

    print("   Backup complete.")


# ---------------------------------------------------------------------------
# majority-assign
# ---------------------------------------------------------------------------

def _query_orig_labels_for_type(
    session,
    label_set: Tuple[str, ...],
    prop_keys: Tuple[str, ...],
) -> Counter:
    """
    Return a Counter of _orig_labels (as frozen tuples) for nodes that
    belong to the given inferred type.

    Matching strategy:
      * Non-empty label set → exact label-set match (ignoring OriginalLabel).
      * Empty label set → nodes that are stripped (only OriginalLabel), filtered
        by having exactly the given property keys.
    """
    real_labels = tuple(l for l in label_set if l != MARKER_LABEL)
    counter: Counter = Counter()

    if real_labels:
        where = _label_match_exact("n", real_labels)
        result = session.run(
            f"MATCH (n) WHERE {where} "
            "RETURN n._orig_labels AS orig",
        )
    else:
        # Unlabeled: nodes with only OriginalLabel and exactly these property keys
        pk_list = list(prop_keys)
        result = session.run(
            f"MATCH (n:{_escape(MARKER_LABEL)}) "
            "WHERE size([l IN labels(n) WHERE l <> $marker | l]) = 0 "
            "  AND size(keys(n)) - size([k IN keys(n) WHERE k STARTS WITH '_' | k]) = $npk "
            "  AND all(k IN $pkeys WHERE k IN keys(n)) "
            "RETURN n._orig_labels AS orig",
            {"marker": MARKER_LABEL, "npk": len(pk_list), "pkeys": pk_list},
        )

    for r in result:
        orig = r["orig"]
        if orig is not None:
            counter[tuple(sorted(orig))] += 1
        else:
            counter[("<unlabeled>",)] += 1

    return counter


def cmd_majority_assign(
    session,
    inferred_path: str,
    output_path: str,
    verbose: bool,
) -> None:
    print(f">>> Majority-label assignment")
    print(f"    Input:  {inferred_path}")
    print(f"    Output: {output_path}")

    # Check that backup has been done
    backed = session.run(
        "MATCH (n) WHERE n._orig_labels IS NOT NULL RETURN count(n) AS c LIMIT 1"
    ).single()["c"]
    if backed == 0:
        sys.exit("ERROR: No _orig_labels found. Run `backup` first.")

    with open(inferred_path, encoding="utf-8") as f:
        schema = json.load(f)

    # ---- node types ----
    print("\n  Node types:")
    new_node_types = []
    for nt in schema.get("node_types", []):
        raw_labels = nt.get("labels") or (
            [nt["name"]] if nt.get("name") else []
        )
        label_set = tuple(sorted(raw_labels))
        prop_keys  = tuple(sorted(
            p["name"] if isinstance(p, dict) else p
            for p in (nt.get("properties") or [])
        ) if isinstance(nt.get("properties"), list) else sorted(nt.get("properties", {}).keys()))

        counter = _query_orig_labels_for_type(session, label_set, prop_keys)
        total   = sum(counter.values())

        if total == 0:
            if verbose:
                print(f"    {{{', '.join(label_set)}}}  →  no matching nodes in Neo4j (kept as-is)")
            new_node_types.append(nt)
            continue

        majority_orig, majority_cnt = counter.most_common(1)[0]
        majority_labels = list(majority_orig)

        if verbose:
            purity = majority_cnt / total * 100
            print(
                f"    {{{', '.join(label_set) or '<empty>'}}}  "
                f"→  majority={{{', '.join(majority_labels)}}}  "
                f"({majority_cnt}/{total} = {purity:.1f}%)"
            )
            if len(counter) > 1:
                for orig, cnt in counter.most_common():
                    print(f"        {dict(zip(orig, orig))}  ×{cnt}")

        updated = dict(nt)
        updated["labels"] = majority_labels
        if updated.get("name") and updated["name"] not in majority_labels:
            updated["name"] = majority_labels[0] if majority_labels else updated["name"]
        new_node_types.append(updated)

    # ---- edge types ----
    print("\n  Edge types:")
    new_edge_types = []
    for et in schema.get("edge_types", []):
        rel = et.get("name") or (et.get("labels") or [""])[0]
        src_labels = tuple(sorted(et.get("source_labels", [])))
        tgt_labels = tuple(sorted(et.get("target_labels", [])))

        # Query majority for src and tgt endpoint label sets
        src_counter = _query_orig_labels_for_type(session, src_labels, ())
        tgt_counter = _query_orig_labels_for_type(session, tgt_labels, ())

        src_majority = list(src_counter.most_common(1)[0][0]) if src_counter else list(src_labels)
        tgt_majority = list(tgt_counter.most_common(1)[0][0]) if tgt_counter else list(tgt_labels)

        if verbose:
            src_str = ", ".join(src_labels) or "<empty>"
            tgt_str = ", ".join(tgt_labels) or "<empty>"
            maj_src = ", ".join(src_majority)
            maj_tgt = ", ".join(tgt_majority)
            print(
                f"    [:{rel}]  {{{src_str}}}→{{{tgt_str}}}"
                f"  →  {{{maj_src}}}→{{{maj_tgt}}}"
            )

        updated = dict(et)
        updated["source_labels"] = src_majority
        updated["target_labels"] = tgt_majority
        new_edge_types.append(updated)

    out = dict(schema)
    out["node_types"] = new_node_types
    out["edge_types"] = new_edge_types
    out["_majority_assigned"] = True

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, default=str)
    print(f"\n  Written → {output_path}")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def cmd_apply(session, pct: float, seed: int, verbose: bool) -> None:
    if not (0.0 <= pct <= 100.0):
        sys.exit("ERROR: percentage must be 0–100.")

    backed = session.run(
        "MATCH (n) WHERE n._orig_labels IS NOT NULL RETURN count(n) AS c"
    ).single()["c"]
    if backed == 0:
        sys.exit("ERROR: No backup found. Run `backup` first.")

    result = session.run(
        "MATCH (n) WHERE n._orig_labels IS NOT NULL "
        "  AND (n._label_stripped IS NULL OR n._label_stripped = false) "
        "RETURN id(n) AS nid, n._orig_labels AS orig"
    )
    candidates = [(r["nid"], list(r["orig"])) for r in result]

    k = max(0, round(len(candidates) * pct / 100.0))
    rng = random.Random(seed)
    selected = rng.sample(candidates, k)

    print(f">>> Applying {pct:.0f}% label availability scenario …")
    print(f"    Eligible: {len(candidates):,}  |  to strip: {k:,}  |  seed={seed}")

    if not selected:
        print("    Nothing to do.")
        return

    label_to_ids: Dict[str, List[int]] = defaultdict(list)
    for nid, orig in selected:
        for lbl in orig:
            if lbl != MARKER_LABEL:
                label_to_ids[lbl].append(nid)

    for lbl, ids in sorted(label_to_ids.items()):
        esc = _escape(lbl)
        _run_batched(session, f"MATCH (n) WHERE id(n) IN $ids REMOVE n:{esc}", ids)
        if verbose:
            print(f"    removed :{lbl} from {len(ids):,} nodes")

    selected_ids = [nid for nid, _ in selected]
    _run_batched(
        session,
        "MATCH (n) WHERE id(n) IN $ids SET n._label_stripped = true",
        selected_ids,
    )
    print(f"    Done — {k:,} nodes now unlabeled.")


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

def cmd_restore(session, verbose: bool) -> None:
    print(">>> Restoring labels …")

    result = session.run(
        "MATCH (n) WHERE n._label_stripped = true "
        "RETURN id(n) AS nid, n._orig_labels AS orig"
    )
    rows = [(r["nid"], list(r["orig"])) for r in result]

    if not rows:
        print("    No stripped nodes — nothing to restore.")
        return

    label_to_ids: Dict[str, List[int]] = defaultdict(list)
    for nid, orig in rows:
        for lbl in orig:
            if lbl != MARKER_LABEL:
                label_to_ids[lbl].append(nid)

    for lbl, ids in sorted(label_to_ids.items()):
        esc = _escape(lbl)
        _run_batched(session, f"MATCH (n) WHERE id(n) IN $ids SET n:{esc}", ids)
        if verbose:
            print(f"    restored :{lbl} to {len(ids):,} nodes")

    stripped_ids = [nid for nid, _ in rows]
    _run_batched(
        session,
        "MATCH (n) WHERE id(n) IN $ids REMOVE n._label_stripped",
        stripped_ids,
    )
    print(f"    Restored {len(rows):,} nodes.")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(session) -> None:
    print(">>> Label-noise status")
    total   = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    backed  = session.run("MATCH (n) WHERE n._orig_labels IS NOT NULL RETURN count(n) AS c").single()["c"]
    marked  = session.run(f"MATCH (n:{_escape(MARKER_LABEL)}) RETURN count(n) AS c").single()["c"]
    stripped = session.run("MATCH (n) WHERE n._label_stripped = true RETURN count(n) AS c").single()["c"]

    print(f"    Total nodes:            {total:>10,}")
    print(f"    Backed-up nodes:        {backed:>10,}")
    print(f"    :{MARKER_LABEL} marked: {marked:>10,}")
    print(f"    Labels stripped:        {stripped:>10,}")
    if backed:
        print(f"    Current label noise:    {stripped/backed*100:>9.1f}%")

    print("\n    Current label distribution (top 20, excluding OriginalLabel):")
    result = session.run(
        "MATCH (n) "
        "WITH [l IN labels(n) WHERE l <> $m | l] AS lbls "
        "RETURN lbls, count(*) AS cnt ORDER BY cnt DESC LIMIT 20",
        {"m": MARKER_LABEL},
    )
    for r in result:
        combo = "{" + ", ".join(r["lbls"]) + "}" if r["lbls"] else "<unlabeled>"
        print(f"      {combo:<50s}  {r['cnt']:>8,}")


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

def cmd_clean(session, verbose: bool) -> None:
    print(">>> Cleaning up all backup metadata …")
    esc = _escape(MARKER_LABEL)
    for prop in (PROP_ORIG_LABELS, PROP_ORIG_CONCAT, PROP_STRIPPED):
        cnt = session.run(
            f"MATCH (n) WHERE n.`{prop}` IS NOT NULL RETURN count(n) AS c"
        ).single()["c"]
        if cnt:
            print(f"    Removing {prop} from {cnt:,} nodes …")
            session.run(f"MATCH (n) WHERE n.`{prop}` IS NOT NULL REMOVE n.`{prop}`")
    cnt = session.run(f"MATCH (n:{esc}) RETURN count(n) AS c").single()["c"]
    if cnt:
        print(f"    Removing :{MARKER_LABEL} from {cnt:,} nodes …")
        session.run(f"MATCH (n:{esc}) REMOVE n:{esc}")
    print("    Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    # Shared connection args live on a parent parser (add_help=False so they
    # don't produce a duplicate -h flag on each subparser).
    conn = argparse.ArgumentParser(add_help=False)
    conn.add_argument("--uri",      default=os.getenv("NEO4J_URI",      "bolt://localhost:7687"))
    conn.add_argument("--user",     default=os.getenv("NEO4J_USER",     "neo4j"))
    conn.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    conn.add_argument("--database", default=os.getenv("NEO4J_DATABASE", None))
    conn.add_argument("--quiet",    action="store_true")

    parser = argparse.ArgumentParser(
        description="PG-HIVE unlabeled-data evaluation support",
        parents=[conn],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("backup",  parents=[conn], help="Snapshot labels into _orig_labels + OriginalLabel marker")
    sub.add_parser("restore", parents=[conn], help="Re-add labels to stripped nodes")
    sub.add_parser("status",  parents=[conn], help="Print graph label state")
    sub.add_parser("clean",   parents=[conn], help="Remove all backup metadata")

    p_apply = sub.add_parser("apply", parents=[conn], help="Strip labels from N pct of nodes")
    p_apply.add_argument("pct", type=float, help="Percentage to strip, 0 to 100")
    p_apply.add_argument("--seed", type=int, default=42)

    p_ma = sub.add_parser(
        "majority-assign",
        parents=[conn],
        help="Rewrite inferred schema with majority-voted original labels",
    )
    p_ma.add_argument("inferred", help="Path to inferred schema JSON")
    p_ma.add_argument("--output", default=None, help="Output path (default: adds _majority suffix)")

    args = parser.parse_args()
    verbose = not args.quiet

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    session_kw: dict = {"fetch_size": 1000}
    if args.database:
        session_kw["database"] = args.database

    try:
        with driver.session(**session_kw) as session:
            if args.command == "backup":
                cmd_backup(session, verbose)
            elif args.command == "majority-assign":
                inf = args.inferred
                if args.output:
                    out = args.output
                else:
                    inp = Path(inf)
                    out = str(inp.parent / (inp.stem + "_majority" + inp.suffix))
                cmd_majority_assign(session, inf, out, verbose)
            elif args.command == "apply":
                cmd_apply(session, args.pct, args.seed, verbose)
            elif args.command == "restore":
                cmd_restore(session, verbose)
            elif args.command == "status":
                cmd_status(session)
            elif args.command == "clean":
                cmd_clean(session, verbose)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
