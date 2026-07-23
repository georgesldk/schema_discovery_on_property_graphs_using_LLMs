"""Validate the PG-SB-compatible schema metric against PG-SB itself.

This test imports PG-SB's read-only evaluation.py and compares its
compute_schema_metrics result with the JSON adapter plus local port.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PGSB_ROOT = PROJECT_ROOT.parent / "PG-SB"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pg_schema_llm.evaluation.adapters import adapt_schema, pgsb_schema_csv_rows
from pg_schema_llm.evaluation.pgsb_metrics import compute_schema_metrics


def _load_pgsb_evaluation():
    spec = importlib.util.spec_from_file_location("pgsb_evaluation", PGSB_ROOT / "evaluation.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load PG-SB evaluation.py")
    module = importlib.util.module_from_spec(spec)
    sentinel = object()
    old_sklearn = sys.modules.get("sklearn", sentinel)
    sys.modules["sklearn"] = None
    try:
        spec.loader.exec_module(module)
    finally:
        if old_sklearn is sentinel:
            sys.modules.pop("sklearn", None)
        else:
            sys.modules["sklearn"] = old_sklearn
    return module


def _node(name: str, labels: list[str] | None = None, props: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"name": name, "labels": labels or [name], "properties": props or []}


def _edge(
    name: str,
    src: list[str],
    dst: list[str],
    props: list[dict[str, Any]] | None = None,
    canonical: bool = True,
    cardinality: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "source_labels": src,
        "target_labels": dst,
        "is_canonical": canonical,
        "cardinality": cardinality,
        "properties": props or [],
    }


def _schema(
    nodes: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
    mined_patterns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {"node_types": nodes or [], "edge_types": edges or []}
    if mined_patterns is not None:
        out["_mined_patterns"] = mined_patterns
    return out


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _base_nodes() -> list[dict[str, Any]]:
    return [_node("A"), _node("B"), _node("C"), _node("D")]


def _fixtures() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    props_a = [{"name": "p", "type": "STRING", "constraint": "MANDATORY"}]
    props_b = [{"name": "q", "type": "INTEGER", "constraint": "OPTIONAL"}]
    base = _base_nodes()
    edge_ab = _edge("R", ["A"], ["B"], cardinality="1..1 : 0..N")
    edge_ac = _edge("R", ["A"], ["C"])
    edge_cb = _edge("R", ["C"], ["B"])
    edge_db = _edge("R", ["D"], ["B"])
    return [
        ("perfect_match", _schema(base, [edge_ab]), _schema(base, [edge_ab])),
        ("missing_node_type", _schema(base, [edge_ab]), _schema([_node("A"), _node("B")], [edge_ab])),
        ("extra_node_type", _schema([_node("A"), _node("B")], [edge_ab]), _schema(base, [edge_ab])),
        ("missing_edge_type", _schema(base, [edge_ab, edge_ac]), _schema(base, [edge_ab])),
        ("extra_edge_type", _schema(base, [edge_ab]), _schema(base, [edge_ab, edge_ac])),
        ("incorrect_edge_source", _schema(base, [edge_ab]), _schema(base, [edge_cb])),
        ("incorrect_edge_target", _schema(base, [edge_ab]), _schema(base, [edge_ac])),
        ("missing_node_property", _schema([_node("A", props=props_a), _node("B")], [edge_ab]), _schema(base, [edge_ab])),
        ("extra_node_property", _schema(base, [edge_ab]), _schema([_node("A", props=props_a), _node("B")], [edge_ab])),
        ("incorrect_property_data_type", _schema([_node("A", props=props_a), _node("B")], [edge_ab]), _schema([_node("A", props=props_b), _node("B")], [edge_ab])),
        ("mandatory_optional_mismatch", _schema([_node("A", props=props_a), _node("B")], [edge_ab]), _schema([_node("A", props=[{"name": "p", "type": "STRING", "constraint": "OPTIONAL"}]), _node("B")], [edge_ab])),
        ("multi_label_node", _schema([_node("AB", labels=["A", "B"]), _node("C")], [_edge("R", ["A", "B"], ["C"])]), _schema([_node("AB", labels=["A", "B"]), _node("C")], [_edge("R", ["A", "B"], ["C"])])),
        ("same_labels_different_property_patterns", _schema([_node("A", props=props_a), _node("B")], [edge_ab]), _schema([_node("A", props=props_b), _node("B")], [edge_ab])),
        ("empty_inferred_schema", _schema(base, [edge_ab]), _schema([], [])),
        ("empty_ground_truth_schema", _schema([], []), _schema(base, [edge_ab])),
        ("zero_denominator_cases", _schema([], []), _schema([], [])),
        ("duplicate_elements", _schema(base, [edge_ab, edge_ab]), _schema(base, [edge_ab, edge_ab])),
        ("weighted_count_differences", _schema(base, [edge_ab], {"edge_types": [{"labels": ["R"], "count": 10}]}), _schema(base, [edge_ab], {"edge_types": [{"labels": ["R"], "count": 999}]})),
        ("cardinality_mismatch", _schema(base, [edge_ab]), _schema(base, [_edge("R", ["A"], ["B"], cardinality="0..N : 0..N")])),
        ("canonical_derived_edge_entries", _schema(base, [edge_ab]), _schema(base, [edge_ab, _edge("R", ["D"], ["B"], canonical=False)])),
        ("legacy_topology", _schema(base, [{"name": "R", "properties": [], "topology": [{"allowed_sources": ["A"], "allowed_targets": ["B"]}]}]), _schema(base, [edge_ab])),
        ("relationship_comma_prediction", _schema(base, [_edge("R", ["A"], ["B"])]), _schema(base, [_edge("R,ALT", ["A"], ["B"])])),
        ("case_sensitive_labels", _schema([_node("A"), _node("B")], [_edge("R", ["A"], ["B"])]), _schema([_node("a"), _node("B")], [_edge("R", ["a"], ["B"])])),
    ]


def main() -> None:
    pgsb_eval = _load_pgsb_evaluation()
    with tempfile.TemporaryDirectory(prefix="pgsb_compat_", dir=PROJECT_ROOT) as tmp_name:
        tmp = Path(tmp_name)
        for name, gt, inf in _fixtures():
            gt_csv = tmp / f"{name}_original_edges.csv"
            pred_csv = tmp / f"{name}_predicted_edges.csv"
            _write_csv(gt_csv, pgsb_schema_csv_rows(gt, "ground_truth"), ["srcType", "relationshipType", "dstType"])
            _write_csv(pred_csv, pgsb_schema_csv_rows(inf, "predicted"), ["srcLabels", "relationshipTypes", "dstLabels"])

            with redirect_stdout(io.StringIO()):
                expected = pgsb_eval.compute_schema_metrics(str(gt_csv), str(pred_csv))

            gt_adapted = adapt_schema(gt, "ground_truth")
            inf_adapted = adapt_schema(inf, "predicted")
            actual = compute_schema_metrics(gt_adapted.triples, inf_adapted.triples)

            assert actual.edit_distance == expected.edit_distance, name
            assert abs(actual.precision - expected.precision) < 1e-9, name
            assert abs(actual.recall - expected.recall) < 1e-9, name
            assert abs(actual.f1 - expected.f1) < 1e-9, name
    print(f"Validated {len(_fixtures())} PG-SB schema fixtures against original evaluation.py")


if __name__ == "__main__":
    main()
