"""Runner for PG-SB-compatible evaluation from generated schema JSON."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .adapters import adapt_schema, load_schema_json
from .pgsb_metrics import compute_schema_metrics

PGSB_NODE_WARNING = (
    "PG-SB node metrics require original_nodes CSV rows with _nodeId/original_label "
    "and predicted_nodes CSV cluster membership. The generated schema JSON has no "
    "node IDs or inferred cluster membership, so these metrics cannot be reconstructed."
)
PGSB_EDGE_WARNING = (
    "PG-SB edge metrics require original_edges CSV rows with srcId/dstId and "
    "predicted_edges CSV edgeIdsInCluster membership. The generated schema JSON has "
    "no edge IDs or inferred cluster membership, so these metrics cannot be reconstructed."
)


def _repo_commit(repo_path: Path) -> str | None:
    try:
        safe_path = str(repo_path).replace("\\", "/")
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={safe_path}", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def unavailable_metric(reason: str) -> dict[str, Any]:
    """Return a stable marker for PG-SB metrics unavailable from schema JSON."""
    return {
        "status": "not_comparable_from_schema_json",
        "reason": reason,
        "true_positives": None,
        "false_positives": None,
        "false_negatives": None,
        "precision": None,
        "recall": None,
        "micro_f1": None,
        "macro_f1": None,
    }


def evaluate_json_schemas(
    gt_path: str | Path,
    inf_path: str | Path,
    dataset: str,
    output_path: str | Path | None = None,
    pgsb_repo_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate generated schemas using the PG-SB-compatible schema metric."""
    gt_file = Path(gt_path)
    inf_file = Path(inf_path)
    pgsb_path = Path(pgsb_repo_path) if pgsb_repo_path else Path(__file__).resolve().parents[4] / "PG-SB"

    gt = load_schema_json(gt_file)
    inf = load_schema_json(inf_file)

    gt_adapted = adapt_schema(gt, "ground_truth")
    inf_adapted = adapt_schema(inf, "predicted")
    schema_metrics = compute_schema_metrics(gt_adapted.triples, inf_adapted.triples)

    warnings: list[str] = [PGSB_NODE_WARNING, PGSB_EDGE_WARNING]
    if gt_adapted.skipped_edges:
        warnings.append("Ground-truth edges skipped during PG-SB adaptation: " + "; ".join(gt_adapted.skipped_edges))
    if inf_adapted.skipped_edges:
        warnings.append("Inferred edges skipped during PG-SB adaptation: " + "; ".join(inf_adapted.skipped_edges))
    if gt_adapted.duplicate_triples:
        warnings.append(f"Ground truth contains {len(gt_adapted.duplicate_triples)} duplicate schema triples; PG-SB set semantics ignore duplicates.")
    if inf_adapted.duplicate_triples:
        warnings.append(f"Inferred schema contains {len(inf_adapted.duplicate_triples)} duplicate schema triples; PG-SB set semantics ignore duplicates.")
    warnings.append(
        "PG-SB's executable evaluator does not evaluate node property patterns, edge property patterns, property data types, "
        "property optionality, cardinality, or mined support counts as schema metrics."
    )

    payload: dict[str, Any] = {
        "dataset": dataset,
        "evaluation_mode": "pgsb_compatible",
        "pgsb_reference": {
            "repository_path": str(pgsb_path),
            "commit": _repo_commit(pgsb_path),
            "source_file": "evaluation.py",
            "source_function": "compute_schema_metrics",
        },
        "inputs": {
            "ground_truth": str(gt_file),
            "inferred": str(inf_file),
        },
        "metrics": {
            "nodes": unavailable_metric(PGSB_NODE_WARNING),
            "edges": unavailable_metric(PGSB_EDGE_WARNING),
            "schema": schema_metrics.to_json(),
        },
        "intermediate_counts": {
            "schema": {
                "ground_truth_triples": sorted([list(item) for item in gt_adapted.triples]),
                "predicted_triples": sorted([list(item) for item in inf_adapted.triples]),
                "intersection": sorted([list(item) for item in gt_adapted.triples & inf_adapted.triples]),
                "only_ground_truth": sorted([list(item) for item in gt_adapted.triples - inf_adapted.triples]),
                "only_predicted": sorted([list(item) for item in inf_adapted.triples - gt_adapted.triples]),
            }
        },
        "warnings": warnings,
        "comparability": {
            "fully_comparable": [
                "schema.edit_distance",
                "schema.precision",
                "schema.recall",
                "schema.f1",
            ],
            "partially_comparable": [],
            "not_comparable": [
                "nodes.precision",
                "nodes.recall",
                "nodes.micro_f1",
                "nodes.macro_f1",
                "nodes.ami",
                "nodes.rand_index",
                "nodes.accuracy",
                "edges.precision",
                "edges.recall",
                "edges.micro_f1",
                "edges.macro_f1",
                "node_property_patterns",
                "edge_property_patterns",
                "property_data_types",
                "property_optionality",
                "cardinality",
                "support_or_frequency_weighting",
            ],
        },
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return payload


def print_summary(payload: dict[str, Any]) -> None:
    """Print a compact terminal summary."""
    schema = payload["metrics"]["schema"]
    print("\nPG-SB-Compatible Evaluation")
    print(f"  Dataset: {payload['dataset']}")
    print(f"  PG-SB commit: {payload['pgsb_reference'].get('commit') or 'unavailable'}")
    print("  Schema triples:")
    print(f"    GT={schema['ground_truth_count']} Pred={schema['predicted_count']} TP={schema['true_positives']} FP={schema['false_positives']} FN={schema['false_negatives']}")
    print(f"    Edit distance={schema['edit_distance']} P={schema['precision']:.6f} R={schema['recall']:.6f} F1={schema['f1']:.6f}")
    print("  Not comparable from schema JSON: PG-SB node and edge clustering metrics.")
    if payload.get("warnings"):
        print("  Warnings:")
        for warning in payload["warnings"]:
            print(f"    - {warning}")

