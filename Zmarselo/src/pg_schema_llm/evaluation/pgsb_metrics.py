"""Minimal port of PG-SB's executable schema metric formula."""
from __future__ import annotations

from .models import PGSBSchemaMetrics, SchemaTriple


def safe_div(a: float, b: float) -> float:
    """Equivalent to PG-SB evaluation._safe_div."""
    return a / b if b else 0.0


def compute_schema_metrics(gt_schema: set[SchemaTriple], pred_schema: set[SchemaTriple]) -> PGSBSchemaMetrics:
    """Compute PG-SB schema edit distance, precision, recall, and F1.

    PG-SB treats a schema as a set of (srcType, relType, dstType) triples:
    edit_distance = |GT - Pred| + |Pred - GT|;
    precision = |GT intersection Pred| / |Pred|;
    recall = |GT intersection Pred| / |GT|;
    F1 = 2PR / (P + R), with 0.0 for zero denominators.
    """
    if not gt_schema and not pred_schema:
        return PGSBSchemaMetrics(
            edit_distance=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            gt_count=0,
            predicted_count=0,
        )

    intersection = gt_schema & pred_schema
    false_positives = len(pred_schema - gt_schema)
    false_negatives = len(gt_schema - pred_schema)
    precision = safe_div(len(intersection), len(pred_schema))
    recall = safe_div(len(intersection), len(gt_schema))
    f1 = safe_div(2 * precision * recall, precision + recall)

    return PGSBSchemaMetrics(
        edit_distance=false_positives + false_negatives,
        true_positives=len(intersection),
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        gt_count=len(gt_schema),
        predicted_count=len(pred_schema),
    )

