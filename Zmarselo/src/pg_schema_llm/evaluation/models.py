"""Data models for PG-SB-compatible schema evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SchemaTriple = tuple[str, str, str]


@dataclass(frozen=True)
class AdaptedSchema:
    """Schema data adapted to PG-SB's schema-level set representation."""

    triples: set[SchemaTriple]
    duplicate_triples: list[SchemaTriple] = field(default_factory=list)
    skipped_edges: list[str] = field(default_factory=list)
    property_keys_by_parent: dict[str, set[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class PGSBSchemaMetrics:
    """Exact PG-SB schema metric fields plus auditable counts."""

    edit_distance: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    gt_count: int
    predicted_count: int

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "edit_distance": self.edit_distance,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "ground_truth_count": self.gt_count,
            "predicted_count": self.predicted_count,
        }

