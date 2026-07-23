"""Dataset and output-scenario naming helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetScenario:
    """A dataset name plus an optional output/noise scenario suffix."""

    raw: str
    base: str
    scenario: str
    noise_percent: int | None = None


_NOISE_SUFFIX_RE = re.compile(r"^(?P<base>.+)-(?P<noise>\d+)$")


def parse_dataset_scenario(name: str) -> DatasetScenario:
    """Parse names such as ``starwars-10`` into base and scenario parts.

    The base is used for static inputs such as ``02_pgs/pg_data_<base>`` and
    ground truth JSON. The scenario is used for generated outputs.
    """
    raw = name.lower().strip()
    match = _NOISE_SUFFIX_RE.fullmatch(raw)
    if not match:
        return DatasetScenario(raw=raw, base=raw, scenario=raw)
    return DatasetScenario(
        raw=raw,
        base=match.group("base"),
        scenario=raw,
        noise_percent=int(match.group("noise")),
    )


def gt_schema_path(dataset: DatasetScenario) -> str:
    """Ground truth path for a scenario, always based on the clean dataset."""
    return f"03_outputs/schemas/ground_truth/{dataset.base}/gt_{dataset.base}.json"


def inferred_schema_path(dataset: DatasetScenario) -> str:
    """Inferred schema path for the exact requested scenario."""
    return f"03_outputs/schemas/inferred/{dataset.scenario}/inf_{dataset.scenario}.json"


def pgsb_metrics_path(dataset: DatasetScenario) -> str:
    """PG-SB-compatible metrics path for the exact requested scenario."""
    return f"03_outputs/evaluation/pgsb/{dataset.scenario}/metrics.json"

