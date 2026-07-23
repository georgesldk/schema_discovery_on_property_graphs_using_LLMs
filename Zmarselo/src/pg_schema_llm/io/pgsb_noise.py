"""PG-SB-compatible noise mutations for a loaded Neo4j dataset.

This module mirrors the executable Cypher semantics in PG-SB's ``benchmark.py``.
It mutates the currently loaded database; it does not load or reset Neo4j
dumps. PG-SB uses APOC for batching, but these mutations also run on Neo4j
instances without APOC by executing equivalent plain Cypher statements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from neo4j import Driver

INTERNAL_NODE_PROPS = frozenset({
    "original_label",
    "_orig_labels",
    "_orig_label_concat",
    "_label_stripped",
})
INTERNAL_NODE_LABELS = frozenset({"OriginalLabel"})


@dataclass(frozen=True)
class PGSBNoiseConfig:
    """Noise settings matching PG-SB's benchmark config semantics."""

    property_noise_percent: float
    label_fraction: float = 0.0
    batch_size: int = 10000
    database: str | None = None


def _escape_name(name: str) -> str:
    return name.replace("`", "``")


def _session_kwargs(database: str | None) -> dict:
    kwargs = {"fetch_size": 1000}
    if database:
        kwargs["database"] = database
    return kwargs


def _collect_node_properties(driver: Driver, database: str | None) -> list[str]:
    with driver.session(**_session_kwargs(database)) as session:
        result = session.run(
            "MATCH (n) UNWIND keys(n) AS key RETURN DISTINCT key AS key ORDER BY key"
        )
        return [
            row["key"]
            for row in result
            if row["key"] not in INTERNAL_NODE_PROPS
        ]


def _collect_edge_properties(driver: Driver, database: str | None) -> list[str]:
    with driver.session(**_session_kwargs(database)) as session:
        result = session.run(
            "MATCH ()-[r]->() UNWIND keys(r) AS key RETURN DISTINCT key AS key ORDER BY key"
        )
        return [row["key"] for row in result]


def _collect_node_labels(driver: Driver, database: str | None) -> list[str]:
    with driver.session(**_session_kwargs(database)) as session:
        result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
        return [
            row["label"]
            for row in result
            if row["label"] not in INTERNAL_NODE_LABELS
        ]


def _run_write(driver: Driver, database: str | None, cypher: str) -> None:
    with driver.session(**_session_kwargs(database)) as session:
        session.run(cypher).consume()


def set_original_labels(driver: Driver, *, batch_size: int = 10000, database: str | None = None) -> None:
    """Save original labels exactly as PG-SB does before label removal."""
    del batch_size
    query = "MATCH (n) SET n.original_label = labels(n)"
    _run_write(driver, database, query)


def remove_node_property(
    driver: Driver,
    prop: str,
    fraction: float,
    *,
    batch_size: int = 10000,
    database: str | None = None,
) -> None:
    """Remove one node property with PG-SB's random Cypher predicate."""
    del batch_size
    prop_escaped = _escape_name(prop)
    query = (
        f"MATCH (n) WHERE rand() < {fraction} "
        f"AND '{prop_escaped}' IN keys(n) REMOVE n.`{prop_escaped}`"
    )
    _run_write(driver, database, query)


def remove_edge_property(
    driver: Driver,
    prop: str,
    fraction: float,
    *,
    batch_size: int = 10000,
    database: str | None = None,
) -> None:
    """Remove one relationship property with PG-SB's random Cypher predicate."""
    del batch_size
    prop_escaped = _escape_name(prop)
    query = (
        f"MATCH ()-[r]-() WHERE rand() < {fraction} "
        f"AND '{prop_escaped}' IN keys(r) REMOVE r.`{prop_escaped}`"
    )
    _run_write(driver, database, query)


def remove_labels(
    driver: Driver,
    node_labels: Iterable[str],
    fraction: float,
    *,
    batch_size: int = 10000,
    database: str | None = None,
) -> None:
    """Remove all known labels from nodes selected by PG-SB's rand predicate."""
    del batch_size
    labels = [label for label in node_labels if label not in INTERNAL_NODE_LABELS]
    if not labels:
        return
    labels_str = ":".join(f"`{_escape_name(label)}`" for label in labels)
    query = f"MATCH (n) WHERE rand() < {fraction} REMOVE n:{labels_str}"
    _run_write(driver, database, query)


def apply_pgsb_noise(driver: Driver, config: PGSBNoiseConfig) -> dict:
    """Apply PG-SB property noise and label removal to the current database."""
    if not (0.0 <= config.property_noise_percent <= 100.0):
        raise ValueError("property_noise_percent must be between 0 and 100")
    if not (0.0 <= config.label_fraction <= 1.0):
        raise ValueError("label_fraction must be between 0.0 and 1.0")

    node_properties = _collect_node_properties(driver, config.database)
    edge_properties = _collect_edge_properties(driver, config.database)
    node_labels = _collect_node_labels(driver, config.database)
    fraction = config.property_noise_percent / 100.0

    set_original_labels(driver, batch_size=config.batch_size, database=config.database)

    for prop in node_properties:
        remove_node_property(
            driver,
            prop,
            fraction,
            batch_size=config.batch_size,
            database=config.database,
        )
    for prop in edge_properties:
        remove_edge_property(
            driver,
            prop,
            fraction,
            batch_size=config.batch_size,
            database=config.database,
        )
    remove_labels(
        driver,
        node_labels,
        config.label_fraction,
        batch_size=config.batch_size,
        database=config.database,
    )

    return {
        "property_noise_percent": config.property_noise_percent,
        "label_fraction": config.label_fraction,
        "node_properties": node_properties,
        "edge_properties": edge_properties,
        "node_labels": node_labels,
        "batch_size": config.batch_size,
        "execution": "plain_cypher_no_apoc",
    }
