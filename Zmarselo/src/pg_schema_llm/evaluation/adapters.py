"""Adapters from Zmarselo schema JSON to PG-SB evaluation inputs."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Literal

from .models import AdaptedSchema, SchemaTriple
from .normalization import property_map, parse_label_list

SchemaSide = Literal["ground_truth", "predicted"]


def load_schema_json(path: str | Path) -> dict[str, Any]:
    """Load a schema JSON file without changing its contents."""
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def _labels_for_node_name(schema: dict[str, Any]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for node in schema.get("node_types", []) or []:
        if not isinstance(node, dict):
            continue
        labels = [str(label) for label in (node.get("labels") or []) if str(label)]
        if not labels and node.get("name"):
            labels = [str(node["name"])]
        for key in [node.get("name"), node.get("type_name"), *labels]:
            if key:
                lookup[str(key)] = labels
    return lookup


def _edge_relationship_name(edge: dict[str, Any], side: SchemaSide) -> str:
    rel = edge.get("name")
    if not rel:
        labels = edge.get("labels") or []
        rel = labels[0] if labels else ""
    rel_text = str(rel).strip()
    if side == "predicted":
        rel_text = rel_text.split(",", 1)[0]
    return rel_text


def _endpoint_string(labels: list[str], side: SchemaSide) -> str:
    if side == "ground_truth":
        return ",".join(sorted(labels))
    return ":".join(sorted(parse_label_list(",".join(sorted(labels)), sep=",")))


def _edge_rows(schema: dict[str, Any], side: SchemaSide) -> tuple[list[SchemaTriple], list[str]]:
    rows: list[SchemaTriple] = []
    skipped: list[str] = []
    node_lookup = _labels_for_node_name(schema)

    for index, edge in enumerate(schema.get("edge_types", []) or []):
        if not isinstance(edge, dict):
            skipped.append(f"edge_types[{index}] is not an object")
            continue
        if side == "predicted" and edge.get("is_canonical") is False:
            continue

        rel = _edge_relationship_name(edge, side)
        if not rel:
            skipped.append(f"edge_types[{index}] has no relationship type")
            continue

        topology = edge.get("topology")
        if topology:
            for topo_index, topo in enumerate(topology):
                src_names = topo.get("allowed_sources", []) or []
                dst_names = topo.get("allowed_targets", []) or []
                for src_name, dst_name in product(src_names, dst_names):
                    src_labels = node_lookup.get(str(src_name), [str(src_name)])
                    dst_labels = node_lookup.get(str(dst_name), [str(dst_name)])
                    src = _endpoint_string(src_labels, side)
                    dst = _endpoint_string(dst_labels, side)
                    if src and rel and dst:
                        rows.append((src, rel, dst))
                    else:
                        skipped.append(f"edge_types[{index}].topology[{topo_index}] has an empty endpoint")
            continue

        src_labels = edge.get("source_labels")
        dst_labels = edge.get("target_labels")
        if src_labels is None:
            src_name = edge.get("source") or edge.get("start_node") or ""
            src_labels = node_lookup.get(str(src_name), [str(src_name)] if src_name else [])
        if dst_labels is None:
            dst_name = edge.get("target") or edge.get("end_node") or ""
            dst_labels = node_lookup.get(str(dst_name), [str(dst_name)] if dst_name else [])

        src = _endpoint_string([str(label) for label in src_labels or []], side)
        dst = _endpoint_string([str(label) for label in dst_labels or []], side)
        if src and rel and dst:
            rows.append((src, rel, dst))
        else:
            skipped.append(f"edge_types[{index}] has an empty source, relationship, or target")

    return rows, skipped


def adapt_schema(schema: dict[str, Any], side: SchemaSide) -> AdaptedSchema:
    """Adapt a Zmarselo schema JSON object to PG-SB schema triples."""
    rows, skipped = _edge_rows(schema, side)
    triples: set[SchemaTriple] = set()
    duplicates: list[SchemaTriple] = []
    for row in rows:
        if row in triples:
            duplicates.append(row)
        triples.add(row)

    property_keys: dict[str, set[str]] = {}
    for node in schema.get("node_types", []) or []:
        if isinstance(node, dict):
            labels = node.get("labels") or ([node.get("name")] if node.get("name") else [])
            parent = ":".join(sorted(str(label) for label in labels if str(label)))
            property_keys[parent] = set(property_map(node.get("properties", {})))
    for edge in schema.get("edge_types", []) or []:
        if isinstance(edge, dict):
            rel = _edge_relationship_name(edge, side)
            if rel:
                property_keys[f"rel:{rel}"] = set(property_map(edge.get("properties", {})))

    return AdaptedSchema(
        triples=triples,
        duplicate_triples=duplicates,
        skipped_edges=skipped,
        property_keys_by_parent=property_keys,
    )


def pgsb_schema_csv_rows(schema: dict[str, Any], side: SchemaSide) -> list[dict[str, str]]:
    """Return CSV rows equivalent to the fields PG-SB uses for schema metrics."""
    rows, _ = _edge_rows(schema, side)
    if side == "ground_truth":
        return [
            {"srcType": src, "relationshipType": rel, "dstType": dst}
            for src, rel, dst in rows
        ]
    return [
        {
            "srcLabels": src.replace(":", ","),
            "relationshipTypes": rel,
            "dstLabels": dst.replace(":", ","),
        }
        for src, rel, dst in rows
    ]

