"""Normalization helpers that mirror PG-SB's executable evaluator."""
from __future__ import annotations

from typing import Any


TYPE_ALIASES: dict[str, str] = {
    "LONG": "INTEGER",
    "INT": "INTEGER",
    "INTEGER": "INTEGER",
    "DOUBLE": "DOUBLE",
    "FLOAT": "DOUBLE",
    "STRING": "STRING",
    "BOOLEAN": "BOOLEAN",
    "BOOL": "BOOLEAN",
    "DATE": "DATE",
    "DATETIME": "DATE",
    "POINT": "STRING",
    "STRINGARRAY": "LIST",
    "LIST": "LIST",
    "ARRAY": "LIST",
}


def parse_label_list(value: Any, sep: str = ":") -> list[str]:
    """Equivalent to PG-SB evaluation._parse_label_list."""
    return [item for item in str(value).split(sep) if item]


def normalize_property_type(value: Any) -> str | None:
    """Normalize a legacy or generic property type for preservation in adapters."""
    if value is None or value == "":
        return None
    raw = str(value).strip().upper()
    return TYPE_ALIASES.get(raw, raw)


def normalize_constraint(prop: dict[str, Any]) -> str | None:
    """Normalize legacy mandatory booleans and current constraint strings."""
    constraint = prop.get("constraint")
    if constraint:
        return str(constraint).strip().upper()
    if "mandatory" in prop:
        return "MANDATORY" if bool(prop["mandatory"]) else "OPTIONAL"
    return None


def property_map(properties: Any) -> dict[str, dict[str, Any]]:
    """Return normalized property descriptors keyed by exact property name."""
    if isinstance(properties, dict):
        raw_items = properties.items()
    else:
        raw_items = []
        for prop in properties or []:
            if isinstance(prop, dict) and prop.get("name"):
                raw_items.append((str(prop["name"]), prop))

    normalized: dict[str, dict[str, Any]] = {}
    for key, value in raw_items:
        if not key:
            continue
        info = value if isinstance(value, dict) else {}
        out = dict(info)
        out["name"] = key
        constraint = normalize_constraint(out)
        if constraint:
            out["constraint"] = constraint
        prop_type = normalize_property_type(out.get("data_type") or out.get("type"))
        if prop_type:
            out["data_type"] = prop_type
        normalized[key] = out
    return normalized

