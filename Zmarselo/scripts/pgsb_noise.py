#!/usr/bin/env python3
"""Apply PG-SB-compatible noise to the currently loaded Neo4j database."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pg_schema_llm.io.pgsb_noise import PGSBNoiseConfig, apply_pgsb_noise


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Apply PG-SB-compatible property/label noise")
    parser.add_argument("noise", type=float, help="Property noise percent, e.g. 10 for PG-SB noise=10")
    parser.add_argument(
        "--label-fraction",
        type=float,
        default=0.0,
        help="Fraction of nodes that lose all labels, matching PG-SB label_percents values",
    )
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", None))
    parser.add_argument("--metadata-out", help="Optional JSON file recording the applied noise")
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        metadata = apply_pgsb_noise(
            driver,
            PGSBNoiseConfig(
                property_noise_percent=args.noise,
                label_fraction=args.label_fraction,
                batch_size=args.batch_size,
                database=args.database,
            ),
        )
    finally:
        driver.close()

    print("Applied PG-SB-compatible noise:")
    print(f"  property_noise_percent: {metadata['property_noise_percent']}")
    print(f"  label_fraction: {metadata['label_fraction']}")
    print(f"  node_properties: {len(metadata['node_properties'])}")
    print(f"  edge_properties: {len(metadata['edge_properties'])}")
    print(f"  node_labels: {len(metadata['node_labels'])}")

    if args.metadata_out:
        out = Path(args.metadata_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"  metadata: {out}")


if __name__ == "__main__":
    main()

