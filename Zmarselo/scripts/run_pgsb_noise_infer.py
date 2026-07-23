#!/usr/bin/env python3
"""Apply PG-SB-compatible noise and write inference to a scenario folder."""
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
from pg_schema_llm.pipeline.infer_schema import run_infer_schema
from pg_schema_llm.utils.datasets import inferred_schema_path, parse_dataset_scenario


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Apply PG-SB-style noise to the loaded graph, then infer into <dataset>-<noise>/"
    )
    parser.add_argument("dataset", help="Base dataset name, e.g. starwars")
    parser.add_argument("noise", type=int, help="Property noise percent, e.g. 10")
    parser.add_argument("--label-fraction", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--skip-apply", action="store_true", help="Only write to the scenario folder; assume noise is already applied")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", None))
    args = parser.parse_args()

    scenario = parse_dataset_scenario(f"{args.dataset}-{args.noise}")
    data_dir = f"02_pgs/pg_data_{scenario.base}"
    output_path = inferred_schema_path(scenario)
    metadata_path = Path(output_path).with_name(f"noise_{scenario.scenario}.json")

    metadata = {
        "property_noise_percent": args.noise,
        "label_fraction": args.label_fraction,
        "skipped_apply": args.skip_apply,
    }

    if not args.skip_apply:
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

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Inferring scenario {scenario.scenario} from base dataset {scenario.base}")
    print(f"Output: {output_path}")
    run_infer_schema(data_dir, output_path)


if __name__ == "__main__":
    main()

