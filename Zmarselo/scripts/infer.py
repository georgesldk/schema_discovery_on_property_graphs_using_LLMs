import argparse
import sys
from pathlib import Path

# Add src to sys.path automatically
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pg_schema_llm.pipeline.infer_schema import run_infer_schema
from pg_schema_llm.utils.datasets import inferred_schema_path, parse_dataset_scenario


def main():
    parser = argparse.ArgumentParser(description="Infer schema from the currently loaded Neo4j graph")

    # NEW: positional dataset name
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g. fib25, mb6, starwars)"
    )

    args = parser.parse_args()

    dataset = parse_dataset_scenario(args.dataset)
    data_dir = f"02_pgs/pg_data_{dataset.base}"
    out_file = inferred_schema_path(dataset)

    run_infer_schema(data_dir, out_file)


if __name__ == "__main__":
    main()
