import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pg_schema_llm.pipeline.compare import run_compare
from pg_schema_llm.utils.datasets import gt_schema_path, inferred_schema_path, parse_dataset_scenario


def main():
    parser = argparse.ArgumentParser(description="Compare inferred schema against GT")
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g. fib25, mb6, starwars, ldbc, pole)"
    )
    parser.add_argument("--gt",  help="Override GT file path")
    parser.add_argument("--inf", help="Override inferred file path")

    args = parser.parse_args()
    ds = parse_dataset_scenario(args.dataset)

    gt_path  = args.gt  or gt_schema_path(ds)
    inf_path = args.inf or inferred_schema_path(ds)

    run_compare(gt_path, inf_path)


if __name__ == "__main__":
    main()
