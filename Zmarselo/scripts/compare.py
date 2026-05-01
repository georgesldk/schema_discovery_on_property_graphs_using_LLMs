import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pg_schema_llm.pipeline.compare import run_compare


def main():
    parser = argparse.ArgumentParser(description="Compare inferred schema against GT")
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g. fib25, mb6, starwars, ldbc, pole)"
    )
    parser.add_argument("--gt",  help="Override GT file path")
    parser.add_argument("--inf", help="Override inferred file path")

    args = parser.parse_args()
    ds = args.dataset.lower()

    gt_path  = args.gt  or f"03_outputs/schemas/ground_truth/{ds}/gt_{ds}.json"
    inf_path = args.inf or f"03_outputs/schemas/inferred/{ds}/inf_{ds}.json"

    run_compare(gt_path, inf_path)


if __name__ == "__main__":
    main()