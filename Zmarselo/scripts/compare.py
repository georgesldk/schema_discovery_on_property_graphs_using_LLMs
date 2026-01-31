import argparse
from pg_schema_llm.pipeline.compare import run_compare


def main():
    parser = argparse.ArgumentParser(description="Compare inferred schema against GT")

    # Minimal: positional dataset name, like extract/infer
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g. fib25, mb6, starwars, ldbc)"
    )

    args = parser.parse_args()
    ds = args.dataset.lower()

    gt_path = f"03_outputs/schemas/ground_truth/{ds}/gt_{ds}.json"
    inferred_path = f"03_outputs/schemas/inferred/{ds}/inf_{ds}.json"

    run_compare(gt_path, inferred_path)


if __name__ == "__main__":
    main()
