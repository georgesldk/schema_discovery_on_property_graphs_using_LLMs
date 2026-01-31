import argparse
from pg_schema_llm.pipeline.extract_gt import run_extract_gt


def main():
    parser = argparse.ArgumentParser(description="Extract GT schema for a dataset")
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g. fib25, mb6, starwars, ldbc)"
    )

    args = parser.parse_args()
    ds = args.dataset.lower()

    input_dir = f"01_gts/gt_data_{ds}"
    output_file = f"03_outputs/schemas/ground_truth/{ds}/gt_{ds}.json"

    run_extract_gt(input_dir, output_file)


if __name__ == "__main__":
    main()
