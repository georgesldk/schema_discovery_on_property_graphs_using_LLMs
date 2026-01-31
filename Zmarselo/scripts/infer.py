import argparse
from pg_schema_llm.pipeline.infer_schema import run_infer_schema


def main():
    parser = argparse.ArgumentParser(description="Infer schema from raw PG CSV data")

    # NEW: positional dataset name
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g. fib25, mb6, starwars)"
    )

    args = parser.parse_args()

    data_dir = f"02_pgs/pg_data_{args.dataset.lower()}"
    out_file = f"03_outputs/schemas/inferred/{args.dataset.lower()}/inf_{args.dataset.lower()}.json"

    run_infer_schema(data_dir, out_file)


if __name__ == "__main__":
    main()
