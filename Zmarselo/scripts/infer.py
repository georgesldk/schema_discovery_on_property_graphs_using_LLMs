import argparse
from pg_schema_llm.pipeline.infer_schema import run_infer_schema


def main():
    parser = argparse.ArgumentParser(description="Infer schema from raw PG CSV data")
    parser.add_argument("--data", required=True, help="Raw dataset folder (CSV files)")
    parser.add_argument("--output", required=True, help="Output folder for inferred schema")

    args = parser.parse_args()
    run_infer_schema(args.data, args.output)


if __name__ == "__main__":
    main()
