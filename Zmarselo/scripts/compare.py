import argparse
from pg_schema_llm.pipeline.compare import run_compare


def main():
    parser = argparse.ArgumentParser(description="Compare inferred schema against GT")
    parser.add_argument("--gt", required=True, help="Golden truth schema JSON")
    parser.add_argument("--inferred", required=True, help="Inferred schema JSON")

    args = parser.parse_args()
    run_compare(args.gt, args.inferred)


if __name__ == "__main__":
    main()
