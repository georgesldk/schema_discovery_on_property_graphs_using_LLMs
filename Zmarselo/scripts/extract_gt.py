import argparse
from pg_schema_llm.pipeline.extract_gt import run_extract_gt

# Reads your Ground Truth inputs (.pgs and optional CSV like edge_types.csv) and 
# writes a normalized GT schema JSON.
# This is what you later compare against.

def main():
    parser = argparse.ArgumentParser(description="Extract Ground Truth schema")
    parser.add_argument("--input", required=True, help="Ground truth input folder")
    parser.add_argument("--output", required=True, help="Output folder for GT schema")

    args = parser.parse_args()
    run_extract_gt(args.input, args.output)


if __name__ == "__main__":
    main()
