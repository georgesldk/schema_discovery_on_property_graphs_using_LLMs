import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pg_schema_llm.evaluation.runner import evaluate_json_schemas, print_summary
from pg_schema_llm.utils.datasets import (
    gt_schema_path,
    inferred_schema_path,
    parse_dataset_scenario,
    pgsb_metrics_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated schema JSON with PG-SB-compatible metrics")
    parser.add_argument("dataset", nargs="?", help="Dataset name (e.g. fib25, mb6, starwars, ldbc, pole)")
    parser.add_argument("--gt", help="Ground-truth schema JSON path")
    parser.add_argument("--inf", help="Inferred schema JSON path")
    parser.add_argument("--out", help="Output metrics JSON path")
    parser.add_argument(
        "--pgsb-repo",
        default=str(Path(__file__).resolve().parents[2] / "PG-SB"),
        help="Read-only PG-SB repository path",
    )
    args = parser.parse_args()

    if not args.dataset and not (args.gt and args.inf):
        parser.error("provide a dataset or both --gt and --inf")

    dataset = parse_dataset_scenario(args.dataset or "custom")
    gt_path = args.gt or gt_schema_path(dataset)
    inf_path = args.inf or inferred_schema_path(dataset)
    out_path = args.out or pgsb_metrics_path(dataset)

    payload = evaluate_json_schemas(
        gt_path=gt_path,
        inf_path=inf_path,
        dataset=dataset.scenario,
        output_path=out_path,
        pgsb_repo_path=args.pgsb_repo,
    )
    print_summary(payload)
    print(f"  Metrics JSON: {out_path}")


if __name__ == "__main__":
    main()
