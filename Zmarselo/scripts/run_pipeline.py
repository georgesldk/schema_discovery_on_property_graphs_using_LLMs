from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def newest_mtime(path: str) -> float:
    """Newest modification time in a file or folder (recursive)."""
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getmtime(path)

    newest = 0.0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                newest = max(newest, os.path.getmtime(fp))
            except OSError:
                pass
    return newest


def run_command(cmd: list[str], env: dict) -> None:
    print(f"\n[EXEC] {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run extract_gt -> infer -> compare for a dataset")
    ap.add_argument("dataset", help="Dataset name, e.g. fib25 / mb6 / starwars / ldbc")
    ap.add_argument("--skip-gt", action="store_true", help="Skip GT extraction")
    ap.add_argument("--skip-infer", action="store_true", help="Skip inference")
    ap.add_argument("--skip-compare", action="store_true", help="Skip comparison")
    ap.add_argument("--cache", action="store_true", help="Skip steps if outputs are up-to-date")
    args = ap.parse_args()

    dataset = args.dataset.lower()

    # Inputs
    pg_dir = f"pg_data_{dataset}"
    gt_dir = f"gt_data_{dataset}"

    # Outputs
    gt_out_dir = Path("outputs") / "schemas" / "ground_truth" / dataset
    inf_out_dir = Path("outputs") / "schemas" / "inferred" / dataset

    gt_file = gt_out_dir / f"gt_{dataset}.json"
    inf_file = inf_out_dir / f"inf_{dataset}.json"

    print("==================================================")
    print(f"   RUNNING PIPELINE FOR: {dataset.upper()}")
    print("==================================================")

    # Make imports work without manual terminal setup
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    # ---- STEP 1: Extract GT ----
    if not args.skip_gt and os.path.exists(gt_dir):
        gt_out_dir.mkdir(parents=True, exist_ok=True)

        should_run = True
        if args.cache and gt_file.exists():
            should_run = newest_mtime(gt_dir) > gt_file.stat().st_mtime

        if should_run:
            print("\n>>> Step 1: Extracting Ground Truth...")
            run_command(
                [sys.executable, "scripts/extract_gt.py", "--input", gt_dir, "--output", str(gt_file)],
                env=env,
            )
        else:
            print("\n[CACHE] GT is up-to-date -> skipping extract.")
    else:
        print(f"\n[INFO] GT step skipped (missing '{gt_dir}' or --skip-gt).")

    # ---- STEP 2: Infer Schema ----
    if not args.skip_infer:
        if not os.path.exists(pg_dir):
            print(f"\n[ERROR] Data folder '{pg_dir}' does not exist.")
            return 1

        inf_out_dir.mkdir(parents=True, exist_ok=True)

        should_run = True
        if args.cache and inf_file.exists():
            should_run = newest_mtime(pg_dir) > inf_file.stat().st_mtime

        if should_run:
            print("\n>>> Step 2: Inferring Schema...")
            run_command(
                [sys.executable, "scripts/infer.py", "--data", pg_dir, "--output", str(inf_file)],
                env=env,
            )
        else:
            print("\n[CACHE] Inferred schema is up-to-date -> skipping infer.")
    else:
        print("\n[INFO] Inference skipped (--skip-infer).")

    # ---- STEP 3: Compare ----
    if not args.skip_compare and gt_file.exists():
        print("\n>>> Step 3: Comparing...")
        run_command(
            [sys.executable, "scripts/compare.py", "--gt", str(gt_file), "--inferred", str(inf_file)],
            env=env,
        )
    else:
        print("\n[INFO] Comparison skipped (no GT file or --skip-compare).")

    print(f"\n? Done: {dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
