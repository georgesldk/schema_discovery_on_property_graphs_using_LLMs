from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], env: dict) -> None:
    print(f"\n[EXEC] {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run extract_gt -> infer -> compare for a dataset"
    )
    ap.add_argument("dataset", help="Dataset name, e.g. fib25 / mb6 / starwars / ldbc / pole")
    ap.add_argument("--skip-gt",      action="store_true", help="Skip GT extraction")
    ap.add_argument("--skip-infer",   action="store_true", help="Skip inference")
    ap.add_argument("--skip-compare", action="store_true", help="Skip comparison")
    args = ap.parse_args()

    dataset = args.dataset.lower()

    # Paths (matching the individual scripts)
    gt_dir  = f"01_gts/gt_data_{dataset}"
    gt_file = f"03_outputs/schemas/ground_truth/{dataset}/gt_{dataset}.json"
    inf_file = f"03_outputs/schemas/inferred/{dataset}/inf_{dataset}.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    print("=" * 50)
    print(f"   PIPELINE: {dataset.upper()}")
    print("=" * 50)

    # ---- STEP 1: Extract GT ----
    if not args.skip_gt:
        if os.path.exists(gt_dir):
            print("\n>>> Step 1: Extracting Ground Truth...")
            run_command(
                [sys.executable, "scripts/extract_gt.py", dataset],
                env=env,
            )
        else:
            print(f"\n[SKIP] No GT directory: {gt_dir}")
    else:
        print("\n[SKIP] GT extraction (--skip-gt)")

    # ---- STEP 2: Infer Schema ----
    if not args.skip_infer:
        print("\n>>> Step 2: Inferring Schema...")
        run_command(
            [sys.executable, "scripts/infer.py", dataset],
            env=env,
        )
    else:
        print("\n[SKIP] Inference (--skip-infer)")

    # ---- STEP 3: Compare ----
    if not args.skip_compare:
        if os.path.exists(gt_file) or not args.skip_gt:
            print("\n>>> Step 3: Comparing...")
            run_command(
                [sys.executable, "scripts/compare.py", dataset],
                env=env,
            )
        else:
            print(f"\n[SKIP] No GT file: {gt_file}")
    else:
        print("\n[SKIP] Comparison (--skip-compare)")

    print(f"\nDone: {dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())