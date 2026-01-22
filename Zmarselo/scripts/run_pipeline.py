import argparse
import os
import subprocess
import sys

def run_command(cmd):
    """Prints and runs a command, exiting if it fails."""
    print(f"\n[EXEC] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!!! COMMAND FAILED !!!")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Master Switch for Schema Discovery")
    parser.add_argument("dataset", help="Dataset name (e.g., 'fib25' or 'mb6')")
    parser.add_argument("--skip-gt", action="store_true", help="Skip Ground Truth extraction")
    args = parser.parse_args()

    # --- CONFIGURATION ---
    dataset = args.dataset.lower()
    
    # Inputs
    pg_dir = f"pg_data_{dataset}"
    gt_dir = f"gt_data_{dataset}"
    
    # Outputs (Organized by Type, then Dataset)
    gt_out_dir = os.path.join("outputs", "schemas", "ground_truth", dataset)
    inf_out_dir = os.path.join("outputs", "schemas", "inferred", dataset)
    
    gt_file = os.path.join(gt_out_dir, f"golden_truth_gt_data_{dataset}.json")
    inf_file = os.path.join(inf_out_dir, "inferred_schema.json")

    # --- CHECKS ---
    if not os.path.exists(pg_dir):
        print(f"Error: Data folder '{pg_dir}' does not exist.")
        sys.exit(1)

    print(f"==================================================")
    print(f"   RUNNING PIPELINE FOR: {dataset.upper()}")
    print(f"==================================================")

    # --- STEP 1: EXTRACT GROUND TRUTH ---
    if os.path.exists(gt_dir) and not args.skip_gt:
        # Create output folder first to prevent 'FileNotFound' errors in older scripts
        os.makedirs(gt_out_dir, exist_ok=True)
        
        print(f"\n>>> Step 1: Extracting Ground Truth...")
        run_command([
            "python", "scripts/extract_gt.py",
            "--input", gt_dir,
            "--output", gt_file
        ])
    else:
        print(f"\n[INFO] GT folder '{gt_dir}' not found (or skipped).")

    # --- STEP 2: INFER SCHEMA ---
    # Create output folder first
    os.makedirs(inf_out_dir, exist_ok=True)
    
    print(f"\n>>> Step 2: Inferring Schema...")
    run_command([
        "python", "scripts/infer.py",
        "--data", pg_dir,
        "--output", inf_file
    ])

    # --- STEP 3: COMPARE ---
    if os.path.exists(gt_file):
        print(f"\n>>> Step 3: Comparing...")
        run_command([
            "python", "scripts/compare.py",
            "--gt", gt_file,
            "--inferred", inf_file
        ])
    else:
        print("\n[INFO] No GT file found. Skipping comparison.")

if __name__ == "__main__":
    main()