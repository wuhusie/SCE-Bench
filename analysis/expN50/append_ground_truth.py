import pandas as pd
from pathlib import Path
import os
import sys

import re
import ast

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared preprocessing
from src.analysis.common.preprocessing import clean_distribution_data

# Configuration
# Using absolute paths as per environment
base_dir = Path(r"/root/autodl-fs")
CACHE_DIR = base_dir / "data" / "sce" / "cache"
# Support for specific model subdirectory if needed
# For now, default to base exp1.3, but user might be running in subdirs.
# Let's keep it generic or follow the previous pattern. 
RESULT_DIR_BASE = base_dir / "result" / "exp1.3"
OUTPUT_DIR_BASE = base_dir / "result_cleaned" / "exp1.3"




# Task Definitions
# For Exp1.3, we look for files matching the pattern: {experiment_name}_*_probe_N*.csv
TASKS = [
    {
        "name": "spending",
        "gt_file": "sceProfile.pkl",
        "gt_col": "Q26v2part2"
    },
    {
        "name": "labor",
        "gt_file": "labor_original.pkl",
        "gt_col": "oo2c3"
    },
    {
        "name": "credit",
        "gt_file": "credit_original.pkl",
        "gt_col": "N17b_2"
    }
]

def main():
    # Allow recursive search to handle model subdirectories
    # e.g. /result/exp1.3/ModelName/file.csv
    
    if not RESULT_DIR_BASE.exists():
        print(f"❌ Result directory not found: {RESULT_DIR_BASE}")
        return

    print(f"📂 Searching in: {RESULT_DIR_BASE}")
    
    # Recursive glob for all csvs
    all_csvs = list(RESULT_DIR_BASE.rglob("*.csv"))
    print(f"📄 Found {len(all_csvs)} CSV files in total.")
    
    for task in TASKS:
        print(f"\nProcessing Task: {task['name']}")
        
        # Filter files for this task
        # Pattern: {task_name}_*_probe_N*.csv
        task_files = [f for f in all_csvs if f.name.startswith(f"{task['name']}_") and "_probe_N" in f.name]
        
        if not task_files:
            print(f"⚠️ No result files found for task '{task['name']}'")
            continue
            
        print(f"  Found {len(task_files)} matching files.")

        # 2. Load Ground Truth (PKL)
        gt_path = CACHE_DIR / task["gt_file"]
        if not gt_path.exists():
            print(f"❌ GT cache file not found: {gt_path}")
            continue
            
        print(f"  Reading GT cache: {gt_path.name}")
        try:
            df_gt = pd.read_pickle(gt_path)
        except Exception as e:
            print(f"❌ Failed to read pickle: {e}")
            continue

        # 3. Validation
        target_col = task['gt_col']
        if target_col not in df_gt.columns:
            print(f"❌ Column '{target_col}' not found in {task['gt_file']}")
            continue
        
        # Pre-process GT
        df_gt['userid'] = df_gt['userid'].astype(str)
        if 'date' in df_gt.columns:
            df_gt['date'] = pd.to_numeric(df_gt['date'], errors='coerce').fillna(0).astype(int)
        
        gt_subset = df_gt[['userid', 'date', target_col]].copy()

        # Process each result file
        for result_file in task_files:
            print(f"  -> Processing: {result_file.name}")
            
            try:
                df_res = pd.read_csv(result_file)
            except Exception as e:
                print(f"     ❌ Failed to load CSV: {e}")
                continue
                
            # Merge GT
            # 5. Save
            # Preserve directory structure relative to result base
            rel_path = result_file.relative_to(RESULT_DIR_BASE)
            output_subdir = OUTPUT_DIR_BASE / rel_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            output_filename = result_file.stem + "_withHumanData.csv"
            output_path = output_subdir / output_filename
            
            # --- Cleaning & Merging ---
            # 1. Preliminary Formatting
            if 'date' in df_res.columns:
                df_res['date'] = pd.to_numeric(df_res['date'], errors='coerce').fillna(0).astype(int)
            df_res['userid'] = df_res['userid'].astype(str)
            
            # 2. Merge (Left Join)
            # We merge first to get the GT column, then clean based on it
            merged_df = pd.merge(df_res, gt_subset, on=['userid', 'date'], how='left')
            
            # 3. Apply Standard Cleaning
            # Identifying the LLM column (assuming 'llm_response')
            llm_col = 'llm_response'
            if llm_col not in merged_df.columns:
                print("     ⚠️ 'llm_response' missing, skipping cleaning.")
                merged_df.to_csv(output_path, index=False)
                continue

            cleaned_df, stats = clean_distribution_data(
                merged_df, 
                task['name'], 
                llm_col, 
                target_col
            )
            
            # Printing Stats (as requested)
            rows_dropped_human = stats['human_null']
            rows_dropped_llm = stats['llm_invalid']
            total_dropped = stats['original_rows'] - stats['final_rows']
            
            print(f"     Merge & Clean Stats:")
            print(f"       - Original: {stats['original_rows']}")
            print(f"       - Dropped (Missing GT): {rows_dropped_human}")
            print(f"       - Dropped (Invalid LLM): {rows_dropped_llm}")
            print(f"       - Final Count: {stats['final_rows']} (Coverage: {stats['final_rows']/stats['original_rows']:.2%})")
            
            if stats['final_rows'] == 0:
                print("     ⚠️ No valid data left. Skipping save.")
                continue

            cleaned_df.to_csv(output_path, index=False)
            print(f"     ✅ Saved to: {output_path}")

if __name__ == "__main__":
    main()
