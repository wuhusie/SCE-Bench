"""
Data Preprocessing and Caching Script.

Loads large CSV files, aggregates statistics by month, and serializes them into pkl caches.
Subsequent plotting scripts read the cache directly to avoid repeated loading of large files.

Usage:
    python prepare_cache.py
"""
import sys
import pickle
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR,
    CACHE_MONTHLY_STATS,
    CACHE_RAW_SAMPLES,
    CACHE_ERRORS,
    TASK_CONFIGS,
)


def clean_llm_response(text) -> str:
    """Clean LLM response text, removing <think> tags, etc."""
    if pd.isna(text):
        return text
    text = str(text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def load_and_aggregate(
    file_path: Path,
    llm_col: str,
    human_col: str,
    task_name: str
) -> Dict[str, Any]:
    """
    Load a single CSV file, clean, and aggregate by month.

    Returns:
        {
            "llm_stats": DataFrame with columns [date, mean, std, count],
            "human_stats": DataFrame with columns [date, mean, std, count],
            "llm_samples": np.ndarray (Raw samples, used for JS divergence calculation),
            "human_samples": np.ndarray (Raw samples, used for JS divergence calculation),
            "errors": np.ndarray (Error = LLM - Human, used for error distribution plots)
        }
    """
    print(f"    Loading: {file_path.name}")

    try:
        df = pd.read_csv(file_path, on_bad_lines='warn')
    except Exception as e:
        print(f"    ✗ Read failed: {e}")
        return None

    # Clean LLM response
    df[llm_col] = df[llm_col].apply(clean_llm_response)

    # Convert to numeric
    df[llm_col] = pd.to_numeric(df[llm_col], errors='coerce')
    df[human_col] = pd.to_numeric(df[human_col], errors='coerce')

    # Remove missing values
    df = df.dropna(subset=[llm_col, human_col])

    # Range filtering (Only for labor and credit)
    task_config = TASK_CONFIGS.get(task_name, {})
    if task_config.get("cleaning_method") == "range":
        min_val, max_val = task_config.get("valid_range", (0, 100))
        df = df[(df[llm_col] >= min_val) & (df[llm_col] <= max_val)]

    # IQR filtering (Only for spending)
    elif task_config.get("cleaning_method") == "iqr":
        # Human IQR
        q1_h, q3_h = df[human_col].quantile([0.25, 0.75])
        iqr_h = q3_h - q1_h
        human_mask = (df[human_col] >= q1_h - 1.5*iqr_h) & (df[human_col] <= q3_h + 1.5*iqr_h)

        # LLM IQR
        q1_l, q3_l = df[llm_col].quantile([0.25, 0.75])
        iqr_l = q3_l - q1_l
        llm_mask = (df[llm_col] >= q1_l - 1.5*iqr_l) & (df[llm_col] <= q3_l + 1.5*iqr_l)

        df = df[human_mask & llm_mask]

    if df.empty:
        print(f"    ✗ No valid data after cleaning")
        return None

    # Extract raw samples (Used for JS divergence calculation)
    llm_samples = df[llm_col].values.copy()
    human_samples = df[human_col].values.copy()

    # Calculate errors (LLM - Human)
    errors = llm_samples - human_samples

    # Convert date format (YYYYMM -> datetime)
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m')

    # Aggregate by month
    llm_stats = df.groupby('date')[llm_col].agg(['mean', 'std', 'count']).reset_index()
    human_stats = df.groupby('date')[human_col].agg(['mean', 'std', 'count']).reset_index()

    print(f"    ✓ Sample count: {len(df):,}, Month count: {len(llm_stats)}")

    return {
        "llm_stats": llm_stats,
        "human_stats": human_stats,
        "llm_samples": llm_samples,
        "human_samples": human_samples,
        "errors": errors
    }


def process_all_data() -> tuple:
    """
    Process data for all models and tasks.

    Returns:
        (all_stats, all_samples, all_errors) tuple:

        all_stats: {
            "spending": {
                "Human": DataFrame (Monthly stats),
                "GPT-3.5": DataFrame,
                ...
            },
            ...
        }

        all_samples: {
            "spending": {
                "Human": np.ndarray (Raw samples),
                "GPT-3.5": np.ndarray,
                ...
            },
            ...
        }

        all_errors: {
            "spending": {
                "GPT-3.5": np.ndarray (Error = LLM - Human),
                ...
            },
            ...
        }
    """
    all_stats = {}
    all_samples = {}
    all_errors = {}

    for task_name, task_config in TASK_CONFIGS.items():
        print(f"\n[{task_name.upper()}]")

        llm_col = task_config["llm_col"]
        human_col = task_config["human_col"]
        file_pattern = task_config["file_pattern"]

        task_stats = {}
        task_samples = {}
        task_errors = {}
        human_stats = None
        human_samples = None

        # Iterate through all model directories
        for model_dir in sorted(DATA_DIR.iterdir()):
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name  # Use raw directory name as key

            # Find matching files
            files = list(model_dir.glob(file_pattern))
            if not files:
                print(f"  [{model_name}] File not found")
                continue

            file_path = files[0]  # Take the first match
            print(f"  [{model_name}]")

            result = load_and_aggregate(file_path, llm_col, human_col, task_name)

            if result:
                task_stats[model_name] = result["llm_stats"]
                task_samples[model_name] = result["llm_samples"]
                task_errors[model_name] = result["errors"]

                # Save Human data (Only need to save once, as Human data is the same for all models)
                if human_stats is None:
                    human_stats = result["human_stats"]
                    human_samples = result["human_samples"]

        # Add Human data
        if human_stats is not None:
            task_stats["Human"] = human_stats
            task_samples["Human"] = human_samples

        all_stats[task_name] = task_stats
        all_samples[task_name] = task_samples
        all_errors[task_name] = task_errors

    return all_stats, all_samples, all_errors


def save_cache(data: Dict, path: Path):
    """Save cache to pkl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n✓ Cache saved: {path}")


def main():
    print("=" * 60)
    print("  Data Preprocessing and Caching")
    print("=" * 60)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Monthly Stats Cache: {CACHE_MONTHLY_STATS}")
    print(f"Raw Samples Cache: {CACHE_RAW_SAMPLES}")
    print(f"Error Cache: {CACHE_ERRORS}")

    # Process data
    all_stats, all_samples, all_errors = process_all_data()

    # Save cache
    save_cache(all_stats, CACHE_MONTHLY_STATS)
    save_cache(all_samples, CACHE_RAW_SAMPLES)
    save_cache(all_errors, CACHE_ERRORS)

    # Print summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for task_name, task_stats in all_stats.items():
        models = [k for k in task_stats.keys() if k != "Human"]
        human_n = len(all_samples[task_name].get("Human", []))
        print(f"  {task_name}: {len(models)} models + Human ({human_n:,} samples)")

    print("\nCompleted! Plotting scripts can now be run.")


if __name__ == "__main__":
    main()
