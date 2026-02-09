"""
Data Preprocessing Module.

Provides data cleaning functions:
- clean_task_data: Data cleaning for single-point evaluation
- clean_distribution_data: Data cleaning for distribution evaluation
"""

import re
import ast
import numpy as np
import pandas as pd
from typing import Tuple
from .task_config import get_task_config


def clean_llm_response(text) -> str:
    """
    Clean LLM response text.

    Cleaning steps:
    1. Remove <think>...</think> tags and their content
    2. Remove leading/trailing whitespace, newlines, and empty lines

    Parameters:
        text: Raw LLM response text

    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return text

    text = str(text)

    # Remove <think>...</think> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Remove leading/trailing whitespace (spaces, newlines, tabs, etc.)
    text = text.strip()

    return text


def extract_samples(text):
    """
    Extract a list of numbers from text.

    Handles formats like Markdown code blocks, text prefixes, etc.
    """
    if pd.isna(text):
        return None

    text = str(text).strip()

    try:
        val = ast.literal_eval(text)
        if isinstance(val, list):
            return val
    except:
        pass

    match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if match:
        try:
            list_str = f"[{match.group(1)}]"
            val = ast.literal_eval(list_str)
            if isinstance(val, list):
                return val
        except:
            pass

    return None


def clean_task_data(
    df: pd.DataFrame,
    task_name: str,
    llm_col: str,
    human_col: str
) -> Tuple[pd.DataFrame, dict]:
    """
    Data cleaning for single-point evaluation.

    Cleaning steps:
    1. Clean LLM response (remove <think> tags, whitespace)
    2. Remove rows where Human column is null
    3. Convert to numeric types
    4. Remove rows where any column is NaN

    Parameters:
        df: Original DataFrame
        task_name: Task name
        llm_col: LLM response column name
        human_col: Human response column name

    Returns:
        (Cleaned DataFrame, Data cleaning statistics dict)
    """
    df_clean = df.copy()
    stats = {
        "original_rows": len(df),
        "human_null": 0,
        "llm_invalid": 0,
        "rows_removed_range": 0,
        "rows_removed_iqr": 0,
        "final_rows": 0
    }

    # 1. Basic column cleaning
    # Clean LLM response
    df_clean[llm_col] = df_clean[llm_col].apply(clean_llm_response)

    # Human null check
    human_null_mask = df_clean[human_col].isna()
    if df_clean[human_col].dtype == object:
        human_null_mask = human_null_mask | (df_clean[human_col].astype(str).str.strip() == '')
    stats["human_null"] = int(human_null_mask.sum())
    df_clean = df_clean[~human_null_mask].copy()

    # Convert to numeric types
    df_clean[llm_col] = pd.to_numeric(df_clean[llm_col], errors='coerce')
    df_clean[human_col] = pd.to_numeric(df_clean[human_col], errors='coerce')

    # LLM invalid value check (Strict Numeric)
    llm_invalid_mask = df_clean[llm_col].isna()
    stats["llm_invalid"] = int(llm_invalid_mask.sum())
    df_clean = df_clean[~llm_invalid_mask].copy()

    # 2. Advanced cleaning strategies
    task_config = get_task_config(task_name)
    cleaning_method = task_config.get("cleaning_method", "none")

    # Range Filtering (Suitable for Labor and Credit tasks)
    if cleaning_method == "range":
        min_val, max_val = task_config.get("valid_range", (0, 100))
        # Keep values within closed interval [min, max], including boundaries (e.g., 0 and 100 are valid)
        range_mask = (df_clean[llm_col] >= min_val) & (df_clean[llm_col] <= max_val)
        removed_count = (~range_mask).sum()
        stats["rows_removed_range"] = int(removed_count)
        df_clean = df_clean[range_mask].copy()

    # IQR Outlier Filtering (Suitable for Spending tasks)
    elif cleaning_method == "iqr":
        # Calculate IQR (Interquartile Range) for Human data
        q1_h = df_clean[human_col].quantile(0.25)
        q3_h = df_clean[human_col].quantile(0.75)
        iqr_h = q3_h - q1_h
        # Normal range mask for Human data (Only upper bound, no lower bound)
        human_inlier_mask = df_clean[human_col] <= (q3_h + 15 * iqr_h)

        # Calculate IQR for LLM data
        q1_l = df_clean[llm_col].quantile(0.25)
        q3_l = df_clean[llm_col].quantile(0.75)
        iqr_l = q3_l - q1_l
        # Normal range mask for LLM data (Only upper bound, no lower bound)
        llm_inlier_mask = df_clean[llm_col] <= (q3_l + 15 * iqr_l)

        # Intersection: Keep rows only if both Human and LLM data are within normal range
        combined_inlier_mask = human_inlier_mask & llm_inlier_mask
        removed_count = (~combined_inlier_mask).sum()
        stats["rows_removed_iqr"] = int(removed_count)
        df_clean = df_clean[combined_inlier_mask].copy()

    # Final cleanup (Ensure no NaNs missed due to special cases)
    df_clean = df_clean.dropna(subset=[llm_col, human_col])
    stats["final_rows"] = int(len(df_clean))

    return df_clean, stats


def clean_distribution_data(
    df: pd.DataFrame,
    task_name: str,
    llm_col: str,
    human_col: str
) -> Tuple[pd.DataFrame, dict]:
    """
    Data cleaning for distribution evaluation.

    Cleaning steps:
    1. Clean LLM response (remove <think> tags, whitespace)
    2. Remove rows where Human column is null
    3. Parse LLM distribution column (string -> list)
    4. Remove rows where LLM parsing failed or Human is null

    Parameters:
        df: Original DataFrame
        task_name: Task name
        llm_col: LLM response column name
        human_col: Human response column name

    Returns:
        (Cleaned DataFrame, statistics dict)
    """
    df_clean = df.copy()
    stats = {
        "original_rows": len(df),
        "human_null": 0,
        "llm_invalid": 0,
        "rows_removed_range": 0,
        "rows_removed_iqr": 0,
        "final_rows": 0
    }

    # 1. Basic column cleaning
    # Clean LLM response
    df_clean[llm_col] = df_clean[llm_col].apply(clean_llm_response)

    # Human null check
    human_null_mask = df_clean[human_col].isna()
    if df_clean[human_col].dtype == object:
        human_null_mask = human_null_mask | (df_clean[human_col].astype(str).str.strip() == '')
    stats["human_null"] = int(human_null_mask.sum())
    df_clean = df_clean[~human_null_mask].copy()

    # Parse LLM list
    if llm_col in df_clean.columns:
        df_clean[llm_col] = df_clean[llm_col].apply(extract_samples)
        llm_invalid_mask = df_clean[llm_col].isna()
        stats["llm_invalid"] = int(llm_invalid_mask.sum())
        df_clean = df_clean[~llm_invalid_mask].copy()

    # Convert to numeric types
    df_clean[human_col] = pd.to_numeric(df_clean[human_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[human_col])

    # 2. Advanced cleaning strategies (Range filtering for samples only)
    task_config = get_task_config(task_name)
    cleaning_method = task_config.get("cleaning_method", "none")

    if cleaning_method == "range":
        min_val, max_val = task_config.get("valid_range", (0, 100))

        def filter_samples(samples):
            """Internal function: Filter invalid samples"""
            if not isinstance(samples, list): return []
            # Keep values within closed interval [min, max], including boundaries (e.g., 0 and 100 are valid)
            return [x for x in samples if isinstance(x, (int, float)) and min_val <= x <= max_val]

        # Apply filtering: Clean sample list row by row
        df_clean[llm_col] = df_clean[llm_col].apply(filter_samples)

        # Remove rows where sample list became empty after filtering
        empty_mask = df_clean[llm_col].apply(len) == 0
        removed_count = empty_mask.sum()
        stats["rows_removed_range"] = int(removed_count)
        df_clean = df_clean[~empty_mask].copy()

    # IQR Outlier Filtering (Suitable for Spending distribution tasks)
    elif cleaning_method == "iqr":
        # 1. IQR Filtering for Human data
        q1_h = df_clean[human_col].quantile(0.25)
        q3_h = df_clean[human_col].quantile(0.75)
        iqr_h = q3_h - iqr_h
        human_inlier_mask = df_clean[human_col] <= (q3_h + 15 * iqr_h)

        # 2. IQR Filtering for LLM data (Based on list mean)
        # Calculate mean for each sample list
        llm_means = df_clean[llm_col].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)

        # Calculate IQR on the mean sequence
        q1_l = llm_means.quantile(0.25)
        q3_l = llm_means.quantile(0.75)
        iqr_l = q3_l - q1_l

        # Find rows where mean is within normal range (Only upper bound, no lower bound)
        llm_inlier_mask = llm_means <= (q3_l + 15 * iqr_l)
        
        # Handle potential NaNs (e.g. mean is NaN due to empty list)
        llm_inlier_mask = llm_inlier_mask.fillna(False)

        # 3. Intersection
        combined_inlier_mask = human_inlier_mask & llm_inlier_mask
        removed_count = (~combined_inlier_mask).sum()
        stats["rows_removed_iqr"] = int(removed_count)
        df_clean = df_clean[combined_inlier_mask].copy()

    stats["final_rows"] = int(len(df_clean))
    return df_clean, stats
