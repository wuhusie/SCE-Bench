"""
Pointwise Evaluator.

Evaluation Metrics:
- JS Divergence (Jensen-Shannon Divergence)
- Spearman's ρ (Spearman's rank correlation coefficient)
- RMSE (Root Mean Square Error)

Reused from:
- src/analysis/common/metrics.py: compute_js_divergence, compute_temporal_spearman, compute_rmse
- src/analysis/common/preprocessing.py: clean_task_data
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from .base import BaseEvaluator
from ..common.metrics import (
    compute_js_divergence,
    compute_temporal_spearman,
    compute_rmse,
    compute_mae,
)
from ..common.preprocessing import clean_task_data


class PointwiseEvaluator(BaseEvaluator):
    """
    Pointwise Evaluator.

    Used to evaluate the alignment between LLM pointwise predictions and human ground truth.

    Metrics:
    - js_divergence: Distribution discrepancy (lower is better)
    - spearman_rho: Temporal rank correlation (higher is better, close to 1)
    - rmse: Root Mean Square Error (lower is better)
    """

    def evaluate_task(
        self,
        task_name: str,
        input_dir: Path,
        output_dir: Path,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate pointwise metrics for a single task.

        Args:
            task_name: Task name
            input_dir: Input data directory
            output_dir: Output directory
            task_config: Task configuration

        Returns:
            A dictionary containing js_divergence, spearman_rho, spearman_p, rmse, etc.
        """
        llm_col = task_config["llm_col"]
        human_col = task_config["human_col"]
        file_pattern = task_config["file_pattern"]

        # 1. Load data
        file_path = self.find_task_file(input_dir, file_pattern)
        self.log(f"Loading file: {file_path.name}")

        df = pd.read_csv(file_path)
        self.log(f"Original rows: {len(df):,}")

        # 2. Data preprocessing
        df_clean, clean_stats = clean_task_data(
            df=df,
            task_name=task_name,
            llm_col=llm_col,
            human_col=human_col
        )

        n_samples = clean_stats["final_rows"]
        self.log(f"Human null values removed: {clean_stats['human_null']:,}")
        self.log(f"Samples after cleaning: {n_samples:,}")

        if n_samples == 0:
            return {
                "error": "No valid samples after cleaning",
                "cleaning_stats": clean_stats
            }

        # 3. Extract values
        human_vals = df_clean[human_col].values
        llm_vals = df_clean[llm_col].values

        # 4. Calculate RMSE
        rmse = compute_rmse(human_vals, llm_vals)

        # 5. Aggregate by time and calculate Spearman
        if 'date' in df_clean.columns:
            daily_stats = df_clean.groupby('date')[[llm_col, human_col]].mean().reset_index()
            n_time_points = len(daily_stats)
            self.log(f"Number of time points: {n_time_points}")

            spearman_rho, spearman_p = compute_temporal_spearman(
                daily_stats[human_col].values,
                daily_stats[llm_col].values
            )
        else:
            # If no date column, calculate directly
            n_time_points = 0
            spearman_rho, spearman_p = compute_temporal_spearman(human_vals, llm_vals)
            self.log("No date column, using raw samples to calculate Spearman", level="warning")

        # 6. Calculate JS Divergence
        js_div = compute_js_divergence(human_vals, llm_vals)

        # 7. Calculate MAE
        mae = compute_mae(human_vals, llm_vals)

        # 8. Construct results
        result = {
            "spearman_rho": round(spearman_rho, 4) if not np.isnan(spearman_rho) else None,
            "spearman_p": round(spearman_p, 6) if not np.isnan(spearman_p) else None,
            "js_divergence": round(js_div, 4) if not np.isnan(js_div) else None,
            "rmse": round(rmse, 4) if not np.isnan(rmse) else None,
            "mae": round(mae, 2) if not np.isnan(mae) else None,
            "n_samples": int(n_samples),
            "n_time_points": int(n_time_points),
            "cleaning_stats": clean_stats
        }

        # 9. Output results
        self.log(f"Spearman ρ: {result['spearman_rho']}", level="success")
        self.log(f"JS Divergence: {result['js_divergence']}", level="success")
        self.log(f"RMSE: {result['rmse']}", level="success")
        self.log(f"MAE: {result['mae']}", level="success")

        return result
