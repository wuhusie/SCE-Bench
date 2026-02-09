"""
Distribution Evaluator.

Evaluation Metrics:
- MAE (Mean Absolute Error)
- Coverage Rate (90% CI Coverage Rate)

Reused from common module:
- common/metrics.py: compute_mae, calculate_ecdf
- common/preprocessing.py: clean_distribution_data
"""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .base import BaseEvaluator
from ..common.metrics import compute_mae, calculate_ecdf
from ..common.preprocessing import clean_distribution_data


class DistributionEvaluator(BaseEvaluator):
    """
    Distribution Evaluator.

    Used to evaluate the calibration of LLM sampling distribution predictions.

    Metrics:
    - mae: Mean Absolute Error (based on sample mean vs. ground truth)
    - coverage_rate: 90% CI coverage rate (target: ~90%)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.confidence_level = self.config.get("confidence_level", 0.90)

    def evaluate_task(
        self,
        task_name: str,
        input_dir: Path,
        output_dir: Path,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate distribution metrics for a single task.

        Args:
            task_name: Task name
            input_dir: Input data directory
            output_dir: Output directory
            task_config: Task configuration

        Returns:
            A dictionary containing mae, coverage_rate
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
        df_clean, clean_stats = clean_distribution_data(
            df=df,
            task_name=task_name,
            llm_col=llm_col,
            human_col=human_col
        )

        n_samples = clean_stats["final_rows"]
        self.log(f"Human null values removed: {clean_stats['human_null']:,}")
        self.log(f"LLM parsing failed: {clean_stats['llm_invalid']:,}")
        self.log(f"Samples after cleaning: {n_samples:,}")

        if n_samples == 0:
            return {
                "error": "No valid samples after cleaning",
                "cleaning_stats": clean_stats
            }

        # 3. Calculate ECDF and Coverage
        eval_results = []

        for _, row in df_clean.iterrows():
            samples = row[llm_col]
            truth = row[human_col]

            if not isinstance(samples, list):
                continue

            ecdf = calculate_ecdf(samples, truth)
            if ecdf is None:
                continue

            # 90% CI: 0.05 <= ECDF <= 0.95
            lower_bound = (1 - self.confidence_level) / 2
            upper_bound = (1 + self.confidence_level) / 2
            is_hit = lower_bound <= ecdf <= upper_bound

            eval_results.append({
                'is_hit': is_hit,
                'truth': truth,
                'mean_pred': np.mean(samples)
            })

        if not eval_results:
            return {
                "error": "No valid ECDF calculation results",
                "cleaning_stats": clean_stats
            }

        df_eval = pd.DataFrame(eval_results)

        # 4. Calculate metrics
        total_valid = len(df_eval)
        hit_count = df_eval['is_hit'].sum()
        coverage_rate = hit_count / total_valid

        # MAE: sample mean vs. ground truth
        mae = compute_mae(df_eval['truth'].values, df_eval['mean_pred'].values)

        # 5. Construct results
        result = {
            "mae": round(mae, 2) if not np.isnan(mae) else None,
            "coverage_rate": round(coverage_rate, 4),
            "n_samples": int(total_valid),
            "cleaning_stats": clean_stats
        }

        # 6. Output results
        self.log(f"MAE: {result['mae']}", level="success")
        self.log(f"Coverage Rate: {result['coverage_rate']:.2%} (target: ~90%)", level="success")

        return result
