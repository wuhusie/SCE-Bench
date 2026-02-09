"""
Base evaluator class.

Defines the abstract interface for evaluators.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

import pandas as pd


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators.

    All evaluators must implement the evaluate_task method.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.

        Args:
            config: Optional evaluator configuration
        """
        self.config = config or {}

    @abstractmethod
    def evaluate_task(
        self,
        task_name: str,
        input_dir: Path,
        output_dir: Path,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single task.

        Args:
            task_name: Task name
            input_dir: Input data directory
            output_dir: Output results directory
            task_config: Task configuration (column mapping, etc.)

        Returns:
            A dictionary containing evaluation metrics
        """
        pass

    def find_task_file(self, input_dir: Path, file_pattern: str) -> Path:
        """
        Find task data file based on a pattern.

        Args:
            input_dir: Input directory
            file_pattern: File matching pattern (glob format)

        Returns:
            Matched file path (returns the latest if multiple matches)

        Raises:
            FileNotFoundError: If no matching file is found
        """
        matches = list(input_dir.glob(file_pattern))

        if not matches:
            raise FileNotFoundError(
                f"No matching file found. Directory: {input_dir}, Pattern: {file_pattern}"
            )

        # Return the most recently modified file if multiple matches are found
        return max(matches, key=lambda p: p.stat().st_mtime)

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        filename: str = "metrics.json"
    ) -> Path:
        """
        Save evaluation results to a JSON file.

        Args:
            results: Evaluation results dictionary
            output_dir: Output directory
            filename: Output filename

        Returns:
            Path to the saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return output_path

    def log(self, message: str, level: str = "info"):
        """Simple log output."""
        prefix = {
            "info": "  ",
            "success": "  ✓",
            "warning": "  ⚠",
            "error": "  ✗"
        }.get(level, "  ")
        print(f"{prefix} {message}")
