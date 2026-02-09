"""
Task Configuration Module.

Defines column name mappings and file patterns for each task.
"""

from typing import Dict, Any, Optional

# Task Configuration: Defines column name mappings for each task
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "spending": {
        "llm_col": "llm_response",
        "human_col": "Q26v2part2",
        "file_pattern": "spending_*_withHumanData.csv",
        "description": "Spending expectation change",
        "cleaning_method": "iqr",
    },
    "labor": {
        "llm_col": "llm_response",
        "human_col": "oo2c3",
        "file_pattern": "labor_*_withHumanData.csv",
        "description": "Job acceptance probability (0-100)",
        "cleaning_method": "range",
        "valid_range": (0, 100),
    },
    "credit": {
        "llm_col": "llm_response",
        "human_col": "N17b_2",
        "file_pattern": "credit_*_withHumanData.csv",
        "description": "Loan application probability (0-100)",
        "cleaning_method": "range",
        "valid_range": (0, 100),
    }
}


def get_task_config(task_name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Retrieve task configuration, with support for overriding default values.

    Parameters:
        task_name: Task name (spending/labor/credit)
        overrides: Optional dictionary of configuration overrides

    Returns:
        Task configuration dictionary

    Raises:
        ValueError: Unknown task name
    """
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task name: {task_name}, available options: {list(TASK_CONFIGS.keys())}")

    config = TASK_CONFIGS[task_name].copy()

    if overrides:
        config.update(overrides)

    return config


def list_available_tasks() -> list:
    """Returns a list of all available task names."""
    return list(TASK_CONFIGS.keys())
