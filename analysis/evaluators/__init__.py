"""
Evaluator module.

Provides two evaluators:
- PointwiseEvaluator: Pointwise evaluation (JS Divergence, Spearman, RMSE)
- DistributionEvaluator: Distribution evaluation (MAE, Coverage Rate)
"""

from .base import BaseEvaluator
from .pointwise import PointwiseEvaluator
from .distribution import DistributionEvaluator

__all__ = [
    "BaseEvaluator",
    "PointwiseEvaluator",
    "DistributionEvaluator",
]
