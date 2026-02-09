"""
Shared Metrics Calculation Module.

Provides 5 core evaluation metrics:
- compute_js_divergence: Jensen-Shannon Divergence (Point evaluation)
- compute_temporal_spearman: Spearman's Rank Correlation Coefficient (Point evaluation)
- compute_rmse: Root Mean Square Error (Point evaluation)
- compute_mape: Mean Absolute Percentage Error (Distribution evaluation)
- calculate_ecdf: Empirical Cumulative Distribution Function value (Distribution evaluation - Coverage Rate)
"""

import numpy as np
from scipy.stats import spearmanr, entropy
from typing import Tuple, Optional


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Formula: mean(|y_true - y_pred|)

    Parameters:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Formula: mean(|y_true - y_pred| / |y_true|) * 100

    Parameters:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAPE value as a percentage (0-100+)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE).

    Formula: sqrt(mean((y_true - y_pred)^2))

    Parameters:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan

    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def compute_temporal_spearman(
    time_series_true: np.ndarray,
    time_series_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate Spearman's Rank Correlation Coefficient for time series.

    Parameters:
        time_series_true: Time-aggregated mean sequence (Human)
        time_series_pred: Time-aggregated mean sequence (LLM)

    Returns:
        (Correlation coefficient, p-value) tuple
    """
    time_series_true = np.asarray(time_series_true, dtype=float)
    time_series_pred = np.asarray(time_series_pred, dtype=float)

    valid_mask = ~(np.isnan(time_series_true) | np.isnan(time_series_pred))
    true_valid = time_series_true[valid_mask]
    pred_valid = time_series_pred[valid_mask]

    if len(true_valid) < 2:
        return np.nan, np.nan

    corr, p_value = spearmanr(true_valid, pred_valid)
    return corr, p_value


def compute_js_divergence(
    p_vals: np.ndarray,
    q_vals: np.ndarray,
    bins: int = 50,
    range_vals: Optional[Tuple[float, float]] = None
) -> float:
    """
    Calculate Jensen-Shannon Divergence between two distributions.

    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)

    Parameters:
        p_vals: Samples of distribution P (Human responses)
        q_vals: Samples of distribution Q (LLM responses)
        bins: Number of bins for histogram discretization
        range_vals: Optional (min, max) range

    Returns:
        JS Divergence value (0 = identical, higher means more different)
    """
    p_vals = np.asarray(p_vals, dtype=float)
    q_vals = np.asarray(q_vals, dtype=float)

    p_vals = p_vals[~np.isnan(p_vals)]
    q_vals = q_vals[~np.isnan(q_vals)]

    if len(p_vals) == 0 or len(q_vals) == 0:
        return np.nan

    if range_vals is None:
        min_val = min(p_vals.min(), q_vals.min())
        max_val = max(p_vals.max(), q_vals.max())
        range_vals = (min_val, max_val)

    p_hist, _ = np.histogram(p_vals, bins=bins, range=range_vals, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, range=range_vals, density=True)

    epsilon = 1e-10
    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon

    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    m_hist = 0.5 * (p_hist + q_hist)

    js_div = 0.5 * entropy(p_hist, m_hist) + 0.5 * entropy(q_hist, m_hist)

    return js_div


def calculate_ecdf(samples: list, truth: float) -> Optional[float]:
    """
    Calculate Empirical Cumulative Distribution Function (ECDF) value.

    Formula: F_hat(h) = (count_less + 0.5 * count_equal) / N

    Used for Coverage Rate calculation: If 0.05 <= ECDF <= 0.95, 
    the ground truth falls within the 90% confidence interval.

    Parameters:
        samples: List of samples predicted by LLM
        truth: Ground truth value

    Returns:
        ECDF value (0-1), or None on failure
    """
    if not samples or len(samples) == 0:
        return None

    try:
        samples = [float(x) for x in samples]
        truth = float(truth)
    except (ValueError, TypeError):
        return None

    count_less = sum(1 for s in samples if s < truth)
    count_equal = sum(1 for s in samples if s == truth)

    return (count_less + 0.5 * count_equal) / len(samples)
