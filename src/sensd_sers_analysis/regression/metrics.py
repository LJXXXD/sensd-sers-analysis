"""
Regression metrics for concentration prediction (log10 scale).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """
    Compute RMSE, MAE, and R² on the same scale as ``y_true`` / ``y_pred``.

    Parameters
    ----------
    y_true, y_pred:
        1-D arrays of equal length.

    Returns
    -------
    tuple[float, float, float]
        ``(rmse, mae, r2)``.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2
