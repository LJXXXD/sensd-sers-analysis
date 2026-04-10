"""
Group-based train/test splits for concentration regression (sensor holdout).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


def group_train_test_indices(
    groups: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split row indices so no ``sensor_id`` (group) appears in both train and test.

    Parameters
    ----------
    groups:
        Group label per row (e.g. ``sensor_id``), same length as the design matrix.
    test_size:
        Fraction of **groups** reserved for test (``GroupShuffleSplit`` semantics).
    random_state:
        RNG seed for the split.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(train_idx, test_idx)`` positional indices into the frame used to build
        ``groups``.

    Raises
    ------
    ValueError
        If there are fewer than two unique groups or the split is degenerate.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("No rows for group split.")
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Need at least two distinct sensor_id values for a group holdout split.")

    X_dummy = np.zeros((n, 1))
    y_dummy = np.zeros(n)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X_dummy, y_dummy, groups))

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("GroupShuffleSplit produced an empty train or test set.")

    logger.info(
        "Group split: %d train rows (%d sensors), %d test rows (%d sensors)",
        train_idx.size,
        np.unique(groups[train_idx]).size,
        test_idx.size,
        np.unique(groups[test_idx]).size,
    )
    return train_idx, test_idx


def assert_disjoint_group_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    group_col: str = "sensor_id",
) -> None:
    """
    Verify that train and test rows share no common group labels.

    Parameters
    ----------
    df:
        Dataframe indexed consistently with ``train_idx`` / ``test_idx``.
    train_idx, test_idx:
        Positional indices.
    group_col:
        Grouping column (default ``sensor_id``).

    Raises
    ------
    ValueError
        If any group appears in both splits.
    """
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r}.")
    g_train = set(df.iloc[train_idx][group_col].astype(str).unique())
    g_test = set(df.iloc[test_idx][group_col].astype(str).unique())
    overlap = g_train & g_test
    if overlap:
        raise ValueError(f"Group leakage: sensors in both splits: {sorted(overlap)[:10]}...")
