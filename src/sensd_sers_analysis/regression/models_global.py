"""
Paradigm 1: global concentration regressors (serotype-blind, pooled positives).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sensd_sers_analysis.config import (
    REGRESSION_HYPERPARAMETER_TUNING,
    REGRESSION_RANDOM_STATE,
    REGRESSION_RF_N_ESTIMATORS,
    REGRESSION_RF_SEARCH_MAX_DEPTH,
    REGRESSION_RF_SEARCH_MIN_SAMPLES_LEAF,
    REGRESSION_RF_SEARCH_N_ESTIMATORS,
    REGRESSION_SVR_SEARCH_C,
    REGRESSION_SVR_SEARCH_EPSILON,
    REGRESSION_SVR_SEARCH_GAMMA,
    REGRESSION_TUNING_GROUP_KFOLD_SPLITS,
    REGRESSION_TUNING_MIN_TRAIN_SAMPLES,
    REGRESSION_TUNING_RANDOM_SEARCH_ITER,
)
from sensd_sers_analysis.regression.metrics import regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class SingleRegressorResult:
    """Held-out evaluation for one global regressor."""

    model_name: str
    model: object
    y_true: np.ndarray
    y_pred: np.ndarray
    rmse: float
    mae: float
    r2: float
    feature_names: list[str]
    scaler: StandardScaler
    best_params: Optional[dict[str, Any]] = None


def _effective_group_cv_splits(n_groups: int, requested: int) -> int:
    if n_groups < 2:
        return 0
    return max(2, min(int(requested), n_groups))


def _should_run_regression_search(n_train: int, n_groups: int) -> bool:
    if not REGRESSION_HYPERPARAMETER_TUNING:
        return False
    if n_train < REGRESSION_TUNING_MIN_TRAIN_SAMPLES:
        logger.info(
            "Regression hyperparameter search skipped: train size %d < %d",
            n_train,
            REGRESSION_TUNING_MIN_TRAIN_SAMPLES,
        )
        return False
    if _effective_group_cv_splits(n_groups, REGRESSION_TUNING_GROUP_KFOLD_SPLITS) == 0:
        logger.info(
            "Regression hyperparameter search skipped: need >=2 train sensor groups for GroupKFold."
        )
        return False
    return True


def _fit_random_forest_regressor(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int,
    *,
    run_search: bool,
) -> tuple[RandomForestRegressor, Optional[dict[str, Any]]]:
    if run_search:
        n_splits = _effective_group_cv_splits(
            int(np.unique(groups).size),
            REGRESSION_TUNING_GROUP_KFOLD_SPLITS,
        )
        cv = GroupKFold(n_splits=n_splits)
        base = RandomForestRegressor(random_state=random_state)
        param_distributions = {
            "n_estimators": list(REGRESSION_RF_SEARCH_N_ESTIMATORS),
            "max_depth": list(REGRESSION_RF_SEARCH_MAX_DEPTH),
            "min_samples_leaf": list(REGRESSION_RF_SEARCH_MIN_SAMPLES_LEAF),
        }
        search = RandomizedSearchCV(
            base,
            param_distributions=param_distributions,
            n_iter=min(REGRESSION_TUNING_RANDOM_SEARCH_ITER, 48),
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X, y, groups=groups)
        return search.best_estimator_, dict(search.best_params_)

    rf = RandomForestRegressor(
        random_state=random_state,
        n_estimators=REGRESSION_RF_N_ESTIMATORS,
    )
    rf.fit(X, y)
    return rf, None


def _fit_svr_regressor(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int,
    *,
    run_search: bool,
) -> tuple[SVR, Optional[dict[str, Any]]]:
    if run_search:
        n_splits = _effective_group_cv_splits(
            int(np.unique(groups).size),
            REGRESSION_TUNING_GROUP_KFOLD_SPLITS,
        )
        cv = GroupKFold(n_splits=n_splits)
        base = SVR(kernel="rbf")
        param_distributions = {
            "C": list(REGRESSION_SVR_SEARCH_C),
            "gamma": list(REGRESSION_SVR_SEARCH_GAMMA),
            "epsilon": list(REGRESSION_SVR_SEARCH_EPSILON),
        }
        search = RandomizedSearchCV(
            base,
            param_distributions=param_distributions,
            n_iter=min(REGRESSION_TUNING_RANDOM_SEARCH_ITER, 24),
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X, y, groups=groups)
        return search.best_estimator_, dict(search.best_params_)

    svr = SVR(kernel="rbf")
    svr.fit(X, y)
    return svr, None


def train_global_regressors(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    target_col: str = "log_concentration",
    group_col: str = "sensor_id",
    random_state: int = REGRESSION_RANDOM_STATE,
) -> tuple[SingleRegressorResult, SingleRegressorResult]:
    """
    Train serotype-blind RF and SVR regressors with one shared group-aware split.

    StandardScaler is fit on the training rows only. Hyperparameter search uses
    ``GroupKFold`` on training sensors when enabled.

    Parameters
    ----------
    df:
        Clean regression dataframe (reset index).
    feature_cols:
        Feature column names (must not include serotype one-hot for Paradigm 1).
    train_idx, test_idx:
        Positional indices from :func:`~sensd_sers_analysis.regression.splits.group_train_test_indices`.
    target_col:
        Regression target (default ``log_concentration``).
    group_col:
        Sensor column for grouped CV during tuning.
    random_state:
        RNG seed.

    Returns
    -------
    tuple[SingleRegressorResult, SingleRegressorResult]
        ``(rf_result, svr_result)``.
    """
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(f"No feature columns found. Needed: {feature_cols}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column {target_col!r}.")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r}.")

    X_all = df[available].fillna(0).to_numpy(dtype=np.float64, copy=False)
    y_all = df[target_col].to_numpy(dtype=np.float64, copy=False)
    groups_all = df[group_col].astype(str).to_numpy(dtype=object, copy=False)

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    groups_train = groups_all[train_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_groups_train = int(np.unique(groups_train).size)
    run_search = _should_run_regression_search(len(X_train_s), n_groups_train)

    rf, rf_params = _fit_random_forest_regressor(
        X_train_s, y_train, groups_train, random_state, run_search=run_search
    )
    y_pred_rf = rf.predict(X_test_s)
    rmse_rf, mae_rf, r2_rf = regression_metrics(y_test, y_pred_rf)

    svr, svr_params = _fit_svr_regressor(
        X_train_s, y_train, groups_train, random_state, run_search=run_search
    )
    y_pred_svm = svr.predict(X_test_s)
    rmse_sv, mae_sv, r2_sv = regression_metrics(y_test, y_pred_svm)

    rf_result = SingleRegressorResult(
        model_name="Random Forest",
        model=rf,
        y_true=y_test,
        y_pred=y_pred_rf,
        rmse=rmse_rf,
        mae=mae_rf,
        r2=r2_rf,
        feature_names=available,
        scaler=scaler,
        best_params=rf_params,
    )
    svm_result = SingleRegressorResult(
        model_name="SVM (RBF)",
        model=svr,
        y_true=y_test,
        y_pred=y_pred_svm,
        rmse=rmse_sv,
        mae=mae_sv,
        r2=r2_sv,
        feature_names=available,
        scaler=scaler,
        best_params=svr_params,
    )
    return rf_result, svm_result
