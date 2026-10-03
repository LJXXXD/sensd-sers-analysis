"""
Paradigm 2: serotype classification then serotype-specific regression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sensd_sers_analysis.modeling import feature_matrix, select_by_training_cv
from sensd_sers_analysis.splits import assert_disjoint_group_split

from sensd_sers_analysis.classification.models import (
    ClassificationResult,
    train_classifiers_on_arrays,
)
from sensd_sers_analysis.regression.metrics import regression_metrics
from sensd_sers_analysis.regression.models_global import (
    _fit_random_forest_regressor,
    _should_run_regression_search,
)

logger = logging.getLogger(__name__)


@dataclass
class SerotypeRegressorBundle:
    """One serotype-specific RF pipeline, including training-fold scaling."""

    serotype: str
    model: Any
    scaler: StandardScaler | None = None
    best_params: Optional[dict[str, Any]] = None


@dataclass
class TwoStageRegressionOutputs:
    """
    Training outputs for the two-stage pipeline (test-set evaluation).
    """

    stage1_rf: ClassificationResult
    stage1_svm: ClassificationResult
    stage1_best: ClassificationResult
    regressor_bundles: dict[str, SerotypeRegressorBundle]
    y_true_reg: np.ndarray
    y_pred_routed: np.ndarray
    y_pred_oracle: np.ndarray
    routing_accuracy: float
    rmse_routed: float
    mae_routed: float
    r2_routed: float
    rmse_oracle: float
    mae_oracle: float
    r2_oracle: float
    y_true_cls: np.ndarray
    serotype_true_test: np.ndarray


def _fit_per_serotype_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    serotype: str,
    random_state: int,
) -> SerotypeRegressorBundle:
    n_groups = int(np.unique(groups_train).size) if len(groups_train) else 0
    run_search = _should_run_regression_search(len(X_train), n_groups)
    rf, params, _ = _fit_random_forest_regressor(
        X_train, y_train, groups_train, random_state, run_search=run_search
    )
    return SerotypeRegressorBundle(
        serotype=serotype,
        model=rf,
        scaler=None,
        best_params=params,
    )


def _predict_routed_row(
    x_row: np.ndarray,
    predicted_serotype: str,
    bundles: dict[str, SerotypeRegressorBundle],
) -> float:
    if predicted_serotype not in bundles:
        raise ValueError(
            f"No stage-2 regressor for predicted serotype {predicted_serotype!r}. "
            f"Available: {sorted(bundles)!r}."
        )
    b = bundles[predicted_serotype]
    return float(b.model.predict(x_row.reshape(1, -1))[0])


def train_two_stage_regressors(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    target_col: str = "log_concentration",
    class_col: str = "target",
    group_col: str = "sensor_id",
    random_state: int = 42,
) -> TwoStageRegressionOutputs:
    """
    Stage 1: RF + SVM multi-class serotype discrimination on the training sensors.

    Stage 2: one Random Forest regressor per serotype observed in the training
    split. Test predictions use training-CV stage-1 selection, with RF as the fixed reference when CV is infeasible.
    ``oracle`` predictions route using the true serotype (diagnostic true-serotype routing).
    """
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(f"No feature columns found. Needed: {feature_cols}")

    X_all = feature_matrix(df, feature_cols)
    assert_disjoint_group_split(df, train_idx, test_idx, group_col=group_col)
    y_reg_all = df[target_col].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(y_reg_all).all():
        raise ValueError("Regression targets must be finite log concentrations.")
    y_cls_all = df[class_col].astype(str).to_numpy(dtype=object, copy=False)
    groups_all = df[group_col].astype(str).str.strip().to_numpy(dtype=object, copy=False)

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_reg_train, y_reg_test = y_reg_all[train_idx], y_reg_all[test_idx]
    y_cls_train, y_cls_test = y_cls_all[train_idx], y_cls_all[test_idx]
    groups_train = groups_all[train_idx]

    min_cls = 2
    labels_union = sorted(
        set(np.unique(y_cls_train)).union(np.unique(y_cls_test)),
        key=str,
    )
    for label in labels_union:
        n_tr = int((y_cls_train == label).sum())
        n_te = int((y_cls_test == label).sum())
        if n_tr < min_cls or n_te < min_cls:
            raise ValueError(
                f"Need at least {min_cls} train and {min_cls} test rows for serotype "
                f"{label!r} (got train={n_tr}, test={n_te}) for two-stage stage 1."
            )

    train_serotypes = sorted(set(np.unique(y_cls_train).tolist()), key=str)
    test_serotypes = set(np.unique(y_cls_test).tolist())
    orphan = test_serotypes - set(train_serotypes)
    if orphan:
        raise ValueError(
            "Held-out split has serotype(s) with no training rows; cannot fit stage-2 "
            f"regressors for: {sorted(orphan)!r}. Add sensors per serotype or adjust "
            "the group holdout."
        )

    stage1_rf, stage1_svm = train_classifiers_on_arrays(
        X_train,
        X_test,
        y_cls_train,
        y_cls_test,
        available,
        random_state=random_state,
        groups_train=groups_train,
    )
    stage1_best = select_by_training_cv(stage1_rf, stage1_svm)
    y_cls_pred = stage1_best.y_pred

    regressor_bundles: dict[str, SerotypeRegressorBundle] = {}
    for sero in train_serotypes:
        mask = y_cls_train == sero
        if mask.sum() == 0:
            continue
        regressor_bundles[str(sero)] = _fit_per_serotype_regressor(
            X_train[mask],
            y_reg_train[mask],
            groups_train[mask],
            str(sero),
            random_state,
        )

    if not regressor_bundles:
        raise ValueError("No serotype-specific training rows for stage-2 regressors.")

    y_pred_routed = np.empty(len(X_test), dtype=np.float64)
    y_pred_oracle = np.empty(len(X_test), dtype=np.float64)
    for i in range(len(X_test)):
        y_pred_routed[i] = _predict_routed_row(X_test[i], str(y_cls_pred[i]), regressor_bundles)
        y_pred_oracle[i] = _predict_routed_row(X_test[i], str(y_cls_test[i]), regressor_bundles)

    routing_accuracy = float(np.mean(y_cls_pred == y_cls_test))

    rmse_r, mae_r, r2_r = regression_metrics(y_reg_test, y_pred_routed)
    rmse_o, mae_o, r2_o = regression_metrics(y_reg_test, y_pred_oracle)

    logger.info(
        "Two-stage: routing accuracy=%.3f, RMSE routed=%.4f, RMSE oracle=%.4f",
        routing_accuracy,
        rmse_r,
        rmse_o,
    )

    return TwoStageRegressionOutputs(
        stage1_rf=stage1_rf,
        stage1_svm=stage1_svm,
        stage1_best=stage1_best,
        regressor_bundles=regressor_bundles,
        y_true_reg=y_reg_test,
        y_pred_routed=y_pred_routed,
        y_pred_oracle=y_pred_oracle,
        routing_accuracy=routing_accuracy,
        rmse_routed=rmse_r,
        mae_routed=mae_r,
        r2_routed=r2_r,
        rmse_oracle=rmse_o,
        mae_oracle=mae_o,
        r2_oracle=r2_o,
        y_true_cls=y_cls_test,
        serotype_true_test=y_cls_test,
    )
