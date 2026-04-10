"""Unit tests for concentration regression splits, metrics, and training smoke."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sensd_sers_analysis.classification.models import train_classifiers_on_arrays
from sensd_sers_analysis.regression.metrics import regression_metrics
from sensd_sers_analysis.regression.models_global import train_global_regressors
from sensd_sers_analysis.regression.models_mtl import train_mtl_regressor
from sensd_sers_analysis.regression.models_two_stage import train_two_stage_regressors
from sensd_sers_analysis.regression.splits import (
    assert_disjoint_group_split,
    group_train_test_indices,
)


def test_group_train_test_indices_no_overlap():
    """Sensors must not appear in both train and test."""
    n = 40
    rng = np.random.default_rng(0)
    sensors = np.array([f"S{i // 8}" for i in range(n)], dtype=object)
    df = pd.DataFrame(
        {
            "sensor_id": sensors,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "log_concentration": rng.uniform(0.5, 3.0, size=n),
            "target": np.where(rng.random(n) > 0.45, "SE", "ST"),
        }
    )
    groups = df["sensor_id"].astype(str).to_numpy(dtype=object, copy=False)
    train_idx, test_idx = group_train_test_indices(
        groups,
        test_size=0.25,
        random_state=42,
    )
    assert_disjoint_group_split(df, train_idx, test_idx, group_col="sensor_id")
    assert len(set(train_idx) & set(test_idx)) == 0


def test_regression_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rmse, mae, r2 = regression_metrics(y, y)
    assert math.isclose(rmse, 0.0, abs_tol=1e-9)
    assert math.isclose(mae, 0.0, abs_tol=1e-9)
    assert math.isclose(r2, 1.0, abs_tol=1e-9)


def test_train_classifiers_on_arrays_binary():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(30, 3))
    X_test = rng.normal(size=(12, 3))
    y_train = np.array(["ST"] * 15 + ["SE"] * 15, dtype=object)
    y_test = np.array(["ST"] * 6 + ["SE"] * 6, dtype=object)
    rf, svm = train_classifiers_on_arrays(
        X_train,
        X_test,
        y_train,
        y_test,
        ["a", "b", "c"],
        random_state=0,
    )
    assert rf.accuracy >= 0.0
    assert svm.accuracy >= 0.0


def _synth_regression_df(*, n_per: int = 12) -> pd.DataFrame:
    """Minimal clean-like frame for sklearn / torch smoke tests."""
    rows = []
    for s in ("T1", "T2", "T3", "E1", "E2", "E3"):
        for _ in range(n_per):
            rows.append(s)
    n = len(rows)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "sensor_id": rows,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
            "log_concentration": rng.uniform(0.5, 3.0, size=n),
            "target": np.array(
                ["ST" if str(x).startswith("T") else "SE" for x in rows],
                dtype=object,
            ),
        }
    )


def test_train_global_regressors_smoke():
    df = _synth_regression_df()
    groups = df["sensor_id"].astype(str).to_numpy(dtype=object, copy=False)
    train_idx, test_idx = group_train_test_indices(
        groups,
        test_size=0.33,
        random_state=0,
    )
    rf, svm = train_global_regressors(
        df.reset_index(drop=True),
        ["f1", "f2", "f3"],
        train_idx,
        test_idx,
        random_state=0,
    )
    assert rf.y_true.shape == rf.y_pred.shape
    assert svm.y_true.shape == svm.y_pred.shape


def test_train_two_stage_regressors_smoke():
    # Four sensor groups: two ST and two SE, split so train/test each contain both classes.
    rng = np.random.default_rng(3)
    rows = []
    for sid, lab in (("TA", "ST"), ("TB", "ST"), ("EA", "SE"), ("EB", "SE")):
        for _ in range(24):
            rows.append(
                {
                    "sensor_id": sid,
                    "target": lab,
                    "f1": rng.normal(),
                    "f2": rng.normal(),
                    "f3": rng.normal(),
                    "log_concentration": float(rng.uniform(0.5, 3.0)),
                }
            )
    df = pd.DataFrame(rows).reset_index(drop=True)
    tr_sensors = {"TA", "EA"}
    mask_train = df["sensor_id"].isin(tr_sensors).to_numpy()
    train_idx = np.flatnonzero(mask_train)
    test_idx = np.flatnonzero(~mask_train)
    assert_disjoint_group_split(df, train_idx, test_idx, group_col="sensor_id")
    out = train_two_stage_regressors(
        df,
        ["f1", "f2", "f3"],
        train_idx,
        test_idx,
        random_state=0,
    )
    assert out.y_true_reg.shape == out.y_pred_routed.shape
    assert out.y_true_reg.shape == out.y_pred_oracle.shape


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_mtl_regressor_smoke():
    pytest.importorskip("torch")
    df = _synth_regression_df(n_per=20)
    groups = df["sensor_id"].astype(str).to_numpy(dtype=object, copy=False)
    train_idx, test_idx = group_train_test_indices(
        groups,
        test_size=0.33,
        random_state=2,
    )
    out = train_mtl_regressor(
        df.reset_index(drop=True),
        ["f1", "f2", "f3"],
        train_idx,
        test_idx,
        random_state=0,
    )
    assert out.y_true_reg.shape == out.y_pred_reg.shape
    assert 0.0 <= out.clf_accuracy <= 1.0
