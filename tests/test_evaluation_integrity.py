"""Independent regressions for selection, preprocessing and sensor boundaries."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

from sensd_sers_analysis.application.classification_service import (
    build_classification_clean_dataset,
)
from sensd_sers_analysis.application.regression_service import (
    build_concentration_regression_dataset,
)
from sensd_sers_analysis.classification.models import train_classifiers_on_arrays
from sensd_sers_analysis.modeling import feature_matrix, fit_model_pipeline, select_by_training_cv
from sensd_sers_analysis.processing import add_pca_features
from sensd_sers_analysis.regression import models_mtl
from sensd_sers_analysis.splits import assert_disjoint_group_split, group_train_test_indices


def test_selected_missing_features_fail_with_source_context_and_zero_is_measured():
    frame = pd.DataFrame(
        {"measured": [0.0, 2.0], "unused_peak": [np.nan, np.inf]}, index=["source-A", "source-B"]
    )
    np.testing.assert_array_equal(feature_matrix(frame, ["measured"]), [[0.0], [2.0]])
    frame.loc["source-B", "measured"] = np.nan
    with pytest.raises(ValueError, match="source-B.*measured"):
        feature_matrix(frame, ["measured"])
    with pytest.raises(ValueError, match="Missing selected"):
        feature_matrix(frame, ["absent"])


def test_scaling_is_fit_inside_each_cv_fold(monkeypatch):
    """Inner means 1 and 101 differ from all-training mean 51."""
    import sensd_sers_analysis.modeling as modeling

    real_search = modeling.RandomizedSearchCV
    monkeypatch.setattr(
        modeling, "RandomizedSearchCV", lambda *a, **kw: real_search(*a, **{**kw, "n_jobs": 1})
    )
    means = []
    fit = StandardScaler.fit

    def observe(self, X, y=None, **kwargs):
        result = fit(self, X, y, **kwargs)
        means.append(self.mean_.copy())
        return result

    monkeypatch.setattr(StandardScaler, "fit", observe)
    X = np.array([[0.0], [2.0], [100.0], [102.0]])
    model, _, score = fit_model_pipeline(
        DummyRegressor(),
        X,
        np.arange(4.0),
        cv=[(np.array([0, 1]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1]))],
        parameters={"strategy": ["mean"]},
        scoring="neg_root_mean_squared_error",
        n_iter=1,
        random_state=1,
    )
    np.testing.assert_array_equal(np.asarray(means).ravel(), [1.0, 101.0, 51.0])
    assert np.isfinite(score)
    np.testing.assert_array_equal(model.predict([[1000.0]]), [1.5])


def test_selection_ignores_opposite_test_rankings():
    rf = SimpleNamespace(cv_score=0.9, f1=0.1, rmse=10.0)
    svm = SimpleNamespace(cv_score=0.8, f1=1.0, rmse=0.0)
    assert select_by_training_cv(rf, svm) is rf
    rf.cv_score = svm.cv_score = None
    assert select_by_training_cv(rf, svm) is rf
    svm.cv_score = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_by_training_cv(rf, svm)


def test_failed_search_and_nonfinite_fold_score_abort(monkeypatch):
    import sensd_sers_analysis.modeling as modeling

    real_search = modeling.RandomizedSearchCV
    monkeypatch.setattr(
        modeling, "RandomizedSearchCV", lambda *a, **kw: real_search(*a, **{**kw, "n_jobs": 1})
    )
    X = np.arange(4.0).reshape(-1, 1)
    folds = [(np.array([0, 1]), np.array([2, 3]))]
    with pytest.raises(ValueError):
        fit_model_pipeline(
            DummyRegressor(),
            X,
            X.ravel(),
            cv=folds,
            parameters={"strategy": ["invalid"]},
            scoring="neg_root_mean_squared_error",
            n_iter=1,
            random_state=0,
        )
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="non-finite"):
        fit_model_pipeline(
            DummyRegressor(),
            X,
            X.ravel(),
            cv=folds,
            parameters={"strategy": ["mean"]},
            scoring=lambda *args: np.nan,
            n_iter=1,
            random_state=0,
        )


def test_model_eligibility_does_not_depend_on_response_qa_or_pca():
    frame = pd.DataFrame(
        {
            "sample_type": ["Bacteria sample", "Bacteria sample", "Rinsate control", "Unknown"],
            "serotype": ["ST", "SE", "ST", "ST"],
            "sensor_id": ["S1", "S2", "S1", "S3"],
            "concentration": [10.0, 100.0, 0.0, 1.0],
            "log_concentration": [1.0, 2.0, np.nan, 0.0],
            "integral_area": [np.nan, 1e9, 0.0, 2.0],
            "PC1": [np.nan] * 4,
        },
        index=[41, 7, 28, 16],
    )
    classified = build_classification_clean_dataset(frame)
    assert classified.index.tolist() == [41, 7, 28]
    assert classified.target.tolist() == ["ST", "SE", "Rinsate"]
    assert build_concentration_regression_dataset(frame).index.tolist() == [41, 7]
    altered = frame.assign(integral_area=-999.0, PC1=1000.0)
    assert build_classification_clean_dataset(altered).index.tolist() == classified.index.tolist()
    assert build_concentration_regression_dataset(altered).index.tolist() == [41, 7]


def test_exploratory_pca_never_imputes_incomplete_spectra():
    frame = pd.DataFrame(
        {"rs_500.00": [1.0, 3.0, np.nan], "rs_600.00": [2.0, 4.0, 9.0]}, index=[4, 1, 8]
    )
    result = add_pca_features(frame)
    assert result.index.tolist() == [4, 1, 8]
    assert result.loc[8].isna().all()
    assert np.isfinite(result.loc[[4, 1], ["PC1", "PC2"]]).all().all()
    assert add_pca_features(frame.iloc[[0]]).isna().all().all()
    constant = add_pca_features(frame.iloc[:2].replace({1.0: 3.0, 2.0: 4.0}))
    assert constant["PC1_var_ratio"].isna().all()


def test_group_contract_rejects_missing_ids_and_overlap():
    with pytest.raises(ValueError, match="non-missing"):
        group_train_test_indices(np.array(["A", None], dtype=object), test_size=0.5, random_state=0)
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_group_split(
            pd.DataFrame({"sensor_id": [" A", "A "]}), np.array([0]), np.array([1])
        )


def test_classifier_weights_do_not_depend_on_outer_test_values(monkeypatch):
    from sensd_sers_analysis.classification import models

    monkeypatch.setattr(models, "CLASSIFICATION_HYPERPARAMETER_TUNING", False)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["ST", "SE", "ST", "SE"])
    first = train_classifiers_on_arrays(
        X, np.array([[0.5], [2.5]]), y, np.array(["ST", "SE"]), ["f"]
    )
    second = train_classifiers_on_arrays(
        X, np.array([[1000.0], [-1000.0]]), y, np.array(["SE", "ST"]), ["f"]
    )
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.model.predict(X), b.model.predict(X))
        np.testing.assert_array_equal(a.model.named_steps["scale"].mean_, [1.5])


def test_mtl_nested_sensor_split_and_outer_test_invariance(monkeypatch):
    import torch

    monkeypatch.setattr(models_mtl, "REGRESSION_MTL_MAX_EPOCHS", 2)
    monkeypatch.setattr(models_mtl, "REGRESSION_MTL_HIDDEN_DIMS", (4,))
    frame = pd.DataFrame(
        {
            "sensor_id": np.repeat(["A", "B", "C", "D", "E", "F"], 2),
            "target": ["ST", "SE"] * 6,
            "f": np.arange(12.0),
            "log_concentration": np.arange(12.0) / 5,
        }
    )
    train, test = np.arange(10), np.arange(10, 12)
    a = models_mtl.train_mtl_regressor(frame, ["f"], train, test, random_state=3)
    inner, validation = a.early_stop_train_indices, a.early_stop_validation_indices
    assert set(frame.iloc[inner].sensor_id).isdisjoint(frame.iloc[validation].sensor_id)
    assert set(frame.iloc[inner].sensor_id).isdisjoint(frame.iloc[test].sensor_id)
    np.testing.assert_array_equal(a.scaler.mean_, frame.iloc[inner][["f"]].mean().to_numpy())
    altered = frame.copy()
    altered.loc[test, "f"] = [1000.0, -1000.0]
    altered.loc[test, "target"] = ["SE", "ST"]
    altered.loc[test, "log_concentration"] = [10.0, 20.0]
    b = models_mtl.train_mtl_regressor(altered, ["f"], train, test, random_state=3)
    assert a.train_loss_history == b.train_loss_history
    assert a.val_loss_history == b.val_loss_history
    for name, weights in a.model.state_dict().items():
        assert torch.equal(weights, b.model.state_dict()[name])
    altered.loc[test, "target"] = "UNSEEN"
    with pytest.raises(ValueError, match="Unknown serotype"):
        models_mtl.train_mtl_regressor(altered, ["f"], train, test, random_state=3)


def test_exploratory_peak_discovery_preserves_missing_measurements():
    from sensd_sers_analysis.processing.peak_features import (
        _compute_peak_windows_for_serotype,
        _find_peaks_on_spectrum,
    )
    from sensd_sers_analysis.assessment.degradation import add_sequence_column

    x = np.arange(15.0, dtype=float)
    observed = np.array([0.0, 0.0, 1.0, 4.0, 8.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    invalid = observed.copy()
    invalid[3] = np.nan
    assert not _find_peaks_on_spectrum(x, invalid, 1).size
    frame = pd.DataFrame(
        {
            "sample_type": ["Bacteria sample", "Bacteria sample"],
            "concentration_group": ["1000 CFU", "1000 CFU"],
        }
    )
    for i in range(len(x)):
        frame[f"rs_{i}"] = [observed[i], invalid[i]]
    _, mean = _compute_peak_windows_for_serotype(frame, x, 1, "concentration_group", 0.0)
    np.testing.assert_array_equal(mean, observed)
    _, unavailable = _compute_peak_windows_for_serotype(
        frame.iloc[[1]], x, 1, "concentration_group", 0.0
    )
    assert np.isnan(unavailable).all()
    sequence = add_sequence_column(pd.DataFrame({"signal_index": [0.5, np.nan, np.inf]}))[
        "sequence"
    ]
    assert sequence.iloc[0] == 0.5 and sequence.iloc[1:].isna().all()
