"""Regressions for sample identity and honest validation failures."""

import numpy as np
import pandas as pd
import pytest

from sensd_sers_analysis.assessment import validation_metrics as validation
from sensd_sers_analysis.classification.data_prep import prepare_classification_dataset
from sensd_sers_analysis.regression.data_prep import prepare_concentration_regression_data


def _identity_frame():
    """Return uniquely indexed controls interleaved with positive samples."""
    return pd.DataFrame(
        {
            "sensor_id": ["S1", "S1", "S2", "S2"],
            "serotype": ["ST"] * 4,
            "test_id": ["T1"] * 4,
            "filename": ["first.xlsx", "first.xlsx", "second.xlsx", "second.xlsx"],
            "signal_index": [0, 1, 0, 1],
            "sample_type": ["Rinsate control", "Bacteria sample"] * 2,
            "concentration": [0.0, 10.0, 0.0, 100.0],
            "concentration_group": ["0 CFU", "10 CFU", "0 CFU", "100 CFU"],
            "target_concentration": [0.0, 10.0, 0.0, 100.0],
            "log_concentration": [np.nan, 1.0, np.nan, 2.0],
            "integral_area": [0.1, 1.0, 0.2, 2.0],
        },
        index=["row-z", "row-a", "row-y", "row-b"],
    )


def test_preparation_preserves_source_identity_and_order():
    source = _identity_frame()
    original = source.copy(deep=True)
    classification = prepare_classification_dataset(source, excluded_map={})
    regression = prepare_concentration_regression_data(source, excluded_map={})
    assert classification.index.tolist() == ["row-z", "row-a", "row-y", "row-b"]
    assert classification["target"].tolist() == ["Rinsate", "ST", "Rinsate", "ST"]
    assert regression.index.tolist() == ["row-a", "row-b"]
    assert regression["concentration"].tolist() == [10.0, 100.0]
    assert regression[["filename", "signal_index"]].values.tolist() == [
        ["first.xlsx", 1],
        ["second.xlsx", 1],
    ]
    pd.testing.assert_frame_equal(source, original)


def test_regression_preparation_uses_requested_class_column():
    source = _identity_frame()
    original = source.copy(deep=True)
    regression = prepare_concentration_regression_data(
        source, excluded_map={}, target_col="class_label"
    )
    assert regression["class_label"].tolist() == ["ST", "ST"]
    assert "target" not in regression.columns
    pd.testing.assert_frame_equal(source, original)


def test_custom_class_column_cannot_overwrite_measurements():
    source = _identity_frame()
    with pytest.raises(ValueError, match="already exists"):
        prepare_concentration_regression_data(source, excluded_map={}, target_col="concentration")
    assert source["concentration"].tolist() == [0.0, 10.0, 0.0, 100.0]


def test_regression_preparation_requires_finite_positive_targets(monkeypatch):
    prepared = pd.DataFrame(
        {
            "target": ["ST"] * 8 + ["Rinsate", "ST", "ST"],
            "log_concentration": [1.0, np.inf, -np.inf, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0],
            "concentration": [10.0, 10.0, 10.0, 10.0, np.inf, -np.inf, 0.0, -1.0, 10.0, 0.1, 1.0],
        },
        index=list("abcdefghijk"),
    )
    monkeypatch.setattr(
        "sensd_sers_analysis.regression.data_prep.prepare_classification_dataset",
        lambda *args, **kwargs: prepared.copy(),
    )
    actual = prepare_concentration_regression_data(pd.DataFrame())
    assert actual.index.tolist() == ["a", "j", "k"]
    assert actual["concentration"].tolist() == [10.0, 0.1, 1.0]


def _fixed_split(groups, **kwargs):
    return [(np.array([0, 1]), np.array([2, 3]))]


def test_validation_predictions_map_to_original_spectra(monkeypatch):
    source = _identity_frame()
    monkeypatch.setattr(validation, "iter_group_train_test_indices", _fixed_split)
    monkeypatch.setattr(
        validation,
        "_fit_global_classifier_predict_all",
        lambda *args, **kwargs: np.array(["ST", "ST", "Rinsate", "ST"]),
    )
    # Two training positive rows permit the regressor; predictions identify the spectrum.
    source.loc["row-z", "sample_type"] = "Bacteria sample"
    source.loc["row-z", "concentration"] = 1.0
    source.loc["row-z", "log_concentration"] = 0.0
    classification = prepare_classification_dataset(source, excluded_map={})
    regression = prepare_concentration_regression_data(source, excluded_map={}).iloc[::-1]
    monkeypatch.setattr(
        validation,
        "_fit_global_regressor_predict_all",
        lambda work, *args, **kwargs: work["log_concentration"].to_numpy() + 0.25,
    )
    artifacts = validation.build_validation_tables(
        classification, regression, feature_cols=["integral_area"]
    )
    fold = artifacts.predictions.folds[0]
    np.testing.assert_allclose(fold.pred_log_conc, [0.25, 1.25, np.nan, 2.25], equal_nan=True)
    np.testing.assert_array_equal(fold.eval_mask, [False, False, True, True])


def test_validation_regressor_failure_is_explicit(monkeypatch):
    source = _identity_frame()
    source["sample_type"] = "Bacteria sample"
    source["concentration"] = [1.0, 10.0, 1.0, 100.0]
    source["log_concentration"] = [0.0, 1.0, 0.0, 2.0]
    classification = prepare_classification_dataset(source, excluded_map={})
    regression = prepare_concentration_regression_data(source, excluded_map={})
    monkeypatch.setattr(validation, "iter_group_train_test_indices", _fixed_split)
    monkeypatch.setattr(
        validation,
        "_fit_global_classifier_predict_all",
        lambda *args, **kwargs: np.array(["ST"] * 4),
    )

    def fail(*args, **kwargs):
        raise ValueError("invalid regression features")

    monkeypatch.setattr(validation, "_fit_global_regressor_predict_all", fail)
    with pytest.raises(ValueError, match="Validation regressor fold failed"):
        validation.fit_validation_predictions(classification, regression, ["integral_area"])


@pytest.mark.parametrize("bad_index", [["row-a", "row-a"], ["unknown"]])
def test_validation_rejects_ambiguous_source_identity(bad_index):
    classification = prepare_classification_dataset(_identity_frame(), excluded_map={})
    regression = pd.DataFrame(index=bad_index)
    with pytest.raises(ValueError, match="indices"):
        validation.fit_validation_predictions(classification, regression, ["integral_area"])


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_validation_classifier_failure_never_becomes_true_label_predictions(
    monkeypatch, fail_on_call
):
    classification = prepare_classification_dataset(_identity_frame(), excluded_map={})
    monkeypatch.setattr(
        validation,
        "iter_group_train_test_indices",
        lambda *args, **kwargs: _fixed_split(None) * 2,
    )
    calls = 0

    def predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_on_call:
            raise ValueError("invalid feature matrix")
        return np.array(["ST", "ST", "ST", "ST"])

    monkeypatch.setattr(validation, "_fit_global_classifier_predict_all", predict)
    with pytest.raises(ValueError, match="Validation classifier fold failed.*invalid feature"):
        validation.fit_validation_predictions(classification, pd.DataFrame(), ["integral_area"])


def test_unavailable_holdout_does_not_train_a_fallback_model(monkeypatch):
    classification = prepare_classification_dataset(_identity_frame(), excluded_map={})
    classification["sensor_id"] = "S1"
    monkeypatch.setattr(
        validation,
        "_fit_global_classifier_predict_all",
        lambda *args, **kwargs: pytest.fail("No held-out sensors; no model fit is needed."),
    )
    artifacts = validation.build_validation_tables(
        classification, pd.DataFrame(), feature_cols=["integral_area"]
    )
    assert not artifacts.predictions.sensor_holdout_available
    assert artifacts.predictions.folds == ()
    assert artifacts.predictions.n_splits == 0
    assert artifacts.concentration_repeatability["Eval N"].eq(0).all()
    assert artifacts.concentration_repeatability["Accuracy (% Correct)"].eq("").all()
    assert artifacts.quantification["Meet Target?"].eq("N/A").all()
