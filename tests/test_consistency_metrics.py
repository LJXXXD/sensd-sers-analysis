"""Unit tests for the unified consistency core (signal + concentration CV)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sensd_sers_analysis.assessment.consistency import (
    coefficient_of_variation,
    compute_consistency_metrics,
)


def _grouped_df() -> pd.DataFrame:
    """One sensor/serotype/target group with a stable signal but spread CFU."""
    return pd.DataFrame(
        {
            "sensor_id": ["S1", "S1", "S1", "S1"],
            "serotype": ["ST", "ST", "ST", "ST"],
            "target_concentration_group": ["1000 CFU"] * 4,
            "integral_area": [10.0, 10.5, 9.8, 10.1],
            "concentration": [850.0, 1200.0, 900.0, 1100.0],
        }
    )


def test_coefficient_of_variation_matches_definition() -> None:
    """CV equals population sigma over absolute mean."""
    values = np.array([10.0, 12.0, 14.0])
    expected = float(np.std(values) / abs(np.mean(values)))
    assert math.isclose(coefficient_of_variation(values), expected, rel_tol=1e-9)


def test_coefficient_of_variation_zero_mean_is_nan() -> None:
    """A zero mean yields NaN rather than dividing by zero."""
    assert math.isnan(coefficient_of_variation(np.array([-1.0, 1.0])))


def test_coefficient_of_variation_single_value_is_nan() -> None:
    """A single measurement has no spread; CV is NaN, not a flattering 0."""
    assert math.isnan(coefficient_of_variation(np.array([10.0])))
    assert math.isnan(coefficient_of_variation(np.array([10.0, np.nan])))


def test_consistency_single_replicate_group_reports_nan_variability() -> None:
    """A group with one row reports NaN std/CV instead of a misleading 0."""
    df = pd.DataFrame(
        {
            "sensor_id": ["S1"],
            "serotype": ["ST"],
            "target_concentration_group": ["1000 CFU"],
            "integral_area": [10.0],
            "concentration": [1000.0],
        }
    )
    metrics = compute_consistency_metrics(
        df,
        "integral_area",
        group_cols=["sensor_id", "serotype", "target_concentration_group"],
    )
    row = metrics.iloc[0]
    assert row["n_total"] == 1
    assert math.isnan(row["cv_raw"])
    assert math.isnan(row["std_raw"])
    assert math.isnan(row["cv_filtered"])
    assert math.isnan(row["conc_cv_raw"])
    # The single measured value itself is still reported (it is not misleading).
    assert math.isclose(row["mean_raw"], 10.0, rel_tol=1e-9)


def test_consistency_reports_signal_and_concentration_cv() -> None:
    """The core returns both the signal CV and the actual-concentration CV."""
    metrics = compute_consistency_metrics(
        _grouped_df(),
        "integral_area",
        group_cols=["sensor_id", "serotype", "target_concentration_group"],
    )
    assert len(metrics) == 1
    row = metrics.iloc[0]
    for col in ("cv_raw", "conc_cv_raw", "conc_mean", "conc_cv_filtered"):
        assert col in metrics.columns

    expected_signal = coefficient_of_variation([10.0, 10.5, 9.8, 10.1])
    expected_conc = coefficient_of_variation([850.0, 1200.0, 900.0, 1100.0])
    assert math.isclose(row["cv_raw"], expected_signal, rel_tol=1e-9)
    assert math.isclose(row["conc_cv_raw"], expected_conc, rel_tol=1e-9)
    # In this fixture the signal is far more stable than the sample CFU spread.
    assert row["cv_raw"] < row["conc_cv_raw"]
    assert math.isclose(row["conc_mean"], 1012.5, rel_tol=1e-9)


def test_consistency_concentration_cv_nan_without_column() -> None:
    """Concentration CV is NaN when no concentration column is present."""
    df = _grouped_df().drop(columns=["concentration"])
    metrics = compute_consistency_metrics(
        df,
        "integral_area",
        group_cols=["sensor_id"],
    )
    assert math.isnan(metrics.iloc[0]["conc_cv_raw"])
