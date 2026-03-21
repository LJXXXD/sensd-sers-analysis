"""Tests for model-based sensor consistency (concentration regression)."""

import numpy as np
import pandas as pd
import pytest

from sensd_sers_analysis.assessment import (
    fit_concentration_regression,
    get_global_model_consistency,
    get_zero_cfu_baseline,
)
from sensd_sers_analysis.visualization import plot_multi_sensor_regression


@pytest.fixture
def df_with_log_conc():
    """DataFrame with log_concentration, feature, and 0 CFU group."""
    return pd.DataFrame(
        {
            "log_concentration": [0.0, 1.0, 2.0, 3.0, np.nan],
            "max_intensity": [10.0, 20.0, 30.0, 40.0, 5.0],
            "concentration_group": ["1 CFU", "10 CFU", "100 CFU", "1000 CFU", "0 CFU"],
        }
    )


class TestFitConcentrationRegression:
    def test_perfect_fit(self, df_with_log_conc):
        """Perfect linear relationship yields R²=1, RMSE≈0."""
        result = fit_concentration_regression(df_with_log_conc, "max_intensity")
        assert result is not None
        assert result.r2 == pytest.approx(1.0, rel=1e-6)
        assert result.rmse == pytest.approx(0.0, abs=1e-10)
        assert result.n_samples == 4  # 0 CFU excluded
        assert result.slope == pytest.approx(10.0)
        assert result.intercept == pytest.approx(10.0)

    def test_insufficient_data_returns_none(self):
        df = pd.DataFrame({"log_concentration": [1.0], "max_intensity": [10.0]})
        assert fit_concentration_regression(df, "max_intensity") is None

    def test_all_nan_returns_none(self):
        df = pd.DataFrame({"log_concentration": [np.nan, np.nan], "max_intensity": [10.0, 20.0]})
        assert fit_concentration_regression(df, "max_intensity") is None

    def test_missing_columns_returns_none(self):
        df = pd.DataFrame({"max_intensity": [10, 20]})
        assert fit_concentration_regression(df, "max_intensity") is None


class TestGetZeroCfuBaseline:
    def test_uses_concentration_group(self, df_with_log_conc):
        baseline = get_zero_cfu_baseline(df_with_log_conc, "max_intensity")
        assert baseline == pytest.approx(5.0)  # Only 0 CFU row has max_intensity=5

    def test_no_zero_cfu_returns_none(self):
        df = pd.DataFrame(
            {
                "max_intensity": [10, 20, 30],
                "concentration_group": ["1 CFU", "10 CFU", "100 CFU"],
            }
        )
        assert get_zero_cfu_baseline(df, "max_intensity") is None

    def test_fallback_to_concentration_column(self):
        df = pd.DataFrame({"concentration": [0, 0, 10], "max_intensity": [5.0, 7.0, 20.0]})
        baseline = get_zero_cfu_baseline(df, "max_intensity")
        assert baseline == pytest.approx(6.0)  # mean of 5 and 7

    def test_missing_feature_returns_none(self, df_with_log_conc):
        assert get_zero_cfu_baseline(df_with_log_conc, "nonexistent") is None


class TestGetGlobalModelConsistency:
    def test_returns_table_with_expected_columns(self):
        df = pd.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s2", "s2"],
                "serotype": ["ST", "ST", "ST", "ST"],
                "log_concentration": [0, 1, 0, 1],
                "max_intensity": [10, 20, 12, 22],
            }
        )
        tbl = get_global_model_consistency(df)
        assert list(tbl.columns) == [
            "sensor_id",
            "serotype",
            "feature",
            "n_points",
            "R_squared",
            "RMSE",
        ]
        assert len(tbl) == 2  # s1 and s2, each with max_intensity
        assert set(tbl["sensor_id"]) == {"s1", "s2"}

    def test_empty_df_returns_empty_table(self):
        df = pd.DataFrame(columns=["sensor_id", "serotype", "log_concentration"])
        tbl = get_global_model_consistency(df)
        assert tbl.empty
        assert "sensor_id" in tbl.columns and "RMSE" in tbl.columns

    def test_missing_required_columns_returns_empty(self):
        df = pd.DataFrame({"max_intensity": [10, 20]})
        tbl = get_global_model_consistency(df)
        assert tbl.empty


class TestPlotMultiSensorRegression:
    def test_plot_smoke(self):
        df = pd.DataFrame(
            {
                "sensor_id": ["s1", "s1", "s2", "s2"],
                "serotype": ["ST", "ST", "ST", "ST"],
                "log_concentration": [0, 1, 0, 1],
                "max_intensity": [10, 20, 12, 22],
            }
        )
        fig = plot_multi_sensor_regression(df, "ST", "max_intensity")
        assert fig is not None

    def test_raises_on_empty_filtered_data(self):
        df = pd.DataFrame(
            {
                "sensor_id": ["s1"],
                "serotype": ["ST"],
                "log_concentration": [np.nan],  # 0 CFU - excluded
                "max_intensity": [5],
            }
        )
        with pytest.raises(ValueError, match="No valid data"):
            plot_multi_sensor_regression(df, "ST", "max_intensity")
