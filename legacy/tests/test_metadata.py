"""Tests for metadata preprocessing (log_concentration, concentration_group)."""

import numpy as np
import pandas as pd
import pytest

from sensd_sers_analysis.processing import (
    add_concentration_group,
    add_log_concentration,
    preprocess_metadata,
)


@pytest.fixture
def df_scalar_conc():
    """DataFrame with scalar concentration per row."""
    return pd.DataFrame(
        {
            "concentration": [0, 1, 10, 100, 1000, 5, 50, 500],
            "serotype": ["ST"] * 8,
        }
    )


@pytest.fixture
def df_list_conc():
    """DataFrame with concentration as list per row (wide format)."""
    return pd.DataFrame(
        {
            "concentration": [[0, 10], [1, 100], [10, 1000]],
            "signal_index": [0, 1, 0],
            "serotype": ["ST"] * 3,
        }
    )


class TestAddLogConcentration:
    def test_positive_concentrations(self, df_scalar_conc):
        out = add_log_concentration(df_scalar_conc)
        assert "log_concentration" in out.columns
        expected = [np.nan, 0, 1, 2, 3, np.log10(5), np.log10(50), np.log10(500)]
        np.testing.assert_allclose(
            out["log_concentration"].values,
            expected,
            rtol=1e-9,
            equal_nan=True,
        )

    def test_zero_concentration_is_nan(self):
        df = pd.DataFrame({"concentration": [0, 0.0]})
        out = add_log_concentration(df)
        assert out["log_concentration"].isna().all()

    def test_negative_concentration_is_nan(self):
        df = pd.DataFrame({"concentration": [-1, -0.1]})
        out = add_log_concentration(df)
        assert out["log_concentration"].isna().all()

    def test_list_concentration_uses_signal_index(self, df_list_conc):
        out = add_log_concentration(df_list_conc)
        # Row 0: signal_index 0 -> conc 0 -> nan
        # Row 1: signal_index 1 -> conc 100 -> log10(100)=2
        # Row 2: signal_index 0 -> conc 10 -> log10(10)=1
        assert np.isnan(out["log_concentration"].iloc[0])
        assert np.isclose(out["log_concentration"].iloc[1], 2.0)
        assert np.isclose(out["log_concentration"].iloc[2], 1.0)

    def test_no_concentration_column(self):
        df = pd.DataFrame({"serotype": ["ST"]})
        out = add_log_concentration(df)
        assert "log_concentration" not in out.columns


class TestAddConcentrationGroup:
    def test_zero_gets_0_cfu(self):
        df = pd.DataFrame({"concentration": [0, 0.0]})
        out = add_concentration_group(df)
        assert list(out["concentration_group"]) == ["0 CFU", "0 CFU"]

    def test_nearest_log_centers(self, df_scalar_conc):
        out = add_concentration_group(df_scalar_conc)
        # 0->0 CFU, 1->1 CFU, 10->10 CFU, 100->100 CFU, 1000->1000 CFU
        # 5: log10(5)~0.70, nearer to 1 than 0 -> 10 CFU
        # 50: log10(50)~1.70, nearer to 2 than 1 -> 100 CFU
        # 500: log10(500)~2.70, nearer to 3 than 2 -> 1000 CFU
        expected = [
            "0 CFU",
            "1 CFU",
            "10 CFU",
            "100 CFU",
            "1000 CFU",
            "10 CFU",  # 5
            "100 CFU",  # 50
            "1000 CFU",  # 500
        ]
        assert list(out["concentration_group"]) == expected

    def test_boundary_near_3_16_goes_to_1_or_10_cfu(self):
        """sqrt(10) ~ 3.16 is boundary between 1 CFU and 10 CFU in log space."""
        df = pd.DataFrame({"concentration": [3.0, 3.5, 4.0]})
        out = add_concentration_group(df)
        # 3.0: log10~0.48 < 0.5 -> 1 CFU; 3.5, 4.0: > 0.5 -> 10 CFU
        assert list(out["concentration_group"]) == ["1 CFU", "10 CFU", "10 CFU"]

    def test_boundary_near_31_6_goes_to_10_or_100_cfu(self):
        df = pd.DataFrame({"concentration": [30.0, 32.0, 35.0]})
        out = add_concentration_group(df)
        # 30: log10=1.48, nearest 1 -> 10 CFU
        # 32: log10=1.51, nearest 2 -> 100 CFU (boundary ~31.6)
        # 35: log10=1.54, nearest 2 -> 100 CFU
        assert out["concentration_group"].iloc[0] == "10 CFU"
        assert out["concentration_group"].iloc[1] == "100 CFU"
        assert out["concentration_group"].iloc[2] == "100 CFU"

    def test_list_concentration(self, df_list_conc):
        out = add_concentration_group(df_list_conc)
        # Row 0: conc 0 -> 0 CFU
        # Row 1: conc 100 -> 100 CFU
        # Row 2: conc 10 -> 10 CFU
        assert out["concentration_group"].iloc[0] == "0 CFU"
        assert out["concentration_group"].iloc[1] == "100 CFU"
        assert out["concentration_group"].iloc[2] == "10 CFU"

    def test_missing_concentration_gets_unknown(self):
        df = pd.DataFrame({"serotype": ["ST"]})
        out = add_concentration_group(df)
        assert list(out["concentration_group"]) == ["Unknown"]

    def test_nan_concentration_gets_unknown(self):
        df = pd.DataFrame({"concentration": [np.nan, 10], "serotype": ["ST"] * 2})
        out = add_concentration_group(df)
        assert out["concentration_group"].iloc[0] == "Unknown"
        assert out["concentration_group"].iloc[1] == "10 CFU"


class TestPreprocessMetadata:
    def test_adds_all_columns(self, df_scalar_conc):
        out = preprocess_metadata(df_scalar_conc)
        assert "log_concentration" in out.columns
        assert "concentration_group" in out.columns

    def test_normalizes_date(self):
        df = pd.DataFrame(
            {
                "concentration": [10],
                "date": ["2024-03-15 10:30:00"],
            }
        )
        out = preprocess_metadata(df)
        assert out["date"].iloc[0] == "2024-03-15"

    def test_date_coerces_invalid_to_empty(self):
        df = pd.DataFrame({"concentration": [10], "date": ["not-a-date"]})
        out = preprocess_metadata(df)
        assert out["date"].iloc[0] == ""
