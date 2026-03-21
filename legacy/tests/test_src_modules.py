"""Quick smoke tests for data.io, processing.features, processing.filters, visualization."""

import numpy as np
import pandas as pd
import pytest

from sensd_sers_analysis.data import (
    RS_COL_PREFIX,
    count_unique_spectra,
    get_metadata_columns,
    get_raman_shift,
    get_signals_matrix,
    load_sers_data,
    load_sers_data_as_wide_and_tidy,
    wide_to_tidy,
)
from sensd_sers_analysis.processing import (
    extract_basic_features,
    filter_sers_data,
    get_feature_metadata_columns,
    get_filter_options,
    get_filterable_columns,
    get_plot_hue_columns,
    pick_preferred_column,
    trim_raman_shift,
)
from sensd_sers_analysis.utils import format_column_label


def _make_wide_df(
    n_samples: int = 3,
    n_wavenumbers: int = 5,
    rs_min: float = 400,
    rs_max: float = 500,
) -> pd.DataFrame:
    """Create synthetic wide-format DataFrame with rs_* columns."""
    rs = np.linspace(rs_min, rs_max, n_wavenumbers)
    meta = pd.DataFrame(
        {
            "sensor_id": [f"s{i}" for i in range(n_samples)],
            "serotype": ["ST", "SE", "ST"][:n_samples],
            "concentration": [[10.0, 100.0]] * n_samples,
            "filename": [f"f{i}.xlsx" for i in range(n_samples)],
            "signal_index": list(range(n_samples)),
        }
    )
    signals = np.random.randn(n_samples, n_wavenumbers).cumsum(axis=1) + 100
    rs_cols = {f"{RS_COL_PREFIX}{r:.2f}": signals[:, i] for i, r in enumerate(rs)}
    return pd.concat([meta, pd.DataFrame(rs_cols)], axis=1)


def _make_tidy_df(n_spectra: int = 4, n_points: int = 10) -> pd.DataFrame:
    """Create synthetic tidy DataFrame for plotting."""
    n_rows = n_spectra * n_points
    return pd.DataFrame(
        {
            "raman_shift": np.tile(np.linspace(400, 500, n_points), n_spectra),
            "intensity": np.random.randn(n_rows).cumsum() + 100,
            "filename": np.repeat([f"f{i}.xlsx" for i in range(n_spectra)], n_points),
            "signal_index": np.repeat(range(n_spectra), n_points),
            "serotype": np.repeat(["ST", "SE"] * ((n_spectra // 2) + 1), n_points)[:n_rows],
        }
    )


# ---------------------------------------------------------------------------
# data.io
# ---------------------------------------------------------------------------


class TestDataIO:
    def test_get_signals_matrix(self):
        wide = _make_wide_df(2, 4)
        mat = get_signals_matrix(wide)
        assert mat.shape == (2, 4)
        assert mat.dtype in (np.float32, np.float64)

    def test_get_raman_shift(self):
        wide = _make_wide_df(2, 4)
        rs = get_raman_shift(wide)
        assert len(rs) == 4
        assert np.all(rs >= 400) and np.all(rs <= 500)

    def test_get_metadata_columns(self):
        wide = _make_wide_df(2)
        meta = get_metadata_columns(wide)
        assert "sensor_id" in meta.columns
        assert "serotype" in meta.columns
        assert len(meta) == 2

    def test_wide_to_tidy(self):
        wide = _make_wide_df(2, 3)
        tidy = wide_to_tidy(wide)
        assert "raman_shift" in tidy.columns
        assert "intensity" in tidy.columns
        assert len(tidy) == 2 * 3

    def test_load_sers_data_empty_path(self):
        df = load_sers_data("/nonexistent/path/xyz")
        assert df.empty

    def test_load_sers_data_as_wide_and_tidy_empty(self):
        wide, tidy = load_sers_data_as_wide_and_tidy([])
        assert wide.empty and tidy.empty

    def test_count_unique_spectra(self):
        tidy = _make_tidy_df(4)
        n = count_unique_spectra(tidy)
        assert n == 4

    def test_count_unique_spectra_empty(self):
        assert count_unique_spectra(pd.DataFrame()) == 0

    def test_load_sers_data_from_example(self):
        """Load from example_data if available."""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        dec = root / "example_data" / "SERS Data 7 (Mar 2025)" / "December Signals"
        if not dec.exists():
            pytest.skip("Example data not found")
        # December Signals uses embedded format; exclude _metadata.xlsx via default pattern
        df = load_sers_data(dec)
        if df.empty:
            pytest.skip("No valid signal files in December Signals")
        assert "serotype" in df.columns
        assert any(c.startswith(RS_COL_PREFIX) for c in df.columns)


# ---------------------------------------------------------------------------
# processing.features
# ---------------------------------------------------------------------------


class TestProcessingFeatures:
    def test_extract_basic_features(self):
        wide = _make_wide_df(3, 10)
        feat = extract_basic_features(wide)
        assert "max_intensity" in feat.columns
        assert "mean_intensity" in feat.columns
        assert "integral_area" in feat.columns
        assert len(feat) == 3

    def test_extract_basic_features_empty(self):
        wide = pd.DataFrame()
        feat = extract_basic_features(wide)
        assert feat.empty


# ---------------------------------------------------------------------------
# processing.alignment (trim_raman_shift)
# ---------------------------------------------------------------------------


class TestTrimRamanShift:
    def test_trim_both_bounds(self):
        """Trim to 400–500 when data spans 300–600."""
        wide = _make_wide_df(3, 31, rs_min=300, rs_max=600)
        rs_before = get_raman_shift(wide)
        assert rs_before.min() <= 350 and rs_before.max() >= 550
        trimmed = trim_raman_shift(wide, min_shift=400, max_shift=500)
        rs_after = get_raman_shift(trimmed)
        assert np.all(rs_after >= 400) and np.all(rs_after <= 500)

    def test_trim_min_only(self):
        wide = _make_wide_df(2, 21, rs_min=300, rs_max=500)
        trimmed = trim_raman_shift(wide, min_shift=400, max_shift=None)
        rs = get_raman_shift(trimmed)
        assert np.all(rs >= 400)

    def test_trim_max_only(self):
        wide = _make_wide_df(2, 21, rs_min=400, rs_max=600)
        trimmed = trim_raman_shift(wide, min_shift=None, max_shift=500)
        rs = get_raman_shift(trimmed)
        assert np.all(rs <= 500)

    def test_trim_both_none_unchanged(self):
        wide = _make_wide_df(2, 5)
        trimmed = trim_raman_shift(wide, min_shift=None, max_shift=None)
        assert len(trimmed.columns) == len(wide.columns)

    def test_trim_metadata_preserved(self):
        wide = _make_wide_df(2, 10)
        trimmed = trim_raman_shift(wide, min_shift=420, max_shift=480)
        for col in ["sensor_id", "serotype", "concentration", "filename", "signal_index"]:
            assert col in trimmed.columns

    def test_trim_empty_no_op(self):
        wide = pd.DataFrame()
        trimmed = trim_raman_shift(wide, min_shift=400, max_shift=1800)
        assert trimmed.empty


# ---------------------------------------------------------------------------
# processing.filters
# ---------------------------------------------------------------------------


class TestProcessingFilters:
    def test_get_filterable_columns(self):
        df = pd.DataFrame(
            {"serotype": ["ST"], "concentration_group": ["10 CFU"], "raman_shift": [400]}
        )
        cols = get_filterable_columns(df)
        assert "serotype" in cols
        assert "concentration_group" in cols
        assert "raman_shift" not in cols

    def test_get_filter_options_cascading(self):
        df = pd.DataFrame(
            {
                "serotype": ["ST", "SE", "ST"],
                "concentration_group": ["1 CFU", "10 CFU", "1 CFU"],
            }
        )
        cols = ["serotype", "concentration_group"]
        opts = get_filter_options(df, cols, {})
        assert "serotype" in opts
        assert "concentration_group" in opts
        assert "ST" in opts["serotype"]
        assert "SE" in opts["serotype"]

    def test_filter_sers_data_include(self):
        df = pd.DataFrame({"serotype": ["ST", "SE", "ST"]})
        filtered = filter_sers_data(df, {"serotype": (["ST"], False)})
        assert len(filtered) == 2

    def test_filter_sers_data_exclude(self):
        df = pd.DataFrame({"serotype": ["ST", "SE", "ST"]})
        filtered = filter_sers_data(df, {"serotype": (["SE"], True)})
        assert len(filtered) == 2

    def test_get_plot_hue_columns(self):
        df = pd.DataFrame(
            {"serotype": ["ST"], "concentration_group": ["10 CFU"], "raman_shift": [400]}
        )
        cols = get_plot_hue_columns(df)
        assert "serotype" in cols
        assert "concentration_group" in cols
        assert "raman_shift" not in cols

    def test_pick_preferred_column(self):
        assert pick_preferred_column(["a", "b"], ("b", "a")) == "b"
        assert pick_preferred_column(["a"], ("b", "a")) == "a"
        assert pick_preferred_column([], ("a",)) is None

    def test_get_feature_metadata_columns(self):
        df = pd.DataFrame(
            {
                "max_intensity": [100],
                "serotype": ["ST"],
                "concentration_group": ["10 CFU"],
            }
        )
        meta = get_feature_metadata_columns(df)
        assert "max_intensity" not in meta
        assert "serotype" in meta


# ---------------------------------------------------------------------------
# utils.labels
# ---------------------------------------------------------------------------


class TestUtilsLabels:
    def test_format_column_label(self):
        assert format_column_label("concentration_group") == "Concentration Group"
        assert format_column_label("sensor_id") == "Sensor Id"


# ---------------------------------------------------------------------------
# visualization.plots (smoke only)
# ---------------------------------------------------------------------------


class TestVisualizationPlots:
    def test_plot_spectra_smoke(self):
        import matplotlib

        matplotlib.use("Agg")
        from sensd_sers_analysis.visualization import plot_spectra

        tidy = _make_tidy_df(4)
        fig = plot_spectra(tidy, hue="serotype", figsize=(6, 4))
        assert fig is not None


# ---------------------------------------------------------------------------
# visualization.stats (smoke only)
# ---------------------------------------------------------------------------


class TestVisualizationStats:
    def test_plot_feature_distribution_smoke(self):
        import matplotlib

        matplotlib.use("Agg")
        from sensd_sers_analysis.visualization import plot_feature_distribution

        df = pd.DataFrame(
            {
                "max_intensity": [100, 120, 90],
                "serotype": ["ST", "SE", "ST"],
                "concentration_group": ["1 CFU", "10 CFU", "1 CFU"],
            }
        )
        fig = plot_feature_distribution(df, "max_intensity", x="serotype")
        assert fig is not None
