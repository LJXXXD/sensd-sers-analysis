"""Tests for feature-analysis visualization helpers in ``visualization.stats``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from sensd_sers_analysis.visualization.stats import (
    compute_feature_log_concentration_correlations,
    plot_feature_correlation_heatmap,
    plot_feature_log_concentration_correlation_bars,
    plot_feature_log_concentration_scatter,
)


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Synthetic samples with correlated integral_area and log concentration."""
    rng = np.random.default_rng(0)
    n = 40
    log_c = np.linspace(2.0, 6.0, n) + rng.normal(0.0, 0.05, n)
    return pd.DataFrame(
        {
            "log_concentration": log_c,
            "integral_area": 5.0 * log_c + rng.normal(0.0, 1.0, n),
            "max_intensity": rng.normal(100.0, 10.0, n),
            "serotype": ["Typhimurium"] * (n // 2) + ["Enteritidis"] * (n - n // 2),
        }
    )


def test_compute_feature_log_concentration_correlations_orders_by_abs_r(
    feature_df: pd.DataFrame,
) -> None:
    out = compute_feature_log_concentration_correlations(
        feature_df,
        ["integral_area", "max_intensity"],
    )
    assert not out.empty
    assert list(out.columns) == ["feature", "pearson_r", "p_value", "n"]
    assert (out["n"] >= 3).all()
    abs_r = out["pearson_r"].abs().tolist()
    assert abs_r == sorted(abs_r, reverse=True)


def test_plot_feature_correlation_heatmap_returns_figure(feature_df: pd.DataFrame) -> None:
    fig = plot_feature_correlation_heatmap(
        feature_df,
        ["integral_area", "max_intensity"],
    )
    try:
        assert fig.axes[0].get_title()
    finally:
        plt.close(fig)


def test_correlation_heatmap_requires_two_columns(feature_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least two"):
        plot_feature_correlation_heatmap(feature_df, ["integral_area"])


def test_bar_and_scatter_plots(feature_df: pd.DataFrame) -> None:
    fig_b = plot_feature_log_concentration_correlation_bars(
        feature_df,
        ["integral_area", "max_intensity"],
    )
    plt.close(fig_b)
    fig_s = plot_feature_log_concentration_scatter(feature_df, "integral_area")
    plt.close(fig_s)
    fig_h = plot_feature_log_concentration_scatter(
        feature_df,
        "integral_area",
        hue_col="serotype",
    )
    plt.close(fig_h)
