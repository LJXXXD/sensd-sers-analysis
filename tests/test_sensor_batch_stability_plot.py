"""Tests for the honest between-sensor batch stability plot."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from sensd_sers_analysis.assessment import (  # noqa: E402
    compute_batch_variance,
    get_consistency_summary_table,
)
from sensd_sers_analysis.visualization import (  # noqa: E402
    plot_sensor_batch_stability,
    plot_signal_vs_concentration_cv,
)


@pytest.fixture
def batch_df() -> pd.DataFrame:
    """Three sensors: two with 2 reads (one deviating high), one with a single read."""
    return pd.DataFrame(
        {
            "sensor_id": ["A", "A", "B", "B", "C"],
            "integral_area": [10.0, 10.4, 20.0, 21.0, 10.2],
        }
    )


def test_plot_sensor_batch_stability_returns_figure(batch_df: pd.DataFrame) -> None:
    """The plot renders one axes without fabricating boxplot quartiles."""
    batch_table = compute_batch_variance(batch_df, "integral_area", group_cols=None)
    fig = plot_sensor_batch_stability(
        batch_df,
        batch_table,
        "integral_area",
        z_threshold=1.0,
    )
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # One x tick per sensor.
    assert len(ax.get_xticks()) == 3


def test_plot_sensor_batch_stability_requires_rows(batch_df: pd.DataFrame) -> None:
    """An empty batch table raises rather than drawing an empty figure."""
    empty_table = compute_batch_variance(batch_df, "integral_area", group_cols=None).iloc[0:0]
    with pytest.raises(ValueError):
        plot_sensor_batch_stability(batch_df, empty_table, "integral_area")


def test_plot_signal_vs_concentration_cv_returns_figure() -> None:
    """Signal-vs-concentration CV bars render one axes over the long CV table."""
    df = pd.DataFrame(
        {
            "sensor_id": ["A", "A", "B", "B"],
            "serotype": ["ST"] * 4,
            "target_concentration_group": ["1000 CFU"] * 4,
            "integral_area": [10.0, 10.4, 20.0, 21.0],
            "max_intensity": [4.0, 4.3, 8.0, 8.4],
            "concentration": [980.0, 1100.0, 900.0, 1200.0],
        }
    )
    table = get_consistency_summary_table(
        df,
        feature_cols=["integral_area", "max_intensity"],
        group_cols=["sensor_id", "serotype", "target_concentration_group"],
    )
    fig = plot_signal_vs_concentration_cv(table)
    assert len(fig.axes) == 1
    # Two features + concentration = three bar containers.
    assert len(fig.axes[0].containers) == 3
