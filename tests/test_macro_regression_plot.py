"""Pooled regression provenance and reuse of fitted plotting artifacts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sensd_sers_analysis.assessment import compute_macro_batch_regression
from sensd_sers_analysis.visualization import assessment_plots


@pytest.fixture
def parallel_responses() -> pd.DataFrame:
    """Two sensors with slope two and intercepts one and two."""
    return pd.DataFrame(
        {
            "sensor_id": ["A"] * 3 + ["B"] * 3,
            "serotype": ["ST"] * 6,
            "log_concentration": [0.0, 1.0, 2.0] * 2,
            "integral_area": [1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
        }
    )


def test_macro_fit_preserves_sensor_point_provenance(parallel_responses):
    """A pooled fit has the analytic mean intercept and labels for every point."""
    result = compute_macro_batch_regression(parallel_responses, "ST", "integral_area", {"A", "B"})
    assert result is not None
    assert result.slope == pytest.approx(2.0)
    assert result.intercept == pytest.approx(1.5)
    assert result.clean_batch_rmse == pytest.approx(0.5)
    assert result.n_sensors == 2
    assert result.n_points == 6
    assert result.n_macro_outliers == 0
    for sensor, intercept in (("A", 1.0), ("B", 2.0)):
        mask = result.pooled_sensor_ids == sensor
        assert mask.sum() == 3
        np.testing.assert_allclose(result.y_pooled[mask], 2.0 * result.x_pooled[mask] + intercept)


def test_macro_plot_uses_supplied_points_without_refitting(parallel_responses, monkeypatch):
    """A supplied fit owns both the points and the line drawn in the plot."""
    result = compute_macro_batch_regression(parallel_responses, "ST", "integral_area", {"A", "B"})
    assert result is not None

    def unexpected_fit(*args, **kwargs):
        """Fail if plotting attempts to fit a supplied artifact again."""
        raise AssertionError("Precomputed macro regression must not be refitted.")

    monkeypatch.setattr(assessment_plots, "compute_macro_batch_regression", unexpected_fit)
    changed_frame = parallel_responses.assign(integral_area=1000.0)
    figure, returned_result = assessment_plots.plot_macro_batch_regression(
        changed_frame, "ST", "integral_area", {"A", "B"}, macro_result=result
    )
    try:
        assert returned_result is result
        points = np.concatenate(
            [collection.get_offsets() for collection in figure.axes[0].collections]
        )
        points = points[np.lexsort((points[:, 1], points[:, 0]))]
        expected = np.array([[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]])
        np.testing.assert_allclose(points, expected)
        clean_line = next(
            line for line in figure.axes[0].lines if line.get_label().startswith("Clean (")
        )
        np.testing.assert_allclose(clean_line.get_ydata(), 1.5 + 2.0 * clean_line.get_xdata())
    finally:
        plt.close(figure)
