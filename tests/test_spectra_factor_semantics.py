"""Plot factors use experimental meaning rather than numeric storage dtype."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensd_sers_analysis.processing.filters import get_plot_hue_columns
from sensd_sers_analysis.visualization.plots import plot_spectra


def test_target_is_discrete_and_legend_matches_lines():
    """Target levels have one legend entry each and no continuous colorbar."""
    frame = pd.DataFrame(
        {
            "filename": ["a", "a", "b", "b"],
            "signal_index": [0] * 4,
            "raman_shift": [500, 600, 500, 600],
            "intensity": [1.0, 2.0, 3.0, 4.0],
            "target_concentration": [1.0, 1.0, 1000.0, 1000.0],
            "concentration": [1.0, 1.0, 950.0, 950.0],
            "target_concentration_group": ["1 CFU"] * 4,
            "concentration_group": ["1 CFU"] * 4,
        }
    )
    options = get_plot_hue_columns(frame)
    assert "target_concentration" in options
    assert "target_concentration_group" not in options
    assert "concentration_group" not in options
    figure = plot_spectra(frame, hue="target_concentration")
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    handles, labels = axis.get_legend_handles_labels()
    assert labels == ["1", "1000"]
    lines = [line for line in axis.lines if len(line.get_xdata())]
    for line, handle in zip(lines, handles, strict=True):
        np.testing.assert_allclose(line.get_color(), handle.get_color())
    plt.close(figure)
    figure = plot_spectra(frame, hue="concentration")
    assert len(figure.axes) == 2
    plt.close(figure)
