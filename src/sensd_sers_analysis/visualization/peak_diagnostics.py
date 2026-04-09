"""
Peak-diagnostics plotting helpers for Streamlit verification views.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from sensd_sers_analysis.application.peak_diagnostics_service import SignalVerificationArtifact
from sensd_sers_analysis.processing import PeakWindowInfo


def plot_peak_anchor_summary(
    raman_x,
    mean_spectrum,
    peak_infos: list[PeakWindowInfo],
    serotype: str,
    *,
    figsize: tuple[float, float] = (14, 4),
    legend_fontsize: int = 8,
    grid_alpha: float = 0.3,
    span_alpha: float = 0.15,
    axvline_alpha: float = 0.8,
):
    """
    Plot the mean spectrum and search windows for one serotype.

    Parameters
    ----------
    raman_x:
        Raman-shift grid for the selected serotype.
    mean_spectrum:
        Mean spectrum used to derive the anchors.
    peak_infos:
        Peak-window metadata for the selected serotype.
    serotype:
        Serotype label used in the figure title.
    figsize:
        Matplotlib figure size in inches.
    legend_fontsize:
        Legend font size.
    grid_alpha:
        Grid transparency.
    span_alpha:
        Window shading transparency.
    axvline_alpha:
        Anchor-line transparency.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered peak-anchor summary figure.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        raman_x,
        mean_spectrum,
        color="C0",
        linewidth=1.5,
        label=f"Mean ({serotype}, high conc)",
    )
    for i, info in enumerate(peak_infos):
        ax.axvline(
            info.center,
            color=f"C{(i % 9) + 1}",
            linestyle="--",
            linewidth=1.2,
            alpha=axvline_alpha,
            label=f"{info.peak_name} @ {info.center:.0f} cm⁻¹",
        )
        ax.axvspan(
            info.window_min,
            info.window_max,
            alpha=span_alpha,
            color=f"C{(i % 9) + 1}",
        )
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{serotype} | Dashed = Voted Centers, Shaded = Search Windows")
    ax.legend(loc="upper right", fontsize=legend_fontsize, ncol=2)
    ax.grid(True, alpha=grid_alpha)
    fig.tight_layout()
    return fig


def plot_signal_level_peak_verification(
    artifact: SignalVerificationArtifact,
    *,
    figsize: tuple[float, float] = (14, 5),
    legend_fontsize: int = 8,
    grid_alpha: float = 0.3,
    span_alpha: float = 0.12,
):
    """
    Plot one selected spectrum with shaded peak windows and detected peaks.

    Parameters
    ----------
    artifact:
        Signal-level verification payload.
    figsize:
        Matplotlib figure size in inches.
    legend_fontsize:
        Legend font size.
    grid_alpha:
        Grid transparency.
    span_alpha:
        Window shading transparency.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered signal-level verification figure.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(artifact.x_plot, artifact.y_plot, color="C0", linewidth=1.2, label="Raw spectrum")

    for i, info in enumerate(artifact.row_peak_infos):
        ax.axvspan(
            info.window_min,
            info.window_max,
            alpha=span_alpha,
            color=f"C{(i % 9) + 1}",
        )
    for peak_index, peak_x, peak_y in artifact.detected_points:
        ax.scatter(
            peak_x,
            peak_y,
            marker="*",
            s=200,
            color="green",
            edgecolors="darkgreen",
            linewidths=1.5,
            zorder=5,
            label="Detected" if peak_index == 0 else None,
        )

    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(
        "Signal: "
        f"{artifact.selected_sensor} @ {artifact.selected_concentration} "
        f"({artifact.row_serotype}) | Green ★ = detected peaks"
    )
    ax.legend(loc="upper right", fontsize=legend_fontsize)
    ax.grid(True, alpha=grid_alpha)
    fig.tight_layout()
    return fig
