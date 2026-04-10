"""
Plots for fixed-anchor (targeted) peak extraction verification.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_targeted_mean_spectrum_markers(
    raman_x: np.ndarray,
    mean_spectrum: np.ndarray,
    anchor_cm1: Sequence[float],
    detected_shifts: np.ndarray,
    detected_raw_heights: np.ndarray,
    serotype: str,
    *,
    figsize: tuple[float, float] = (14, 4),
    legend_fontsize: int = 8,
    grid_alpha: float = 0.3,
    axvline_alpha: float = 0.75,
):
    """
    Plot a serotype mean spectrum with target anchors and detected peak positions.

    Parameters
    ----------
    raman_x:
        Raman-shift axis (cm⁻¹).
    mean_spectrum:
        Mean intensity vector aligned with ``raman_x``.
    anchor_cm1:
        User target anchors (cm⁻¹).
    detected_shifts:
        Raman shifts of local maxima inside each anchor window (NaN allowed).
    detected_raw_heights:
        Raw spectrum intensity at each detected shift (NaN allowed).
    serotype:
        Title label for the serotype.
    figsize:
        Matplotlib figure size in inches.
    legend_fontsize:
        Legend font size.
    grid_alpha:
        Grid line alpha.
    axvline_alpha:
        Vertical guide line alpha.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered figure.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        raman_x,
        mean_spectrum,
        color="C0",
        linewidth=1.5,
        label=f"Mean ({serotype})",
    )
    for i, anchor in enumerate(anchor_cm1):
        color = f"C{(i % 9) + 1}"
        ax.axvline(
            float(anchor),
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=axvline_alpha,
            label=f"Target {i + 1}: {float(anchor):.1f} cm⁻¹",
        )
        if i < len(detected_shifts) and np.isfinite(detected_shifts[i]):
            ds = float(detected_shifts[i])
            yy = (
                float(detected_raw_heights[i])
                if i < len(detected_raw_heights) and np.isfinite(detected_raw_heights[i])
                else float("nan")
            )
            if np.isfinite(yy):
                ax.scatter(
                    ds,
                    yy,
                    marker="*",
                    s=200,
                    color="green",
                    edgecolors="darkgreen",
                    linewidths=1.2,
                    zorder=5,
                    label="Detected (mean)" if i == 0 else None,
                )
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(
        f"{serotype} | Dashed = target anchors (cm⁻¹), green ★ = local max in search window"
    )
    ax.legend(loc="upper right", fontsize=legend_fontsize, ncol=2)
    ax.grid(True, alpha=grid_alpha)
    fig.tight_layout()
    return fig


def plot_targeted_signal_verification(
    raman_x: np.ndarray,
    intensity: np.ndarray,
    anchor_cm1: Sequence[float],
    detected_shifts: np.ndarray,
    detected_heights: np.ndarray,
    *,
    title: str,
    figsize: tuple[float, float] = (14, 5),
    legend_fontsize: int = 8,
    grid_alpha: float = 0.3,
    axvline_alpha: float = 0.75,
):
    """
    Plot one spectrum with target anchors and per-window detections.

    Parameters
    ----------
    raman_x:
        Raman-shift axis (cm⁻¹).
    intensity:
        Intensity vector for one spectrum.
    anchor_cm1:
        Target anchors (cm⁻¹).
    detected_shifts:
        Raman shift of the maximum in each search window (baseline-adjusted
        search is not redrawn here; markers use raw spectrum height at the
        detected shift).
    detected_heights:
        Intensity (raw) at ``detected_shifts`` for marker placement.
    title:
        Figure title.
    figsize:
        Matplotlib figure size in inches.
    legend_fontsize:
        Legend font size.
    grid_alpha:
        Grid alpha.
    axvline_alpha:
        Vertical guide alpha.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered figure.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(raman_x, intensity, color="C0", linewidth=1.2, label="Spectrum")
    for i, anchor in enumerate(anchor_cm1):
        color = f"C{(i % 9) + 1}"
        ax.axvline(
            float(anchor),
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=axvline_alpha,
        )
    for i in range(len(anchor_cm1)):
        if i < len(detected_shifts) and np.isfinite(detected_shifts[i]):
            ds = float(detected_shifts[i])
            yy = float(detected_heights[i]) if i < len(detected_heights) else float("nan")
            if np.isfinite(ds) and np.isfinite(yy):
                ax.scatter(
                    ds,
                    yy,
                    marker="*",
                    s=200,
                    color="green",
                    edgecolors="darkgreen",
                    linewidths=1.2,
                    zorder=5,
                    label="Detected" if i == 0 else None,
                )
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=legend_fontsize)
    ax.grid(True, alpha=grid_alpha)
    fig.tight_layout()
    return fig
