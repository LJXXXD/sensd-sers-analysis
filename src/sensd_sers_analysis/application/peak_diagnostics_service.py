"""
Application-layer helpers for peak-diagnostics orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sensd_sers_analysis.application.contracts import PeakArtifacts
from sensd_sers_analysis.data import get_raman_shift, get_signals_matrix
from sensd_sers_analysis.processing import PeakWindowInfo
from sensd_sers_analysis.utils import order_concentration_labels


@dataclass(slots=True)
class PeakAnchorOverview:
    """
    Peak-anchor overview for one serotype.

    Parameters
    ----------
    serotype:
        Serotype represented by the overview.
    peak_infos:
        Peak metadata used to render the diagnostics.
    mean_spectrum:
        Mean spectrum used for anchor discovery.
    """

    serotype: str
    peak_infos: list[PeakWindowInfo]
    mean_spectrum: np.ndarray


@dataclass(slots=True)
class PeakDiagnosticContext:
    """
    Shared context for the peak-diagnostics tab.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe.
    wide_filtered:
        Wide dataframe aligned to `filtered_features`.
    peak_artifacts:
        Shared peak-extraction artifacts.
    sensor_col:
        Column used for sensor selection.
    concentration_col:
        Column used for concentration selection.
    serotype_col:
        Column used for serotype selection.
    """

    filtered_features: pd.DataFrame
    wide_filtered: pd.DataFrame
    peak_artifacts: PeakArtifacts
    sensor_col: str | None
    concentration_col: str | None
    serotype_col: str | None


@dataclass(slots=True)
class PeakSignalOptions:
    """
    UI options for selecting a signal in the diagnostics tab.

    Parameters
    ----------
    sensor_options:
        Available sensor identifiers for the selected serotype.
    concentration_options:
        Available concentration groups for the selected serotype.
    signal_labels:
        Display labels for matching signals.
    row_indices:
        Feature-dataframe indices associated with each signal label.
    """

    sensor_options: tuple[str, ...]
    concentration_options: tuple[str, ...]
    signal_labels: tuple[str, ...]
    row_indices: tuple[int, ...]


@dataclass(slots=True)
class SignalVerificationArtifact:
    """
    Signal-level peak-verification payload.

    Parameters
    ----------
    row_idx:
        Selected dataframe index.
    row_serotype:
        Serotype used for peak-window lookup.
    selected_sensor:
        Selected sensor identifier.
    selected_concentration:
        Selected concentration group.
    row_peak_infos:
        Peak windows used for the selected signal.
    x_plot:
        Raman-shift values used for plotting.
    y_plot:
        Intensity values used for plotting.
    detected_points:
        Detected peak markers as `(peak_index, x, y)` tuples.
    """

    row_idx: int
    row_serotype: str
    selected_sensor: str
    selected_concentration: str
    row_peak_infos: list[PeakWindowInfo]
    x_plot: np.ndarray
    y_plot: np.ndarray
    detected_points: list[tuple[int, float, float]]


def build_peak_diagnostic_context(
    filtered_features: pd.DataFrame,
    wide_df: pd.DataFrame,
    peak_artifacts: PeakArtifacts,
) -> PeakDiagnosticContext | None:
    """
    Build the shared context used by the peak-diagnostics tab.

    Parameters
    ----------
    filtered_features:
        Filtered feature dataframe.
    wide_df:
        Wide dataframe from the main application pipeline.
    peak_artifacts:
        Shared peak-extraction artifacts.

    Returns
    -------
    PeakDiagnosticContext | None
        Shared context, or None when peak artifacts are unavailable.

    Raises
    ------
    KeyError
        If `filtered_features` is not index-aligned with `wide_df`.
    """

    if peak_artifacts.is_empty:
        return None

    if filtered_features.empty:
        wide_filtered = wide_df
    else:
        missing_indices = filtered_features.index.difference(wide_df.index)
        if not missing_indices.empty:
            raise KeyError("filtered_features index is not aligned with the wide dataframe")
        wide_filtered = wide_df.loc[filtered_features.index]

    sensor_col = "sensor_id" if "sensor_id" in filtered_features.columns else None
    concentration_col = (
        "concentration_group" if "concentration_group" in filtered_features.columns else None
    )
    serotype_col = "serotype" if "serotype" in filtered_features.columns else None
    return PeakDiagnosticContext(
        filtered_features=filtered_features,
        wide_filtered=wide_filtered,
        peak_artifacts=peak_artifacts,
        sensor_col=sensor_col,
        concentration_col=concentration_col,
        serotype_col=serotype_col,
    )


def build_peak_anchor_overviews(peak_artifacts: PeakArtifacts) -> list[PeakAnchorOverview]:
    """
    Build per-serotype peak-anchor overviews.

    Parameters
    ----------
    peak_artifacts:
        Shared peak artifacts produced by dynamic peak extraction.

    Returns
    -------
    list[PeakAnchorOverview]
        Sorted per-serotype peak overviews.
    """

    overviews: list[PeakAnchorOverview] = []
    for serotype in sorted(peak_artifacts.peak_infos_by_serotype):
        peak_infos = peak_artifacts.peak_infos_by_serotype.get(serotype, [])
        mean_spectrum = peak_artifacts.mean_spec_by_serotype.get(
            serotype, np.array([], dtype=float)
        )
        if peak_infos and mean_spectrum.size > 0:
            overviews.append(
                PeakAnchorOverview(
                    serotype=serotype,
                    peak_infos=peak_infos,
                    mean_spectrum=mean_spectrum,
                )
            )
    return overviews


def build_peak_anchor_table(peak_infos: list[PeakWindowInfo]) -> pd.DataFrame:
    """
    Build the tabular anchor summary shown under each mean-spectrum figure.

    Parameters
    ----------
    peak_infos:
        Peak-window metadata for one serotype.

    Returns
    -------
    pd.DataFrame
        Diagnostic table for the selected peak set.
    """

    return pd.DataFrame(
        [
            {
                "Peak Name": info.peak_name,
                "Center (cm⁻¹)": f"{info.center:.1f}",
                "Window Range": f"[{info.window_min:.1f}, {info.window_max:.1f}]",
                "Detection Success Rate (%)": f"{info.success_rate * 100:.1f}",
            }
            for info in peak_infos
        ]
    )


def build_signal_selection_options(
    context: PeakDiagnosticContext,
    selected_serotype: str,
) -> PeakSignalOptions:
    """
    Build signal-selection options for the chosen serotype.

    Parameters
    ----------
    context:
        Shared diagnostics context.
    selected_serotype:
        Serotype selected in the UI.

    Returns
    -------
    PeakSignalOptions
        Available sensors, concentrations, signal labels, and row indices.
    """

    if context.serotype_col:
        df_ver = context.filtered_features[
            context.filtered_features[context.serotype_col].astype(str) == selected_serotype
        ]
    else:
        df_ver = context.filtered_features

    if context.sensor_col is None or context.concentration_col is None or df_ver.empty:
        return PeakSignalOptions(
            sensor_options=(),
            concentration_options=(),
            signal_labels=(),
            row_indices=(),
        )

    sensor_options = tuple(sorted(df_ver[context.sensor_col].dropna().astype(str).tolist()))
    concentration_values = df_ver[context.concentration_col].dropna().astype(str).tolist()
    concentration_options = tuple(order_concentration_labels(concentration_values))
    return PeakSignalOptions(
        sensor_options=tuple(dict.fromkeys(sensor_options)),
        concentration_options=tuple(dict.fromkeys(concentration_options)),
        signal_labels=(),
        row_indices=(),
    )


def build_matching_signal_options(
    context: PeakDiagnosticContext,
    *,
    selected_serotype: str,
    selected_sensor: str,
    selected_concentration: str,
) -> PeakSignalOptions:
    """
    Build signal labels and row indices for a selected sensor/concentration pair.

    Parameters
    ----------
    context:
        Shared diagnostics context.
    selected_serotype:
        Serotype selected in the UI.
    selected_sensor:
        Sensor selected in the UI.
    selected_concentration:
        Concentration selected in the UI.

    Returns
    -------
    PeakSignalOptions
        Signal labels and row indices for the current selection.
    """

    if context.sensor_col is None or context.concentration_col is None:
        return PeakSignalOptions((), (), (), ())

    if context.serotype_col:
        df_ver = context.filtered_features[
            context.filtered_features[context.serotype_col].astype(str) == selected_serotype
        ]
    else:
        df_ver = context.filtered_features

    matches = df_ver[
        (df_ver[context.sensor_col].astype(str) == selected_sensor)
        & (df_ver[context.concentration_col].astype(str) == selected_concentration)
    ]
    signal_labels: list[str] = []
    row_indices: list[int] = []
    for row_idx in matches.index:
        filename = (
            context.wide_filtered.loc[row_idx, "filename"]
            if "filename" in context.wide_filtered.columns
            else str(row_idx)
        )
        signal_labels.append(f"{filename} (idx {row_idx})")
        row_indices.append(int(row_idx))

    return PeakSignalOptions(
        sensor_options=(),
        concentration_options=(),
        signal_labels=tuple(signal_labels),
        row_indices=tuple(row_indices),
    )


def build_signal_verification_artifact(
    context: PeakDiagnosticContext,
    *,
    selected_serotype: str,
    selected_sensor: str,
    selected_concentration: str,
    signal_position: int = 0,
) -> SignalVerificationArtifact | None:
    """
    Build the selected signal and detected-peak markers for verification.

    Parameters
    ----------
    context:
        Shared diagnostics context.
    selected_serotype:
        Serotype selected in the UI.
    selected_sensor:
        Sensor selected in the UI.
    selected_concentration:
        Concentration selected in the UI.
    signal_position:
        Positional index into the matching signal list.

    Returns
    -------
    SignalVerificationArtifact | None
        Signal-level plotting payload, or None when no signal matches.
    """

    matching_options = build_matching_signal_options(
        context,
        selected_serotype=selected_serotype,
        selected_sensor=selected_sensor,
        selected_concentration=selected_concentration,
    )
    if not matching_options.row_indices:
        return None

    row_idx = matching_options.row_indices[
        min(signal_position, len(matching_options.row_indices) - 1)
    ]
    if context.serotype_col and context.serotype_col in context.wide_filtered.columns:
        row_serotype = str(context.wide_filtered.loc[row_idx, context.serotype_col])
    else:
        row_serotype = selected_serotype

    row_peak_infos = context.peak_artifacts.peak_infos_by_serotype.get(row_serotype)
    if not row_peak_infos and context.peak_artifacts.default_serotype is not None:
        row_peak_infos = context.peak_artifacts.peak_infos_by_serotype.get(
            context.peak_artifacts.default_serotype
        )
    if not row_peak_infos:
        row_peak_infos = next(iter(context.peak_artifacts.peak_infos_by_serotype.values()), [])

    spec_row = context.wide_filtered.loc[[row_idx]]
    signal_matrix = get_signals_matrix(spec_row)
    raman_shift = get_raman_shift(spec_row)
    y_spec = signal_matrix[0]
    x_spec = np.asarray(raman_shift, dtype=float)

    valid = np.isfinite(y_spec.astype(float))
    x_plot = x_spec[valid]
    y_plot = np.asarray(y_spec, dtype=float)[valid]
    sort_idx = np.argsort(x_plot)
    x_plot = x_plot[sort_idx]
    y_plot = y_plot[sort_idx]

    detected_points: list[tuple[int, float, float]] = []
    for peak_index, info in enumerate(row_peak_infos):
        peak_column = info.peak_name
        if peak_column not in context.filtered_features.columns:
            continue
        peak_value = context.filtered_features.loc[row_idx, peak_column]
        if pd.notna(peak_value) and np.isfinite(peak_value):
            mask = (x_spec >= info.window_min) & (x_spec <= info.window_max)
            window_y = np.where(mask, y_spec.astype(float), np.nan)
            if mask.any() and np.any(np.isfinite(window_y)):
                local_idx = int(np.nanargmax(window_y))
                detected_points.append(
                    (peak_index, float(x_spec[local_idx]), float(y_spec[local_idx]))
                )

    return SignalVerificationArtifact(
        row_idx=row_idx,
        row_serotype=row_serotype,
        selected_sensor=selected_sensor,
        selected_concentration=selected_concentration,
        row_peak_infos=list(row_peak_infos),
        x_plot=x_plot,
        y_plot=y_plot,
        detected_points=detected_points,
    )
