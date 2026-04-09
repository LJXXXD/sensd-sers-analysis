"""
Application-layer dataset pipeline for the Streamlit app.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from sensd_sers_analysis.application.contracts import (
    DerivedDataBundle,
    LoadedDataBundle,
    PeakArtifacts,
)
from sensd_sers_analysis.data import load_sers_data_as_wide_and_tidy, wide_to_tidy
from sensd_sers_analysis.processing import (
    extract_basic_features,
    extract_dynamic_peak_features,
    get_peak_height_columns,
    preprocess_metadata,
    trim_raman_shift,
)


def load_uploaded_bundle(
    files_data: tuple[tuple[str, bytes], ...],
) -> LoadedDataBundle:
    """
    Parse uploaded Excel files into wide and tidy dataframes.

    Parameters
    ----------
    files_data:
        Tuple of `(filename, file_bytes)` pairs from the Streamlit uploader.

    Returns
    -------
    LoadedDataBundle
        Parsed bundle containing the raw wide and tidy dataframes.
    """

    if not files_data:
        return LoadedDataBundle(wide_df=pd.DataFrame(), tidy_df=pd.DataFrame())

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        paths = [tmp_path / name for name, _ in files_data]
        for (name, content), file_path in zip(files_data, paths):
            del name
            file_path.write_bytes(content)
        wide_df, tidy_df = load_sers_data_as_wide_and_tidy([str(file_path) for file_path in paths])

    return LoadedDataBundle(wide_df=wide_df, tidy_df=tidy_df)


def build_derived_bundle(
    loaded_bundle: LoadedDataBundle,
    *,
    min_shift: float | None = None,
    max_shift: float | None = None,
    n_peaks: int = 6,
    n_peaks_by_serotype: dict[str, int] | None = None,
) -> DerivedDataBundle:
    """
    Build the full derived dataframe bundle used across the Streamlit app.

    Parameters
    ----------
    loaded_bundle:
        Raw loaded wide/tidy dataframes.
    min_shift:
        Lower Raman-shift trimming bound.
    max_shift:
        Upper Raman-shift trimming bound.
    n_peaks:
        Default peak count used when serotype-specific counts are not provided.
    n_peaks_by_serotype:
        Optional mapping of serotype to peak count.

    Returns
    -------
    DerivedDataBundle
        Derived bundle containing trimmed/preprocessed dataframes and peak
        artifacts.
    """

    if loaded_bundle.wide_df.empty:
        empty_artifacts = PeakArtifacts(
            peak_infos_by_serotype={},
            mean_spec_by_serotype={},
            default_serotype=None,
            raman_x=pd.Series(dtype=float).to_numpy(),
        )
        return DerivedDataBundle(
            wide_df=loaded_bundle.wide_df.copy(),
            tidy_df=loaded_bundle.tidy_df.copy(),
            features_df=pd.DataFrame(),
            peak_df=pd.DataFrame(),
            peak_artifacts=empty_artifacts,
        )

    wide_df = preprocess_metadata(loaded_bundle.wide_df)
    wide_df = trim_raman_shift(wide_df, min_shift=min_shift, max_shift=max_shift)

    tidy_df = wide_to_tidy(wide_df)
    tidy_df = preprocess_metadata(tidy_df)

    features_df = extract_basic_features(wide_df)
    peak_df, peak_by_serotype, mean_by_serotype, default_serotype, raman_x = (
        extract_dynamic_peak_features(
            wide_df,
            n_peaks=int(n_peaks),
            n_peaks_by_serotype=n_peaks_by_serotype,
        )
    )

    if peak_by_serotype:
        first_infos = next(iter(peak_by_serotype.values()))
        peak_cols = get_peak_height_columns(first_infos)
        features_df = features_df.join(peak_df[peak_cols], how="left")

    peak_artifacts = PeakArtifacts(
        peak_infos_by_serotype=peak_by_serotype,
        mean_spec_by_serotype=mean_by_serotype,
        default_serotype=default_serotype,
        raman_x=raman_x,
    )
    return DerivedDataBundle(
        wide_df=wide_df,
        tidy_df=tidy_df,
        features_df=features_df,
        peak_df=peak_df,
        peak_artifacts=peak_artifacts,
    )
