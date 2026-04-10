"""
Join fixed-anchor peak features into filtered application bundles.
"""

from __future__ import annotations

from collections.abc import Sequence

from sensd_sers_analysis.application.contracts import FilteredBundle
from sensd_sers_analysis.processing.targeted_peak_features import (
    extract_targeted_peak_height_features,
)


def merge_targeted_peaks_into_filtered_bundle(
    filtered_bundle: FilteredBundle,
    wide_df,
    anchor_cm1: Sequence[float],
) -> FilteredBundle:
    """
    Append ``peak_near_*`` columns to the filtered feature matrix.

    Targeted heights are recomputed on ``wide_df`` rows aligned to the
    filtered feature index so downstream tabs share one consistent feature
    table.

    Parameters
    ----------
    filtered_bundle:
        Filtered views from :func:`sensd_sers_analysis.application.apply_filters`.
    wide_df:
        Trimmed wide dataframe (same row labels as the derived feature table).
    anchor_cm1:
        Target Raman shifts (cm⁻¹) in UI order.

    Returns
    -------
    FilteredBundle
        Copy of ``filtered_bundle`` with targeted peak columns joined.
    """

    idx = filtered_bundle.filtered_features_df.index
    wide_aligned = wide_df.reindex(idx)
    targeted = extract_targeted_peak_height_features(wide_aligned, anchor_cm1)
    merged = filtered_bundle.filtered_features_df.join(targeted, how="left")
    return FilteredBundle(
        filtered_tidy_df=filtered_bundle.filtered_tidy_df,
        filtered_features_df=merged,
        n_unique_spectra=filtered_bundle.n_unique_spectra,
    )
