"""Unit tests for fixed-anchor targeted peak feature extraction."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from sensd_sers_analysis.application.targeted_peak_service import (
    merge_targeted_peaks_into_filtered_bundle,
)
from sensd_sers_analysis.application.contracts import FilteredBundle
from sensd_sers_analysis.processing.targeted_peak_features import (
    detect_targeted_peaks_on_spectrum_row,
    extract_targeted_peak_height_features,
    list_targeted_peak_feature_columns,
    parse_feature_name_to_anchor,
    target_anchor_to_feature_name,
)


class TargetedPeakFeatureTests(unittest.TestCase):
    """Tests for naming, parsing, and window extraction."""

    def test_target_anchor_to_feature_name_examples(self) -> None:
        self.assertEqual(target_anchor_to_feature_name(501.8), "peak_near_501_8")
        self.assertEqual(target_anchor_to_feature_name(1066.5), "peak_near_1066_5")

    def test_parse_feature_name_round_trip(self) -> None:
        for anchor in (501.8, 613.7, 1066.5, 1196.8):
            col = target_anchor_to_feature_name(anchor)
            parsed = parse_feature_name_to_anchor(col)
            self.assertIsNotNone(parsed)
            assert parsed is not None
            self.assertTrue(math.isclose(parsed, anchor, rel_tol=0.0, abs_tol=1e-6))

    def test_extract_finds_max_in_window(self) -> None:
        wide = pd.DataFrame(
            {
                "sensor_id": ["s1"],
                "rs_490.0": [0.0],
                "rs_500.0": [1.0],
                "rs_502.0": [10.0],
                "rs_505.0": [2.0],
                "rs_510.0": [0.0],
            }
        )
        out = extract_targeted_peak_height_features(wide, [502.0], half_width_cm1=5.0)
        self.assertIn("peak_near_502", out.columns)
        val = float(out.iloc[0, 0])
        self.assertGreater(val, 5.0)

    def test_merge_into_filtered_bundle_aligns_rows(self) -> None:
        wide = pd.DataFrame(
            {
                "sensor_id": ["a", "b"],
                "rs_500.0": [1.0, 2.0],
                "rs_600.0": [5.0, 6.0],
            },
            index=[10, 20],
        )
        features = pd.DataFrame({"sensor_id": ["a", "b"]}, index=[10, 20])
        fb = FilteredBundle(
            filtered_tidy_df=pd.DataFrame(),
            filtered_features_df=features,
            n_unique_spectra=2,
        )
        merged = merge_targeted_peaks_into_filtered_bundle(fb, wide, (550.0,))
        cols = list_targeted_peak_feature_columns(merged.filtered_features_df.columns)
        self.assertEqual(len(cols), 1)
        self.assertEqual(len(merged.filtered_features_df.columns), 2)

    def test_detect_row_matches_vectorized_column(self) -> None:
        x = np.array([500.0, 502.0, 504.0], dtype=float)
        y = np.array([0.0, 8.0, 1.0], dtype=float)
        sh, raw, adj = detect_targeted_peaks_on_spectrum_row(y, x, [502.0], half_width_cm1=3.0)
        self.assertTrue(math.isclose(float(sh[0]), 502.0, rel_tol=0.0, abs_tol=0.1))
        self.assertTrue(math.isclose(float(raw[0]), 8.0, rel_tol=0.0, abs_tol=1e-9))
        self.assertGreater(float(adj[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
