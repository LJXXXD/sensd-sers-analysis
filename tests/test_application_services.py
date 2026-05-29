"""
Parity-focused tests for the application-layer service refactor.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from sensd_sers_analysis.application.assessment_service import (
    build_sensor_assessment_artifacts,
)
from sensd_sers_analysis.application.classification_service import (
    run_classification_training,
)
from sensd_sers_analysis.application.contracts import (
    FilterSelection,
    LoadedDataBundle,
    ModelConsistencySelection,
    SensorAssessmentSelection,
)
from sensd_sers_analysis.application.dataset_pipeline import build_derived_bundle
from sensd_sers_analysis.application.filtering_service import apply_filters
from sensd_sers_analysis.application.sensor_assessment_service import (
    build_global_qa_artifacts,
    build_overlay_artifacts,
    build_single_sensor_consistency_artifacts,
)
from sensd_sers_analysis.assessment import (
    ASSESSMENT_GROUP_COLS,
    compute_batch_variance,
    compute_degradation,
    fit_concentration_regression_cleaned,
    get_consistency_summary_table,
    get_global_model_consistency_qa,
    get_zero_cfu_baseline,
    identify_deviating_sensors,
    prepare_degradation_data,
)
from sensd_sers_analysis.classification import train_classifiers
from sensd_sers_analysis.config import BATCH_DEVIATION_Z_THRESHOLD
from sensd_sers_analysis.data import count_unique_spectra, wide_to_tidy
from sensd_sers_analysis.processing import (
    extract_basic_features,
    extract_dynamic_peak_features,
    filter_sers_data,
    preprocess_metadata,
    snap_spectra_to_master_grid,
    trim_raman_shift,
)


def _make_sample_loaded_bundle() -> LoadedDataBundle:
    wide_df = pd.DataFrame(
        {
            "sensor_model": ["M1", "M1", "M2", "M2"],
            "sensor_id": ["S1", "S1", "S2", "S2"],
            "test_id": ["T1", "T2", "T1", "T2"],
            "connection_id": ["C1", "C1", "C2", "C2"],
            "serotype": ["ST", "ST", "SE", "SE"],
            "date": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"],
            "operator": ["op", "op", "op", "op"],
            "concentration": [1000, 1000, 1000, 0],
            "filename": ["f1.xlsx", "f2.xlsx", "f3.xlsx", "f4.xlsx"],
            "signal_index": [0, 0, 0, 0],
            "rs_400.00": [1.0, 1.0, 1.0, 0.0],
            "rs_500.00": [3.0, 2.0, 8.0, 1.0],
            "rs_600.00": [10.0, 9.0, 3.0, 0.5],
            "rs_700.00": [3.0, 2.0, 1.0, 0.2],
            "rs_800.00": [1.0, 1.0, 0.0, 0.1],
        }
    )
    return LoadedDataBundle(wide_df=wide_df, tidy_df=wide_to_tidy(wide_df))


def _make_assessment_features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor_id": ["A", "A", "B", "B", "C", "C"],
            "serotype": ["ST", "ST", "ST", "ST", "SE", "SE"],
            "concentration_group": [
                "1000 CFU",
                "1000 CFU",
                "1000 CFU",
                "1000 CFU",
                "1000 CFU",
                "1000 CFU",
            ],
            "concentration": [1000, 1000, 1000, 1000, 1000, 1000],
            "test_id": ["T1", "T2", "T1", "T2", "T1", "T2"],
            "date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-01",
                "2025-01-02",
                "2025-01-01",
                "2025-01-02",
            ],
            "integral_area": [1.0, 1.2, 0.9, 1.1, 2.0, 2.1],
            "max_intensity": [4.0, 4.3, 3.8, 4.1, 5.0, 5.2],
            "mean_intensity": [2.0, 2.1, 1.9, 2.0, 2.5, 2.6],
            "PC1": [0.1, 0.2, -0.1, 0.0, 1.0, 1.1],
            "PC2": [0.5, 0.4, 0.3, 0.2, 1.5, 1.4],
        }
    )


def _make_model_consistency_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor_id": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "serotype": ["ST", "ST", "ST", "ST", "ST", "ST"],
            "concentration_group": [
                "0 CFU",
                "10 CFU",
                "100 CFU",
                "0 CFU",
                "10 CFU",
                "100 CFU",
            ],
            "concentration": [0, 10, 100, 0, 10, 100],
            "log_concentration": [np.nan, 1.0, 2.0, np.nan, 1.0, 2.0],
            "integral_area": [0.2, 1.0, 2.0, 0.1, 0.9, 1.9],
            "max_intensity": [0.3, 1.1, 2.1, 0.2, 1.0, 2.0],
            "mean_intensity": [0.1, 0.5, 1.0, 0.1, 0.4, 0.9],
            "PC1": [0.0, 0.2, 0.4, 0.0, 0.2, 0.4],
            "PC2": [0.0, -0.1, -0.2, 0.0, -0.1, -0.2],
        }
    )


def _make_classification_clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [
                "ST",
                "ST",
                "ST",
                "ST",
                "ST",
                "SE",
                "SE",
                "SE",
                "SE",
                "SE",
                "Rinsate",
                "Rinsate",
                "Rinsate",
                "Rinsate",
                "Rinsate",
            ],
            "integral_area": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_intensity": [
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2.5,
                2.6,
                2.7,
                2.8,
                2.9,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ],
            "mean_intensity": [
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
            ],
            "PC1": [
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                -0.8,
                -0.7,
                -0.6,
                -0.5,
                -0.4,
            ],
            "PC2": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                -0.5,
                -0.4,
                -0.3,
                -0.2,
                -0.1,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
            ],
        }
    )


class ApplicationServiceTests(unittest.TestCase):
    """Parity checks for the application-service refactor."""

    def test_build_derived_bundle_matches_manual_pipeline(self) -> None:
        loaded_bundle = _make_sample_loaded_bundle()

        manual_wide = preprocess_metadata(loaded_bundle.wide_df)
        manual_wide = snap_spectra_to_master_grid(manual_wide)
        manual_wide = trim_raman_shift(manual_wide, min_shift=450.0, max_shift=750.0)
        manual_tidy = preprocess_metadata(wide_to_tidy(manual_wide))
        manual_features = extract_basic_features(manual_wide)
        (
            manual_peak_df,
            manual_peak_by_sero,
            manual_mean_by_sero,
            manual_default_sero,
            manual_raman_x,
        ) = extract_dynamic_peak_features(
            manual_wide,
            n_peaks=1,
            n_peaks_by_serotype={"ST": 1, "SE": 1},
        )
        bundle = build_derived_bundle(
            loaded_bundle,
            min_shift=450.0,
            max_shift=750.0,
            n_peaks=1,
            n_peaks_by_serotype={"ST": 1, "SE": 1},
        )

        assert_frame_equal(bundle.wide_df, manual_wide)
        assert_frame_equal(bundle.tidy_df, manual_tidy)
        assert_frame_equal(bundle.features_df, manual_features)
        assert_frame_equal(bundle.peak_df, manual_peak_df)
        self.assertEqual(bundle.peak_artifacts.default_serotype, manual_default_sero)
        self.assertEqual(
            bundle.peak_artifacts.peak_infos_by_serotype.keys(),
            manual_peak_by_sero.keys(),
        )
        self.assertEqual(
            bundle.peak_artifacts.mean_spec_by_serotype.keys(),
            manual_mean_by_sero.keys(),
        )
        self.assertTrue(np.allclose(bundle.peak_artifacts.raman_x, manual_raman_x))

    def test_snap_spectra_to_master_grid_respects_native_bounds(self) -> None:
        wide_df = pd.DataFrame(
            {
                "sensor_id": ["wide", "narrow"],
                "rs_400.0": [1.0, np.nan],
                "rs_500.0": [2.0, 20.0],
                "rs_600.0": [3.0, 30.0],
                "rs_700.0": [4.0, np.nan],
                "rs_800.0": [5.0, np.nan],
            }
        )
        snapped = snap_spectra_to_master_grid(wide_df)
        narrow = snapped.loc[snapped["sensor_id"] == "narrow"].iloc[0]
        rs_cols = sorted(
            (c for c in snapped.columns if isinstance(c, str) and c.startswith("rs_")),
            key=lambda c: float(c[3:]),
        )
        for col in rs_cols:
            shift = float(col[3:])
            val = narrow[col]
            if 500.0 <= shift <= 600.0:
                self.assertTrue(np.isfinite(val), msg=col)
            else:
                self.assertTrue(pd.isna(val), msg=col)

    def test_apply_filters_matches_processing_layer(self) -> None:
        loaded_bundle = _make_sample_loaded_bundle()
        derived_bundle = build_derived_bundle(loaded_bundle, n_peaks=1)
        filter_state = {"serotype": FilterSelection(selected_values=("ST",), exclude=False)}

        filtered_bundle = apply_filters(derived_bundle, filter_state)
        manual_state = {"serotype": (["ST"], False)}
        manual_tidy = filter_sers_data(derived_bundle.tidy_df, manual_state)
        manual_features = filter_sers_data(derived_bundle.features_df, manual_state)

        assert_frame_equal(filtered_bundle.filtered_tidy_df, manual_tidy)
        assert_frame_equal(filtered_bundle.filtered_features_df, manual_features)
        self.assertEqual(filtered_bundle.n_unique_spectra, count_unique_spectra(manual_tidy))

    def test_sensor_assessment_artifacts_match_direct_calls(self) -> None:
        filtered_features = _make_assessment_features_df()
        selection = SensorAssessmentSelection(
            serotype="ST",
            concentration_group="1000 CFU",
            feature="integral_area",
            outlier_method="iqr",
            batch_feature="max_intensity",
        )
        artifacts = build_sensor_assessment_artifacts(filtered_features, selection)

        assessment_df = filtered_features[
            (filtered_features["serotype"] == "ST")
            & (filtered_features["concentration_group"] == "1000 CFU")
        ].copy()
        group_cols = [column for column in ASSESSMENT_GROUP_COLS if column in assessment_df.columns]
        expected_display_consistency = get_consistency_summary_table(
            assessment_df,
            feature_cols=[selection.feature],
            group_cols=group_cols,
            outlier_method=selection.outlier_method,
        )
        expected_pdf_consistency = get_consistency_summary_table(
            assessment_df,
            group_cols=group_cols,
            outlier_method=selection.outlier_method,
        )
        expected_deg_input = prepare_degradation_data(assessment_df, selection.feature)
        expected_deg_table = compute_degradation(
            expected_deg_input,
            selection.feature,
            "test_ordinal",
            group_cols=["sensor_id"],
        )
        expected_display_batch = compute_batch_variance(
            assessment_df,
            selection.batch_feature,
            sensor_col="sensor_id",
            group_cols=None,
        )
        expected_display_deviating = identify_deviating_sensors(
            expected_display_batch,
            z_threshold=BATCH_DEVIATION_Z_THRESHOLD,
            sensor_col="sensor_id",
        )
        expected_pdf_batch = compute_batch_variance(
            assessment_df,
            selection.feature,
            sensor_col="sensor_id",
            group_cols=None,
        )
        expected_pdf_deviating = identify_deviating_sensors(
            expected_pdf_batch,
            z_threshold=BATCH_DEVIATION_Z_THRESHOLD,
            sensor_col="sensor_id",
        )

        self.assertIsNone(artifacts.consistency_error)
        self.assertIsNone(artifacts.degradation_error)
        self.assertIsNone(artifacts.batch_error)
        assert_frame_equal(artifacts.assessment_df, assessment_df)
        self.assertEqual(artifacts.consistency_group_cols, group_cols)
        assert_frame_equal(artifacts.display_consistency_table, expected_display_consistency)
        assert_frame_equal(artifacts.pdf_consistency_table, expected_pdf_consistency)
        assert_frame_equal(artifacts.degradation_input_df, expected_deg_input)
        assert_frame_equal(artifacts.degradation_table, expected_deg_table)
        assert_frame_equal(artifacts.display_batch_table, expected_display_batch)
        assert_frame_equal(
            artifacts.display_deviating_sensors_table,
            expected_display_deviating,
        )
        assert_frame_equal(artifacts.pdf_batch_table, expected_pdf_batch)
        assert_frame_equal(artifacts.pdf_deviating_sensors_table, expected_pdf_deviating)

    def test_model_consistency_and_classification_services_match_direct_calls(self) -> None:
        filtered_features = _make_model_consistency_df()
        selection = ModelConsistencySelection(
            sensor_id="S1",
            serotype="ST",
            feature="integral_area",
        )
        single_artifacts = build_single_sensor_consistency_artifacts(
            filtered_features,
            selection,
        )
        direct_subset = filtered_features[
            (filtered_features["sensor_id"] == "S1") & (filtered_features["serotype"] == "ST")
        ].copy()
        direct_regression = fit_concentration_regression_cleaned(
            direct_subset,
            "integral_area",
        )
        direct_zero_baseline = get_zero_cfu_baseline(direct_subset, "integral_area")

        assert_frame_equal(single_artifacts.model_df, direct_subset)
        self.assertEqual(single_artifacts.zero_cfu_baseline, direct_zero_baseline)
        self.assertIsNotNone(single_artifacts.regression_result)
        self.assertIsNotNone(direct_regression)
        self.assertEqual(
            single_artifacts.regression_result.clean_rmse,
            direct_regression.clean_rmse,
        )
        self.assertEqual(
            single_artifacts.regression_result.clean_r2,
            direct_regression.clean_r2,
        )

        qa_artifacts = build_global_qa_artifacts(filtered_features, ("integral_area",))
        direct_qa_table, direct_excluded_map = get_global_model_consistency_qa(
            filtered_features,
            feature_cols=["integral_area"],
        )
        assert_frame_equal(qa_artifacts.table, direct_qa_table)
        self.assertEqual(qa_artifacts.excluded_map, direct_excluded_map)

        overlay_artifacts = build_overlay_artifacts(
            filtered_features,
            ("ST",),
            ("integral_area",),
            qa_artifacts.excluded_map,
        )
        self.assertEqual(len(overlay_artifacts), 1)
        expected_all_sensors = frozenset({"S1", "S2"})
        self.assertEqual(
            overlay_artifacts[0].excluded_sensors,
            frozenset(direct_excluded_map.get(("ST", "integral_area"), set())),
        )
        self.assertEqual(
            overlay_artifacts[0].pass_sensors,
            expected_all_sensors - overlay_artifacts[0].excluded_sensors,
        )

        clean_classification_df = _make_classification_clean_df()
        feature_columns = (
            "integral_area",
            "max_intensity",
            "mean_intensity",
            "PC1",
            "PC2",
        )
        direct_rf, direct_svm = train_classifiers(clean_classification_df, list(feature_columns))
        artifacts = run_classification_training(clean_classification_df, feature_columns)

        assert_frame_equal(artifacts.clean_classification_df, clean_classification_df)
        self.assertEqual(artifacts.feature_columns, feature_columns)
        self.assertEqual(artifacts.rf_result.accuracy, direct_rf.accuracy)
        self.assertEqual(artifacts.rf_result.f1, direct_rf.f1)
        self.assertEqual(artifacts.svm_result.accuracy, direct_svm.accuracy)
        self.assertEqual(artifacts.svm_result.f1, direct_svm.f1)
        self.assertEqual(artifacts.best_result.f1, max(direct_rf.f1, direct_svm.f1))


if __name__ == "__main__":
    unittest.main()
