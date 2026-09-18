"""
Unit tests for SENS-D validation metric tables (Metrics-docx Tables 1–3).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sensd_sers_analysis.assessment.validation_metrics import (
    TABLE1_COLUMNS,
    TABLE2_COLUMNS,
    TABLE3_COLUMNS,
    _add_target_concentration_group,
    build_validation_tables,
    fit_validation_predictions,
)
from sensd_sers_analysis.config import VALIDATION_N_SPLITS


def _make_validation_classification_df() -> pd.DataFrame:
    """Synthetic clean classification rows spanning two serovars and concentrations."""
    return pd.DataFrame(
        {
            "sensor_id": ["S1", "S1", "S1", "S2", "S2", "S3", "S3", "S3", "S4"],
            "serotype": ["ST", "ST", "ST", "ST", "ST", "SE", "SE", "SE", "SE"],
            "test_id": ["T1", "T1", "T2", "T1", "T2", "T1", "T2", "T3", "T1"],
            "concentration": [1000, 1000, 0, 1000, 1000, 100, 100, 100, 0],
            "target_concentration": [1000, 1000, 0, 1000, 1000, 100, 100, 100, 0],
            "concentration_group": [
                "1000 CFU",
                "1000 CFU",
                "0 CFU",
                "1000 CFU",
                "1000 CFU",
                "100 CFU",
                "100 CFU",
                "100 CFU",
                "0 CFU",
            ],
            "target_concentration_group": [
                "1000 CFU",
                "1000 CFU",
                "0 CFU",
                "1000 CFU",
                "1000 CFU",
                "100 CFU",
                "100 CFU",
                "100 CFU",
                "0 CFU",
            ],
            "log_concentration": [3.0, 3.0, np.nan, 3.0, 3.0, 2.0, 2.0, 2.0, np.nan],
            "target": ["ST", "ST", "Rinsate", "ST", "ST", "SE", "SE", "SE", "Rinsate"],
            "integral_area": [10.0, 10.5, 0.2, 9.8, 10.1, 5.0, 5.2, 5.1, 0.15],
            "max_intensity": [4.0, 4.1, 0.3, 3.9, 4.0, 2.5, 2.6, 2.5, 0.2],
            "mean_intensity": [2.0, 2.1, 0.1, 1.9, 2.0, 1.2, 1.3, 1.2, 0.1],
            "PC1": [0.5, 0.6, 0.0, 0.4, 0.5, 1.0, 1.1, 1.0, 0.0],
            "PC2": [0.2, 0.3, 0.0, 0.2, 0.2, 0.8, 0.9, 0.8, 0.0],
        }
    )


class TestValidationMetrics(unittest.TestCase):
    """Validation table builders."""

    def setUp(self) -> None:
        self.clf_df = _make_validation_classification_df()
        self.reg_df = self.clf_df[self.clf_df["target"] != "Rinsate"].copy()
        self.feat_cols = [
            "integral_area",
            "max_intensity",
            "mean_intensity",
            "PC1",
            "PC2",
        ]

    def test_repeated_sensor_holdout_folds(self) -> None:
        work = _add_target_concentration_group(self.clf_df.copy()).reset_index(drop=True)
        predictions = fit_validation_predictions(work, self.reg_df, self.feat_cols)
        self.assertTrue(predictions.sensor_holdout_available)
        self.assertEqual(predictions.n_splits, VALIDATION_N_SPLITS)
        self.assertEqual(len(predictions.folds), VALIDATION_N_SPLITS)
        self.assertGreater(predictions.n_test_sensors, 0)
        self.assertGreater(predictions.n_train_sensors, 0)
        self.assertGreater(predictions.n_eval_rows, 0)
        for fold in predictions.folds:
            self.assertEqual(len(fold.y_pred), len(work))
            self.assertEqual(len(fold.eval_mask), len(work))
            self.assertGreater(int(fold.eval_mask.sum()), 0)
            self.assertLess(int(fold.eval_mask.sum()), len(work))

    def test_three_tables_match_docx_column_schemas(self) -> None:
        artifacts = build_validation_tables(
            self.clf_df,
            self.reg_df,
            feature_cols=self.feat_cols,
        )
        self.assertEqual(list(artifacts.concentration_repeatability.columns), TABLE1_COLUMNS)
        self.assertEqual(list(artifacts.quantification.columns), TABLE2_COLUMNS)
        self.assertEqual(list(artifacts.consistency_reusability.columns), TABLE3_COLUMNS)
        self.assertFalse(artifacts.concentration_repeatability.empty)
        self.assertFalse(artifacts.quantification.empty)

    def test_table1_overall_row_and_no_merged_quant_columns(self) -> None:
        artifacts = build_validation_tables(
            self.clf_df,
            self.reg_df,
            feature_cols=self.feat_cols,
        )
        table = artifacts.concentration_repeatability
        st_rows = table[table["Serovar / Sample Group"] == "ST"]
        self.assertIn("Overall", st_rows["Concentration (CFU/mL)"].tolist())
        overall = st_rows[st_rows["Concentration (CFU/mL)"] == "Overall"].iloc[0]
        self.assertEqual(overall["No. of Sensors Tested"], 2)
        self.assertNotIn("Quantification Accuracy", table.columns)
        self.assertNotIn("Meet Target?", table.columns)
        self.assertIn("Meet Target?", artifacts.quantification.columns)

    def test_table2_quantification_range_when_eval_sufficient(self) -> None:
        artifacts = build_validation_tables(
            self.clf_df,
            self.reg_df,
            feature_cols=self.feat_cols,
        )
        table = artifacts.quantification
        st_1000 = table[
            (table["Serovar / Sample Group"] == "ST") & (table["Concentration (CFU/mL)"] == "1000")
        ]
        self.assertFalse(st_1000.empty)
        quant = st_1000.iloc[0]["Quantification Accuracy"]
        if quant:
            self.assertTrue(str(quant).startswith("~"), quant)

    def test_table1_groups_by_target_concentration_not_raw(self) -> None:
        """Grouping follows the nominal target even when actual CFU varies widely."""
        df = self.clf_df.copy()
        df.loc[0, "concentration"] = 620
        df.loc[1, "concentration"] = 1480
        artifacts = build_validation_tables(df, self.reg_df, feature_cols=self.feat_cols)
        st_concs = artifacts.concentration_repeatability.loc[
            artifacts.concentration_repeatability["Serovar / Sample Group"] == "ST",
            "Concentration (CFU/mL)",
        ].tolist()
        self.assertIn("1000", st_concs)
        self.assertNotIn("620", st_concs)
        self.assertNotIn("1480", st_concs)

    def test_table1_reports_concentration_cv_column(self) -> None:
        """Table 1 exposes actual-concentration CV alongside signal CV."""
        df = self.clf_df.copy()
        df.loc[0, "concentration"] = 620
        df.loc[1, "concentration"] = 1480
        df.loc[3, "concentration"] = 900
        df.loc[4, "concentration"] = 1100
        artifacts = build_validation_tables(df, self.reg_df, feature_cols=self.feat_cols)
        self.assertIn("Concentration CV%", artifacts.concentration_repeatability.columns)
        st_1000 = artifacts.concentration_repeatability[
            (artifacts.concentration_repeatability["Serovar / Sample Group"] == "ST")
            & (artifacts.concentration_repeatability["Concentration (CFU/mL)"] == "1000")
        ]
        self.assertFalse(st_1000.empty)
        self.assertTrue(st_1000.iloc[0]["Concentration CV%"].endswith("%"))

    def test_ml_metrics_blanked_below_min_eval_rows(self) -> None:
        """Tiny held-out groups do not report misleading accuracy numbers."""
        artifacts = build_validation_tables(
            self.clf_df,
            self.reg_df,
            feature_cols=self.feat_cols,
        )
        table = artifacts.concentration_repeatability
        self.assertIn("Eval N", table.columns)
        small = table[table["Eval N"] < 5]
        self.assertTrue((small["Accuracy (% Correct)"] == "").all())

    def test_table3_requires_repeated_tests(self) -> None:
        artifacts = build_validation_tables(
            self.clf_df,
            self.reg_df,
            feature_cols=self.feat_cols,
        )
        table = artifacts.consistency_reusability
        self.assertEqual(list(table.columns), TABLE3_COLUMNS)
        if not table.empty:
            se_row = table[table["Serovar"] == "SE"]
            self.assertGreaterEqual(float(se_row["Repeated Tests per Sensor (n)"].iloc[0]), 2.0)

    def test_build_validation_tables_artifact(self) -> None:
        artifacts = build_validation_tables(
            self.clf_df,
            self.reg_df,
            feature_cols=self.feat_cols,
        )
        self.assertEqual(artifacts.n_classification_rows, len(self.clf_df))
        self.assertEqual(artifacts.n_regression_rows, len(self.reg_df))
        self.assertFalse(artifacts.concentration_repeatability.empty)
        self.assertFalse(artifacts.quantification.empty)
        self.assertIsNotNone(artifacts.predictions)


if __name__ == "__main__":
    unittest.main()
