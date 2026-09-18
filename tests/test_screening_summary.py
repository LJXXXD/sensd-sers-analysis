"""Screening count conservation and fit-point provenance."""

import numpy as np
import pandas as pd
import pytest

from sensd_sers_analysis.application.contracts import ModelConsistencySelection
from sensd_sers_analysis.application.sensor_assessment_service import (
    build_screening_counts,
    build_screening_point_ledger,
    build_single_sensor_consistency_artifacts,
)
from sensd_sers_analysis.application.inventory_service import (
    build_inventory_chart_tables,
    metadata_issues,
)


def test_screening_counts_partial_sensor_and_unknown():
    """Partial sensor exclusion and unavailable fits retain their own counts."""
    features = pd.DataFrame(
        {"sensor_id": ["A", "A", "A", "B", "C"], "serotype": ["ST", "ST", "SE", "ST", "SE"]}
    )
    qa = pd.DataFrame(
        {
            "sensor_id": ["A", "A", "C"],
            "serotype": ["ST", "SE", "SE"],
            "status": ["Pass", "Excluded", "Pass"],
            "clean_r2": [0.9, 0.1, np.nan],
            "clean_rmse": [1.0, 2.0, 0.0],
        }
    )
    counts = build_screening_counts(features, qa)
    assert counts.spectra.sum() == len(features)
    assert counts.groupby("status").spectra.sum().to_dict() == {
        "Excluded": 1,
        "Not assessed": 2,
        "Pass": 2,
    }
    with pytest.raises(ValueError):
        build_screening_counts(features, pd.concat([qa, qa]))


def test_fit_ledger_matches_regression_mask():
    """Ledger distinguishes zero CFU and outliers without changing source rows."""
    frame = pd.DataFrame(
        {
            "sensor_id": ["A"] * 9,
            "serotype": ["ST"] * 9,
            "log_concentration": [np.nan, 0, 0, 1, 1, 2, 2, 3, 3],
            "integral_area": [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 90.0],
            "concentration": [0, 1, 1, 10, 10, 100, 100, 1000, 1000],
        }
    )
    artifacts = build_single_sensor_consistency_artifacts(
        frame, ModelConsistencySelection("A", "ST", "integral_area")
    )
    ledger = build_screening_point_ledger(artifacts)
    assert len(ledger) == len(frame)
    assert ledger.fit_use.str.startswith("Outlier").sum() == artifacts.regression_result.n_outliers
    assert ledger.fit_use.str.startswith("Outside").sum() == 1
    assert "fit_use" not in frame


def test_inventory_dimensions_and_date_diagnostics():
    """Mixed valid date formats are accepted and every chart dimension conserves counts."""
    frame = pd.DataFrame(
        {
            "filename": ["a", "b", "c"],
            "date": ["2026-01-02", "2/3/2026", "bad"],
            "serotype": ["ST", "SE", "ST"],
            "operator": ["A", "B", "A"],
        }
    )
    issues = metadata_issues(frame)
    dates = issues.loc[issues.field == "date"]
    assert dates.filename.tolist() == ["c"]
    for dimension in ("serotype", "operator", "sensor_id", "sample_type"):
        table = build_inventory_chart_tables(frame, group_by=dimension)["composition"]
        assert table.spectra.sum() == 3


def test_preprocessing_preserves_mixed_valid_dates():
    """A timestamp suffix must not erase otherwise valid acquisition dates."""
    from sensd_sers_analysis.processing.metadata import preprocess_metadata

    frame = pd.DataFrame({"date": ["2026-01-01", "2026-04-01 00:00:00"]})
    result = preprocess_metadata(frame)
    assert result.date.tolist() == ["2026-01-01", "2026-04-01"]
