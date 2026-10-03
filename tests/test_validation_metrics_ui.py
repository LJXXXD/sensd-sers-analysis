"""Validation failures are visible and stop table/download presentation."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_validation_fit_error_is_displayed(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))
    from tabs import validation_metrics

    def fail(*args, **kwargs):
        raise ValueError("Validation classifier fold failed: invalid feature matrix")

    monkeypatch.setattr(validation_metrics, "build_cached_validation_artifacts", fail)
    app = AppTest.from_string(
        """
import pandas as pd
from tabs.validation_metrics import render
render(pd.DataFrame({
    "sensor_id": ["S1"], "serotype": ["ST"], "test_id": ["T1"],
    "sample_type": ["Bacteria sample"], "concentration_group": ["10 CFU"], "integral_area": [1.0], "PC1": [0.1],
}), None)
"""
    ).run(timeout=30)
    assert not app.exception
    assert len(app.error) == 1
    assert "Validation classifier fold failed" in app.error[0].value
    assert not app.dataframe
    assert not app.get("download_button")
