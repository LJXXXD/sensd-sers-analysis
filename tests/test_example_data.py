"""Bundled dataset selection and unload persistence checks."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from sensd_sers_analysis.config.example_data import EXAMPLE_DATA_DIRECTORIES


def test_bundled_scope_and_source_controls():
    """Load real dilution files, persist unloading, and restore the bundled source."""
    assert all(path.is_dir() for path in EXAMPLE_DATA_DIRECTORIES)
    app_directory = Path(__file__).resolve().parents[1] / "apps"
    script = f"""
import sys
sys.path.insert(0, {str(app_directory)!r})
import streamlit as st
from components.data_loading import render_data_source
bundle, count = render_data_source()
st.metric("Files", count)
st.metric("Spectra", len(bundle.wide_df) if bundle is not None else 0)
"""
    app = AppTest.from_string(script).run(timeout=60)
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["125", "580"]
    app.button[0].click().run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["0", "0"]
    app.run()
    assert app.metric[0].value == "0"
    app.button[1].click().run(timeout=60)
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["125", "580"]
