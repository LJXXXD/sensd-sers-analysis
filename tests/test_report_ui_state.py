"""Report downloads belong to the current inputs and a successful request."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_report_input_change_and_failed_request_clear_download_bytes():
    app_dir = Path(__file__).resolve().parents[1] / "apps"
    script = f"""
import sys
sys.path.insert(0, {str(app_dir)!r})
import streamlit as st
import pandas as pd
from components.shared_ui import report_context_key, render_pdf_download_section
changed = st.checkbox("Change scope")
fail = st.checkbox("Fail generation")
frame = pd.DataFrame({{"measurement": [2 if changed else 1]}})
def generate():
    if fail:
        raise RuntimeError("Requested report failed")
    return b"verified-callback-bytes"
render_pdf_download_section("report", "report.pdf", generate, context_key=report_context_key(frame))
"""
    app = AppTest.from_string(script).run()
    assert "report" not in app.session_state
    app.button[0].click().run()
    assert app.session_state["report"] == b"verified-callback-bytes"
    assert len(app.get("download_button")) == 1
    app.checkbox[0].check().run()
    assert "report" not in app.session_state
    assert not app.get("download_button")
    app.button[0].click().run()
    assert len(app.get("download_button")) == 1
    app.checkbox[1].check().run()
    app.button[0].click().run()
    assert "report" not in app.session_state
    assert not app.get("download_button")
    assert "Requested report failed" in app.error[0].value
