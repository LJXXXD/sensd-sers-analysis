"""Filter presentation preserves selection behavior across cascading option sets."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_search_filters_preserve_selection_exclude_and_reset():
    """Long labels remain searchable as options shrink; controls keep their state."""
    apps = Path(__file__).resolve().parents[1] / "apps"
    app = AppTest.from_string(f"""
import sys
sys.path.insert(0, {str(apps)!r})
import streamlit as st
from components.filter_ui import _render_filter, _use_search_selector
assert not _use_search_selector("sensor_id", [f"S{{i}}" for i in range(150)])
assert not _use_search_selector("date", [f"2026-{{i}}" for i in range(150)])
assert not _use_search_selector("extra", [f"X{{i}}" for i in range(150)])
assert not _use_search_selector("extra", ["a" * 40])
assert _use_search_selector("extra", ["a" * 40] * 20)
assert not _use_search_selector("filename", ["a.xlsx"])
assert not _use_search_selector("source_txt_filename", ["a.txt"])
assert _use_search_selector("filename", ["experiment_filename_2026_09_27.xlsx"] * 20)
assert _use_search_selector("source_txt_filename", ["original_spectrum_2026_09_27.txt"] * 20)
full = [f"long_metadata_value_for_experiment_{{i}}" for i in range(20)]
short = st.checkbox("Reduce available options")
options = full[:2] if short else full
selected, excluded = _render_filter(
    "extra", "Extra", options, [], False, st,
    layout_options=full, reset_button_key="reset_extra",
)
st.session_state["result"] = (selected, excluded)
""").run(timeout=30)
    assert not app.exception
    value = "long_metadata_value_for_experiment_0"
    app.multiselect(key="filter_extra").select(value).run()
    app.toggle(key="filter_extra_exclude").set_value(True).run()
    app.checkbox[0].check().run()
    assert not app.exception
    assert app.session_state["result"] == ([value], True)
    assert len(app.multiselect) == 1
    app.button(key="reset_extra").click().run()
    assert not app.exception
    assert app.session_state["result"] == ([], False)
