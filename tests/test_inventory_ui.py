"""Inventory chart controls preserve stable choices and curated defaults."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_inventory_add_remove_and_dimension_changes():
    """Chart actions preserve choices and expose independent group/color dimensions."""
    apps = Path(__file__).resolve().parents[1] / "apps"
    app = AppTest.from_string(f"""
import sys
sys.path.insert(0, {str(apps)!r})
import pandas as pd
from tabs.data_inventory import render
frame = pd.DataFrame({{
    "filename": ["a", "b", "c"], "signal_index": [0, 0, 0],
    "sensor_id": ["S1", "S2", "S1"], "serotype": ["ST", "SE", "ST"],
    "target_concentration": [1, 10, 100], "date": ["2026-01-01"] * 3,
    "operator": ["A", "B", "A"], "raman_shift": [500] * 3, "intensity": [1.] * 3,
}})
render(frame)
""").run(timeout=30)
    assert not app.exception
    assert [app.session_state[f"inventory_count_{i}_group"] for i in range(3)] == [
        "serotype",
        "sensor_id",
        "month",
    ]
    assert app.session_state["inventory_count_0_color"] == "target_concentration"
    app.selectbox(key="inventory_count_0_color").set_value(None).run()
    assert not app.exception
    assert app.session_state["inventory_count_0_color"] is None
    app.run()
    assert app.session_state["inventory_count_0_color"] is None
    app.selectbox(key="inventory_count_0_color").select("operator").run()
    assert not app.exception
    assert app.session_state["inventory_count_0_group"] == "serotype"
    assert app.session_state["inventory_count_0_color"] == "operator"
    app.selectbox(key="inventory_count_0_group").select("operator").run()
    assert not app.exception
    app.button(key="remove_count_1").click().run()
    assert not app.exception
    assert app.session_state["inventory_count_0_group"] == "operator"
    next(button for button in app.button if button.label == "+ Add count chart").click().run()
    assert not app.exception
    ids = app.session_state["inventory_count_ids"]
    choices = [app.session_state[f"inventory_count_{i}_group"] for i in ids]
    assert len(choices) == len(set(choices)) == 3
    app.selectbox(key="inventory_coverage_0_row").select("serotype").run()
    assert not app.exception
    dims = [
        app.session_state[f"inventory_coverage_0_{suffix}"] for suffix in ("row", "column", "facet")
    ]
    assert len(set(dims)) == 3
    next(button for button in app.button if button.label == "+ Add coverage chart").click().run()
    assert not app.exception
    assert len(app.session_state["inventory_coverage_ids"]) == 2
