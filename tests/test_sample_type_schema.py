"""Behavioral coverage for sample identity, migration and generator presets."""

import importlib
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from sensd_sers_analysis.application.metadata_presets import validate_signal_preset
from sensd_sers_analysis.assessment.validation_metrics import _group_rinsate_positive_masks
from sensd_sers_analysis.classification.data_prep import prepare_classification_dataset
from sensd_sers_analysis.data.io import _load_signal_file
from sensd_sers_analysis.data.metadata_migration import migrate_rows, spectral_rows
from sensd_sers_analysis.processing.metadata import sample_type_masks
from sensd_sers_analysis.processing.peak_features import _exclude_rinsate_controls


def prep_module():
    """Import the app utility using its documented apps module search path."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))
    return importlib.import_module("txt_to_excel")


def test_identity_does_not_depend_on_concentration():
    """Zero bacteria and positive-count controls retain explicit identities."""
    frame = pd.DataFrame(
        {
            "sample_type": ["Bacteria sample", "Rinsate control", ""],
            "concentration": [0.0, 15.0, 0.0],
            "concentration_group": ["0 CFU", "10 CFU", "0 CFU"],
            "sensor_id": ["A"] * 3,
            "serotype": ["ST"] * 3,
            "integral_area": [1.0, 2.0, 3.0],
            "log_concentration": [np.nan, 1.0, np.nan],
        }
    )
    controls, bacteria = sample_type_masks(frame)
    assert controls.tolist() == [False, True, False]
    assert bacteria.tolist() == [True, False, False]
    masks = _group_rinsate_positive_masks(frame)
    assert masks[0].tolist() == controls.tolist()
    assert masks[1].tolist() == bacteria.tolist()
    assert _exclude_rinsate_controls(frame).index.tolist() == [0]
    clean = prepare_classification_dataset(frame, excluded_map={})
    assert clean.set_index("concentration")["target"].to_dict() == {0.0: "ST", 15.0: "Rinsate"}


def test_plain_workbook_roundtrip_preserves_per_signal_identity(tmp_path):
    """Exporter writes plain values; loader keeps identity independent of dose."""
    prep = prep_module()
    metadata = {label: "test" for _, label, _ in prep.METADATA_FIELD_SPECS}
    rows = prep.build_embedded_workbook_rows(
        metadata,
        target_concentrations=[1000, 0],
        actual_concentrations=[0, 0],
        source_txt_filenames=["heat.txt", "control.txt"],
        special_treatments=["Heat treated", ""],
        sample_types=["Bacteria sample", "Rinsate control"],
        raman_shift=np.array([500.0, 501.0]),
        intensity_columns=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    )
    payload = prep.embedded_workbook_to_excel_bytes(rows)
    workbook = openpyxl.load_workbook(BytesIO(payload))
    assert not workbook.active.data_validations.dataValidation
    assert workbook.active["A20"].value == "Initial Target Concentration (CFU/mL)"
    path = tmp_path / "new.xlsx"
    path.write_bytes(payload)
    frame = _load_signal_file(path)
    assert frame.sample_type.tolist() == ["Bacteria sample", "Rinsate control"]
    np.testing.assert_allclose(frame.target_concentration, [1000, 0])
    np.testing.assert_allclose(frame.concentration, [0, 0])


def test_preset_labels_roundtrip_and_reload(monkeypatch):
    """Preset import restores named signals; batch reset clears labels and counts."""
    prep = prep_module()
    labels = {"a.txt": {"sample_type": "Bacteria sample", "special_treatment": "PAA 50 ppm"}}
    payload = prep.build_template_export_payload({}, frozenset(), labels)
    valid, warnings = validate_signal_preset(payload["signal_labels"])
    assert valid == labels and not warnings
    assert validate_signal_preset({"bad": {"sample_type": "oops"}})[0] == {}
    state = {
        prep.TEMPLATE_IMPORT_PENDING_VALUES_KEY: {prep.PER_SIGNAL_PRESET_KEY: valid},
        "txt2excel_sample_type_a.txt": "Rinsate control",
    }
    monkeypatch.setattr(prep.st, "session_state", state)
    prep._apply_pending_template_import_values()
    assert state[prep.PER_SIGNAL_PRESET_KEY] == labels
    assert "txt2excel_sample_type_a.txt" not in state
    state.update(
        {
            "txt2excel_sample_type_a.txt": "Bacteria sample",
            "txt2excel_actual_a.txt": "15",
            "txt2excel_treatment_a.txt": "PAA 50 ppm",
            "txt2excel_meta_operator": "Adheesha",
        }
    )
    prep.clear_prep_uploads()
    assert "txt2excel_actual_a.txt" not in state
    assert "txt2excel_treatment_a.txt" not in state
    assert "txt2excel_sample_type_a.txt" not in state
    assert state[prep.PERSISTENT_METADATA_SNAPSHOT_KEY]["txt2excel_meta_operator"] == "Adheesha"
    assert state[prep.PER_SIGNAL_PRESET_KEY] == labels


def test_migration_preserves_actual_and_spectra_flags_conflicts():
    """PAA initials use confirmed policy; contradictions remain explicitly unknown."""
    rows = [
        ["File Name", "rinsate.txt", "PAA100 ppm.txt", "heat killed.txt"],
        ["Special Treatment", "Rinsate", "100 ppm PAA treated", "Untreated 1000 CFU"],
        ["Target Concentration (CFU/mL)", 0, None, 1000],
        ["Actual Concentration (CFU/mL)", 0, 15, 950],
        ["Raman Shift", "I", "I", "I"],
        [500.0, 1.0, 2.0, 3.0],
    ]
    output, changes = migrate_rows(rows, collection="PAA")
    assert spectral_rows(output) == spectral_rows(rows)
    assert changes[1]["initial_target"] == 1000
    assert changes[1]["actual"] == 15
    assert changes[2]["sample_type"] == ""
    assert "REVIEW" in changes[2]["basis"]
    assert migrate_rows(output, collection="PAA") == (output, [])


def test_generator_dropdowns_render_and_change_independently():
    """The generator renders categorical choices per file rather than globally."""
    from streamlit.testing.v1 import AppTest

    app_directory = str(Path(__file__).resolve().parents[1] / "apps")
    script = f"""
import sys
sys.path.insert(0, {app_directory!r})
import streamlit as st
from types import SimpleNamespace
import txt_to_excel as prep
files = [SimpleNamespace(name=name, getvalue=lambda: b"600\\t1\\n601\\t2\\n") for name in ["a.txt", "b.txt"]]
original = st.file_uploader
st.file_uploader = lambda label, **kwargs: files if label == "Upload Raman TXT Files" else None
try:
    prep.render_prep_mode()
finally:
    st.file_uploader = original
"""
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.selectbox(key="txt2excel_sample_type_a.txt").value is None
    assert app.selectbox(key="txt2excel_treatment_a.txt").value == "None"
    app.selectbox(key="txt2excel_sample_type_a.txt").select("Bacteria sample")
    app.selectbox(key="txt2excel_treatment_a.txt").select("PAA 100 ppm")
    app.run()
    assert not app.exception
    assert app.selectbox(key="txt2excel_treatment_a.txt").value == "PAA 100 ppm"
    assert app.selectbox(key="txt2excel_treatment_b.txt").value == "None"
    assert app.selectbox(key="txt2excel_sample_type_b.txt").value is None
