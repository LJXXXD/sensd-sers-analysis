"""Independent workbook and dataset integrity regressions."""

from io import BytesIO
import re

import numpy as np
import pandas as pd
import pytest

from sensd_sers_analysis.application.contracts import FilterSelection
from sensd_sers_analysis.application.dataset_pipeline import (
    build_derived_bundle,
    load_uploaded_bundle,
)
from sensd_sers_analysis.application.filtering_service import apply_filters
from sensd_sers_analysis.data.io import get_metadata_columns


def _workbook_bytes(
    *, shifts=(100.0, 101.0, 103.0), first_signal=(1.0, 2.0, 4.0), extra_metadata=()
):
    """Write literal worksheet cells, independent of the converter implementation."""
    rows = [
        ["Sensor ID", "S1", None],
        ["Test ID", "T1", None],
        ["Connection ID", "C1", None],
        ["Serotype", "ST", None],
        ["Lab Batch", "batch-A", None],
        *[[label, value, None] for label, value in extra_metadata],
        ["Sample Type", "Bacteria sample", "Bacteria sample"],
        ["Target Concentration", 10, 100],
        ["Actual Concentration", 12, 90],
        ["Raman Shift", None, None],
        *[[x, y, y + 1] for x, y in zip(shifts, first_signal)],
    ]
    buffer = BytesIO()
    pd.DataFrame(rows).to_excel(buffer, index=False, header=False)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../escape.xlsx", "/escape.xlsx", r"dir\a.xlsx", "C:a.xlsx", "a\0.xlsx"]
)
def test_upload_requires_simple_filename(name):
    with pytest.raises(ValueError, match="filename"):
        load_uploaded_bundle(((name, _workbook_bytes()),))


@pytest.mark.parametrize("names", [("same.xlsx", "same.xlsx"), ("same.xlsx", "SAME.xlsx")])
def test_duplicate_upload_names_are_rejected_before_parsing(monkeypatch, names):
    def unexpected_load(paths):
        pytest.fail("An ambiguous upload must be rejected before parsing.")

    monkeypatch.setattr(
        "sensd_sers_analysis.application.dataset_pipeline.load_sers_data_as_wide_and_tidy",
        unexpected_load,
    )
    with pytest.raises(ValueError, match="Duplicate upload filename"):
        load_uploaded_bundle(((names[0], b"first"), (names[1], b"second")))


def test_extra_metadata_survives_tidy_conversion_and_filtering():
    loaded = load_uploaded_bundle((("sample.xlsx", _workbook_bytes()),))
    assert loaded.load_report.loaded_files == ("sample.xlsx",)
    assert loaded.wide_df["lab_batch"].tolist() == ["batch-A", "batch-A"]
    assert loaded.tidy_df["lab_batch"].tolist() == ["batch-A"] * 6
    assert get_metadata_columns(loaded.wide_df)["lab_batch"].tolist() == ["batch-A"] * 2
    derived = build_derived_bundle(loaded)
    filtered = apply_filters(
        derived, {"lab_batch": FilterSelection(selected_values=("batch-A",), exclude=True)}
    )
    assert filtered.filtered_tidy_df.empty
    assert filtered.filtered_features_df.empty
    assert filtered.n_unique_spectra == 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"shifts": (100.0, np.inf, 103.0)}, "Raman shifts must be finite"),
        ({"shifts": (100.0, np.nan, 103.0)}, "Missing Raman shift.*signal data"),
        ({"shifts": (100.001, 100.002, 103.0)}, "Raman shifts collide"),
        ({"first_signal": (1.0, np.inf, 4.0)}, "Non-finite.*signals"),
        ({"first_signal": (1.0, np.nan, 4.0)}, "Non-finite.*signals"),
    ],
)
def test_invalid_spectral_values_are_reported_per_file(kwargs, message):
    loaded = load_uploaded_bundle((("invalid.xlsx", _workbook_bytes(**kwargs)),))
    assert loaded.wide_df.empty
    assert loaded.tidy_df.empty
    assert loaded.load_report.n_skipped == 1
    assert loaded.load_report.skipped_files[0][0] == "invalid.xlsx"
    assert re.search(message, loaded.load_report.skipped_files[0][1])


def test_fully_blank_spectral_separator_has_no_measurement_to_drop():
    loaded = load_uploaded_bundle(
        (
            (
                "separator.xlsx",
                _workbook_bytes(shifts=(100.0, np.nan, 103.0), first_signal=(1.0, np.nan, 4.0)),
            ),
        )
    )
    assert loaded.load_report.loaded_files == ("separator.xlsx",)
    assert loaded.wide_df[["rs_100.00", "rs_103.00"]].to_numpy().tolist() == [[1, 4], [2, 5]]


@pytest.mark.parametrize(
    "label", ["Target", "Intensity", "rs 100", "Integral Area", "peak near 501 8"]
)
def test_metadata_cannot_overwrite_spectra_or_computed_fields(label):
    loaded = load_uploaded_bundle(
        (("reserved.xlsx", _workbook_bytes(extra_metadata=((label, "raw-value"),))),)
    )
    assert loaded.wide_df.empty
    assert loaded.load_report.n_skipped == 1
    assert "Reserved or unusable" in loaded.load_report.skipped_files[0][1]


@pytest.mark.parametrize("label", ["Lab Batch", "Lab-Batch"])
def test_duplicate_metadata_fields_are_reported(label):
    loaded = load_uploaded_bundle(
        (("duplicate.xlsx", _workbook_bytes(extra_metadata=((label, "batch-B"),))),)
    )
    assert loaded.wide_df.empty
    assert "Duplicate" in loaded.load_report.skipped_files[0][1]
