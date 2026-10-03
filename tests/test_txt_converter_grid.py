"""Uploaded TXT grids must match before converter merging."""

from pathlib import Path

import numpy as np


def test_unequal_grid_lengths_return_filename_error(monkeypatch):
    """Different point counts produce a mismatch message rather than broadcasting failure."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "apps"))
    from sensd_sers_analysis.data.txt_converter import validate_common_shift

    grid, error = validate_common_shift(
        {
            "two_points.txt": "500\t1\n600\t2\n",
            "three_points.txt": "500\t3\n600\t4\n700\t5\n",
        }
    )

    assert grid is None
    assert error == "Raman shift mismatch in file: three_points.txt"


def test_equal_grids_accept_different_intensities(monkeypatch):
    """Matching coordinates are mergeable regardless of their measured intensities."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "apps"))
    from sensd_sers_analysis.data.txt_converter import validate_common_shift

    grid, error = validate_common_shift(
        {"first.txt": "500\t1\n600\t2\n", "second.txt": "500\t8\n600\t9\n"}
    )

    assert error is None
    np.testing.assert_allclose(grid, [500.0, 600.0])


def test_txt_measurements_are_not_silently_dropped():
    import pytest
    from sensd_sers_analysis.data.txt_converter import parse_txt_content

    for content in [
        "500\t1\n600\tbad\n",
        "500\t1\n\t2\n",
        "500\t1\n600\tinf\n",
        "500\t1\t2\n600\t3\t4\n",
        "500\t1\n500\t2\n",
    ]:
        with pytest.raises(ValueError):
            parse_txt_content(content)


def test_common_grid_tolerance_has_no_relative_slack():
    from sensd_sers_analysis.data.txt_converter import validate_common_shift

    grid, error = validate_common_shift(
        {"a.txt": "1500\t1\n1600\t2\n", "b.txt": "1500.01\t3\n1600.01\t4\n"}
    )
    assert grid is None
    assert error == "Raman shift mismatch in file: b.txt"


def test_trim_must_retain_real_coordinates_and_bounds_are_finite():
    from sensd_sers_analysis.data.txt_converter import merge_txt_spectra

    inputs = {"a.txt": "500\t1\n600\t2\n"}
    for lower, upper in [(700.0, 800.0), (np.nan, 700.0), (550.0, 650.0)]:
        grid, signals, error = merge_txt_spectra(inputs, min_shift=lower, max_shift=upper)
        assert grid is None and signals is None and error


def test_preset_numbers_and_compact_time_do_not_hide_invalid_values():
    from sensd_sers_analysis.data.txt_converter import (
        apply_template_import_to_values,
        parse_metadata_time,
    )

    for text in ["12345pm", "text3pm", "PM", "270pm", "13pm"]:
        assert parse_metadata_time(text) is None
    values, warnings = apply_template_import_to_values(
        {
            "version": 1,
            "fields": {
                "txt2excel_meta_disk_diameter_nm": np.inf,
                "txt2excel_meta_date": 123,
                "txt2excel_meta_testing_time": 123,
            },
        },
        {},
    )
    assert values == {}
    assert len(warnings) == 3


def test_instrument_provenance_header_is_supported_without_dropping_bad_rows():
    from sensd_sers_analysis.data.txt_converter import parse_txt_content
    import pytest

    frame = parse_txt_content("Data from specimen.txt Node\n\n501.2\t3\n502.3\t4\n")
    assert frame["Value"].tolist() == [3.0, 4.0]
    with pytest.raises(ValueError):
        parse_txt_content("Data from specimen.txt Node\n501.2\t3\n502.3\tbroken\n")
