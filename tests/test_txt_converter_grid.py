"""Uploaded TXT grids must match before converter merging."""

from pathlib import Path

import numpy as np


def test_unequal_grid_lengths_return_filename_error(monkeypatch):
    """Different point counts produce a mismatch message rather than broadcasting failure."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "apps"))
    from txt_to_excel import _validate_common_shift

    grid, error = _validate_common_shift(
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
    from txt_to_excel import _validate_common_shift

    grid, error = _validate_common_shift(
        {"first.txt": "500\t1\n600\t2\n", "second.txt": "500\t8\n600\t9\n"}
    )

    assert error is None
    np.testing.assert_allclose(grid, [500.0, 600.0])
