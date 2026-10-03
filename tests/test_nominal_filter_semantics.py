"""Nominal grouping and filters preserve measured values and sample identity."""

import numpy as np
import pandas as pd

from sensd_sers_analysis.assessment import get_zero_cfu_baseline
from sensd_sers_analysis.processing.filters import get_filterable_columns, get_filter_options
from sensd_sers_analysis.processing.metadata import preprocess_metadata, sample_type_masks


def test_nominal_groups_never_rebin_measured_values():
    """Measured zero remains bacterial, and off-target counts keep intended dose."""
    frame = pd.DataFrame(
        {
            "target_concentration": [1, 100, 100, 100, 250, np.nan, 0],
            "concentration": [0, 78, 180, 230, 900, 1000, 0],
            "sample_type": ["Bacteria sample"] * 6 + ["Rinsate control"],
            "integral_area": [9, 10, 11, 12, 13, 14, 2],
        }
    )
    result = preprocess_metadata(frame)
    expected = ["1 CFU", "100 CFU", "100 CFU", "100 CFU", "250 CFU", "Unknown", "0 CFU"]
    assert result["target_concentration_group"].tolist() == expected
    assert result["concentration_group"].tolist() == expected
    np.testing.assert_allclose(result["concentration"], frame["concentration"])
    controls, bacteria = sample_type_masks(result)
    assert bacteria.iloc[0] and not controls.iloc[0]
    assert np.isclose(get_zero_cfu_baseline(result, "integral_area"), 2)


def test_sidebar_uses_initial_target_and_hides_derived_groups():
    """The first five filters retain the intended explicit nominal dimension."""
    frame = pd.DataFrame(
        {
            "serotype": ["ST"] * 3,
            "target_concentration": [1000, 1, 100],
            "date": ["2026-01-01"] * 3,
            "sensor_id": ["A"] * 3,
            "test_id": [1] * 3,
            "concentration": [950, 0, 230],
            "concentration_group": ["1 CFU"] * 3,
            "target_concentration_group": ["1 CFU"] * 3,
            "log_concentration": [0] * 3,
        }
    )
    columns = get_filterable_columns(frame)
    assert columns[:5] == ["serotype", "target_concentration", "date", "sensor_id", "test_id"]
    assert not {"concentration_group", "target_concentration_group", "log_concentration"} & set(
        columns
    )
    assert get_filter_options(frame, columns, {})["target_concentration"] == ["1", "100", "1000"]
