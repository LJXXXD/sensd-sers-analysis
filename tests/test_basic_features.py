"""Integrated area retains its mathematical meaning when other rows are incomplete."""

import numpy as np
import pandas as pd

from sensd_sers_analysis.processing.features import extract_basic_features


def test_missing_spectrum_does_not_change_valid_integrals(caplog):
    """Constant and piecewise-linear curves retain their analytical areas."""
    frame = pd.DataFrame(
        {
            "sensor_id": ["constant", "incomplete", "linear_segments"],
            "rs_500": [2.0, 1.0, 3.0],
            "rs_600": [2.0, np.nan, 4.0],
            "rs_700": [2.0, 3.0, 7.0],
        },
        index=[4, 9, 17],
    )
    original = frame.copy(deep=True)

    result = extract_basic_features(frame)

    np.testing.assert_allclose(result.integral_area, [400.0, np.nan, 900.0])
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_series_equal(result.sensor_id, frame.sensor_id)
    assert "Integral area unavailable for 1 spectra" in caplog.text


def test_single_coordinate_has_no_assessable_integral(caplog):
    """One point provides no interval over which to integrate a spectrum."""
    frame = pd.DataFrame({"rs_500": [2.0, 4.0]})

    result = extract_basic_features(frame)

    assert result.integral_area.isna().all()
    np.testing.assert_allclose(result.max_intensity, [2.0, 4.0])
    assert "Integral area unavailable for 2 spectra" in caplog.text
