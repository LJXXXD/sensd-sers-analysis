"""Exposure normalization preserves measurements and routes scaled features consistently."""

import numpy as np
import pandas as pd
import pytest

from sensd_sers_analysis.processing.normalization import normalize_integration_time
from sensd_sers_analysis.processing.features import extract_basic_features


def test_exposure_scaling_preserves_input_and_metadata():
    """Equal signals per unit time agree after scaling, including integrated area."""
    raw = pd.DataFrame(
        {
            "integration_time_ms": [100, 200, 300],
            "scan_average": [1, 2, 3],
            "rs_500": [2.0, 4.0, 6.0],
            "rs_600": [4.0, 8.0, 12.0],
        }
    )
    original = raw.copy(deep=True)
    scaled = normalize_integration_time(raw)
    np.testing.assert_allclose(scaled[["rs_500", "rs_600"]], [[2, 4]] * 3)
    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(scaled.iloc[:, :2], original.iloc[:, :2])
    features = extract_basic_features(scaled)
    np.testing.assert_allclose(features.integral_area, [300.0] * 3)


@pytest.mark.parametrize("value", [0, -100, None, "bad", np.inf])
def test_invalid_exposure_is_never_silently_mixed(value):
    """Invalid exposure values block normalization rather than retaining unscaled spectra."""
    with pytest.raises(ValueError, match="invalid integration time"):
        normalize_integration_time(pd.DataFrame({"integration_time_ms": [value], "rs_500": [1.0]}))


def test_pipeline_normalizes_before_shared_feature_extraction(monkeypatch):
    """The option scales spectra and basic features through the same source frame."""
    from sensd_sers_analysis.application import dataset_pipeline
    from sensd_sers_analysis.application.contracts import LoadedDataBundle

    raw = pd.DataFrame({"integration_time_ms": [200], "rs_500": [4.0], "rs_600": [8.0]})
    monkeypatch.setattr(
        dataset_pipeline,
        "extract_dynamic_peak_features",
        lambda *args, **kwargs: (pd.DataFrame(), {}, {}, None, np.array([500.0, 600.0])),
    )
    bundle = LoadedDataBundle(wide_df=raw, tidy_df=pd.DataFrame())
    original = dataset_pipeline.build_derived_bundle(bundle)
    scaled = dataset_pipeline.build_derived_bundle(bundle, normalize_exposure=True)
    np.testing.assert_allclose(
        scaled.features_df.integral_area, original.features_df.integral_area / 2
    )
    np.testing.assert_allclose(scaled.tidy_df.intensity, original.tidy_df.intensity / 2)
    np.testing.assert_allclose(bundle.wide_df.rs_500, [4.0])
