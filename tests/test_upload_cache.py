"""Regression coverage for upload cache identity and missing sample metadata."""

import importlib
import sys
from pathlib import Path

import pandas as pd

from sensd_sers_analysis.application.contracts import LoadedDataBundle
from sensd_sers_analysis.processing.metadata import sample_type_masks


def test_missing_sample_type_is_unknown_not_rinsate():
    """An older cached frame without Sample Type remains inspectable and unknown."""
    frame = pd.DataFrame({"concentration": [0, 1000]}, index=[7, 9])
    controls, bacteria = sample_type_masks(frame)
    assert controls.index.equals(frame.index)
    assert int((~(controls | bacteria)).sum()) == 2
    assert not controls.any() and not bacteria.any()


def test_upload_cache_keys_include_names_and_contents(monkeypatch):
    """Different files and changed bytes reload; identical uploads reuse parsing."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))
    module = importlib.import_module("components.data_loading")
    calls = []

    def load(files_data):
        """Return distinguishable bundles without workbook I/O."""
        calls.append(files_data)
        frame = pd.DataFrame({"filename": [files_data[0][0]], "content": [files_data[0][1]]})
        return LoadedDataBundle(wide_df=frame, tidy_df=frame.copy())

    monkeypatch.setattr(module, "load_uploaded_bundle", load)
    module.load_from_uploaded.clear()
    try:
        first = module.load_from_uploaded((("a.xlsx", b"old"),))
        repeat = module.load_from_uploaded((("a.xlsx", b"old"),))
        changed = module.load_from_uploaded((("a.xlsx", b"new"),))
        renamed = module.load_from_uploaded((("b.xlsx", b"new"),))
        empty = module.load_from_uploaded(())
        assert len(calls) == 3
        pd.testing.assert_frame_equal(first.wide_df, repeat.wide_df)
        assert changed.wide_df.iloc[0]["content"] == b"new"
        assert renamed.wide_df.iloc[0]["filename"] == "b.xlsx"
        assert empty.wide_df.empty
    finally:
        module.load_from_uploaded.clear()
