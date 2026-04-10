"""Tests for serotype canonicalization in metadata preprocessing."""

from __future__ import annotations

import unittest

import pandas as pd

from sensd_sers_analysis.processing.metadata import (
    preprocess_metadata,
    sorted_unique_canonical_serotypes,
)


class SerotypeCanonicalizationTests(unittest.TestCase):
    """Serotype labels are merged case-insensitively."""

    def test_sorted_unique_canonical_serotypes_dedupes_case(self) -> None:
        df = pd.DataFrame({"serotype": ["st", "ST", "se", "SE", "st"]})
        self.assertEqual(sorted_unique_canonical_serotypes(df), ["SE", "ST"])

    def test_preprocess_metadata_uppercases_serotype(self) -> None:
        df = pd.DataFrame(
            {
                "serotype": ["abc", "ABC", "xYz"],
                "concentration": [100, 100, 100],
            }
        )
        out = preprocess_metadata(df)
        self.assertListEqual(out["serotype"].tolist(), ["ABC", "ABC", "XYZ"])

    def test_preprocess_metadata_blank_serotype_becomes_na(self) -> None:
        df = pd.DataFrame(
            {
                "serotype": ["  ", "nan", "ST"],
                "concentration": [1, 1, 1],
            }
        )
        out = preprocess_metadata(df)
        self.assertTrue(pd.isna(out["serotype"].iloc[0]))
        self.assertTrue(pd.isna(out["serotype"].iloc[1]))
        self.assertEqual(out["serotype"].iloc[2], "ST")


if __name__ == "__main__":
    unittest.main()
