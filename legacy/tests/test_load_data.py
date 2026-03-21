import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.sensd_sers_analysis.data.sers_io import (
    load_dataset_by_serotypes,
    load_dataset_by_filenames,
)


class TestSERSDataLoading(unittest.TestCase):
    def setUp(self):
        # Define paths to test data
        self.data_folder = "../example_data/SERS Data 7 (Mar 2025)"
        self.signals_folders = ["SERS signals-Adheesha"]
        self.serotypes = ["SE", "ST"]

    def test_load_dataset_by_serotypes_structure(self):
        """Test that load_dataset_by_serotypes returns the expected DataFrame structure."""
        try:
            df = load_dataset_by_serotypes(self.data_folder, self.signals_folders, self.serotypes)

            # Check that we get a DataFrame
            self.assertIsInstance(df, pd.DataFrame)

            if not df.empty:  # Only test if we have data
                # Check required columns
                required_columns = {
                    "raman_shift",
                    "signal",
                    "concentration",
                    "log_concentration",
                    "concentration_level",
                    "serotype",
                    "sensor_id",
                    "test_id",
                    "filename",
                    "signal_index",
                }
                self.assertTrue(required_columns.issubset(set(df.columns)))

                # Check data types
                self.assertIsInstance(df["raman_shift"].iloc[0], np.ndarray)
                self.assertIsInstance(df["signal"].iloc[0], np.ndarray)
                self.assertIsInstance(df["concentration"].iloc[0], (int, float))
                self.assertIsInstance(df["log_concentration"].iloc[0], (int, float))
                self.assertIsInstance(df["concentration_level"].iloc[0], str)
                self.assertIsInstance(df["serotype"].iloc[0], str)

        except FileNotFoundError:
            self.skipTest("Test data not available")

    def test_load_dataset_by_filenames_structure(self):
        """Test that load_dataset_by_filenames returns the expected DataFrame structure."""
        try:
            # This is a simplified test since we need actual file names
            # In a real test, you would provide actual filenames
            file_list = []  # Empty list for now
            df = load_dataset_by_filenames(self.data_folder, self.signals_folders[0], file_list)

            # Check that we get a DataFrame
            self.assertIsInstance(df, pd.DataFrame)

        except FileNotFoundError:
            self.skipTest("Test data not available")

    def test_data_consistency(self):
        """Test that loaded data has consistent shapes and types."""
        try:
            df = load_dataset_by_serotypes(self.data_folder, self.signals_folders, self.serotypes)

            if len(df) > 1:
                # Check that all entries have consistent raman_shift
                first_raman_shift = df["raman_shift"].iloc[0]
                for idx in range(1, len(df)):
                    self.assertTrue(np.array_equal(df["raman_shift"].iloc[idx], first_raman_shift))

                # Check that signals have consistent shape
                first_signal_shape = df["signal"].iloc[0].shape
                for idx in range(1, len(df)):
                    self.assertEqual(
                        df["signal"].iloc[idx].shape[0], first_signal_shape[0]
                    )  # Same number of raman shifts

        except FileNotFoundError:
            self.skipTest("Test data not available")


if __name__ == "__main__":
    unittest.main()
