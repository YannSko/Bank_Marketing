import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO
import os
from pipeline.data_ingestion import load_raw_data, preprocess_data, save_cleaned_data


class TestDataIngestion(unittest.TestCase):
    """
    Unit tests for the data_ingestion module.
    """

    def setUp(self):
        """
        Set up mock data for testing.
        """
        self.raw_csv = StringIO("""age;job;marital;education;default;housing;loan;contact;month;day_of_week;duration;campaign;pdays;previous;poutcome;emp.var.rate;cons.price.idx;cons.conf.idx;euribor3m;nr.employed;y
56;admin.;married;unknown;no;yes;no;cellular;may;mon;261;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191;no
57;services;married;high.school;unknown;yes;unknown;cellular;may;mon;149;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191;no
37;services;married;high.school;no;yes;no;cellular;may;mon;226;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191;yes
""")
        self.raw_data = pd.read_csv(self.raw_csv, sep=';')
        self.cleaned_data_path = "./data/processed/test_cleaned_data.csv"

    @patch("data_ingestion.logging.info")
    def test_load_raw_data(self, mock_logging):
        """
        Test that raw data is loaded correctly from a valid CSV file.
        """
        with patch("builtins.open", MagicMock()) as mock_file:
            mock_file.return_value.__enter__.return_value = self.raw_csv
            df = load_raw_data("mock_path.csv")
            self.assertEqual(df.shape, self.raw_data.shape)
            self.assertTrue("age" in df.columns)
            mock_logging.assert_called_with("Raw data successfully loaded with shape: (3, 21).")

    def test_preprocess_data(self):
        """
        Test preprocessing of raw data: handling missing values and encoding.
        """
        processed_data = preprocess_data(self.raw_data)
        # Check shape remains unchanged
        self.assertEqual(processed_data.shape, self.raw_data.shape)
        # Ensure all 'unknown' are replaced with NaN and imputed
        self.assertFalse(processed_data.isnull().any().any())
        # Check categorical columns are encoded
        self.assertTrue((processed_data.dtypes == float).all())

    def test_save_cleaned_data(self):
        """
        Test saving the cleaned data to a CSV file.
        """
        if os.path.exists(self.cleaned_data_path):
            os.remove(self.cleaned_data_path)  # Ensure a clean slate
        save_cleaned_data(self.raw_data, self.cleaned_data_path)
        self.assertTrue(os.path.exists(self.cleaned_data_path))
        saved_data = pd.read_csv(self.cleaned_data_path)
        pd.testing.assert_frame_equal(saved_data, self.raw_data)

    @patch("data_ingestion.logging.error")
    def test_load_raw_data_error(self, mock_logging):
        """
        Test error handling when loading invalid data.
        """
        with self.assertRaises(Exception):
            load_raw_data("invalid_path.csv")
        mock_logging.assert_called()

    @patch("data_ingestion.logging.error")
    def test_preprocess_data_error(self, mock_logging):
        """
        Test error handling during preprocessing.
        """
        with self.assertRaises(Exception):
            preprocess_data(None)
        mock_logging.assert_called()

    @patch("data_ingestion.logging.error")
    def test_save_cleaned_data_error(self, mock_logging):
        """
        Test error handling during saving data.
        """
        with self.assertRaises(Exception):
            save_cleaned_data(None, "invalid_path/test.csv")
        mock_logging.assert_called()

    def tearDown(self):
        """
        Clean up any files created during testing.
        """
        if os.path.exists(self.cleaned_data_path):
            os.remove(self.cleaned_data_path)


if __name__ == "__main__":
    unittest.main()
