import os
import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pipeline.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    """
    Unit tests for the data_preprocessing module.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment by creating mock datasets and directories.
        """
        # Mock input dataset
        cls.mock_data = pd.DataFrame({
            "age": [25, 35, 45, np.nan],
            "income": [50000, 60000, np.nan, 80000],
            "job": ["admin", "technician", "unknown", "blue-collar"],
            "y": [0, 1, 0, 1]
        })

        cls.mock_input_path = "mock_cleaned_data.csv"
        cls.mock_output_train_path = "mock_train_data.csv"
        cls.mock_output_test_path = "mock_test_data.csv"

        # Save mock input dataset to CSV
        cls.mock_data.to_csv(cls.mock_input_path, index=False)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up test environment by removing mock files and directories.
        """
        if os.path.exists(cls.mock_input_path):
            os.remove(cls.mock_input_path)
        if os.path.exists(cls.mock_output_train_path):
            os.remove(cls.mock_output_train_path)
        if os.path.exists(cls.mock_output_test_path):
            os.remove(cls.mock_output_test_path)

    def test_preprocess_data_output(self):
        """
        Test the output of preprocess_data to ensure it matches expectations.
        """
        # Call preprocess_data
        X_train, X_test, y_train, y_test = preprocess_data(
            input_path=self.mock_input_path,
            output_train_path=self.mock_output_train_path,
            output_test_path=self.mock_output_test_path
        )

        # Check that files are created
        self.assertTrue(os.path.exists(self.mock_output_train_path), "Train data file not created.")
        self.assertTrue(os.path.exists(self.mock_output_test_path), "Test data file not created.")

        # Check shape consistency
        train_df = pd.read_csv(self.mock_output_train_path)
        test_df = pd.read_csv(self.mock_output_test_path)
        self.assertEqual(train_df.shape[1], self.mock_data.shape[1], "Train data columns mismatch.")
        self.assertEqual(test_df.shape[1], self.mock_data.shape[1], "Test data columns mismatch.")

    def test_numeric_imputation(self):
        """
        Test if numeric missing values are imputed correctly.
        """
        _, X_test, _, _ = preprocess_data(
            input_path=self.mock_input_path,
            output_train_path=self.mock_output_train_path,
            output_test_path=self.mock_output_test_path
        )
        self.assertFalse(X_test.isnull().any().any(), "Numeric imputation failed.")

    def test_categorical_encoding(self):
        """
        Test if categorical columns are encoded correctly.
        """
        _, X_test, _, _ = preprocess_data(
            input_path=self.mock_input_path,
            output_train_path=self.mock_output_train_path,
            output_test_path=self.mock_output_test_path
        )
        self.assertFalse(X_test.select_dtypes(include=["object"]).any().any(), "Categorical encoding failed.")

    def test_scaling_numeric_columns(self):
        """
        Test if numeric columns are scaled correctly.
        """
        _, X_test, _, _ = preprocess_data(
            input_path=self.mock_input_path,
            output_train_path=self.mock_output_train_path,
            output_test_path=self.mock_output_test_path
        )
        # Ensure numeric columns have a mean close to 0 and std close to 1
        numeric_cols = X_test.select_dtypes(include=["float64", "int64"]).columns
        means = X_test[numeric_cols].mean()
        stds = X_test[numeric_cols].std()
        for col in numeric_cols:
            self.assertAlmostEqual(means[col], 0, delta=1e-1, msg=f"Column {col} mean not scaled.")
            self.assertAlmostEqual(stds[col], 1, delta=1e-1, msg=f"Column {col} std not scaled.")

    def test_split_data(self):
        """
        Test if the data is split into train and test sets correctly.
        """
        X_train, X_test, y_train, y_test = preprocess_data(
            input_path=self.mock_input_path,
            output_train_path=self.mock_output_train_path,
            output_test_path=self.mock_output_test_path
        )

        # Check that train-test split ratio is maintained
        total_records = len(y_train) + len(y_test)
        self.assertEqual(total_records, self.mock_data.shape[0], "Total records mismatch after split.")
        self.assertAlmostEqual(len(y_test) / total_records, 0.2, delta=0.05, msg="Train-test split ratio mismatch.")

if __name__ == "__main__":
    unittest.main()
