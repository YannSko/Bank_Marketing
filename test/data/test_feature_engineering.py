import unittest
import pandas as pd
import numpy as np
import os
from pipeline.feature_engineering import feature_engineering

class TestFeatureEngineering(unittest.TestCase):
    """
    Unit tests for the feature_engineering function.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment by creating synthetic datasets for testing.
        """
        # Create synthetic training data
        cls.train_data = pd.DataFrame({
            'duration': [100, 200, 300],
            'campaign': [1, 2, 3],
            'previous': [0, 1, 5],
            'emp.var.rate': [1.1, -0.1, -1.1],
            'cons.price.idx': [93.994, 92.893, 92.201],
            'cons.conf.idx': [-36.4, -40.0, -42.7],
            'euribor3m': [4.857, 1.313, 0.634],
            'nr.employed': [5191, 5099, 5017],
            'age': [30, 45, 65],
            'pdays': [999, 999, 999]
        })

        # Create synthetic testing data
        cls.test_data = pd.DataFrame({
            'duration': [150, 250, 350],
            'campaign': [2, 3, 4],
            'previous': [1, 2, 6],
            'emp.var.rate': [1.1, -0.3, -1.5],
            'cons.price.idx': [93.994, 92.500, 91.994],
            'cons.conf.idx': [-36.4, -40.3, -43.5],
            'euribor3m': [4.857, 1.400, 0.500],
            'nr.employed': [5191, 5089, 5000],
            'age': [35, 50, 75],
            'pdays': [999, 999, 999]
        })

        # Paths for temporary files
        cls.train_path = "./test_train.csv"
        cls.test_path = "./test_test.csv"
        cls.output_train_path = "./engineered_test_train.csv"
        cls.output_test_path = "./engineered_test_test.csv"

        # Save synthetic data to files
        cls.train_data.to_csv(cls.train_path, index=False)
        cls.test_data.to_csv(cls.test_path, index=False)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up files after tests are run.
        """
        for path in [cls.train_path, cls.test_path, cls.output_train_path, cls.output_test_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_feature_engineering(self):
        """
        Test the main feature engineering function for correctness.
        """
        train_df, test_df = feature_engineering(
            self.train_path,
            self.test_path,
            self.output_train_path,
            self.output_test_path
        )

        # Check that files were created
        self.assertTrue(os.path.exists(self.output_train_path))
        self.assertTrue(os.path.exists(self.output_test_path))

        # Reload the engineered data
        engineered_train = pd.read_csv(self.output_train_path)
        engineered_test = pd.read_csv(self.output_test_path)

        # Check for added features
        self.assertIn('interaction_employed_duration', engineered_train.columns)
        self.assertIn('age_group', engineered_train.columns)
        self.assertNotIn('pdays', engineered_train.columns)

        # Check interaction feature values
        np.testing.assert_array_almost_equal(
            engineered_train['interaction_employed_duration'].values,
            self.train_data['duration'] * self.train_data['nr.employed']
        )

        # Check age binning
        self.assertTrue(set(engineered_train['age_group'].unique()).issubset({'Young', 'Mid-age', 'Senior', 'Elderly'}))

    def test_log_transform(self):
        """
        Verify log transformation on skewed features.
        """
        train_df, test_df = feature_engineering(
            self.train_path,
            self.test_path,
            self.output_train_path,
            self.output_test_path
        )

        # Check log transformation
        for feature in ['duration', 'campaign', 'previous']:
            self.assertTrue((train_df[feature] >= 0).all())
            self.assertTrue((test_df[feature] >= 0).all())

    def test_standardization(self):
        """
        Verify that economic features are standardized.
        """
        train_df, test_df = feature_engineering(
            self.train_path,
            self.test_path,
            self.output_train_path,
            self.output_test_path
        )

        # Check that standardized features have mean ~0 and std ~1
        for feature in ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']:
            mean = train_df[feature].mean()
            std = train_df[feature].std()
            self.assertAlmostEqual(mean, 0, places=5)
            self.assertAlmostEqual(std, 1, places=5)

    def test_error_handling(self):
        """
        Test the function's ability to handle missing or incorrect paths.
        """
        with self.assertRaises(Exception):
            feature_engineering(
                "non_existent_train.csv",
                self.test_path,
                self.output_train_path,
                self.output_test_path
            )

if __name__ == "__main__":
    unittest.main()
