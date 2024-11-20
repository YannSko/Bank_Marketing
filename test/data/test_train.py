import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from models.training.train import load_data, hyperparameter_search, train_and_log_model, MLflowConfig
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class TestTrainModule(unittest.TestCase):
    """
    Unit tests for train.py functions.
    """

    @patch("train.pd.read_csv")
    def test_load_data(self, mock_read_csv):
        """
        Test the load_data function to ensure it correctly loads train and test datasets.
        """
        # Mock train and test data
        train_data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "y": [0, 1, 0]
        })
        test_data = pd.DataFrame({
            "feature1": [7, 8, 9],
            "feature2": [10, 11, 12],
            "y": [1, 0, 1]
        })

        # Mock pd.read_csv return values
        mock_read_csv.side_effect = [train_data, test_data]

        # Call the function
        X_train, X_test, y_train, y_test = load_data("train.csv", "test.csv")

        # Assertions
        pd.testing.assert_frame_equal(X_train, train_data.drop(columns=["y"]))
        pd.testing.assert_frame_equal(X_test, test_data.drop(columns=["y"]))
        pd.testing.assert_series_equal(y_train, train_data["y"])
        pd.testing.assert_series_equal(y_test, test_data["y"])

    def test_hyperparameter_search(self):
        """
        Test the hyperparameter_search function for both grid and bayesian search.
        """
        X_train = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8]})
        y_train = pd.Series([0, 1, 0, 1])

        # Define parameter grid
        param_grid = {
            "max_depth": [2, 3],
            "n_estimators": [10, 20]
        }

        # Test Grid Search
        best_model, best_params = hyperparameter_search(
            X_train, y_train, model_type="xgboost", search_method="grid", param_grid=param_grid, cv_splits=2
        )
        self.assertIsInstance(best_model, XGBClassifier)
        self.assertIn("max_depth", best_params)
        self.assertIn("n_estimators", best_params)

    @patch("train.mlflow.start_run")
    @patch("train.mlflow.sklearn.log_model")
    def test_train_and_log_model(self, mock_log_model, mock_start_run):
        """
        Test the train_and_log_model function for training, logging, and evaluation.
        """
        X_train = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8]})
        y_train = pd.Series([0, 1, 0, 1])
        X_test = pd.DataFrame({"feature1": [9, 10], "feature2": [11, 12]})
        y_test = pd.Series([0, 1])

        # Initialize MLflowConfig
        mlflow_config = MLflowConfig(tracking_uri="./mlruns", experiment_name="test_experiment")

        # Call the function
        best_model, metrics = train_and_log_model(
            X_train,
            y_train,
            X_test,
            y_test,
            model_type="xgboost",
            use_tuning=False,
            params={"max_depth": 3, "n_estimators": 10},
            mlflow_config=mlflow_config
        )

        # Assertions
        self.assertIsInstance(best_model, XGBClassifier)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertGreaterEqual(metrics["f1_score"], 0)
        self.assertGreaterEqual(metrics["roc_auc"], 0)
        mock_start_run.assert_called_once()
        mock_log_model.assert_called_once()

    def test_MLflowConfig(self):
        """
        Test the MLflowConfig class to ensure it sets up MLflow correctly.
        """
        with patch("train.mlflow.set_tracking_uri") as mock_set_uri, patch("train.mlflow.set_experiment") as mock_set_experiment:
            config = MLflowConfig(tracking_uri="test_uri", experiment_name="test_experiment")
            config.initialize()
            mock_set_uri.assert_called_with("test_uri")
            mock_set_experiment.assert_called_with("test_experiment")


if __name__ == "__main__":
    unittest.main()
