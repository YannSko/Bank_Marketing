import unittest
import os
import yaml
import mlflow
from unittest.mock import patch, MagicMock
from models.experiments.mlflow_config import MLflowConfig


class TestMLflowConfig(unittest.TestCase):
    def setUp(self):
        # Create a mock config file for testing
        self.test_config_path = "test_mlflow_config.yaml"
        self.config_data = {
            "mlflow": {
                "tracking_uri": "http://127.0.0.1:5000",
                "experiment_name": "test_experiment",
                "models": {
                    "xgboost": {
                        "registry": "test_xgboost_registry",
                        "description": "Test XGBoost model"
                    }
                },
                "registry": {
                    "default_stage": "Staging",
                    "model_uri_format": "models:/<model_name>/<stage>"
                }
            }
        }
        with open(self.test_config_path, "w") as file:
            yaml.dump(self.config_data, file)

        self.mlflow_config = MLflowConfig(config_path=self.test_config_path)

    def tearDown(self):
        # Remove the mock config file
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def test_load_config(self):
        """Test if the configuration file is loaded correctly."""
        self.assertEqual(self.mlflow_config.config["mlflow"]["tracking_uri"], "http://127.0.0.1:5000")
        self.assertEqual(self.mlflow_config.config["mlflow"]["experiment_name"], "test_experiment")

    @patch("mlflow.set_tracking_uri")
    def test_set_tracking_uri(self, mock_set_tracking_uri):
        """Test if the tracking URI is set correctly."""
        self.mlflow_config._set_tracking_uri()
        mock_set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")

    @patch("mlflow.set_experiment")
    def test_set_experiment(self, mock_set_experiment):
        """Test if the experiment is set correctly."""
        self.mlflow_config.set_experiment()
        mock_set_experiment.assert_called_once_with("test_experiment")

    @patch("mlflow.sklearn.log_model")
    def test_log_model_to_registry(self, mock_log_model):
        """Test if the model is logged to the registry correctly."""
        mock_model =         MagicMock()  # Mock a model object for testing
        self.mlflow_config.log_model_to_registry(mock_model, model_name="xgboost", stage="Staging")
        
        mock_log_model.assert_called_once_with(
            sk_model=mock_model,
            artifact_path="models/xgboost",
            registered_model_name="test_xgboost_registry"
        )

    def test_get_model_uri(self):
        """Test if the correct model URI is returned."""
        uri = self.mlflow_config.get_model_uri("xgboost", stage="Production")
        self.assertEqual(uri, "models:/xgboost/Production")

    def test_get_model_uri_default_stage(self):
        """Test if the default stage is used when no stage is specified."""
        uri = self.mlflow_config.get_model_uri("xgboost")
        self.assertEqual(uri, "models:/xgboost/Staging")

    def test_log_model_to_registry_invalid_model_name(self):
        """Test if logging a model with an invalid name raises an error."""
        with self.assertRaises(ValueError):
            self.mlflow_config.log_model_to_registry(MagicMock(), model_name="invalid_model")

    def test_missing_config_file(self):
        """Test if a missing configuration file raises an error."""
        with self.assertRaises(FileNotFoundError):
            MLflowConfig(config_path="non_existent_config.yaml")

    def test_invalid_model_uri_format(self):
        """Test if an invalid model URI format raises an error."""
        self.mlflow_config.config["mlflow"]["registry"]["model_uri_format"] = None
        with self.assertRaises(TypeError):
            self.mlflow_config.get_model_uri("xgboost")

if __name__ == "__main__":
    unittest.main()

