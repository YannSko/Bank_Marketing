import yaml
import mlflow
import os

class MLflowConfig:
    def __init__(self, config_path="mlflow_config.yaml"):
        self.config = self._load_config(config_path)
        self._set_tracking_uri()
        self.experiment_name = self.config["mlflow"]["experiment_name"]
        self.models = self.config["mlflow"]["models"]
        self.registry = self.config["mlflow"]["registry"]

    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file '{path}' not found.")
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def _set_tracking_uri(self):
        tracking_uri = self.config["mlflow"]["tracking_uri"]
        mlflow.set_tracking_uri(tracking_uri)

    def set_experiment(self):
        """
        Set the MLflow experiment from the config.
        If the experiment doesn't exist, it will be created.
        """
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"Error setting experiment: {e}")
            raise

    def log_model_to_registry(self, model, model_name, description=None, stage="None"):
        """
        Log a model to the MLflow model registry.
        """
        if model_name not in self.models:
            raise ValueError(f"Model name '{model_name}' not found in configuration.")

        registry_info = self.models[model_name]
        registry_name = registry_info.get("registry", model_name)
        description = description or registry_info.get("description", f"{model_name} model")

        # Log the model to the registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"models/{model_name}",
            registered_model_name=registry_name
        )

        print(f"Model '{model_name}' logged to the registry.")

    def get_model_uri(self, model_name, stage=None):
        """
        Get the URI of a model in a specific stage.
        """
        stage = stage or self.registry["default_stage"]
        uri_format = self.registry.get("model_uri_format", "models:/<model_name>/<stage>")
        return uri_format.replace("<model_name>", model_name).replace("<stage>", stage)


if __name__ == "__main__":
    # Example Usage
    config = MLflowConfig()

    # Set the experiment
    config.set_experiment()

    # Example to log a model to the registry (during training)
    # model = trained_model (from train.py)
    # config.log_model_to_registry(model, model_name="xgboost", stage="Staging")
    
    # Fetch model URI for deployment
    xgboost_production_uri = config.get_model_uri("bank_marketing_xgboost", stage="Production")
    print(f"XGBoost Production URI: {xgboost_production_uri}")

