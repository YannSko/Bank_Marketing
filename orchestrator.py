import logging
import yaml
import mlflow
from pipeline.data_ingestion import load_raw_data
from pipeline.data_preprocessing import preprocess_data
from pipeline.feature_engineering import feature_engineering
from train import train_and_log_model, load_data
from mlflow_config import MLflowConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_config(config_path):
    """
    Load the configuration file.
    """
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise

def get_current_model_metrics(model_name, stage="Production"):
    """
    Fetch the current production model's metrics from the MLflow Model Registry.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        model_versions = client.get_latest_versions(model_name, stages=[stage])
        if not model_versions:
            return None  # No current model in production
        run_id = model_versions[0].run_id
        metrics = mlflow.get_run(run_id).data.metrics
        logging.info(f"Fetched metrics for the current production model: {metrics}")
        return metrics
    except Exception as e:
        logging.warning(f"Could not fetch metrics for the current production model: {e}")
        return None

def compare_models(current_model_metrics, new_model_metrics):
    """
    Compare the current production model metrics with the new model's metrics.
    """
    if not current_model_metrics:
        logging.info("No current model found. Automatically promoting new model.")
        return True
    return new_model_metrics["roc_auc"] > current_model_metrics["roc_auc"]

def run_pipeline(config_path="config_model.yaml"):
    """
    Orchestrate the entire MLOps pipeline.
    """
    try:
        # Load configuration
        config = load_config(config_path)

        # Initialize MLflow configuration
        mlflow_config = MLflowConfig(
            tracking_uri=config["mlflow"]["tracking_uri"],
            experiment_name=config["mlflow"]["experiment_name"]
        )

        # Step 1: Data ingestion
        load_raw_data(
            config["data"]["raw_train_path"],
            config["data"]["raw_test_path"],
            config["data"]["processed_train_path"],
            config["data"]["processed_test_path"]
        )

        # Step 2: Data preprocessing
        preprocess_data(
            config["data"]["processed_train_path"],
            config["data"]["preprocessed_train_path"],
            config["data"]["preprocessed_test_path"]
        )

        # Step 3: Feature engineering
        feature_engineering(
            config["data"]["preprocessed_train_path"],
            config["data"]["preprocessed_test_path"],
            config["data"]["engineered_train_path"],
            config["data"]["engineered_test_path"]
        )

        # Step 4: Model training
        X_train, X_test, y_train, y_test = load_data(
            config["data"]["engineered_train_path"],
            config["data"]["engineered_test_path"]
        )

        # Check if hyperparameter tuning is enabled
        if config["training"]["use_tuning"]:
            search_method = config["training"]["search_method"]
            param_grid = config["training"]["param_grid"]
            new_model, new_model_metrics = train_and_log_model(
                X_train, y_train, X_test, y_test,
                model_type=config["training"]["model_type"],
                use_tuning=True,
                search_method=search_method,
                param_grid=param_grid,
                mlflow_config=mlflow_config
            )
        else:
            params = config["training"]["direct_hyperparameters"]
            new_model, new_model_metrics = train_and_log_model(
                X_train, y_train, X_test, y_test,
                model_type=config["training"]["model_type"],
                use_tuning=False,
                params=params,
                mlflow_config=mlflow_config
            )

        # Step 5: Compare models and update registry
        current_model_metrics = get_current_model_metrics(config["training"]["model_type"])
        if compare_models(current_model_metrics, new_model_metrics):
            logging.info("New model is better. Promoting to Production.")
            mlflow_config.log_model_to_registry(new_model, config["training"]["model_type"], stage="Production")
        else:
            logging.info("Current production model is better. No changes made.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()
