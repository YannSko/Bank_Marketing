import pandas as pd
import numpy as np
import logging
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from mlflow_config import MLflowConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log"),
        logging.StreamHandler()
    ]
)

def load_config(config_path):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_data(train_path, test_path):
    """
    Load the preprocessed training and testing datasets.
    """
    try:
        logging.info(f"Loading training data from {train_path}...")
        train_df = pd.read_csv(train_path)
        logging.info(f"Training data loaded successfully with shape: {train_df.shape}.")

        logging.info(f"Loading testing data from {test_path}...")
        test_df = pd.read_csv(test_path)
        logging.info(f"Testing data loaded successfully with shape: {test_df.shape}.")

        X_train = train_df.drop(columns=["y"])
        y_train = train_df["y"]
        X_test = test_df.drop(columns=["y"])
        y_test = test_df["y"]

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def plot_and_log_metrics(y_true, y_pred, model_name):
    """
    Plot and log additional evaluation metrics to MLflow.
    """
    try:
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve for {model_name}")
        plt.legend()
        plt.savefig("precision_recall_curve.png")
        mlflow.log_artifact("precision_recall_curve.png")
        plt.close()

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.crosstab(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

    except Exception as e:
        logging.error(f"Error plotting metrics: {e}")
        raise

def hyperparameter_search(X_train, y_train, config):
    """
    Perform hyperparameter tuning based on the configuration.
    """
    try:
        model_type = config["model"]["type"]
        search_method = config["search"]["method"]
        param_grid = config["search"]["param_grid"]
        cv_splits = config["search"]["cv_splits"]

        if model_type == "xgboost":
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        elif model_type == "random_forest":
            model = RandomForestClassifier(random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logging.info(f"Starting {search_method} search for {model_type}...")
        if search_method == "grid":
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=StratifiedKFold(n_splits=cv_splits),
                n_jobs=-1
            )
        elif search_method == "bayesian":
            search = BayesSearchCV(
                estimator=model,
                search_spaces=param_grid,
                scoring="roc_auc",
                cv=StratifiedKFold(n_splits=cv_splits),
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported search method: {search_method}")

        search.fit(X_train, y_train)

        logging.info(f"Best parameters: {search.best_params_}")
        logging.info(f"Best ROC AUC score during cross-validation: {search.best_score_:.4f}")

        return search.best_estimator_, search.best_params_

    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {e}")
        raise

def train_and_log_model(X_train, y_train, X_test, y_test, config):
    """
    Train and log a model using MLflow.
    """
    try:
        mlflow_config = MLflowConfig(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment_name"])
        mlflow_config.initialize()

        model_type = config["model"]["type"]
        use_tuning = config["search"]["use_tuning"]
        param_grid = config["search"]["param_grid"]

        with mlflow.start_run():
            if use_tuning:
                best_model, best_params = hyperparameter_search(X_train, y_train, config)
                params_to_log = best_params
            else:
                if model_type == "xgboost":
                    best_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, **param_grid)
                elif model_type == "random_forest":
                    best_model = RandomForestClassifier(random_state=42, **param_grid)
                elif model_type == "gradient_boosting":
                    best_model = GradientBoostingClassifier(random_state=42, **param_grid)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

                best_model.fit(X_train, y_train)
                params_to_log = param_grid

            # Evaluate and log metrics
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            mlflow.log_params(params_to_log)
            mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc})

            # Log model
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            # Plot and log metrics
            plot_and_log_metrics(y_test, y_pred, model_type)

            return best_model, {"accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc}

    except Exception as e:
        logging.error(f"Error during training and logging: {e}")
        raise

if __name__ == "__main__":
    config = load_config("config.yaml")
    X_train, X_test, y_train, y_test = load_data(config["data"]["train_path"], config["data"]["test_path"])
    train_and_log_model(X_train, y_train, X_test, y_test, config)
