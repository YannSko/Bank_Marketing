import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/data_preprocessing.log"),
        logging.StreamHandler()
    ]
)

def preprocess_data(input_path, output_train_path, output_test_path):
    """
    Preprocess the cleaned data by imputing missing values, encoding categorical features,
    scaling numeric features, and splitting into training and testing datasets.

    Parameters:
        input_path (str): Path to the cleaned data from data ingestion.
        output_train_path (str): Path to save the preprocessed training data.
        output_test_path (str): Path to save the preprocessed testing data.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    try:
        # Load cleaned data
        logging.info(f"Loading cleaned data from {input_path}...")
        df = pd.read_csv(input_path)
        logging.info(f"Data loaded successfully with shape: {df.shape}.")

        # Replace placeholders for missing values
        logging.info("Replacing 'unknown' and 'nan' with NaN...")
        df.replace(['unknown', 'nan'], np.nan, inplace=True)

        # Split features and target variable
        logging.info("Separating features and target variable...")
        X = df.drop(columns=["y"])
        y = df["y"]

        # Identify numeric and categorical columns
        logging.info("Identifying numeric and categorical columns...")
        numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        logging.info(f"Numeric columns: {numeric_cols}")
        logging.info(f"Categorical columns: {categorical_cols}")

        # Impute missing values with KNNImputer
        logging.info("Imputing missing values using KNNImputer...")
        knn_imputer = KNNImputer(n_neighbors=5)
        X[numeric_cols] = knn_imputer.fit_transform(X[numeric_cols])

        # Encode categorical features with OrdinalEncoder
        if categorical_cols:
            logging.info("Encoding categorical features using OrdinalEncoder...")
            ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols].astype(str))

        # Scale numeric features
        logging.info("Scaling numeric features using StandardScaler...")
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # Split data into training and testing sets
        logging.info("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Save preprocessed data
        logging.info(f"Saving preprocessed training data to {output_train_path}...")
        os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
        train_data = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name="y")], axis=1)
        train_data.to_csv(output_train_path, index=False)

        logging.info(f"Saving preprocessed testing data to {output_test_path}...")
        test_data = pd.concat([pd.DataFrame(X_test), pd.Series(y_test, name="y")], axis=1)
        test_data.to_csv(output_test_path, index=False)

        logging.info("Data preprocessing completed successfully.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise

if __name__ == "__main__":
    # Define input and output file paths
    cleaned_data_path = "./data/processed/bank_data_cleaned.csv"
    train_data_path = "./data/processed/train_data.csv"
    test_data_path = "./data/processed/test_data.csv"

    # Run the preprocessing pipeline
    preprocess_data(cleaned_data_path, train_data_path, test_data_path)
