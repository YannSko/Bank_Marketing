import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/data_ingestion.log"),
        logging.StreamHandler()
    ]
)

def load_raw_data(file_path):
    """
    Load raw data from the given CSV file.
    """
    try:
        logging.info(f"Loading raw data from {file_path}...")
        df = pd.read_csv(file_path, sep=';')
        logging.info(f"Raw data successfully loaded with shape: {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading raw data: {e}")
        raise

def preprocess_data(df):
    """
    Preprocess the raw data: handle missing values and encode categorical features.
    """
    try:
        logging.info("Starting preprocessing of data...")

        # Replace 'unknown' with NaN
        logging.info("Replacing 'unknown' and 'nan' with NaN...")
        df.replace(['unknown', 'nan'], np.nan, inplace=True)

        # Identify numeric and categorical columns
        logging.info("Identifying numeric and categorical columns...")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        logging.info(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns.")

        # Encode categorical features
        logging.info("Encoding categorical features...")
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[categorical_cols] = ordinal_encoder.fit_transform(df[categorical_cols].astype(str))

        # Impute missing values
        logging.info("Imputing missing values using KNN...")
        knn_imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

        # Final check for missing values
        nan_count = df_imputed.isnull().sum().sum()
        if nan_count == 0:
            logging.info("No missing values found after imputation.")
        else:
            logging.warning(f"Found {nan_count} missing values after imputation.")

        logging.info("Preprocessing completed successfully.")
        return df_imputed
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    try:
        logging.info(f"Saving preprocessed data to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Data successfully saved to {output_path}.")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

if __name__ == "__main__":
    # File paths
    raw_data_path = "./data/raw/bank-additional-full.csv"
    processed_data_path = "./data/processed/bank_data_cleaned.csv"

    # Step 1: Load raw data
    raw_data = load_raw_data(raw_data_path)

    # Step 2: Preprocess data
    cleaned_data = preprocess_data(raw_data)

    # Step 3: Save cleaned data
    save_cleaned_data(cleaned_data, processed_data_path)
