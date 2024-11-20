import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/feature_engineering.log"),
        logging.StreamHandler()
    ]
)

def feature_engineering(input_train_path, input_test_path, output_train_path, output_test_path):
    """
    Perform feature engineering on the training and testing datasets.

    Parameters:
        input_train_path (str): Path to the preprocessed training data.
        input_test_path (str): Path to the preprocessed testing data.
        output_train_path (str): Path to save the engineered training data.
        output_test_path (str): Path to save the engineered testing data.

    Returns:
        Tuple of DataFrames: train_df, test_df
    """
    try:
        # Load preprocessed data
        logging.info(f"Loading preprocessed training data from {input_train_path}...")
        train_df = pd.read_csv(input_train_path)
        logging.info(f"Training data loaded successfully with shape: {train_df.shape}.")

        logging.info(f"Loading preprocessed testing data from {input_test_path}...")
        test_df = pd.read_csv(input_test_path)
        logging.info(f"Testing data loaded successfully with shape: {test_df.shape}.")

        # Feature Engineering
        logging.info("Starting feature engineering...")

        # Example 1: Log-transform skewed features
        logging.info("Applying log-transform to skewed features...")
        skewed_features = ['duration', 'campaign', 'previous']
        for feature in skewed_features:
            if feature in train_df.columns:
                train_df[feature] = np.log1p(train_df[feature])
                test_df[feature] = np.log1p(test_df[feature])

        # Example 2: Standardize numeric features (revisit scaling for sensitive variables)
        logging.info("Standardizing key numeric features...")
        economic_features = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        for feature in economic_features:
            train_mean = train_df[feature].mean()
            train_std = train_df[feature].std()
            train_df[feature] = (train_df[feature] - train_mean) / train_std
            test_df[feature] = (test_df[feature] - train_mean) / train_std

        # Example 3: Creating new interaction features
        logging.info("Creating interaction features...")
        train_df['interaction_employed_duration'] = train_df['nr.employed'] * train_df['duration']
        test_df['interaction_employed_duration'] = test_df['nr.employed'] * test_df['duration']

        # Example 4: Binning numeric features (e.g., 'age')
        logging.info("Binning 'age' into categories...")
        bins = [0, 25, 40, 60, np.inf]
        labels = ['Young', 'Mid-age', 'Senior', 'Elderly']
        train_df['age_group'] = pd.cut(train_df['age'], bins=bins, labels=labels)
        test_df['age_group'] = pd.cut(test_df['age'], bins=bins, labels=labels)

        # Example 5: Dropping irrelevant/redundant columns
        logging.info("Dropping irrelevant or redundant columns...")
        redundant_features = ['pdays']  # Example of a feature deemed less informative
        train_df.drop(columns=redundant_features, inplace=True, errors='ignore')
        test_df.drop(columns=redundant_features, inplace=True, errors='ignore')

        # Save the engineered datasets
        logging.info(f"Saving engineered training data to {output_train_path}...")
        os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
        train_df.to_csv(output_train_path, index=False)

        logging.info(f"Saving engineered testing data to {output_test_path}...")
        test_df.to_csv(output_test_path, index=False)

        logging.info("Feature engineering completed successfully.")
        return train_df, test_df

    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise

if __name__ == "__main__":
    # Define input and output file paths
    preprocessed_train_path = "./data/processed/train_data.csv"
    preprocessed_test_path = "./data/processed/test_data.csv"
    engineered_train_path = "./data/processed/engineered_train_data.csv"
    engineered_test_path = "./data/processed/engineered_test_data.csv"

    # Run the feature engineering pipeline
    feature_engineering(preprocessed_train_path, preprocessed_test_path, engineered_train_path, engineered_test_path)
