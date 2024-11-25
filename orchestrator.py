import logging
import subprocess
from pipeline.data_ingestion import load_raw_data
from pipeline.data_preprocessing import preprocess_data
from pipeline.feature_engineering import feature_engineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/orchestrator.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.info("Starting end-to-end pipeline...")

    # Step 1: Data Ingestion
    raw_data_path = "./data/raw/bank-additional-full.csv"
    processed_data_path = "./data/processed/bank_data_cleaned.csv"
    raw_data = load_raw_data(raw_data_path)
    cleaned_data = preprocess_data(raw_data)
    save_cleaned_data(cleaned_data, processed_data_path)
    logging.info("Data ingestion completed.")

    # Step 2: Data Preprocessing
    train_data_path = "./data/processed/train_data.csv"
    test_data_path = "./data/processed/test_data.csv"
    preprocess_data(
        input_path=processed_data_path,
        output_train_path=train_data_path,
        output_test_path=test_data_path
    )
    logging.info("Data preprocessing completed.")

    # Step 3: Feature Engineering
    engineered_train_path = "./data/processed/engineered_train_data.csv"
    engineered_test_path = "./data/processed/engineered_test_data.csv"
    feature_engineering(
        input_train_path=train_data_path,
        input_test_path=test_data_path,
        output_train_path=engineered_train_path,
        output_test_path=engineered_test_path
    )
    logging.info("Feature engineering completed.")

    # Step 4: Model Training with Hyperparameters
    logging.info("Starting model training...")
    training_command = [
        'python', 'models/training/train.py',
        '--max_depth', '10',
        '--learning_rate', '0.1',
        '--n_estimators', '150',
        '--colsample_bytree', '0.8'
    ]
    subprocess.run(training_command, check=True)
    logging.info("Model training and logging completed.")

if __name__ == "__main__":
    main()
