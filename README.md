# Bank_Marketing
Bank Marketing Campaign Analysis

This repository provides an in-depth analysis of a Portuguese banking institution's direct marketing campaigns, aiming to predict client subscription to term deposits and profile potential subscribers.
Project Overview

The primary objective is to identify factors influencing a client's decision to subscribe to a term deposit. By understanding these determinants, the bank can enhance its marketing strategies to improve conversion rates.
Dataset

The analysis utilizes the Bank Marketing Dataset from the UCI Machine Learning Repository. This dataset comprises 45,211 records with 17 attributes, including client information, campaign details, and socio-economic indicators.
Repository Structure

    data/: Contains raw and processed datasets.
    models/: Stores trained machine learning models.
    notebooks/: Includes Jupyter notebooks for exploratory data analysis and model development.
    pipeline/: Scripts for data preprocessing and feature engineering.
    deployment/: Resources for deploying the predictive model.
    streamlit/: Streamlit application for interactive model demonstration.
    test/: Unit tests for code validation.
    EDA.py: Script for exploratory data analysis.
    orchestrator.py: Main script to run the data pipeline and model training.
    requirements.txt: Lists required Python packages.
    config_model.yaml: Configuration file for model parameters.
    mlflow_config.yaml: Configuration for MLflow tracking.

Getting Started
Prerequisites

Ensure Python 3.8+ is installed.
Installation

    Clone the repository:

git clone https://github.com/YannSko/Bank_Marketing.git
cd Bank_Marketing

Create a virtual environment:

python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

Usage

    Data Preprocessing:

python orchestrator.py --stage preprocess

Model Training:

python orchestrator.py --stage train

Model Evaluation:

python orchestrator.py --stage evaluate

Run Streamlit App:

    streamlit run streamlit/app.py

Results

The analysis identified key factors influencing term deposit subscriptions, such as age, job type, and previous campaign outcomes. The predictive model achieved an accuracy of 91%, providing valuable insights for targeted marketing strategies.
Contributing

Contributions are welcome. Please fork the repository and submit a pull request for any enhancements or bug fixes.
License

This project is licensed under the GPL-3.0 License. See the LICENSE file for details.
Acknowledgements

    UCI Machine Learning Repository for the dataset.
    MLflow for experiment tracking.
    Streamlit for the interactive web application framework.

For any questions or issues, please open an issue in this repository.
