# Dynamic Clinical Risk Forecasting

An end-to-end machine learning pipeline for dynamic clinical risk forecasting using the MIMIC-IV dataset. This project automates the extraction of massive clinical datasets from Google BigQuery, processes them through a modular clinical pipeline, and trains predictive models for multiple vital sign trajectories.

## Project Overview

This project provides a robust, production-ready framework for:
- **Scalable Data Extraction**: Efficiently querying and aggregating 50M+ rows of clinical events from BigQuery.
- **Modular Preprocessing**: A decoupled suite of scripts for ingestion, temporal alignment, clinical validation (outlier clipping), and rolling feature engineering.
- **Predictive Modeling**: High-performance XGBoost baselines for 6-hour ahead forecasting of 8 critical vital signs.

## Project Structure

```text
├── data/
│   ├── raw/                # Raw parquet extracts from BigQuery
│   └── processed/          # Modular feature matrices and model artifacts
├── src/
│   ├── config.py           # Centralized project configuration and paths
│   ├── data_extraction/    # Scripts for BigQuery data retrieval
│   ├── preprocessing/      # Modular preprocessing pipeline components
│   └── models/             # Training and evaluation logic
├── requirements.txt        # Core project dependencies
└── README.md               # Project documentation
```

## Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Access to the [MIMIC-IV dataset](https://physionet.org/content/mimiciv/2.2/) on Google BigQuery.
- Google Cloud Project with Billing enabled.

### 2. Google BigQuery Setup
To use the automated data extraction from BigQuery:

1.  **Install Google Cloud SDK**: Follow the instructions at [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install).
2.  **Authenticate**: Run the following command and follow the browser prompts:
    ```bash
    gcloud auth application-default login
    ```
3.  **Project Configuration**: Open `src/config.py` and set your `GOOGLE_CLOUD_PROJECT` ID.
    ```python
    GOOGLE_CLOUD_PROJECT = "your-project-id"
    ```
4.  **Dataset Access**: Ensure your authenticated GCP account has `BigQuery Data Viewer` permissions on the `physionet-data.mimiciv_3_1_icu` and `physionet-data.mimiciv_3_1_hosp` datasets.

### 3. Local Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Install libomp (macOS only, required for XGBoost)
brew install libomp
```

## Running the Pipeline

The pipeline is designed to be run in sequence:

### Step 1: Data Extraction
Extracts cohort, vital signs, and interventions from BigQuery.
```bash
python src/data_extraction/extraction.py
```

### Step 2: Modular Preprocessing
Transforms raw extracts into a feature matrix for modeling.
```bash
python src/preprocessing/pipeline.py
```

### Step 3: Model Training
Trains the baseline XGBoost models and saves performance metrics.
```bash
python src/models/baseline_xgboost.py
```

## Current Progress & Metrics

The baseline models show strong predictive performance across critical vital signs (6-hour horizon):

| Target Vital | MAE | R² (Test) |
| :--- | :--- | :--- |
| **Temperature** | 0.85 °F | **0.78** |
| **FiO2** | 2.75 % | **0.71** |
| **Heart Rate** | 8.09 bpm | **0.61** |
| **Systolic BP** | 11.81 mmHg | **0.49** |

*All artifacts and logs are saved to `data/processed/`.*
