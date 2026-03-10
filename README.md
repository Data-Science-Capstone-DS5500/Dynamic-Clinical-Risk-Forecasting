# Dynamic Clinical Risk Forecasting

An end-to-end machine learning pipeline for dynamic clinical risk forecasting using the MIMIC-IV dataset. This project automates the extraction of massive clinical datasets from Google BigQuery, processes them through a modular clinical pipeline, and trains predictive models for multiple vital sign trajectories.

## Project Overview

This project provides a robust, production-ready framework for:
- **Scalable Data Extraction**: Efficiently querying and aggregating 50M+ rows of clinical events from BigQuery.
- **Modular Preprocessing**: A decoupled suite of scripts for ingestion, temporal alignment, clinical validation (outlier clipping), and rolling feature engineering.
- **Predictive Modeling**: High-performance XGBoost baselines and sequential Bi-LSTM models for 6-hour ahead forecasting of 8 critical vital signs.
- **Interactive Dashboard**: A real-time clinical risk monitoring interface built with Streamlit and Plotly.

## Project Structure

```text
├── data/
│   ├── raw/                # Raw parquet extracts from BigQuery
│   └── processed/          # Modular feature matrices and model artifacts
├── src/
│   ├── config.py           # Centralized project configuration and paths
│   ├── data_extraction/    # Scripts for BigQuery data retrieval
│   ├── preprocessing/      # Modular preprocessing pipeline components
│   ├── models/             # Training and evaluation logic (XGBoost & LSTM)
│   ├── inference/          # Unified inference interface (Predictor)
│   └── dashboard/          # Streamlit application and visualizations
├── requirements.txt        # Core project dependencies
└── README.md               # Project documentation
```

## Dashboard

The project includes a comprehensive interactive dashboard for real-time patient monitoring and risk assessment. 

### 1. Patient Overview
Provides a high-level summary of the patient's current state, temporal trends, and a snapshot grid.
<img width="3024" height="1416" alt="image" src="https://github.com/user-attachments/assets/ac5bda58-27b6-459f-a692-d61ef6285c62" />


### 2. 6-Hour Forecast
Displays predicted vital signs 6 hours into the future, complete with confidence bands and a clinical risk gauge.
<img width="3012" height="1506" alt="image" src="https://github.com/user-attachments/assets/c1c141f1-c1a8-4d81-9f3a-4bb12173aea2" />


### 3. Risk History & Interpretation
Shows the evolution of the clinical risk score and provides feature importance for interpretability.
<img width="3024" height="1508" alt="image" src="https://github.com/user-attachments/assets/8936b13c-34fe-4925-91c6-12e889ba6b4c" />


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
```bash
python src/data_extraction/extraction.py
```

### Step 2: Modular Preprocessing
```bash
python src/preprocessing/pipeline.py
```

### Step 3: Model Training
```bash
python src/models/baseline_xgboost.py
python src/models/lstm_model.py
```

### Step 4: Launch Dashboard
Start the interactive Streamlit application using the project virtual environment:
```bash
./.venv/bin/python3 -m streamlit run src/dashboard/app.py
```

*All artifacts and logs are saved to `data/processed/`.*
