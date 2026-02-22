from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

# Google Cloud settings
GOOGLE_CLOUD_PROJECT = "dynamic-clinical-system"
# Public MIMIC-IV dataset on BigQuery
MIMIC_IV_ICU = "physionet-data.mimiciv_3_1_icu"
MIMIC_IV_HOSP = "physionet-data.mimiciv_3_1_hosp"

# Create directories
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# MIMIC-IV Demo tables we need
TABLES = [
    "patients",
    "admissions", 
    "icustays",
    "chartevents",
    "inputevents",
    "d_items",
]

# Key itemids for demo (MIMIC-IV)
VITALS_ITEMIDS = {
    "heart_rate": [220045],
    "sbp": [220050, 220179],
    "dbp": [220051, 220180],
    "map": [220052, 220181, 225312],
    "resp_rate": [220210, 224690],
    "spo2": [220277],
    "temperature": [223761, 223762],  # F and C
    "fio2": [223835, 227009, 227010],
}

INTERVENTION_ITEMIDS = {
    "norepinephrine": [221906],
    "vasopressin": [222315],
    "epinephrine": [221289],
    "dopamine": [221662],
    "phenylephrine": [221749],
}

# Forecast settings
FORECAST_HORIZON = 6  # hours ahead
LOOKBACK_WINDOW = 12  # hours of history as features