"""
Hyperparameter Tuning for Random Forest
=======================================
Finds optimal RF parameters using GridSearchCV and GroupKFold (by stay_id).
Used for the 6-hour clinical risk forecasting task.

"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import make_scorer, mean_absolute_error

# project imports
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.config import DATA_PROCESSED

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# configuration
SAMPLE_SIZE      = 100000 
TARGET_VITAL     = "heart_rate"
FORECAST_SUFFIX  = "_target"
CV_SPLITS        = 3
RANDOM_STATE     = 42

PARAM_GRID = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [4, 10],
    'max_features': ['sqrt', 'log2']
}

def load_data():
    path = DATA_PROCESSED / "features.parquet"
    df = pd.read_parquet(path)
    log.info(f"Loaded {len(df):,} rows.")
    return df

def main():
    log.info("═══ Random Forest Hyperparameter Tuning ═══")
    
    # 1. Load and sample
    df = load_data()
    target_col = f"{TARGET_VITAL}{FORECAST_SUFFIX}"
    
    # Filter rows with valid targets
    df = df.dropna(subset=[target_col])
    
    # Sampling
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    id_cols = {"subject_id", "hadm_id", "stay_id", "icustay_id", "timestamp", "hour_idx", "intime", "outtime"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in id_cols and not c.endswith(FORECAST_SUFFIX)]
    
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df["stay_id"].values
    
    log.info(f"Tuning on {len(X):,} samples with {len(feature_cols)} features.")
    
    # 2. Setup GridSearch
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    gkf = GroupKFold(n_splits=CV_SPLITS)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=PARAM_GRID,
        cv=gkf,
        scoring='neg_mean_absolute_error',
        verbose=2,
        n_jobs=1 
    )
    
    # 3. Search
    log.info("Starting GridSearchCV...")
    grid_search.fit(X, y, groups=groups)
    
    # 4. Results
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    
    log.info(f"Best MAE: {best_score:.4f}")
    log.info(f"Best Params: {best_params}")
    
    # Save
    out_path = DATA_PROCESSED / "rf_best_params.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info(f"Saved best parameters to {out_path}")

if __name__ == "__main__":
    main()
