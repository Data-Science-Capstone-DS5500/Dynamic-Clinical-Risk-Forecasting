"""
Hyperparameter Tuning for XGBoost
=================================
Finds optimal XGBoost parameters using GridSearchCV and GroupKFold.

"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
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
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'n_estimators': [100, 200]
}

def load_data():
    path = DATA_PROCESSED / "features.parquet"
    df = pd.read_parquet(path)
    log.info(f"Loaded {len(df):,} rows.")
    return df

def main():
    log.info("═══ XGBoost Hyperparameter Tuning ═══")
    
    # 1. Load and sample
    df = load_data()
    target_col = f"{TARGET_VITAL}{FORECAST_SUFFIX}"
    df = df.dropna(subset=[target_col])
    
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    id_cols = {"subject_id", "hadm_id", "stay_id", "icustay_id", "timestamp", "hour_idx", "intime", "outtime"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in id_cols and not c.endswith(FORECAST_SUFFIX)]
    
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df["stay_id"].values
    
    # 2. Setup GridSearchCV
    regr = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1)
    gkf = GroupKFold(n_splits=CV_SPLITS)
    
    grid_search = GridSearchCV(
        estimator=regr,
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
    out_path = DATA_PROCESSED / "xgb_best_params.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info(f"Saved best parameters to {out_path}")

if __name__ == "__main__":
    main()
