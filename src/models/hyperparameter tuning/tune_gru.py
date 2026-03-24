"""
Hyperparameter Tuning for GRU
=============================
Finds optimal GRU parameters using GridSearchCV.
Wraps the PyTorch GRUForecaster in a scikit-learn compatible estimator.

"""

import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_absolute_error

# project imports
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.config import DATA_PROCESSED
from src.models.gru_model import GRUForecaster, build_sequences, make_loader

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# configuration
SAMPLE_STAYS     = 500 
TARGET_VITAL     = "heart_rate"
FORECAST_SUFFIX  = "_target"
CV_SPLITS        = 3
RANDOM_STATE     = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_GRID = {
    'hidden_size': [32, 64],
    'num_layers': [1, 2],
    'lr': [1e-3, 5e-4],
    'epochs': [5]
}

class GRUWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_features=None, hidden_size=64, num_layers=2, lr=1e-3, epochs=5):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def fit(self, X, y):
        if self.n_features is None:
            self.n_features = X.shape[2]
            
        self.model = GRUForecaster(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.HuberLoss()
        
        loader = make_loader(X, y, batch_size=256, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                pred = self.model(xb).squeeze()
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_pt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds = self.model(X_pt).cpu().numpy().flatten()
        return preds

def load_and_prepare_data():
    path = DATA_PROCESSED / "features.parquet"
    df = pd.read_parquet(path)
    
    target_col = f"{TARGET_VITAL}{FORECAST_SUFFIX}"
    df = df.dropna(subset=[target_col])
    
    # Identify features
    id_cols = {"subject_id", "hadm_id", "stay_id", "icustay_id", "timestamp", "hour_idx", "intime", "outtime"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in id_cols and not c.endswith(FORECAST_SUFFIX)]
    
    # Sample by stay_id to keep sequences valid
    unique_stays = df["stay_id"].unique()
    if len(unique_stays) > SAMPLE_STAYS:
        np.random.seed(RANDOM_STATE)
        sampled_stays = np.random.choice(unique_stays, SAMPLE_STAYS, replace=False)
        df = df[df["stay_id"].isin(sampled_stays)]
    
    df = df.sort_values(["stay_id", "timestamp"])
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    log.info(f"Building sequences for {len(df):,} rows ({len(df['stay_id'].unique())} stays)...")
    
    X_list, y_list, groups_list = [], [], []
    for stay_id, group in df.groupby("stay_id"):
        if len(group) < 13: # seq_len + 1
            continue
        X_s, y_s = build_sequences(group, feature_cols, target_col)
        X_list.append(X_s)
        y_list.append(y_s)
        groups_list.append(np.full(len(y_s), stay_id))
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    
    return X, y, groups

def main():
    log.info("═══ GRU Hyperparameter Tuning ═══")
    
    # 1. Prepare data
    X, y, groups = load_and_prepare_data()
    log.info(f"Tuning on {len(X):,} sequences.")
    
    # 2. Setup GridSearchCV
    wrapper = GRUWrapper(n_features=X.shape[2])
    gkf = GroupKFold(n_splits=CV_SPLITS)
    
    grid_search = GridSearchCV(
        estimator=wrapper,
        param_grid=PARAM_GRID,
        cv=gkf,
        scoring='neg_mean_absolute_error',
        verbose=2
    )
    
    # 3. Search
    log.info("Starting GridSearchCV (this may take a while)...")
    grid_search.fit(X, y, groups=groups)
    
    # 4. Results
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    
    log.info(f"Best MAE: {best_score:.4f}")
    log.info(f"Best Params: {best_params}")
    
    # Save
    out_path = DATA_PROCESSED / "gru_best_params.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info(f"Saved best parameters to {out_path}")

if __name__ == "__main__":
    main()
