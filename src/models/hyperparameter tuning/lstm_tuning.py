"""
LSTM Hyperparameter Tuning using GridSearchCV.

"""

import logging
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error

import sys
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.lstm_model import (
    VitalLSTM, LazyVitalDataset, weighted_mae_loss, evaluate, 
    VITAL_COLS, INTERVENTION_COLS, ROLLING_COLS, FEATURE_COLS, TARGET_COLS,
    VITAL_WEIGHTS, DEVICE, BATCH_SIZE, LOOKBACK
)
from src.config import DATA_PROCESSED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Scikit-Learn Wrapper
class LSTMEstimator(BaseEstimator, RegressorMixin):
    """
    Scikit-learn wrapper for VitalLSTM.
    Allows use of GridSearchCV and other sklearn utilities.
    """
    def __init__(self, 
                 hidden_size=128, 
                 lr=1e-3, 
                 epochs=10, 
                 dropout=0.2, 
                 weight_decay=1e-5,
                 n_features=None,
                 n_targets=8,
                 vital_names=None):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.n_features = n_features
        self.n_targets = n_targets
        self.vital_names = vital_names
        self.model = None

    def fit(self, X_stays, y=None):
        
        if self.n_features is None:
            self.n_features = X_stays[0][0].shape[1]
        
        self.model = VitalLSTM(
            n_features=self.n_features, 
            n_targets=self.n_targets, 
            hidden_size=self.hidden_size
        ).to(DEVICE)
        
        train_ds = LazyVitalDataset(X_stays, LOOKBACK)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=True 
        )
        
        w_vec = torch.tensor(
            [VITAL_WEIGHTS.get(v, 1.0) for v in self.vital_names], 
            dtype=torch.float32
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = weighted_mae_loss(pred, y_batch, w_vec)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        
        return self

    def predict(self, X_stays):
        self.model.eval()
        test_ds = LazyVitalDataset(X_stays, LOOKBACK)
        loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                pred = self.model(X_batch.to(DEVICE)).cpu().numpy()
                all_preds.append(pred)
        
        return np.concatenate(all_preds, axis=0)

    def score(self, X_stays, y=None):

        self.model.eval()
        test_ds = LazyVitalDataset(X_stays, LOOKBACK)
        loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False)
        
        w_vec = torch.tensor(
            [VITAL_WEIGHTS.get(v, 1.0) for v in self.vital_names], 
            dtype=torch.float32
        ).to(DEVICE)
        
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = self.model(X_batch)
                loss = weighted_mae_loss(pred, y_batch, w_vec)
                total_loss += loss.item() * len(X_batch)
                n += len(X_batch)
        
        return -(total_loss / n) if n > 0 else -999.0

# Tuning Script
def run_tuning(subset_fraction=0.1):
    start_time = time.time()
    logger.info("Initializing LSTM Hyperparameter Tuning ...")
    
    # 1. Load Data
    features_path = DATA_PROCESSED / "features.parquet"
    if not features_path.exists():
        logger.error("Features not found. Run preprocessing first.")
        return

    df = pd.read_parquet(features_path)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    target_cols  = [c for c in TARGET_COLS  if c in df.columns]
    vital_names  = [t.replace("_target", "") for t in target_cols]

    for col in feature_cols:
        df[col] = df[col].fillna(0.0)

    # 2. Subset for faster tuning
    all_stays = df["stay_id"].unique()
    if subset_fraction < 1.0:
        n_subset = int(len(all_stays) * subset_fraction)
        all_stays = np.random.choice(all_stays, n_subset, replace=False)
        logger.info(f"Using subset of {n_subset} stays ({subset_fraction*100:.0f}%)")

    # 3. Build stay list
    def make_stay_list(stay_ids):
        out = []
        subset_df = df[df["stay_id"].isin(stay_ids)]
        for sid, grp in subset_df.groupby("stay_id"):
            grp = grp.sort_values("hour_idx")
            feat   = grp[feature_cols].values.astype(np.float32)
            target = grp[target_cols].values.astype(np.float32)
            if len(feat) > LOOKBACK:
                out.append((feat, target))
        return out

    stay_list = make_stay_list(all_stays)
    
    # 4. Scale Data
    train_feat_sample = np.vstack([f for f, _ in stay_list])
    scaler = StandardScaler()
    scaler.fit(train_feat_sample)
    
    scaled_stays = []
    for f, t in stay_list:
        scaled_stays.append((scaler.transform(f).astype(np.float32), t))
    
    logger.info(f"Prepared {len(scaled_stays)} stays for tuning.")

    # 5. GridSearchCV Setup
    estimator = LSTMEstimator(
        n_features=len(feature_cols),
        n_targets=len(target_cols),
        vital_names=vital_names,
        epochs=5 # Lower epochs for tuning
    )

    param_grid = {
        'hidden_size': [64, 128, 256],
        'lr': [1e-3, 5e-4],
        'dropout': [0.1, 0.2, 0.3],
    }

    
    logger.info("Starting GridSearchCV ...")
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        scoring=None, 
        verbose=2,
        n_jobs=1 
    )

    X_obj = np.array(scaled_stays, dtype=object)
    
    grid.fit(X_obj)

    # 6. Results
    logger.info(f"Best parameters: {grid.best_params_}")
    logger.info(f"Best score (neg MAE): {grid.best_score_:.4f}")

    # Save results
    results_path = DATA_PROCESSED / "lstm_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
            "cv_results": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in grid.cv_results_.items()}
        }, f, indent=2)
    
    logger.info(f"Tuning results saved → {results_path}")
    logger.info(f"Total time: {(time.time() - start_time)/60:.1f} min")

if __name__ == "__main__":

    run_tuning(subset_fraction=0.5)
