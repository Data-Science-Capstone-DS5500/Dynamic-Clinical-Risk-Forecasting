"""
Multi-Output LSTM Sequence Model
Trains a bidirectional LSTM for 6-hour ahead forecasting of 8 vital signs.
Uses 12-hour lookback windows constructed by sliding over each ICU stay.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import logging
import time
import pickle
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import sys

project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DATA_PROCESSED, FORECAST_HORIZON

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


VITAL_COLS = [
    "heart_rate", "sbp", "dbp", "map",
    "resp_rate", "spo2", "temperature", "fio2",
]
INTERVENTION_COLS = ["vasopressor_on", "vasopressor_rate"]

ROLLING_COLS = []
for v in VITAL_COLS:
    ROLLING_COLS.append(f"{v}_mean_6h")
    ROLLING_COLS.append(f"{v}_std_6h")

FEATURE_COLS = VITAL_COLS + INTERVENTION_COLS + ROLLING_COLS
TARGET_COLS  = [f"{v}_target" for v in VITAL_COLS]

LOOKBACK   = 12   # hours of history per sequence
BATCH_SIZE = 1024
EPOCHS     = 50
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Sequence forecasting settings

# Clinical weights for loss (higher = more important vital)
VITAL_WEIGHTS = {
    "map":        2.0,
    "heart_rate": 1.5,
    "spo2":       1.5,
    "resp_rate":  1.2,
    "sbp":        1.0,
    "dbp":        1.0,
    "temperature": 1.0,
    "fio2":       1.0,
}


class LazyVitalDataset(Dataset):
    """
    Stores one (features, targets) array per stay in RAM.
    Windows are built on-the-fly no pre-allocation of a
    giant sequences array, so memory usage efficient

    Parameters:
    stays       : list of (feat_array, target_array) tuples, each sorted by hour.
    lookback    : number of past hours per sequence window.
    index_map   : pre-computed list of (stay_idx, end_row) tuples for __getitem__.
    scaler      : fitted StandardScaler applied inline (optional, avoids a
                  separate transform pass over the full dataset).
    """
    def __init__(self, stays: list, lookback: int):
        self.stays    = stays
        self.lookback = lookback
        # Build flat index: (stay_idx, end_row_in_stay)
        self.index: list[tuple[int, int]] = []
        for s_idx, (feat, tgt) in enumerate(stays):
            n = len(feat)
            for i in range(lookback, n):
                if not np.isnan(tgt[i - 1]).all():
                    self.index.append((s_idx, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_idx, end = self.index[idx]
        feat, tgt  = self.stays[s_idx]
        window = feat[end - self.lookback: end]          # (lookback, n_feat)
        target = tgt[end - 1]                            # (n_targets,)
        return torch.tensor(window, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


class VitalLSTM(nn.Module):
    """
    Bidirectional 2-layer LSTM encoder with one regression head per vital
    Architecture
    ────────────
    Input  : (batch, lookback, n_features)
    LSTM   : 2 layers, hidden_size=128, bidirectional → output dim 256
    Heads  : 8 independent Linear(256 → 1) layers
    Output : (batch, 8) predicted vital values
    """
    def __init__(self, n_features: int, n_targets: int = 8, hidden_size: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        lstm_out_dim = hidden_size * 2  # bidirectional

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_out_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
            for _ in range(n_targets)
        ])

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        last    = out[:, -1, :]             # take last time-step: (batch, 256)
        preds   = [head(last) for head in self.heads]
        return torch.cat(preds, dim=1)      # (batch, n_targets)


# Training helpers
def weighted_mae_loss(pred: torch.Tensor, target: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
    mask   = ~torch.isnan(target)
    errors = torch.abs(pred - target)
    errors = torch.where(mask, errors, torch.zeros_like(errors))
    weighted = errors * weights.to(errors.device)
    denom = mask.float().sum().clamp(min=1)
    return weighted.sum() / denom


def evaluate(model: VitalLSTM, loader: DataLoader,
             weights: torch.Tensor, device: str) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = weighted_mae_loss(pred, y_batch, weights)
            total_loss += loss.item() * len(X_batch)
            n += len(X_batch)
    return total_loss / n if n > 0 else 0.0


def compute_metrics(model: VitalLSTM, loader: DataLoader,
                    device: str, target_names: list) -> list:
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y_batch.numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    metrics = []
    for i, name in enumerate(target_names):
        y_t = all_true[:, i]
        y_p = all_pred[:, i]
        mask = ~np.isnan(y_t)
        if mask.sum() == 0:
            continue
        y_t, y_p = y_t[mask], y_p[mask]
        metrics.append({
            "target": name,
            "mae":  round(float(mean_absolute_error(y_t, y_p)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_t, y_p))), 4),
            "r2":   round(float(r2_score(y_t, y_p)), 4),
            "n":    int(mask.sum()),
        })
    return metrics


def main():
    start = time.time()
    logger.info(f"Device: {DEVICE}")
    logger.info("  LSTM MULTI-OUTPUT TRAINING PIPELINE")

    # 1. Load features
    features_path = DATA_PROCESSED / "features.parquet"
    if not features_path.exists():
        logger.error(f"Features not found: {features_path}")
        logger.error("Run preprocessing first: python src/preprocessing/pipeline.py")
        return

    df = pd.read_parquet(features_path)
    logger.info(f"Loaded: {df.shape[0]:,} rows, {df['stay_id'].nunique():,} stays")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    target_cols  = [c for c in TARGET_COLS  if c in df.columns]
    for col in feature_cols:
        df[col] = df[col].fillna(0.0)

    n_features  = len(feature_cols)
    n_targets   = len(target_cols)
    vital_names = [t.replace("_target", "") for t in target_cols]
    logger.info(f"Features: {n_features}  |  Targets: {n_targets}")

    # 2. Stay-level train / val / test split BEFORE building any arrays
    #    (GroupShuffleSplit on stay_ids to ensure there is no data leakage)
    all_stays = df["stay_id"].unique()
    rng       = np.random.default_rng(42)

    rng.shuffle(all_stays)
    n_total = len(all_stays)
    n_test  = max(1, int(0.15 * n_total))
    n_val   = max(1, int(0.15 * n_total))
    test_stays  = set(all_stays[:n_test])
    val_stays   = set(all_stays[n_test: n_test + n_val])
    train_stays = set(all_stays[n_test + n_val:])
    logger.info(
        f"Stays — Train: {len(train_stays):,}  Val: {len(val_stays):,}  Test: {len(test_stays):,}"
    )

    # 3. Build per-stay arrays
    def make_stay_list(stay_ids: set):
        out = []
        for sid, grp in df[df["stay_id"].isin(stay_ids)].groupby("stay_id"):
            grp = grp.sort_values("hour_idx")
            feat   = grp[feature_cols].values.astype(np.float32)
            target = grp[target_cols].values.astype(np.float32)
            if len(feat) > LOOKBACK:
                out.append((feat, target))
        return out

    logger.info("Preparing stay arrays …")
    train_list = make_stay_list(train_stays)
    val_list   = make_stay_list(val_stays)
    test_list  = make_stay_list(test_stays)

    # 4. Fit scaler on a FLAT sample of training rows
    train_feat_sample = np.vstack([f for f, _ in train_list])
    scaler = StandardScaler()
    scaler.fit(train_feat_sample)
    del train_feat_sample  

    # 5. Pre-scale stay arrays
    logger.info("Scaling data …")
    def scale_stays(stay_list, sc):
        out = []
        for f, t in stay_list:
            f_scaled = sc.transform(f)
            out.append((f_scaled.astype(np.float32), t))
        return out

    train_list = scale_stays(train_list, scaler)
    val_list   = scale_stays(val_list,   scaler)
    test_list  = scale_stays(test_list,  scaler)

    train_ds = LazyVitalDataset(train_list, LOOKBACK)
    val_ds   = LazyVitalDataset(val_list,   LOOKBACK)
    test_ds  = LazyVitalDataset(test_list,  LOOKBACK)
    logger.info(
        f"Sequences — Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, prefetch_factor=2
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model  = VitalLSTM(n_features=n_features, n_targets=n_targets).to(DEVICE)
    w_vec  = torch.tensor(
        [VITAL_WEIGHTS.get(v, 1.0) for v in vital_names], dtype=torch.float32
    )
    optim  = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched  = CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-5)

    # 6. Training loop
    best_val_loss = float("inf")
    best_weights  = None

    logger.info(f"Training for {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optim.zero_grad()
            pred = model(X_batch)
            loss = weighted_mae_loss(pred, y_batch, w_vec)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item()
            n_batches  += 1
        sched.step()

        if epoch % 5 == 0 or epoch == 1:
            val_loss = evaluate(model, val_loader, w_vec, DEVICE)
            logger.info(
                f"  Epoch {epoch:3d}/{EPOCHS} │ train_loss={epoch_loss/n_batches:.4f}"
                f"  val_loss={val_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
    logger.info(f"Best val loss: {best_val_loss:.4f}")

    # 8. Evaluate
    logger.info("Evaluating on val + test …")
    val_metrics  = compute_metrics(model, val_loader,  DEVICE, vital_names)
    test_metrics = compute_metrics(model, test_loader, DEVICE, vital_names)
    logger.info("─" * 60)
    logger.info(f"{'Vital':<16} {'Val MAE':>8} {'Test MAE':>9} {'Test R²':>8}")
    logger.info("─" * 60)
    for vm, tm in zip(val_metrics, test_metrics):
        logger.info(f"{tm['target']:<16} {vm['mae']:>8.3f} {tm['mae']:>9.3f} {tm['r2']:>8.4f}")

    # 9. Save artefacts
    model_path = DATA_PROCESSED / "lstm_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "n_features":  n_features,
        "n_targets":   n_targets,
        "feature_cols": feature_cols,
        "target_cols":  target_cols,
        "vital_names":  vital_names,
        "lookback":     LOOKBACK,
        "hidden_size":  128,
    }, model_path)
    logger.info(f"Model saved → {model_path}")

    scaler_path = DATA_PROCESSED / "lstm_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved → {scaler_path}")

    all_metrics = {"val": val_metrics, "test": test_metrics}
    metrics_path = DATA_PROCESSED / "lstm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    elapsed = (time.time() - start) / 60
    logger.info(f"Done in {elapsed:.1f} min")


if __name__ == "__main__":
    main()