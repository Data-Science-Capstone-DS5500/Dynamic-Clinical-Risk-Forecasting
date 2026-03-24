"""
GRU Model — Dynamic Clinical Risk Forecasting
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# project imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DATA_PROCESSED  # noqa: E402

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# constants
TARGET_VITALS = [
    "heart_rate", "sbp", "dbp", "map",
    "resp_rate", "spo2", "temperature", "fio2",
]
HORIZON_HOURS   = 6
FORECAST_SUFFIX = "_target"
TRAIN_RATIO     = 0.8
RANDOM_STATE    = 42

SEQ_LEN      = 12     # look-back window (hours)
HIDDEN_SIZE  = 64     
NUM_LAYERS   = 2      # stacked GRU layers
DROPOUT      = 0.2
BATCH_SIZE   = 256
MAX_EPOCHS   = 1
PATIENCE     = 7      # early stopping
LR           = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUForecaster(nn.Module):
    """
    Stacked GRU → fully-connected head for single-step regression.

    Architecture:
    Input  : (batch, seq_len, n_features)
    GRU    : num_layers stacked GRU cells with dropout between layers
    Output : last hidden state → Linear → scalar prediction
    """

    def __init__(self, n_features: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out: (batch, seq_len, hidden)  h_n: (num_layers, batch, hidden)
        _, h_n = self.gru(x)
        # take the last layer's hidden state
        last_hidden = h_n[-1]           
        return self.head(last_hidden).squeeze(-1)


# Helpers
def load_feature_matrix(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in {".csv", ".gz"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    log.info("Loaded feature matrix  shape=%s", df.shape)
    return df


def temporal_train_test_split(df: pd.DataFrame, ratio: float = TRAIN_RATIO):
    time_col = "charttime"
    if time_col in df.columns:
        df = df.sort_values(time_col)
    cut = int(len(df) * ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def get_feature_cols(df: pd.DataFrame, target_cols: list) -> list:
    id_cols = {"subject_id", "hadm_id", "stay_id", "icustay_id",
               "charttime", "storetime"}
    drop    = set(target_cols) | id_cols
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in drop]


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray]:

    clean = df[feature_cols + [target_col]].dropna().values
    feat  = clean[:, :-1]
    tgt   = clean[:, -1]

    X, y = [], []
    for i in range(seq_len, len(feat)):
        X.append(feat[i - seq_len : i])
        y.append(tgt[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_loader(X: np.ndarray, y: np.ndarray,
                batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, pin_memory=(DEVICE.type == "cuda"))



def train_one_target(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list,
    target_col: str,
    vital_name: str,
) -> dict:

    X_tr, y_tr = build_sequences(train_df, feature_cols, target_col)
    X_te, y_te = build_sequences(test_df,  feature_cols, target_col)

    if len(X_tr) < BATCH_SIZE:
        log.warning("Too few training sequences for '%s' — skipping.",
                    target_col)
        return {}

    # scale features
    scaler   = StandardScaler()
    n_tr, sl, nf = X_tr.shape
    X_tr_2d  = X_tr.reshape(-1, nf)
    X_te_2d  = X_te.reshape(-1, nf)
    X_tr_sc  = scaler.fit_transform(X_tr_2d).reshape(n_tr, sl, nf)
    X_te_sc  = scaler.transform(X_te_2d).reshape(len(X_te), sl, nf)

    train_loader = make_loader(X_tr_sc, y_tr, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_te_sc, y_te, BATCH_SIZE, shuffle=False)

    # model, loss, optimiser
    model     = GRUForecaster(n_features=nf).to(DEVICE)
    criterion = nn.HuberLoss()           # robust to ICU outliers
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )

    best_val_loss  = float("inf")
    patience_count = 0
    train_losses   = []
    val_losses     = []
    best_state     = None

    log.info("Training GRU  target=%-30s  train_seq=%d  test_seq=%d  device=%s",
             target_col, len(X_tr), len(X_te), DEVICE)


    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimiser.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(X_batch)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            log.info("  epoch %3d/%d  train=%.5f  val=%.5f",
                     epoch, MAX_EPOCHS, epoch_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info("  Early stopping at epoch %d", epoch)
                break

    if best_state is None:
        return {}
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            all_preds.append(model(X_batch.to(DEVICE)).cpu().numpy())
            all_true.append(y_batch.numpy())

    preds_np = np.concatenate(all_preds)
    true_np  = np.concatenate(all_true)

    rmse = float(np.sqrt(mean_squared_error(true_np, preds_np)))
    mae  = float(mean_absolute_error(true_np, preds_np))
    r2   = float(r2_score(true_np, preds_np))

    log.info("  RMSE=%.4f  MAE=%.4f  R²=%.4f", rmse, mae, r2)

    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "train_losses": train_losses, "val_losses": val_losses,
        "scaler": scaler,
        "model_state": best_state,
    }


def save_artifacts(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_states": {v: r["model_state"] for v, r in results.items()},
        "scalers":      {v: r["scaler"]      for v, r in results.items()},
        "target_vitals": list(results.keys()),
        "horizon":      6,
    }
    bundle_path = out_dir / "gru_models.pt"
    torch.save(bundle, bundle_path)
    log.info("Saved model bundle → %s", bundle_path)

    all_metrics = []
    for vital, r in results.items():
        all_metrics.append({
            "target": vital,
            "rmse": r["rmse"],
            "mae": r["mae"],
            "r2": r["r2"],
        })
    
    metrics_path = out_dir / "gru_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Saved metrics → %s", metrics_path)



def print_summary(metrics: dict) -> None:
    header = f"\n{'Vital Sign':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8}"
    print(header)
    print("─" * len(header))
    for vital, m in metrics.items():
        print(f"{vital:<20} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f}")
    rmse_vals = [m["rmse"] for m in metrics.values()]
    print(f"\n{'Average RMSE':<20} {np.mean(rmse_vals):>8.4f}")


# Main
def main():
    log.info("═══ GRU Model — Clinical Risk Forecasting ═══")
    log.info("Device: %s", DEVICE)

    out_dir = DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load data
    df = load_feature_matrix(DATA_PROCESSED / "features.parquet")

    # 2. columns
    target_cols  = [f"{v}{FORECAST_SUFFIX}" for v in TARGET_VITALS
                    if f"{v}{FORECAST_SUFFIX}" in df.columns]
    feature_cols = get_feature_cols(df, target_cols)
    log.info("Features: %d   Targets: %d", len(feature_cols), len(target_cols))

    # 3. split
    train_df, test_df = temporal_train_test_split(df)
    log.info("Train rows: %d   Test rows: %d", len(train_df), len(test_df))

    # 4. train one GRU per vital sign
    all_results = {}
    all_curves  = {}

    for vital in TARGET_VITALS:
        target_col = f"{vital}{FORECAST_SUFFIX}"
        if target_col not in df.columns:
            log.warning("Skipping '%s' — column not found.", target_col)
            continue

        result = train_one_target(
            train_df, test_df, feature_cols,
            target_col, vital,
        )
        if result:
            all_results[vital] = result
            all_curves[vital]  = {
                "train": result.pop("train_losses"),
                "val":   result.pop("val_losses"),
            }

    # 5. print summary
    print_summary(all_results)

    # 6. save artifacts
    save_artifacts(all_results, out_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
