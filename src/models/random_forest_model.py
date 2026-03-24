"""
Random Forest Baseline — Dynamic Clinical Risk Forecasting
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
HORIZON_HOURS  = 6
FORECAST_SUFFIX = "_target"
TRAIN_RATIO    = 0.8
RANDOM_STATE   = 42

RF_PARAMS = dict(
    n_estimators     = 100,
    max_depth        = 10,
    min_samples_leaf = 10,
    max_features     = "sqrt",
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
    oob_score        = False,
)


# Helpers

def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Load parquet feature matrix produced by the preprocessing pipeline."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in {".csv", ".gz"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    log.info("Loaded feature matrix  shape=%s", df.shape)
    return df


def temporal_train_test_split(df: pd.DataFrame, ratio: float = TRAIN_RATIO):

    time_col = "timestamp" if "timestamp" in df.columns else "charttime"
    if time_col in df.columns:
        df = df.sort_values(time_col)
    cut = int(len(df) * ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def get_feature_cols(df: pd.DataFrame, target_cols: list) -> list:

    id_cols  = {"subject_id", "hadm_id", "stay_id", "icustay_id",
                "charttime", "storetime"}
    drop     = set(target_cols) | id_cols
    feats    = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in drop]
    return feats


def train_random_forest(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list,
) -> dict:

    models      = {}
    metrics_all = {}
    importances_all = pd.DataFrame(index=feature_cols)

    for vital in TARGET_VITALS:
        target_col = f"{vital}{FORECAST_SUFFIX}"

        if target_col not in train_df.columns:
            log.warning("Target column '%s' not found — skipping.", target_col)
            continue

        train_clean = train_df[feature_cols + [target_col]].dropna()
        test_clean  = test_df[feature_cols  + [target_col]].dropna()

        if len(train_clean) < 100:
            log.warning("Too few training rows for '%s' — skipping.", target_col)
            continue

        X_train = train_clean[feature_cols].values
        y_train = train_clean[target_col].values
        X_test  = test_clean[feature_cols].values
        y_test  = test_clean[target_col].values

        log.info("Training RF  target=%-30s  train=%d  test=%d",
                 target_col, len(X_train), len(X_test))

        # fit
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_train, y_train)
        models[vital] = rf

        # evaluate
        preds = rf.predict(X_test)
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae   = float(mean_absolute_error(y_test, preds))
        r2    = float(r2_score(y_test, preds))

        metrics_all[vital] = {"rmse": rmse, "mae": mae, "r2": r2}
        log.info("  RMSE=%.4f  MAE=%.4f  R²=%.4f", rmse, mae, r2)

        # feature importance
        importances_all[vital] = rf.feature_importances_

    importances_all["mean_importance"] = importances_all.mean(axis=1)
    importances_all = importances_all.sort_values("mean_importance", ascending=False)

    return {"models": models, "metrics": metrics_all, "importances": importances_all}



def save_artifacts(results: dict, out_dir: Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "models": results["models"],
        "target_vitals": list(results["models"].keys()),
        "horizon": 6,
    }
    bundle_path = out_dir / "rf_models.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Saved model bundle → %s", bundle_path)

    all_metrics = []
    for vital, m in results["metrics"].items():
        all_metrics.append({
            "target": vital,
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
        })
    
    metrics_path = out_dir / "rf_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Saved metrics → %s", metrics_path)

    imp_path = out_dir / "rf_feature_importance.csv"
    results["importances"].to_csv(imp_path)
    log.info("Saved feature importance → %s", imp_path)


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
    log.info("═══ Random Forest — Clinical Risk Forecasting ═══")

    # 1. load data
    df = load_feature_matrix(DATA_PROCESSED / "features.parquet")

    # 2. resolve feature / target columns
    target_cols  = [f"{v}{FORECAST_SUFFIX}" for v in TARGET_VITALS
                    if f"{v}{FORECAST_SUFFIX}" in df.columns]
    feature_cols = get_feature_cols(df, target_cols)
    log.info("Features: %d   Targets: %d", len(feature_cols), len(target_cols))

    # 3. Handling missing values
    log.info("Imputing missing features with 0.0 (baseline)...")
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # 4. chronological split
    train_df, test_df = temporal_train_test_split(df)
    log.info("Train rows: %d   Test rows: %d", len(train_df), len(test_df))

    # 5. train
    results = train_random_forest(train_df, test_df, feature_cols)

    # 6. print summary
    print_summary(results["metrics"])

    # 7. save artifacts
    save_artifacts(results, DATA_PROCESSED)

    log.info("Done.")


if __name__ == "__main__":
    main()
