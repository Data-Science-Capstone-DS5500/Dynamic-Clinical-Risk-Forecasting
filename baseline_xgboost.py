"""
Baseline XGBoost Model Training
Trains XGBoost regressors for 6-hour vital sign forecasting.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import time
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

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

TARGET_COLS = [f"{v}_target" for v in VITAL_COLS]

PRIMARY_TARGETS = ["map_target", "heart_rate_target"]

META_COLS = ["stay_id", "subject_id", "hadm_id", "timestamp", "hour_idx",
             "intime", "outtime"]


def load_features(path: Path) -> pd.DataFrame:
    """Load preprocessed feature matrix from parquet."""
    logger.info(f"Loading features from {path}")
    df = pd.read_parquet(path)
    logger.info(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"  Stays : {df['stay_id'].nunique():,}")
    return df


def prepare_data(df: pd.DataFrame, target_col: str, feature_cols: list):
    """
    Prepare train / validation / test splits.

    Uses GroupShuffleSplit so that all hours belonging to the same ICU stay
    end up in the same split (prevents data leakage).

    Split ratios: 70% train / 15% validation / 15% test
    """
    mask = df[target_col].notna()
    df_valid = df.loc[mask].copy()
    logger.info(f"  Rows with valid target ({target_col}): {len(df_valid):,}")

    X = df_valid[feature_cols].values
    y = df_valid[target_col].values
    groups = df_valid["stay_id"].values

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    groups_temp = groups[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))

    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    logger.info(f"  Train : {len(X_train):,}  |  Val : {len(X_val):,}  |  Test : {len(X_test):,}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_name: str,
) -> xgb.XGBRegressor:
    """
    Train an XGBoost regressor with early stopping.

    Parameters
    ----------
    X_train, y_train : Training arrays
    X_val, y_val     : Validation arrays (used for early stopping)
    target_name      : Name of the target for logging

    Returns
    -------
    Trained XGBRegressor
    """
    logger.info(f"  Training XGBoost for: {target_name}")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="mae",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration
    logger.info(f"    Best iteration: {best_iter}")

    return model


def evaluate_model(
    model: xgb.XGBRegressor,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    target_name: str,
) -> dict:
    """
    Compute regression metrics on a given split.

    Returns dict with MAE, RMSE, R² and the predictions.
    """
    y_pred = model.predict(X)

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)

    logger.info(
        f"    {split_name:5s} │ MAE {mae:8.3f}  │ RMSE {rmse:8.3f}  │ R² {r2:.4f}"
    )

    return {
        "split": split_name,
        "target": target_name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "n_samples": len(y),
    }


def get_feature_importance(model: xgb.XGBRegressor, feature_names: list) -> pd.DataFrame:
    """
    Extract and sort feature importances.
    """
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def main():
    """
    End-to-end baseline XGBoost training pipeline.

    Steps
    -----
    1. Load preprocessed features  (features.parquet)
    2. For each target vital sign:
       a. Split data (patient-grouped 70/15/15)
       b. Scale features
       c. Train XGBoost with early stopping
       d. Evaluate on val + test
    3. Save models, scaler, feature importances, and metrics
    """
    start = time.time()

    logger.info("  BASELINE XGBOOST TRAINING PIPELINE")

    features_path = DATA_PROCESSED / "features.parquet"
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.error("Run preprocessing first:  python POC/preprocessing.py")
        return

    df = load_features(features_path)

    missing_feats = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feats:
        logger.warning(f"Missing feature columns: {missing_feats}")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    missing_targets = [c for c in TARGET_COLS if c not in df.columns]
    if missing_targets:
        logger.warning(f"Missing target columns: {missing_targets}")

    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    models = {}
    scalers = {}
    all_metrics = []
    all_importances = {}

    for target_col in TARGET_COLS:
        if target_col not in df.columns:
            logger.warning(f"Skipping {target_col} – column not found")
            continue

        vital_name = target_col.replace("_target", "")
        logger.info("")
        logger.info(f"TARGET: {vital_name}   (predict {FORECAST_HORIZON}h ahead)")

        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(df, target_col, feature_cols)

        scaler = fit_scaler(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        model = train_xgboost(X_train_s, y_train, X_val_s, y_val, vital_name)

        val_metrics  = evaluate_model(model, X_val_s,  y_val,  "Val",  vital_name)
        test_metrics = evaluate_model(model, X_test_s, y_test, "Test", vital_name)

        all_metrics.extend([val_metrics, test_metrics])

        fi = get_feature_importance(model, feature_cols)
        all_importances[vital_name] = fi
        logger.info(f"  Top 5 features:")
        for _, row in fi.head(5).iterrows():
            logger.info(f"    {row['feature']:30s}  {row['importance']:.4f}")

        models[vital_name] = model
        scalers[vital_name] = scaler

    logger.info("")
    logger.info("  SAVING ARTEFACTS")

    bundle = {
        "models": models,
        "scalers": scalers,
        "feature_cols": feature_cols,
        "target_cols": TARGET_COLS,
        "forecast_horizon": FORECAST_HORIZON,
    }
    bundle_path = DATA_PROCESSED / "baseline_xgb_models.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"  Models saved   → {bundle_path}")

    metrics_path = DATA_PROCESSED / "baseline_xgb_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"  Metrics saved  → {metrics_path}")

    fi_dir = DATA_PROCESSED / "feature_importance"
    fi_dir.mkdir(exist_ok=True)
    for vital_name, fi_df in all_importances.items():
        fi_path = fi_dir / f"{vital_name}_importance.csv"
        fi_df.to_csv(fi_path, index=False)
    logger.info(f"  Importances    → {fi_dir}/")

    elapsed = time.time() - start
    logger.info("")
    logger.info("  TRAINING COMPLETE")
    logger.info(f"  Time elapsed : {elapsed / 60:.1f} min")
    logger.info(f"  Models       : {len(models)}")
    logger.info(f"  Features     : {len(feature_cols)}")
    logger.info("")

    print("\n" + "=" * 70)
    print(f"{'Target':<16} {'Split':<6} {'MAE':>8} {'RMSE':>8} {'R²':>8}  {'N':>10}")
    print("-" * 70)
    for m in all_metrics:
        print(f"{m['target']:<16} {m['split']:<6} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.4f}  {m['n_samples']:>10,}")
    print("=" * 70)

    print(f"\nModels saved to: {bundle_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Feature importances saved to: {fi_dir}/")


if __name__ == "__main__":
    main()
