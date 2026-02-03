"""
Train XGBoost model to forecast vitals conditioned on interventions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import pickle
from config import DATA_PROCESSED

def prepare_training_data(df: pd.DataFrame, target_col: str):
    """Prepare X, y for training."""
    
    # Feature columns (exclude targets and identifiers)
    exclude_cols = [
        "stay_id", "timestamp", "hour_idx"
    ] + [c for c in df.columns if "_target" in c]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Drop rows with missing target
    valid_mask = df[target_col].notna()
    df_valid = df[valid_mask].copy()
    
    X = df_valid[feature_cols]
    y = df_valid[target_col]
    
    # Handle remaining NaNs in features (simple fill for PoC)
    X = X.fillna(X.median())
    
    return X, y, feature_cols


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost regressor."""
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    return model


def evaluate_model(model, X_test, y_test, target_name):
    """Evaluate model performance."""
    
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"\n{target_name} Forecast Performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²:  {r2:.3f}")
    
    return {"mae": mae, "r2": r2, "predictions": preds, "actuals": y_test.values}


def main():
    print("Loading features...")
    df = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    
    # Train models for key vitals
    targets = ["map_target", "heart_rate_target", "spo2_target"]
    models = {}
    results = {}
    
    for target in targets:
        if target not in df.columns:
            print(f"Skipping {target} - not available")
            continue
            
        print(f"\n{'='*50}")
        print(f"Training model for: {target}")
        
        X, y, feature_cols = prepare_training_data(df, target)
        print(f"Training samples: {len(X):,}")
        
        if len(X) < 100:
            print(f"Insufficient data for {target}, skipping")
            continue
        
        # Split by stay_id to avoid leakage
        stay_ids = df.loc[y.index, "stay_id"].unique()
        train_stays, test_stays = train_test_split(stay_ids, test_size=0.2, random_state=42)
        
        train_mask = df.loc[y.index, "stay_id"].isin(train_stays)
        test_mask = df.loc[y.index, "stay_id"].isin(test_stays)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Further split train for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        model = train_model(X_tr, y_tr, X_val, y_val)
        results[target] = evaluate_model(model, X_test, y_test, target)
        models[target] = model
        
        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        print(f"\nTop 10 features for {target}:")
        print(importance.head(10).to_string(index=False))
    
    # Save models
    model_path = DATA_PROCESSED / "models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols}, f)
    
    print(f"\nModels saved to {model_path}")
    
    return models, results


if __name__ == "__main__":
    models, results = main()