"""
Build hourly feature matrix with vitals + intervention states.
FIXED: Proper timestamp alignment for merging.
"""

import pandas as pd
import numpy as np
from config import DATA_PROCESSED, LOOKBACK_WINDOW, FORECAST_HORIZON

def create_hourly_grid(icustays: pd.DataFrame) -> pd.DataFrame:
    """Create hourly timestamps for each ICU stay."""
    
    rows = []
    for _, stay in icustays.iterrows():
        intime = pd.to_datetime(stay["intime"])
        outtime = pd.to_datetime(stay["outtime"])
        
        # Floor intime to the hour for consistent alignment
        intime_floored = intime.floor("h")
        
        # Generate hourly timestamps
        hours = pd.date_range(start=intime_floored, end=outtime, freq="h")
        
        for i, ts in enumerate(hours):
            rows.append({
                "stay_id": stay["stay_id"],
                "hour_idx": i,
                "timestamp": ts,
            })
    
    return pd.DataFrame(rows)


def aggregate_vitals_hourly(vitals: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vitals to hourly buckets."""
    
    vitals = vitals.copy()
    
    # Floor charttime to hour
    vitals["hour_bucket"] = vitals["charttime"].dt.floor("h")
    
    # Aggregate: mean value per (stay_id, hour, vital)
    hourly_vitals = vitals.groupby(
        ["stay_id", "hour_bucket", "vital_name"]
    )["valuenum"].mean().unstack(fill_value=np.nan)
    
    hourly_vitals = hourly_vitals.reset_index()
    hourly_vitals.rename(columns={"hour_bucket": "timestamp"}, inplace=True)
    
    # Debug: check what we have before merge
    print(f"\nVitals aggregated: {len(hourly_vitals)} rows")
    print(f"Grid rows: {len(grid)}")
    print(f"Vitals stay_ids: {hourly_vitals['stay_id'].nunique()}")
    print(f"Grid stay_ids: {grid['stay_id'].nunique()}")
    
    # Check timestamp overlap
    vitals_ts = set(hourly_vitals["timestamp"].unique())
    grid_ts = set(grid["timestamp"].unique())
    overlap = len(vitals_ts & grid_ts)
    print(f"Timestamp overlap: {overlap} hours")
    
    # Merge
    merged = grid.merge(hourly_vitals, on=["stay_id", "timestamp"], how="left")
    
    # Check merge success
    vital_cols = [c for c in merged.columns if c in [
        "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature", "fio2"
    ]]
    
    print(f"\nPost-merge check:")
    for col in vital_cols:
        if col in merged.columns:
            non_null = merged[col].notna().sum()
            total = len(merged)
            print(f"  {col}: {non_null}/{total} ({non_null/total:.1%}) non-null")
    
    return merged


def create_intervention_flags(interventions: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Create binary vasopressor flags for each hour."""
    
    grid = grid.copy()
    grid["vasopressor_on"] = 0
    grid["vasopressor_rate"] = 0.0
    
    # Convert timestamps
    interventions = interventions.copy()
    interventions["starttime"] = pd.to_datetime(interventions["starttime"])
    interventions["endtime"] = pd.to_datetime(interventions["endtime"])
    
    vaso_count = 0
    for _, row in interventions.iterrows():
        if pd.isna(row["starttime"]) or pd.isna(row["endtime"]):
            continue
            
        mask = (
            (grid["stay_id"] == row["stay_id"]) &
            (grid["timestamp"] >= row["starttime"]) &
            (grid["timestamp"] < row["endtime"])
        )
        
        if mask.sum() > 0:
            vaso_count += mask.sum()
            grid.loc[mask, "vasopressor_on"] = 1
            if pd.notna(row["rate"]):
                grid.loc[mask, "vasopressor_rate"] = np.maximum(
                    grid.loc[mask, "vasopressor_rate"], 
                    row["rate"]
                )
    
    print(f"\nIntervention flags set for {vaso_count} hourly slots")
    
    return grid


def add_lookback_features(df: pd.DataFrame, vital_cols: list) -> pd.DataFrame:
    """Add rolling statistics as features."""
    
    df = df.sort_values(["stay_id", "hour_idx"]).copy()
    
    for col in vital_cols:
        if col not in df.columns:
            continue
        
        # Forward fill within stay first (carry last observation)
        df[col] = df.groupby("stay_id")[col].ffill()
        
        # Rolling stats
        grouped = df.groupby("stay_id")[col]
        
        df[f"{col}_mean_6h"] = grouped.transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df[f"{col}_std_6h"] = grouped.transform(
            lambda x: x.rolling(6, min_periods=1).std()
        )
        df[f"{col}_trend_6h"] = grouped.transform(
            lambda x: x - x.shift(6)
        )
    
    return df


def create_targets(df: pd.DataFrame, vital_cols: list) -> pd.DataFrame:
    """Create forecast targets (t+6h values)."""
    
    df = df.sort_values(["stay_id", "hour_idx"]).copy()
    
    for col in vital_cols:
        if col not in df.columns:
            continue
        
        # Target: value 6 hours ahead
        df[f"{col}_target"] = df.groupby("stay_id")[col].shift(-FORECAST_HORIZON)
    
    # Binary outcome: MAP deterioration
    if "map" in df.columns and "map_target" in df.columns:
        df["deterioration_target"] = (
            df["map_target"] < (df["map"] - 10)
        ).astype(float)
    
    return df


def main():
    print("Loading processed data...")
    vitals = pd.read_parquet(DATA_PROCESSED / "vitals.parquet")
    interventions = pd.read_parquet(DATA_PROCESSED / "interventions.parquet")
    icustays = pd.read_parquet(DATA_PROCESSED / "icustays.parquet")
    
    print(f"ICU stays: {len(icustays)}")
    print(f"Vitals records: {len(vitals):,}")
    print(f"Intervention records: {len(interventions):,}")
    
    # Quick check on vitals data
    print(f"\nVitals sample timestamps:")
    print(vitals["charttime"].head())
    
    print("\nCreating hourly grid...")
    grid = create_hourly_grid(icustays)
    print(f"Total hourly slots: {len(grid):,}")
    print(f"Grid timestamp sample:")
    print(grid["timestamp"].head())
    
    print("\nAggregating vitals...")
    features = aggregate_vitals_hourly(vitals, grid)
    
    print("\nAdding intervention flags...")
    features = create_intervention_flags(interventions, features)
    
    # Identify vital columns
    vital_cols = [c for c in features.columns if c in [
        "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature", "fio2"
    ]]
    print(f"\nVital columns found: {vital_cols}")
    
    print("\nAdding lookback features (with forward-fill)...")
    features = add_lookback_features(features, vital_cols)
    
    print("\nCreating targets...")
    features = create_targets(features, vital_cols)
    
    # Final missing check
    print(f"\n{'='*50}")
    print("FINAL DATA QUALITY CHECK")
    print(f"{'='*50}")
    print(f"Total rows: {len(features):,}")
    print(f"Vasopressor exposure: {features['vasopressor_on'].mean():.1%} of hours")
    
    for col in vital_cols:
        missing = features[col].isna().mean()
        print(f"{col}: {missing:.1%} missing after forward-fill")
    
    # Save
    features.to_parquet(DATA_PROCESSED / "features.parquet", index=False)
    print(f"\nSaved to {DATA_PROCESSED / 'features.parquet'}")
    
    # Also save a sample for quick inspection
    sample = features[features["map"].notna()].head(20)
    print(f"\nSample of rows with MAP data:")
    print(sample[["stay_id", "hour_idx", "timestamp", "map", "heart_rate", "vasopressor_on"]].to_string())


if __name__ == "__main__":
    main()