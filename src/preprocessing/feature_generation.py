"""
Feature Generation Module
Handles rolling window engineering and forecast target creation
"""
import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame, vital_cols: List[str], 
                     window_size: int = 6) -> pd.DataFrame:
    """Generate rolling window features"""
    logger.info(f"Engineering features with {window_size}h rolling window...")
    
    df = df.sort_values(['stay_id', 'hour_idx'])
    
    for col in vital_cols:
        if col not in df.columns:
            continue
        
        grouped = df.groupby('stay_id')[col]
        
        # Rolling mean
        df[f'{col}_mean_{window_size}h'] = grouped.transform(
            lambda x: x.rolling(window_size, min_periods=1).mean()
        )
        
        # Rolling std
        df[f'{col}_std_{window_size}h'] = grouped.transform(
            lambda x: x.rolling(window_size, min_periods=1).std()
        )
    
    logger.info("Feature engineering complete")
    
    return df

def create_forecast_targets(df: pd.DataFrame, vital_cols: List[str],
                           forecast_horizon: int = 6) -> pd.DataFrame:
    """Create forecast targets by shifting future values"""
    logger.info(f"Creating {forecast_horizon}h forecast targets...")
    
    df = df.sort_values(['stay_id', 'hour_idx'])
    
    for col in vital_cols:
        if col not in df.columns:
            continue
        
        df[f'{col}_target'] = df.groupby('stay_id')[col].shift(-forecast_horizon)
    
    logger.info("Forecast targets created")
    
    return df
