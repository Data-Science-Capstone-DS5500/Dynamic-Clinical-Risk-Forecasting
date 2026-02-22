"""
Data Imputation Module
Handles forward-fill and global median imputation
"""
import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

def impute_missing_values(df: pd.DataFrame, vital_cols: List[str]) -> pd.DataFrame:
    """Impute missing values using forward-fill and global median"""
    logger.info("Imputing missing values...")
    
    df = df.sort_values(['stay_id', 'hour_idx'])
    
    for col in vital_cols:
        if col not in df.columns:
            continue
        
        df[col] = df.groupby('stay_id')[col].ffill()
        
        median = df[col].median()
        df[col] = df[col].fillna(median)
    
    logger.info("Missing value imputation complete")
    
    return df
