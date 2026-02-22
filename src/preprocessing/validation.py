"""
Validation Module
Handles data integrity checks and quality reporting
"""
import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame, vital_cols: List[str], intervention_cols: List[str]) -> Dict[str, any]:
    """Validate processed data and generate quality report"""
    logger.info("Validating processed data...")
    
    report = {
        'total_rows': len(df),
        'total_stays': df['stay_id'].nunique(),
        'total_patients': df['subject_id'].nunique(),
        'date_range': (df['timestamp'].min(), df['timestamp'].max()),
        'missing_values': {},
        'value_ranges': {}
    }
    
    for col in vital_cols + intervention_cols:
        if col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            report['missing_values'][col] = f"{missing_pct:.2f}%"
    
    for col in vital_cols:
        if col in df.columns:
            report['value_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median())
            }
    
    logger.info("Validation complete")
    
    return report

def remove_incomplete_sequences(df: pd.DataFrame, 
                               min_sequence_length: int = 12) -> pd.DataFrame:
    """Remove stays with insufficient data for modeling"""
    logger.info(f"Removing stays with < {min_sequence_length} hours of data...")
    
    stay_lengths = df.groupby('stay_id').size()
    valid_stays = stay_lengths[stay_lengths >= min_sequence_length].index
    
    initial_count = len(df)
    df = df[df['stay_id'].isin(valid_stays)]
    removed = initial_count - len(df)
    
    logger.info(f"Removed {removed} rows from short stays. Remaining: {len(df)}")
    
    return df
