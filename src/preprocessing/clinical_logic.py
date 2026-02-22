"""
Clinical Logic Module
Handles vital sign mapping, intervention processing, and outlier clipping
"""
import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Clinical validation ranges
PHYSIOLOGICAL_RANGES = {
    'heart_rate': (20, 250),
    'sbp': (40, 250),
    'dbp': (20, 150),
    'map': (20, 200),
    'resp_rate': (4, 60),
    'spo2': (50, 100),
    'temperature': (80, 110),
    'fio2': (21, 100)
}

def process_vitals(vitals: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Process and merge vital signs into hourly grid"""
    logger.info("Processing vital signs...")
    
    itemid_mapping = {
        220045: 'heart_rate',
        220050: 'sbp', 220179: 'sbp',
        220051: 'dbp', 220180: 'dbp',
        220052: 'map', 220181: 'map', 225312: 'map',
        220210: 'resp_rate', 224690: 'resp_rate',
        220277: 'spo2',
        223761: 'temperature', 223762: 'temperature',
        223835: 'fio2', 227009: 'fio2', 227010: 'fio2'
    }
    
    vitals = vitals.copy()
    vitals['vital_name'] = vitals['itemid'].map(itemid_mapping)
    vitals = vitals.dropna(subset=['vital_name'])
    
    vitals = vitals.rename(columns={'chart_hour': 'timestamp'})
    
    v_pivot = vitals.pivot_table(
        index=['stay_id', 'timestamp'],
        columns='vital_name',
        values='val_mean',
        aggfunc='mean'
    ).reset_index()
    
    features = grid.merge(v_pivot, on=['stay_id', 'timestamp'], how='left')
    
    vital_cols = [col for col in v_pivot.columns if col not in ['stay_id', 'timestamp']]
    logger.info(f"Processed {len(vital_cols)} vital signs: {vital_cols}")
    
    return features, vital_cols

def process_interventions(interventions: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Process and merge interventions into hourly grid"""
    logger.info("Processing interventions...")
    
    valid_stays = set(grid['stay_id'].unique())
    interventions = interventions[interventions['stay_id'].isin(valid_stays)].copy()
    
    interventions['duration_hours'] = (
        (interventions['end_hour'] - interventions['start_hour']).dt.total_seconds() / 3600
    ).astype(int) + 1
    
    int_exploded = interventions.loc[interventions.index.repeat(interventions['duration_hours'])].copy()
    int_exploded['offset'] = int_exploded.groupby(level=0).cumcount()
    int_exploded['timestamp'] = int_exploded['start_hour'] + pd.to_timedelta(int_exploded['offset'], unit='h')
    
    int_hourly = int_exploded.groupby(['stay_id', 'timestamp'], as_index=False)['max_rate'].max()
    int_hourly['vasopressor_on'] = 1
    int_hourly.rename(columns={'max_rate': 'vasopressor_rate'}, inplace=True)
    
    features = grid.merge(int_hourly, on=['stay_id', 'timestamp'], how='left')
    
    features['vasopressor_on'] = features['vasopressor_on'].fillna(0)
    features['vasopressor_rate'] = features['vasopressor_rate'].fillna(0.0)
    
    intervention_cols = ['vasopressor_on', 'vasopressor_rate']
    logger.info(f"Processed interventions: {intervention_cols}")
    
    return features, intervention_cols

def clip_vital_signs(df: pd.DataFrame, vital_cols: List[str]) -> pd.DataFrame:
    """Clip vital signs to physiological ranges to handle sensor errors"""
    logger.info("Applying physiological clipping to vitals...")
    
    for col in vital_cols:
        if col not in df.columns:
            continue
        
        # Handle FiO2 specifically (often mixed 0-1 and 21-100)
        if col == 'fio2':
            mask_fraction = (df[col] > 0) & (df[col] <= 1.0)
            df.loc[mask_fraction, col] = df.loc[mask_fraction, col] * 100
            
        if col in PHYSIOLOGICAL_RANGES:
            v_min, v_max = PHYSIOLOGICAL_RANGES[col]
            
            outliers = ((df[col] < v_min) | (df[col] > v_max)).sum()
            if outliers > 0:
                logger.warning(f"  {col}: Clipping {outliers} outliers to [{v_min}, {v_max}]")
            
            df[col] = df[col].clip(lower=v_min, upper=v_max)
            
    return df
