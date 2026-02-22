"""
Temporal Alignment Module
Handles creation of the continuous hourly timestamp grid
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_hourly_grid(cohort: pd.DataFrame) -> pd.DataFrame:
    """Create continuous hourly timestamp grid for each ICU stay"""
    logger.info("Creating hourly grid...")
    
    rows = []
    for _, stay in cohort.iterrows():
        intime = pd.to_datetime(stay['intime']).floor('h')
        outtime = pd.to_datetime(stay['outtime']).ceil('h')
        
        hours = pd.date_range(start=intime, end=outtime, freq='h')
        
        for hour_idx, timestamp in enumerate(hours):
            rows.append({
                'stay_id': stay['stay_id'],
                'subject_id': stay['subject_id'],
                'hadm_id': stay['hadm_id'],
                'timestamp': timestamp,
                'hour_idx': hour_idx,
                'intime': stay['intime'],
                'outtime': stay['outtime']
            })
    
    grid = pd.DataFrame(rows)
    logger.info(f"Created grid with {len(grid)} hourly observations")
    
    return grid
