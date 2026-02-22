"""
Data Ingestion Module
Handles loading raw data and initial cohort cleaning
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

def load_raw_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw extracted data from parquet files"""
    logger.info("Loading raw data...")
    
    cohort = pd.read_parquet(data_dir / "cohort.parquet")
    vitals = pd.read_parquet(data_dir / "vitals_hourly.parquet")
    interventions = pd.read_parquet(data_dir / "interventions_hourly.parquet")
    
    logger.info(f"Loaded: {len(cohort)} stays, {len(vitals)} vital measurements, {len(interventions)} interventions")
    
    return cohort, vitals, interventions

def clean_cohort_data(cohort: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate cohort data"""
    logger.info("Cleaning cohort data...")
    
    cohort['intime'] = pd.to_datetime(cohort['intime'])
    cohort['outtime'] = pd.to_datetime(cohort['outtime'])
    
    cohort = cohort.dropna(subset=['intime', 'outtime'])
    
    cohort = cohort[cohort['outtime'] > cohort['intime']]
    
    cohort['los_hours'] = (cohort['outtime'] - cohort['intime']).dt.total_seconds() / 3600
    cohort = cohort[cohort['los_hours'] >= 1.0]
    
    removed = initial_count - len(cohort)
    logger.info(f"Removed {removed} invalid stays. Remaining: {len(cohort)}")
    
    return cohort
