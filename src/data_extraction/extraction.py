"""
Data Extraction Module
Extracts clinical data (cohort, vitals, interventions) from Google BigQuery (MIMIC-IV dataset)
and saves them as local parquet files.
"""

import logging
from pathlib import Path
from google.cloud import bigquery
import pandas as pd
from datetime import datetime
import sys

project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))

from config import GOOGLE_CLOUD_PROJECT, MIMIC_IV_ICU, VITALS_ITEMIDS, INTERVENTION_ITEMIDS, DATA_RAW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def extract_cohort(client: bigquery.Client) -> pd.DataFrame:
    """Extract ICU stay cohort from MIMIC-IV"""
    logger.info("Extracting cohort...")
    query = f"""
        SELECT stay_id, subject_id, hadm_id, intime, outtime, los
        FROM `{MIMIC_IV_ICU}.icustays`
    """
    df = client.query(query).to_dataframe()
    logger.info(f"  Extracted {len(df):,} ICU stays")
    return df

def extract_vitals(client: bigquery.Client) -> pd.DataFrame:
    """Extract hourly aggregated vital signs from chartevents"""
    logger.info("Extracting vitals (hourly aggregated)...")
    
    # Flatten itemids
    all_item_ids = []
    for ids in VITALS_ITEMIDS.values():
        all_item_ids.extend(ids)
    ids_str = ",".join(map(str, all_item_ids))
    
    query = f"""
        SELECT
            ce.stay_id,
            TIMESTAMP_TRUNC(ce.charttime, HOUR) as chart_hour,
            ce.itemid,
            AVG(ce.valuenum) as val_mean,
            MIN(ce.valuenum) as val_min,
            MAX(ce.valuenum) as val_max
        FROM `{MIMIC_IV_ICU}.chartevents` ce
        INNER JOIN `{MIMIC_IV_ICU}.icustays` ie ON ce.stay_id = ie.stay_id
        WHERE ce.itemid IN ({ids_str})
        AND ce.valuenum IS NOT NULL
        GROUP BY 1, 2, 3
    """
    df = client.query(query).to_dataframe()
    logger.info(f"  Extracted {len(df):,} vital measurement rows")
    return df

def extract_interventions(client: bigquery.Client) -> pd.DataFrame:
    """Extract interventions (vasopressors) from inputevents"""
    logger.info("Extracting interventions...")
    
    # Flatten itemids
    all_item_ids = []
    for ids in INTERVENTION_ITEMIDS.values():
        all_item_ids.extend(ids)
    ids_str = ",".join(map(str, all_item_ids))
    
    query = f"""
        SELECT
            ie.stay_id,
            TIMESTAMP_TRUNC(ie.starttime, HOUR) as start_hour,
            TIMESTAMP_TRUNC(ie.endtime, HOUR) as end_hour,
            ie.itemid,
            MAX(ie.rate) as max_rate
        FROM `{MIMIC_IV_ICU}.inputevents` ie
        INNER JOIN `{MIMIC_IV_ICU}.icustays` s ON ie.stay_id = s.stay_id
        WHERE ie.itemid IN ({ids_str})
        AND ie.rate IS NOT NULL
        GROUP BY 1, 2, 3, 4
    """
    df = client.query(query).to_dataframe()
    logger.info(f"  Extracted {len(df):,} intervention rows")
    return df

def main():
    """Main extraction routine"""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("STARTING DATA EXTRACTION")
    logger.info("=" * 60)
    
    client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT)
    
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    cohort = extract_cohort(client)
    cohort.to_parquet(DATA_RAW / "cohort.parquet", index=False)
    
    vitals = extract_vitals(client)
    vitals.to_parquet(DATA_RAW / "vitals_hourly.parquet", index=False)
    
    interventions = extract_interventions(client)
    interventions.to_parquet(DATA_RAW / "interventions_hourly.parquet", index=False)
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"  Time elapsed: {elapsed:.2f} minutes")
    logger.info(f"  Files saved to: {DATA_RAW}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
