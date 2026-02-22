"""
Central Preprocessing Pipeline
Orchestrates the modular preprocessing scripts to transform raw MIMIC-IV data 
into a feature matrix for forecasting.
"""
import logging
import time
from pathlib import Path
import pandas as pd

# Import modular components (absolute imports from src)
try:
    from src.preprocessing.data_ingestion import load_raw_data, clean_cohort_data
    from src.preprocessing.temporal_alignment import create_hourly_grid
    from src.preprocessing.clinical_logic import process_vitals, process_interventions, clip_vital_signs
    from src.preprocessing.data_imputation import impute_missing_values
    from src.preprocessing.feature_generation import engineer_features, create_forecast_targets
    from src.preprocessing.validation import validate_data, remove_incomplete_sequences
except ImportError:
    # Fallback for local execution
    from data_ingestion import load_raw_data, clean_cohort_data
    from temporal_alignment import create_hourly_grid
    from clinical_logic import process_vitals, process_interventions, clip_vital_signs
    from data_imputation import impute_missing_values
    from feature_generation import engineer_features, create_forecast_targets
    from validation import validate_data, remove_incomplete_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(data_dir: Path, output_path: Path, 
                 window_size: int = 6, forecast_horizon: int = 6,
                 min_sequence_length: int = 12):
    """
    Run the end-to-end preprocessing pipeline by calling modular scripts
    """
    start_time = time.time()
    logger.info("STARTING MODULAR PREPROCESSING PIPELINE")
    
    # 1. Data Ingestion
    cohort_raw, vitals_raw, interventions_raw = load_raw_data(data_dir)
    cohort = clean_cohort_data(cohort_raw)
    
    # 2. Temporal Alignment
    grid = create_hourly_grid(cohort)
    
    # 3. Clinical Logic (Vitals & Interventions)
    features, vital_cols = process_vitals(vitals_raw, grid)
    features, intervention_cols = process_interventions(interventions_raw, features)
    
    # 4. Outlier Handling (Physiological Clipping)
    features = clip_vital_signs(features, vital_cols)
    
    # 5. Data Imputation
    features = impute_missing_values(features, vital_cols)
    
    # 6. Feature Generation
    features = engineer_features(features, vital_cols, window_size)
    features = create_forecast_targets(features, vital_cols, forecast_horizon)
    
    # 7. Quality Control & Filtering
    features = remove_incomplete_sequences(features, min_sequence_length)
    
    # 8. Validation Report
    report = validate_data(features, vital_cols, intervention_cols)
    
    # Save Final Features
    logger.info(f"Saving processed features to {output_path}")
    features.to_parquet(output_path, index=False)
    
    # Final Summary
    elapsed = (time.time() - start_time) / 60
    logger.info("MODULAR PREPROCESSING COMPLETE")
    logger.info(f"Total rows      : {report['total_rows']:,}")
    logger.info(f"Total stays     : {report['total_stays']:,}")
    logger.info(f"Time elapsed    : {elapsed:.2f} minutes")
    logger.info(f"Output saved to : {output_path}")
    
    return features

if __name__ == "__main__":
    import sys
    
    # Ensure project root is in sys.path
    project_root = Path(__file__).parent.parent.parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    # Import config after path setup
    try:
        from src.config import DATA_RAW, DATA_PROCESSED
    except ImportError:
        # Fallback if running from within src/preprocessing
        sys.path.append(str(project_root / "src"))
        from config import DATA_RAW, DATA_PROCESSED
    
    # Run the pipeline
    # We ingest from RAW and save to PROCESSED
    run_pipeline(
        data_dir=DATA_RAW,
        output_path=DATA_PROCESSED / "features.parquet"
    )

