import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path

try:
    from src.preprocessing.pipeline import run_pipeline
except ImportError:
    pass

@patch('src.preprocessing.pipeline.validate_data')
@patch('src.preprocessing.pipeline.remove_incomplete_sequences')
@patch('src.preprocessing.pipeline.create_forecast_targets')
@patch('src.preprocessing.pipeline.engineer_features')
@patch('src.preprocessing.pipeline.impute_missing_values')
@patch('src.preprocessing.pipeline.clip_vital_signs')
@patch('src.preprocessing.pipeline.process_interventions')
@patch('src.preprocessing.pipeline.process_vitals')
@patch('src.preprocessing.pipeline.create_hourly_grid')
@patch('src.preprocessing.pipeline.clean_cohort_data')
@patch('src.preprocessing.pipeline.load_raw_data')
def test_run_pipeline(mock_load, mock_clean, mock_grid, mock_proc_vits, mock_proc_ints,
                      mock_clip, mock_impute, mock_eng, mock_targ, mock_remove, mock_val):
    
    # Set up basic mock returns
    mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_proc_vits.return_value = (MagicMock(), MagicMock())
    mock_proc_ints.return_value = (MagicMock(), MagicMock())
    mock_val.return_value = {'total_rows': 100, 'total_stays': 5, 'Date Range': 0}
    
    mock_final_df = MagicMock()
    mock_remove.return_value = mock_final_df
    
    run_pipeline(Path("/mock/raw"), Path("/mock/out.parquet"))
    
    # Assert all parts of the pipeline are orchestrated
    mock_load.assert_called_once()
    mock_clean.assert_called_once()
    mock_grid.assert_called_once()
    mock_proc_vits.assert_called_once()
    mock_proc_ints.assert_called_once()
    mock_clip.assert_called_once()
    mock_impute.assert_called_once()
    mock_eng.assert_called_once()
    mock_targ.assert_called_once()
    mock_remove.assert_called_once()
    mock_val.assert_called_once()
    mock_final_df.to_parquet.assert_called_once()
