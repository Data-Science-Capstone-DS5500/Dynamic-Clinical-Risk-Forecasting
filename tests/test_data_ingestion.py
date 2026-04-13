import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from pathlib import Path

try:
    from src.preprocessing.data_ingestion import load_raw_data, clean_cohort_data
except ImportError:
    pass

@patch('pandas.read_parquet')
def test_load_raw_data(mock_read_parquet):
    # Setup mock returns
    mock_df = pd.DataFrame({'test': [1]})
    mock_read_parquet.return_value = mock_df
    
    cohort, vitals, interventions = load_raw_data(Path("/mock/dir"))
    
    assert len(cohort) == 1
    assert mock_read_parquet.call_count == 3

def test_clean_cohort_data():
    cohort_raw = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "intime": ["2020-01-01 10:00", "2020-01-02 10:00", "2020-01-03 10:00"],
        "outtime": ["2020-01-01 15:00", "2020-01-02 10:30", None] # 2 is <1h, 3 is None
    })
    
    cleaned = clean_cohort_data(cohort_raw)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['stay_id'] == 1
    assert 'los_hours' in cleaned.columns
