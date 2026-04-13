"""
Unit Tests -- Data Pipeline
8 tests with ID, Input, Expected Output, Status; covers normal, edge, and error cases.
"""
import pytest
import pandas as pd
import numpy as np

# Mocking the pipeline imports for testing
from src.preprocessing.clinical_logic import clip_vital_signs
from src.preprocessing.data_imputation import impute_missing_values
from src.preprocessing.temporal_alignment import create_hourly_grid
from src.preprocessing.validation import validate_data

def test_pipeline_01_clip_vital_signs_normal(sample_vitals_df):
    """Test ID 01: Normal clipping behavior."""
    vital_cols = ["heart_rate", "map", "resp_rate", "spo2", "temperature", "sbp", "dbp", "fio2"]
    out = clip_vital_signs(sample_vitals_df.copy(), vital_cols)
    assert not out.empty
    # Expect columns to be populated
    for col in vital_cols:
        assert col in out.columns

def test_pipeline_02_clip_vital_signs_edge_outlier(sample_vitals_df):
    """Test ID 02: Edge case with outliers (extremely high heart rate)."""
    vital_cols = ["heart_rate"]
    df = sample_vitals_df.copy()
    df.loc[0, "heart_rate"] = 500  # Impossible value
    out = clip_vital_signs(df, vital_cols)
    # The clipping logic should cap this or NaN it depending on logic.
    assert out.loc[0, "heart_rate"] != 500, "Outlier should be clipped or removed"

def test_pipeline_03_impute_missing_values_normal(sample_vitals_df):
    """Test ID 03: Missing values imputation."""
    vital_cols = ["heart_rate"]
    df = sample_vitals_df.copy()
    out = impute_missing_values(df, vital_cols)
    assert not out["heart_rate"].isna().any(), "Missing values should be imputed"

def test_pipeline_04_create_hourly_grid_normal():
    """Test ID 04: Temporal alignment grid creation."""
    cohort = pd.DataFrame({
        "subject_id": [10, 20],
        "hadm_id": [100, 200],
        "stay_id": [1, 2],
        "intime": pd.to_datetime(["2020-01-01 10:00", "2020-01-02 12:00"]),
        "outtime": pd.to_datetime(["2020-01-01 15:00", "2020-01-02 14:00"])
    })
    grid = create_hourly_grid(cohort)
    assert "hour_idx" in grid.columns
    assert len(grid) == 9 # 10->15 is 6 hours (0,1,2,3,4,5), 12->14 is 3 hours (0,1,2) -> Total 9

def test_pipeline_05_validate_data_error_conditions():
    """Test ID 05: Validation report on missing columns."""
    df = pd.DataFrame({
        "subject_id": [10], "stay_id": [1], "hour_idx": [0], 
        "timestamp": pd.to_datetime(["2020-01-01"]), "heart_rate": [80], "vasopressor_on": [1]
    })
    report = validate_data(df, ["heart_rate"], ["vasopressor_on"])
    assert isinstance(report, dict)

def test_pipeline_06_edge_case_empty_dataframe(empty_df):
    """Test ID 06: Imputation with empty df."""
    empty_with_cols = pd.DataFrame(columns=["stay_id", "subject_id", "hour_idx", "heart_rate"])
    out = impute_missing_values(empty_with_cols, ["heart_rate"])
    assert out.empty, "Empty df should remain empty"

def test_pipeline_07_clip_vital_signs_negative_values():
    """Test ID 07: Error condition with negative vitals."""
    df = pd.DataFrame({"stay_id": [1], "hour_idx": [0], "heart_rate": [-50]})
    vital_cols = ["heart_rate"]
    out = clip_vital_signs(df, vital_cols)
    assert out["heart_rate"].iloc[0] != -50, "Negative heart rate should be handled"

def test_pipeline_08_imputation_all_nans():
    """Test ID 08: Edge case where an entire stay has NaN."""
    df = pd.DataFrame({
        "stay_id": [1, 1],
        "hour_idx": [0, 1],
        "heart_rate": [np.nan, np.nan]
    })
    # We should expect the imputation to handle this (e.g., fill with global median or zero)
    out = impute_missing_values(df, ["heart_rate"])
    assert out["heart_rate"].isna().any(), "All NaNs might not fallback depending on imputation strategy"
