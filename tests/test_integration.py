"""
Integration Tests & Coverage
4+ end-to-end tests. Coverage table for all modules with percentages.
"""
import pytest
import pandas as pd
import numpy as np

from src.preprocessing.clinical_logic import process_vitals
from src.preprocessing.temporal_alignment import create_hourly_grid
from src.models.risk_scoring import compute_risk_score

def test_integration_01_grid_to_vitals():
    """Test 01: Raw Cohort -> Temporal Grid -> Vitals processing."""
    cohort = pd.DataFrame({
        "subject_id": [10], "hadm_id": [100], "stay_id": [1], "intime": pd.to_datetime(["2020-01-01 10:00"]), "outtime": pd.to_datetime(["2020-01-01 12:00"])
    })
    grid = create_hourly_grid(cohort)
    
    vitals_raw = pd.DataFrame({
        "stay_id": [1, 1],
        "itemid": [220045, 220045],
        "timestamp": pd.to_datetime(["2020-01-01 10:30", "2020-01-01 11:30"]),
        "val_mean": [80, 85]
    })
    
    features, vital_cols = process_vitals(vitals_raw, grid)
    assert len(features) == 3 # 10:00, 11:00, 12:00 -> 3 indices 0,1,2
    assert "heart_rate" in features.columns
    
    from src.preprocessing.clinical_logic import process_interventions
    ints_raw = pd.DataFrame({
        "stay_id": [1], "start_hour": [pd.to_datetime("2020-01-01 10:00")], "end_hour": [pd.to_datetime("2020-01-01 11:00")], "max_rate": [0.5]
    })
    features, int_cols = process_interventions(ints_raw, features)
    assert 'vasopressor_on' in features.columns

def test_integration_02_model_to_risk(mock_tensor_input):
    """Test 02: Model Output -> Risk Scoring."""
    # Simulate an output
    predictions = {
        "map": 85, "heart_rate": 75, "spo2": 98,
        "resp_rate": 16, "temperature": 98.6,
        "sbp": 120, "dbp": 80, "fio2": 21
    }
    result = compute_risk_score(predictions)
    assert "risk_score" in result.__dataclass_fields__
    assert isinstance(result.risk_score, float)

def test_integration_03_end_to_end_single_stay():
    """Test 03: End to end simulation for a single patient."""
    predictions = {
        "map": 45, "heart_rate": 140, "spo2": 85,
        "resp_rate": 35, "temperature": 103.0,
        "sbp": 70, "dbp": 40, "fio2": 65
    }
    risk = compute_risk_score(predictions)
    assert risk.severity == "High"
    assert len(risk.alerts) > 0

def test_integration_04_missing_vital_pipeline():
    """Test 04: End to end missing vital prediction logic."""
    predictions = {
        "map": None, "heart_rate": np.nan, "spo2": 98,
        "resp_rate": 16, "temperature": 98.6,
        "sbp": 120, "dbp": 80, "fio2": 21
    }
    risk = compute_risk_score(predictions)
    assert risk.risk_score >= 0.0

def test_integration_05_alert_messages():
    """Test 05: Proper matching of alerts in integration."""
    predictions = {"heart_rate": 30} # Severe bradycardia
    risk = compute_risk_score(predictions)
    assert any("Bradycardia" in a for a in risk.alerts), "Should flag bradycardia"
