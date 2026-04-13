"""
Unit Tests -- Model
7+ model tests covering forward pass shape, gradients, softmax/risk ranges, reproducibility.
"""
import pytest
import torch
import numpy as np

from src.models.lstm_model import VitalLSTM
from src.models.risk_scoring import compute_risk_score

def test_model_01_lstm_forward_pass_shape(mock_tensor_input):
    """Test ID 01: Forward pass shape validation."""
    model = VitalLSTM(n_features=26, n_targets=8, hidden_size=64)
    model.eval()
    with torch.no_grad():
        out = model(mock_tensor_input)
    assert out.shape == (2, 8), "Output shape should be (batch_size, n_targets)"

def test_model_02_lstm_gradients(mock_tensor_input):
    """Test ID 02: Ensure gradients are computed correctly during forward pass."""
    model = VitalLSTM(n_features=26, n_targets=8, hidden_size=64)
    model.train()
    out = model(mock_tensor_input)
    loss = out.sum()
    loss.backward()
    
    # Check if gradients exist for lstm layer
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "Gradients should be populated after backward pass"

def test_model_03_reproducibility():
    """Test ID 03: Reproducibility with set seeds."""
    torch.manual_seed(42)
    model1 = VitalLSTM(n_features=26, n_targets=8, hidden_size=64)
    out1 = model1(torch.ones(1, 12, 26))
    
    torch.manual_seed(42)
    model2 = VitalLSTM(n_features=26, n_targets=8, hidden_size=64)
    out2 = model2(torch.ones(1, 12, 26))
    
    assert torch.allclose(out1, out2), "Outputs should be identical for same seed"

def test_model_04_risk_scoring_normal_range():
    """Test ID 04: Risk score logic inside boundaries."""
    vitals = {
        "map": 85, "heart_rate": 75, "spo2": 98,
        "resp_rate": 16, "temperature": 98.6,
        "sbp": 120, "dbp": 80, "fio2": 21
    }
    result = compute_risk_score(vitals)
    assert result.risk_score == 0.0, "Healthy vitals should result in 0 risk score"
    assert result.severity == "Low"

def test_model_05_risk_scoring_edge_case():
    """Test ID 05: Risk score logic edge case with critical vitals."""
    vitals = {
        "map": 40, "heart_rate": 150, "spo2": 70,
        "resp_rate": 40, "temperature": 105,
        "sbp": 60, "dbp": 30, "fio2": 100
    }
    result = compute_risk_score(vitals)
    assert result.risk_score > 60, "Critical vitals should result in high risk score"
    assert result.severity == "High"

def test_model_06_risk_scoring_missing_vital():
    """Test ID 06: Risk score logic when vitals are missing/NaN."""
    vitals = {
        "map": 85, "heart_rate": np.nan, "spo2": 98,
        "resp_rate": 16, "temperature": None,
        "sbp": 120, "dbp": 80, "fio2": 21
    }
    result = compute_risk_score(vitals)
    assert result.risk_score >= 0, "Should handle missing/None values gracefully"
    assert "heart_rate" not in result.vital_scores, "NaN should not be scored"

def test_model_07_lstm_edge_case_single_timestamp():
    """Test ID 07: LSTM handling minimal lookback."""
    model = VitalLSTM(n_features=26, n_targets=8, hidden_size=64)
    minimal_input = torch.randn(1, 1, 26)
    out = model(minimal_input)
    assert out.shape == (1, 8)

def test_risk_scoring_compute_risk_history():
    from src.models.risk_scoring import compute_risk_history
    import pandas as pd
    df_stay = pd.DataFrame({"stay_id": [1, 1], "hour_idx": [0, 1], "timestamp": ["A", "B"]})
    class MockPredictor:
        def predict(self, stay_id, as_of_hour):
            if as_of_hour == 1:
                raise Exception("Test failure")
            return {"risk_score": 50, "severity": "Mod", "alerts": []}
    
    out = compute_risk_history(df_stay, [], MockPredictor())
    # Should catch exception and only have 1 row
    assert len(out) == 1
    assert out[0]["risk_score"] == 50
