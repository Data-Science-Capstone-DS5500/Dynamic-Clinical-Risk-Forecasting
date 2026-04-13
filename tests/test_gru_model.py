import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path

try:
    from src.models.gru_model import GRUForecaster, main, get_feature_cols, build_sequences
except ImportError:
    pass

@pytest.fixture
def mock_gru_features_df():
    return pd.DataFrame({
        "stay_id": [1]*50 + [2]*50,
        "charttime": pd.date_range("2020-01-01", periods=100, freq="h"),
        "map": np.random.rand(100),
        "heart_rate": np.random.rand(100),
        "map_target": np.random.rand(100),
        "heart_rate_target": np.random.rand(100),
    })

def test_gru_forecaster_shape():
    model = GRUForecaster(n_features=5, hidden_size=16, num_layers=1)
    x = torch.randn(2, 10, 5) # batch, seq, features
    out = model(x)
    assert out.shape == (2,) # Should map to a scalar prediction per sequence

def test_gru_get_feature_cols(mock_gru_features_df):
    target_cols = ["map_target", "heart_rate_target"]
    feat_cols = get_feature_cols(mock_gru_features_df, target_cols)
    assert "map" in feat_cols
    assert "heart_rate" in feat_cols
    assert "stay_id" not in feat_cols
    assert "charttime" not in feat_cols

def test_gru_build_sequences(mock_gru_features_df):
    target_cols = ["map_target", "heart_rate_target"]
    feat_cols = get_feature_cols(mock_gru_features_df, target_cols)
    X, y = build_sequences(mock_gru_features_df, feat_cols, "map_target", seq_len=12)
    assert len(X) == len(mock_gru_features_df) - 12
    assert X.shape[1:] == (12, len(feat_cols))

@patch('src.models.gru_model.DATA_PROCESSED', Path("/mock_path"))
@patch('torch.save')
@patch('builtins.open')
@patch('pathlib.Path.mkdir')
@patch('src.models.gru_model.load_feature_matrix')
@patch('src.models.gru_model.MAX_EPOCHS', 1)
@patch('src.models.gru_model.BATCH_SIZE', 8)
def test_gru_main(mock_load_feature, mock_mkdir, mock_open, mock_save, mock_gru_features_df):
    mock_load_feature.return_value = mock_gru_features_df
    
    # Needs to bypass CUDA checks naturally and run simple iteration
    main()
    
    mock_load_feature.assert_called_once()
