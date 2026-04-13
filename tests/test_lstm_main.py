import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

try:
    from src.models.lstm_model import main
    import src.models.lstm_model as lstm_mod
except ImportError:
    pass

@patch('src.models.lstm_model.EPOCHS', 1) # Force 1 epoch
@patch('src.models.lstm_model.DATA_PROCESSED', Path("/mock_path"))
@patch('src.models.lstm_model.DataLoader')
@patch('pandas.read_parquet')
@patch('torch.save')
@patch('pickle.dump')
@patch('builtins.open')
def test_lstm_main(mock_open, mock_pickle, mock_torch_save, mock_read_parquet, mock_dataloader):
    import pandas as pd
    import numpy as np
    
    # Create fake processed dataset matching the expectations
    df = pd.DataFrame({
        "stay_id": [1]*15 + [2]*15 + [3]*15,
        "hour_idx": list(range(15)) * 3,
        "map": np.random.rand(45),
        "heart_rate": np.random.rand(45),
        "map_target": np.random.rand(45),
        "heart_rate_target": np.random.rand(45),
    })
    
    mock_read_parquet.return_value = df
    
    import torch
    mock_dataloader.return_value = [(torch.randn(2, 12, 2), torch.randn(2, 2))]
    
    # To avoid Path.exists() error
    with patch('pathlib.Path.exists', return_value=True):
        main()
    
    mock_read_parquet.assert_called_once()
    mock_torch_save.assert_called_once()
    mock_pickle.assert_called_once()
