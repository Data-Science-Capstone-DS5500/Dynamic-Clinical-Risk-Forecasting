import pytest
import pandas as pd
import numpy as np
import torch

@pytest.fixture
def sample_vitals_df():
    # Provide a simple dataframe with vitals
    return pd.DataFrame({
        "stay_id": [1, 1, 1, 2, 2],
        "hour_idx": [0, 1, 2, 0, 1],
        "heart_rate": [80, 85, 140, 60, np.nan],
        "map": [70, 65, 45, 90, 85],
        "resp_rate": [16, 18, 32, 12, 14],
        "spo2": [98, 97, 85, 100, 99],
        "temperature": [98.6, 99.1, 102.5, 97.5, 98.0],
        "sbp": [120, 115, 85, 130, 125],
        "dbp": [80, 75, 50, 85, 80],
        "fio2": [21, 21, 65, 21, 21]
    })

@pytest.fixture
def empty_df():
    return pd.DataFrame()

@pytest.fixture
def mock_tensor_input():
    # 2 batches, 12-hour lookback, 26 features
    return torch.randn(2, 12, 26)

@pytest.fixture
def mock_weights():
    # 8 targets
    return torch.ones(8)
