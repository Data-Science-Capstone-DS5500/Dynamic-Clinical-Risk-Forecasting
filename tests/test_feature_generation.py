import pytest
import pandas as pd
import numpy as np

try:
    from src.preprocessing.feature_generation import engineer_features, create_forecast_targets
except ImportError:
    pass

def test_engineer_features():
    df = pd.DataFrame({
        "stay_id": [1, 1, 1],
        "hour_idx": [0, 1, 2],
        "heart_rate": [80, 90, 100]
    })
    
    out = engineer_features(df, ["heart_rate"], window_size=2)
    assert 'heart_rate_mean_2h' in out.columns
    assert 'heart_rate_std_2h' in out.columns
    assert out['heart_rate_mean_2h'].iloc[2] == 95.0 # (90+100)/2
    
def test_create_forecast_targets():
    df = pd.DataFrame({
        "stay_id": [1, 1, 1],
        "hour_idx": [0, 1, 2],
        "heart_rate": [80, 90, 100]
    })
    
    out = create_forecast_targets(df, ["heart_rate"], forecast_horizon=1)
    assert 'heart_rate_target' in out.columns
    assert out['heart_rate_target'].iloc[0] == 90.0
    assert np.isnan(out['heart_rate_target'].iloc[-1]) # Last element has no future
