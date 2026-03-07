"""
ClinicalPredictor — Unified Inference Interface
Loads a trained model bundle (XGBoost or LSTM), accepts a stay_id +
as_of_hour, and returns 6-hour-ahead vital forecasts plus a composite
clinical risk score.
"""

import pickle
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Literal

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DATA_PROCESSED, FORECAST_HORIZON
from src.models.risk_scoring import compute_risk_score, RiskResult

logger = logging.getLogger(__name__)

VITAL_COLS = [
    "heart_rate", "sbp", "dbp", "map",
    "resp_rate", "spo2", "temperature", "fio2",
]
LOOKBACK = 12   # hours — must match lstm_model.py


class ClinicalPredictor:
    """
    Unified inference class for vital sign forecasting.

    Parameters
    ----------
    model_type : "xgboost" or "lstm"
    xgb_bundle_path  : path to baseline_xgb_models.pkl
    lstm_bundle_path : path to lstm_model.pt + lstm_scaler.pkl
    features_path    : path to features.parquet (pre-computed feature matrix)

    Usage
    -----
    >>> predictor = ClinicalPredictor(model_type="xgboost")
    >>> predictor.load_features()
    >>> result = predictor.predict(stay_id=30003600, as_of_hour=10)
    >>> print(result["risk_score"], result["severity"])
    """

    def __init__(
        self,
        model_type: Literal["xgboost", "lstm"] = "xgboost",
        xgb_bundle_path: Optional[Path] = None,
        lstm_bundle_path: Optional[Path] = None,
        lstm_scaler_path: Optional[Path] = None,
        features_path: Optional[Path] = None,
    ):
        self.model_type       = model_type
        self.xgb_bundle_path  = xgb_bundle_path  or (DATA_PROCESSED / "baseline_xgb_models.pkl")
        self.lstm_bundle_path = lstm_bundle_path  or (DATA_PROCESSED / "lstm_model.pt")
        self.lstm_scaler_path = lstm_scaler_path  or (DATA_PROCESSED / "lstm_scaler.pkl")
        self.features_path    = features_path     or (DATA_PROCESSED / "features.parquet")

        self._features_df: Optional[pd.DataFrame] = None
        self._xgb_bundle  = None
        self._lstm_model   = None
        self._lstm_scaler  = None
        self._lstm_meta    = None

        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        if self.model_type == "xgboost":
            self._load_xgboost()
        elif self.model_type == "lstm":
            self._load_lstm()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _load_xgboost(self):
        if not self.xgb_bundle_path.exists():
            raise FileNotFoundError(
                f"XGBoost bundle not found: {self.xgb_bundle_path}\n"
                "Run: python src/models/baseline_xgboost.py"
            )
        with open(self.xgb_bundle_path, "rb") as f:
            self._xgb_bundle = pickle.load(f)
        logger.info(f"XGBoost bundle loaded from {self.xgb_bundle_path}")

    def _load_lstm(self):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for LSTM inference. pip install torch")
        if not self.lstm_bundle_path.exists():
            raise FileNotFoundError(
                f"LSTM checkpoint not found: {self.lstm_bundle_path}\n"
                "Run: python src/models/lstm_model.py"
            )
        from src.models.lstm_model import VitalLSTM
        bundle = torch.load(self.lstm_bundle_path, map_location="cpu", weights_only=False)
        model  = VitalLSTM(
            n_features  = bundle["n_features"],
            n_targets   = bundle["n_targets"],
            hidden_size = bundle.get("hidden_size", 128),
        )
        model.load_state_dict(bundle["model_state"])
        model.eval()
        self._lstm_model = model
        self._lstm_meta  = bundle

        with open(self.lstm_scaler_path, "rb") as f:
            self._lstm_scaler = pickle.load(f)
        logger.info(f"LSTM model loaded from {self.lstm_bundle_path}")

    # ── Feature loading ───────────────────────────────────────────────────────

    def load_features(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Load and cache the feature matrix. Call once before predicting."""
        p = path or self.features_path
        if not p.exists():
            raise FileNotFoundError(
                f"Feature matrix not found: {p}\n"
                "Run the preprocessing pipeline first."
            )
        self._features_df = pd.read_parquet(p)
        logger.info(
            f"Features loaded: {self._features_df.shape[0]:,} rows, "
            f"{self._features_df['stay_id'].nunique():,} stays."
        )
        return self._features_df

    def list_stays(self):
        """Return sorted list of available stay_ids."""
        self._require_features()
        return sorted(self._features_df["stay_id"].unique().tolist())

    # ── Core prediction ───────────────────────────────────────────────────────

    def predict(self, stay_id: int, as_of_hour: int) -> Dict:
        """
        Forecast 6h-ahead vital signs and compute risk score.

        Parameters
        ----------
        stay_id     : ICU stay identifier (must exist in features.parquet)
        as_of_hour  : hour_idx to treat as "now" (0-indexed from ICU admission)

        Returns
        -------
        dict with keys:
            stay_id, as_of_hour, model_type,
            forecasts      : {vital_name: predicted_value, ...}
            risk_score     : float 0–100
            severity       : "Low" | "Moderate" | "High"
            vital_scores   : {vital_name: sub_score, ...}
            alerts         : [alert_message, ...]
            current_vitals : {vital_name: current_value, ...}  (last observed)
        """
        self._require_features()
        df_stay = self._get_stay(stay_id)
        window  = self._get_window(df_stay, as_of_hour)

        if self.model_type == "xgboost":
            forecasts = self._predict_xgboost(window)
        else:
            forecasts = self._predict_lstm(df_stay, as_of_hour)

        risk = compute_risk_score(forecasts)

        current_vitals = {
            v: self._last_observed(window, v) for v in VITAL_COLS
        }

        return {
            "stay_id":       stay_id,
            "as_of_hour":    as_of_hour,
            "model_type":    self.model_type,
            "forecasts":     forecasts,
            "risk_score":    risk.risk_score,
            "severity":      risk.severity,
            "vital_scores":  risk.vital_scores,
            "alerts":        risk.alerts,
            "current_vitals": current_vitals,
        }

    def get_stay_history(self, stay_id: int) -> pd.DataFrame:
        """Return the full hourly feature rows for a single stay."""
        self._require_features()
        return self._get_stay(stay_id)

    def get_feature_importance(self, vital_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Return top-N feature importances for a given vital (XGBoost only).

        Returns empty DataFrame for LSTM (global importance not available).
        """
        if self.model_type != "xgboost":
            logger.warning("Feature importance only available for XGBoost models.")
            return pd.DataFrame(columns=["feature", "importance"])

        fi_path = DATA_PROCESSED / "feature_importance" / f"{vital_name}_importance.csv"
        if fi_path.exists():
            return pd.read_csv(fi_path).head(top_n)

        # Fallback: compute inline from model
        models = self._xgb_bundle["models"]
        feature_cols = self._xgb_bundle["feature_cols"]
        if vital_name not in models:
            raise ValueError(f"No model found for vital '{vital_name}'.")
        importances = models[vital_name].feature_importances_
        return (
            pd.DataFrame({"feature": feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ── XGBoost inference ─────────────────────────────────────────────────────

    def _predict_xgboost(self, window: pd.DataFrame) -> Dict[str, Optional[float]]:
        bundle     = self._xgb_bundle
        models     = bundle["models"]
        scalers    = bundle["scalers"]
        feat_cols  = bundle["feature_cols"]

        # Use the last row of the window as the feature vector
        last_row = window.iloc[[-1]][feat_cols].fillna(0.0).values

        forecasts = {}
        for vital in VITAL_COLS:
            if vital not in models:
                forecasts[vital] = None
                continue
            scaler  = scalers[vital]
            X_s     = scaler.transform(last_row)
            forecasts[vital] = float(models[vital].predict(X_s)[0])
        return forecasts

    # ── LSTM inference ────────────────────────────────────────────────────────

    def _predict_lstm(self, df_stay: pd.DataFrame, as_of_hour: int) -> Dict[str, Optional[float]]:
        import torch

        meta      = self._lstm_meta
        feat_cols = meta["feature_cols"]
        vital_names = meta["vital_names"]

        # Build sequence: last LOOKBACK rows up to as_of_hour
        window    = df_stay[df_stay["hour_idx"] <= as_of_hour].tail(LOOKBACK)
        if len(window) == 0:
            return {v: None for v in VITAL_COLS}

        seq = window[feat_cols].fillna(0.0).values.astype(np.float32)
        # Pad front if shorter than LOOKBACK
        if len(seq) < LOOKBACK:
            pad = np.zeros((LOOKBACK - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, seq])

        n_feat = seq.shape[1]
        seq_2d = seq.reshape(-1, n_feat)
        seq_s  = self._lstm_scaler.transform(seq_2d).reshape(1, LOOKBACK, n_feat)

        with torch.no_grad():
            tensor = torch.tensor(seq_s, dtype=torch.float32)
            pred   = self._lstm_model(tensor).squeeze(0).numpy()

        forecasts = {}
        for val_name in VITAL_COLS:
            if val_name in vital_names:
                idx = vital_names.index(val_name)
                forecasts[val_name] = float(pred[idx])
            else:
                forecasts[val_name] = None
        return forecasts

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_features(self):
        if self._features_df is None:
            raise RuntimeError("Call load_features() before predicting.")

    def _get_stay(self, stay_id: int) -> pd.DataFrame:
        df = self._features_df
        stay = df[df["stay_id"] == stay_id].sort_values("hour_idx")
        if stay.empty:
            raise ValueError(f"stay_id {stay_id} not found in features.")
        return stay

    def _get_window(self, df_stay: pd.DataFrame, as_of_hour: int) -> pd.DataFrame:
        window = df_stay[df_stay["hour_idx"] <= as_of_hour]
        if window.empty:
            raise ValueError(
                f"No data at or before hour {as_of_hour} for this stay."
            )
        return window

    @staticmethod
    def _last_observed(window: pd.DataFrame, vital: str) -> Optional[float]:
        if vital not in window.columns:
            return None
        series = window[vital].dropna()
        return float(series.iloc[-1]) if not series.empty else None
