"""
ClinicalPredictor — Unified Inference Interface
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
LOOKBACK = 12   

ID_COLS_TO_DROP = {
    "stay_id", "subject_id", "hadm_id", "icustay_id", 
    "charttime", "storetime", "timestamp", "intime", "outtime"
}


class ClinicalPredictor:
    
    VERSION = "2.3-FullReload"

    def __init__(
        self,
        model_type: Literal["xgboost", "lstm", "gru"] = "xgboost",
        features_path: Optional[Path] = None,
    ):
        self.model_type    = model_type
        self.features_path = features_path or (DATA_PROCESSED / "features.parquet")
        
        # Model paths
        self.xgb_path  = DATA_PROCESSED / "baseline_xgb_models.pkl"
        self.lstm_path = DATA_PROCESSED / "lstm_model.pt"
        self.gru_path  = DATA_PROCESSED / "gru_models.pt"
        self.scaler_path = DATA_PROCESSED / "lstm_scaler.pkl"

        self._features_df = None
        self._models = {}
        self._scalers = {}
        self._load_model(model_type)

    def _load_model(self, mtype):
        if mtype == "xgboost": self._load_xgboost()
        elif mtype == "lstm": self._load_lstm()
        elif mtype == "gru": self._load_gru()

    def _load_xgboost(self):
        with open(self.xgb_path, "rb") as f:
            self._models["xgboost"] = pickle.load(f)
        logger.info("XGBoost loaded.")

    def _load_gru(self):
        import torch
        from src.models.gru_model import GRUForecaster
        bundle = torch.load(self.gru_path, map_location="cpu", weights_only=False)
        
        target_vitals = bundle["target_vitals"]
        n_feat = bundle["scalers"][target_vitals[0]].n_features_in_
        
        models = {}
        for vital in target_vitals:
            m = GRUForecaster(n_features=n_feat, hidden_size=64, num_layers=2)
            m.load_state_dict(bundle["model_states"][vital])
            m.eval()
            models[vital] = m
            
        self._models["gru"] = {
            "models": models,
            "scalers": bundle["scalers"],
            "vitals": target_vitals,
            "n_feat": n_feat
        }
        logger.info(f"GRU loaded ({len(target_vitals)} targets).")

    def _load_lstm(self):
        import torch
        from src.models.lstm_model import VitalLSTM
        bundle = torch.load(self.lstm_path, map_location="cpu", weights_only=False)
        model = VitalLSTM(n_features=bundle["n_features"], n_targets=bundle["n_targets"])
        model.load_state_dict(bundle["model_state"])
        model.eval()
        with open(self.scaler_path, "rb") as f:
            scaler = pickle.load(f)
        self._models["lstm"] = {"model": model, "scaler": scaler, "vitals": bundle["vital_names"]}
        logger.info("LSTM loaded.")

    def load_features(self):
        self._features_df = pd.read_parquet(self.features_path)
        return self._features_df

    def list_stays(self):
        return sorted(self._features_df["stay_id"].unique().tolist())

    def predict(self, stay_id: int, as_of_hour: int) -> Dict:
        """Standard prediction for a patient at a specific hour."""
        df_stay = self._features_df[self._features_df["stay_id"] == stay_id]
        window = df_stay[df_stay["hour_idx"] <= as_of_hour]
        last_row = window.iloc[-1:]
        
        forecasts = self._run_model_inference(self.model_type, df_stay, as_of_hour, last_row)
        risk = compute_risk_score(forecasts)
        current = {v: self._last_observed(window, v) for v in VITAL_COLS}
        
        return {
            "risk_score": risk.risk_score, "severity": risk.severity,
            "forecasts": forecasts, "current_vitals": current, "alerts": risk.alerts,
            "vital_scores": risk.vital_scores
        }

    def predict_manual(self, feature_vector: pd.DataFrame, stay_id: Optional[int] = None, as_of_hour: Optional[int] = None) -> Dict:
        
        mtype = self.model_type
        
        df_stay = None
        if stay_id is not None and self._features_df is not None:
            df_stay = self._features_df[self._features_df["stay_id"] == stay_id]
            
        forecasts = self._run_model_inference(mtype, df_stay, as_of_hour, feature_vector)
        risk = compute_risk_score(forecasts)
        return {"risk_score": risk.risk_score, "severity": risk.severity}

    def simulate_intervention_evolution(self, stay_id: int, as_of_hour: int, perturbations: Dict[str, float]) -> Dict:
        
        # A. Baseline
        base_result = self.predict(stay_id, as_of_hour)
        base_risk_t6 = base_result["risk_score"]
        
        # B. Current Risk (at T=0)
        df_stay = self._features_df[self._features_df["stay_id"] == stay_id]
        window = df_stay[df_stay["hour_idx"] <= as_of_hour]
        last_row = window.iloc[-1]
        
        # Extract the vitals at T
        current_physio = {v: self._last_observed(window, v) for v in VITAL_COLS}
        # Compute risk score for current hour
        risk_t0 = compute_risk_score(current_physio).risk_score
        
        # C. Simulated Path (T+6)
        feat_vector = window.iloc[-1:].copy()
        for vital, new_val in perturbations.items():
            if vital in feat_vector.columns:
                feat_vector[vital] = new_val
        
        sim_forecasts = self._run_model_inference(self.model_type, df_stay, as_of_hour, feat_vector)
        sim_risk_t6   = compute_risk_score(sim_forecasts).risk_score
        
        return {
            "hour_t0": as_of_hour,
            "hour_t6": as_of_hour + 6,
            "risk_t0": risk_t0,
            "baseline_t6": base_risk_t6,
            "simulated_t6": sim_risk_t6,
            "baseline_forecasts": base_result["forecasts"],
            "simulated_forecasts": sim_forecasts
        }

    def _run_model_inference(self, mtype, df_stay, as_of_hour, last_row):
        if mtype == "xgboost": return self._predict_xgb(last_row)
        if mtype == "gru": return self._predict_rnn("gru", df_stay, as_of_hour, last_row)
        if mtype == "lstm": return self._predict_rnn("lstm", df_stay, as_of_hour, last_row)

    def _predict_xgb(self, row):
        m = self._models["xgboost"]
        f_cols = m["feature_cols"]
        X = row[f_cols].fillna(0.0).values
        return {v: float(m["models"][v].predict(m["scalers"][v].transform(X))[0]) for v in VITAL_COLS}

    def _predict_rnn(self, mtype, df_stay, as_of_hour, last_row=None):
        import torch
        m_info = self._models[mtype]
        
        if df_stay is not None and as_of_hour is not None:
            
            numeric_cols = [c for c in df_stay.columns 
                            if c not in ID_COLS_TO_DROP and not c.endswith("_target")
                            and pd.api.types.is_numeric_dtype(df_stay[c])]
            
            if mtype == "lstm":
                feat_cols = [c for c in numeric_cols if c != "hour_idx"]
            else:
                feat_cols = numeric_cols
                
            history = df_stay[df_stay["hour_idx"] < as_of_hour].tail(LOOKBACK - 1)
            raw_history = history[feat_cols].fillna(0.0).values
            raw_manual  = last_row[feat_cols].fillna(0.0).values
            raw_seq     = np.vstack([raw_history, raw_manual]) if len(raw_history) > 0 else raw_manual
        else:

            feat_cols = [c for c in last_row.columns 
                         if c not in ID_COLS_TO_DROP and not c.endswith("_target")]
            expected = 26 if mtype == "lstm" else 27
            if len(feat_cols) > expected:
                feat_cols = [c for c in feat_cols if c != "hour_idx"] if mtype == "lstm" else feat_cols
            
            raw_seq = last_row[feat_cols].fillna(0.0).values
            
        if len(raw_seq) == 0:
            return {v: None for v in VITAL_COLS}
            
        if len(raw_seq) < LOOKBACK:
            pad = np.zeros((LOOKBACK - len(raw_seq), raw_seq.shape[1]))
            raw_seq = np.vstack([pad, raw_seq])
            
        forecasts = {}
        if mtype == "lstm":
            model, scaler, vitals = m_info["model"], m_info["scaler"], m_info["vitals"]
            seq_s = scaler.transform(raw_seq).reshape(1, LOOKBACK, -1)
            with torch.no_grad():
                pred = model(torch.tensor(seq_s, dtype=torch.float32)).squeeze().numpy()
            forecasts = {v: float(pred[vitals.index(v)]) if v in vitals else None for v in VITAL_COLS}
        
        elif mtype == "gru":
            models, scalers, vitals = m_info["models"], m_info["scalers"], m_info["vitals"]
            for v in VITAL_COLS:
                if v in vitals:
                    m, s = models[v], scalers[v]
                    seq_s = s.transform(raw_seq).reshape(1, LOOKBACK, -1)
                    with torch.no_grad():
                        pred = m(torch.tensor(seq_s, dtype=torch.float32)).item()
                    forecasts[v] = float(pred)
                else:
                    forecasts[v] = None
                    
        return forecasts

    def get_stay_history(self, stay_id: int) -> pd.DataFrame:
        
        self._require_features()
        return self._get_stay(stay_id)

    def get_feature_importance(self, vital_name: str, top_n: int = 10) -> pd.DataFrame:
        
        if self.model_type != "xgboost":
            logger.warning("Feature importance only available for XGBoost models.")
            return pd.DataFrame(columns=["feature", "importance"])

        fi_path = DATA_PROCESSED / "feature_importance" / f"{vital_name}_importance.csv"
        if fi_path.exists():
            return pd.read_csv(fi_path).head(top_n)

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

    # XGBoost inference

    def _predict_xgboost(self, window: pd.DataFrame) -> Dict[str, Optional[float]]:
        bundle     = self._xgb_bundle
        models     = bundle["models"]
        scalers    = bundle["scalers"]
        feat_cols  = bundle["feature_cols"]

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

    # LSTM inference

    def _predict_lstm(self, df_stay: pd.DataFrame, as_of_hour: int) -> Dict[str, Optional[float]]:
        import torch

        meta      = self._lstm_meta
        feat_cols = meta["feature_cols"]
        vital_names = meta["vital_names"]

        window    = df_stay[df_stay["hour_idx"] <= as_of_hour].tail(LOOKBACK)
        if len(window) == 0:
            return {v: None for v in VITAL_COLS}

        seq = window[feat_cols].fillna(0.0).values.astype(np.float32)

        if len(seq) < LOOKBACK:
            pad = np.zeros((LOOKBACK - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, seq])

        n_feat = seq.shape[1]
        seq_2d = seq.reshape(-1, n_feat)
        seq_s  = self._lstm_scaler.transform(seq_2d).reshape(1, LOOKBACK, n_feat)

        with torch.no_grad():
            # Get model device
            device = next(self._lstm_model.parameters()).device
            tensor = torch.tensor(seq_s, dtype=torch.float32).to(device)
            pred   = self._lstm_model(tensor).squeeze(0).cpu().numpy()

        forecasts = {}
        for val_name in VITAL_COLS:
            if val_name in vital_names:
                idx = vital_names.index(val_name)
                forecasts[val_name] = float(pred[idx])
            else:
                forecasts[val_name] = None
        return forecasts

    # Helpers

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
