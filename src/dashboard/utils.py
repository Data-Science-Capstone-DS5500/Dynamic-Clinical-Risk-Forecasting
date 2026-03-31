"""
Dashboard utility helpers — data loading, caching, and formatting.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

VITAL_DISPLAY_NAMES = {
    "heart_rate":  "Heart Rate",
    "sbp":         "Systolic BP",
    "dbp":         "Diastolic BP",
    "map":         "Mean Art. Pressure",
    "resp_rate":   "Resp. Rate",
    "spo2":        "SpO₂",
    "temperature": "Temperature",
    "fio2":        "FiO₂",
}

VITAL_UNITS = {
    "heart_rate":  "bpm",
    "sbp":         "mmHg",
    "dbp":         "mmHg",
    "map":         "mmHg",
    "resp_rate":   "br/min",
    "spo2":        "%",
    "temperature": "°F",
    "fio2":        "%",
}

VITAL_NORMAL_RANGES = {
    "heart_rate":  (60, 100),
    "sbp":         (90, 140),
    "dbp":         (60, 90),
    "map":         (65, 100),
    "resp_rate":   (12, 20),
    "spo2":        (90, 100),
    "temperature": (97, 100),
    "fio2":        (21, 40),
}

SEVERITY_COLORS = {
    "Low":      "#22c55e",   
    "Moderate": "#f59e0b",   
    "High":     "#ef4444",   
}

VITAL_COLORS = {
    "heart_rate":  "#f87171",
    "sbp":         "#60a5fa",
    "dbp":         "#34d399",
    "map":         "#a78bfa",
    "resp_rate":   "#fbbf24",
    "spo2":        "#38bdf8",
    "temperature": "#fb923c",
    "fio2":        "#c084fc",
}


# Cached loaders

@st.cache_resource(show_spinner="Loading predictor …")
def load_predictor(model_type: str, data_processed_dir: str):

    import importlib
    import src.models.risk_scoring
    import src.inference.predictor
    importlib.reload(src.models.risk_scoring)
    importlib.reload(src.inference.predictor)
    from src.inference.predictor import ClinicalPredictor
    
    p = ClinicalPredictor(model_type=model_type)
    p.load_features()
    return p


@st.cache_data(show_spinner="Loading metrics …")
def load_xgb_metrics(metrics_path: str) -> Optional[list]:
    p = Path(metrics_path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading LSTM metrics …")
def load_lstm_metrics(metrics_path: str) -> Optional[dict]:
    p = Path(metrics_path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# Data helpers

def get_stay_vitals(predictor, stay_id: int, last_n_hours: Optional[int] = None
                    ) -> pd.DataFrame:
    
    df = predictor.get_stay_history(stay_id)
    if last_n_hours is not None:
        df = df.tail(last_n_hours)
    return df


def get_vitals_at_hour(df_stay: pd.DataFrame, as_of_hour: int) -> pd.DataFrame:
    
    return df_stay[df_stay["hour_idx"] <= as_of_hour]


def build_risk_history(predictor, stay_id: int, step: int = 2) -> list:
    
    df_stay = predictor.get_stay_history(stay_id)
    max_h = int(df_stay["hour_idx"].max())
    history = []
    
    for h in range(1, max_h + 1, step):
        try:
            res = predictor.predict(stay_id, h)
            row = df_stay[df_stay["hour_idx"] == h]
            history.append({
                "hour_idx": h,
                "timestamp": row["timestamp"].iloc[0] if not row.empty else None,
                "risk_score": res["risk_score"],
                "severity": res["severity"],
            })
        except: continue
    return history


# Formatting helpers

def format_value(vital: str, value: Optional[float]) -> str:
    if value is None or np.isnan(value):
        return "—"
    unit = VITAL_UNITS.get(vital, "")
    return f"{value:.1f} {unit}"


def severity_badge_html(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f'border-radius:999px;font-weight:700;font-size:0.85rem;">{severity}</span>'
    )


def delta_arrow(current: Optional[float], forecast: Optional[float]) -> str:
    if current is None or forecast is None:
        return ""
    diff = forecast - current
    if abs(diff) < 0.5:
        return "→"
    return "↑" if diff > 0 else "↓"


def metrics_to_dataframe(xgb_metrics: list, split: str = "Test") -> pd.DataFrame:
    rows = [m for m in xgb_metrics if m["split"] == split]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)[["target", "mae", "rmse", "r2"]]
    df["target"] = df["target"].str.replace("_target", "", regex=False)
    df.rename(columns={"target": "Vital", "mae": "MAE", "rmse": "RMSE", "r2": "R²"},
              inplace=True)
    return df.reset_index(drop=True)


def get_stay_info(predictor, stay_id: int) -> dict:
    
    df = predictor.get_stay_history(stay_id)
    row = df.iloc[0]
    los_hours = len(df)
    return {
        "stay_id":   stay_id,
        "subject_id": int(row.get("subject_id", 0)),
        "hadm_id":    int(row.get("hadm_id", 0)),
        "intime":     pd.to_datetime(row.get("intime")),
        "outtime":    pd.to_datetime(row.get("outtime")),
        "los_hours":  los_hours,
        "max_hour":   int(df["hour_idx"].max()),
    }

