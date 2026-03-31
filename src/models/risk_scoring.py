"""
Clinical Risk Scoring Module
Converts 8 predicted vital sign values into a composite 0-100 risk index
with severity tiers (Low / Moderate / High) and per-vital alert flags.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

# Clinical reference ranges

CLINICAL_THRESHOLDS = {
    "map":         (50,  65, 100, 130),   # mmHg
    "heart_rate":  (40,  60, 100, 130),   # bpm
    "sbp":         (70,  90, 140, 180),   # mmHg
    "dbp":         (40,  60,  90, 110),   # mmHg
    "resp_rate":   ( 8,  12,  20,  30),   # breaths/min
    "spo2":        (80,  90, 100, 110),   # % (Allowing up to 110 to handle model artifacts above 100)
    "temperature": (95,  97, 100, 104),   # °F
    "fio2":        (21,  21,  60,  100),  # % (elevated FiO2 = intervention signal)
}

# Per-vital clinical weights (higher = more important for composite score)
VITAL_WEIGHTS = {
    "map":         2.0,
    "heart_rate":  1.5,
    "spo2":        1.5,
    "resp_rate":   1.2,
    "sbp":         1.0,
    "dbp":         0.8,
    "temperature": 0.8,
    "fio2":        0.7,
}

SEVERITY_THRESHOLDS = {"Low": 30, "Moderate": 60}   # risk_score boundaries

# Alert messages per vital
ALERT_MESSAGES = {
    "map":         ("Hypotension risk (MAP < 65 mmHg)", "Hypertension risk (MAP > 130 mmHg)"),
    "heart_rate":  ("Bradycardia risk (HR < 40 bpm)",   "Tachycardia risk (HR > 130 bpm)"),
    "sbp":         ("Hypotension risk (SBP < 70 mmHg)", "Severe hypertension (SBP > 180 mmHg)"),
    "dbp":         ("Low DBP (< 40 mmHg)",               "Elevated DBP (> 110 mmHg)"),
    "resp_rate":   ("Bradypnea (RR < 8)",                "Tachypnea (RR > 30)"),
    "spo2":        ("Severe hypoxia (SpO₂ < 80%)",       None),
    "temperature": ("Hypothermia risk (< 95°F)",         "Hyperthermia / Fever (> 104°F)"),
    "fio2":        (None,                                 "High FiO₂ dependency (> 60%)"),
}


@dataclass
class RiskResult:
    
    risk_score: float               # 0–100 composite score
    severity: str                   # "Low" | "Moderate" | "High"
    vital_scores: Dict[str, float]  # per-vital sub-scores (0–100)
    alerts: List[str]               # list of human-readable alert messages
    predicted_vitals: Dict[str, Optional[float]] = field(default_factory=dict)


def _vital_sub_score(vital: str, value: float) -> float:
    
    if vital not in CLINICAL_THRESHOLDS:
        return 0.0

    lo_alert, lo_norm, hi_norm, hi_alert = CLINICAL_THRESHOLDS[vital]

    if lo_norm <= value <= hi_norm:
        return 0.0

    if value < lo_norm:
        if value <= lo_alert:
            return 100.0
        return 100.0 * (lo_norm - value) / max(lo_norm - lo_alert, 1e-9)

    if value >= hi_alert:
        return 100.0
    return 100.0 * (value - hi_norm) / max(hi_alert - hi_norm, 1e-9)


def _severity_label(score: float) -> str:
    if score <= SEVERITY_THRESHOLDS["Low"]:
        return "Low"
    if score <= SEVERITY_THRESHOLDS["Moderate"]:
        return "Moderate"
    return "High"


def compute_risk_score(predicted_vitals: Dict[str, Optional[float]]) -> RiskResult:
    
    vital_scores: Dict[str, float] = {}
    alerts: List[str] = []
    total_weight, weighted_sum = 0.0, 0.0

    for vital, value in predicted_vitals.items():
        if value is None or np.isnan(value):
            continue

        weight    = VITAL_WEIGHTS.get(vital, 1.0)
        sub_score = _vital_sub_score(vital, value)
        vital_scores[vital] = round(sub_score, 2)

        weighted_sum  += weight * sub_score
        total_weight  += weight

        # Generate alert if sub-score is meaningful
        if sub_score >= 50 and vital in ALERT_MESSAGES:
            lo_alert, _, _, hi_alert = CLINICAL_THRESHOLDS[vital]
            msg_lo, msg_hi = ALERT_MESSAGES[vital]
            if value < lo_alert and msg_lo:
                alerts.append(msg_lo)
            elif value > hi_alert and msg_hi:
                alerts.append(msg_hi)
            elif sub_score >= 50:
                # borderline — use whichever direction applies
                _, lo_norm, hi_norm, _ = CLINICAL_THRESHOLDS[vital]
                if value < lo_norm and msg_lo:
                    alerts.append(msg_lo)
                elif value > hi_norm and msg_hi:
                    alerts.append(msg_hi)

    # Composite score
    if total_weight == 0:
        composite = 0.0
    else:
        composite = weighted_sum / total_weight

    composite = float(np.clip(composite, 0, 100))

    return RiskResult(
        risk_score      = round(composite, 2),
        severity        = _severity_label(composite),
        vital_scores    = vital_scores,
        alerts          = list(dict.fromkeys(alerts)),   # deduplicate, preserve order
        predicted_vitals= {k: (round(v, 2) if v is not None and not np.isnan(v) else None)
                           for k, v in predicted_vitals.items()},
    )


def compute_risk_history(df_stay, feature_cols: list, predictor) -> list:
    
    history = []
    stay_id = df_stay["stay_id"].iloc[0]

    for _, row in df_stay.iterrows():
        hour_idx = int(row["hour_idx"])
        try:
            result = predictor.predict(stay_id=stay_id, as_of_hour=hour_idx)
        except Exception:
            continue

        history.append({
            "hour_idx":   hour_idx,
            "timestamp":  row.get("timestamp"),
            "risk_score": result["risk_score"],
            "severity":   result["severity"],
            "alerts":     result["alerts"],
        })

    return history


if __name__ == "__main__":

    print("\n" + "="*50)
    print("CLINICAL RISK SCORING DEMO")
    print("="*50)

    # Example 1: Critical Patient
    critical_vitals = {
        "map": 55,           # Hypotension
        "heart_rate": 135,   # Tachycardia
        "spo2": 85,          # Hypoxia
        "resp_rate": 32,     # Tachypnea
        "temperature": 102,
        "sbp": 85,
        "dbp": 50,
        "fio2": 65           # High FiO2
    }

    # Example 2: Healthy Patient
    stable_vitals = {
        "map": 85,
        "heart_rate": 75,
        "spo2": 98,
        "resp_rate": 16,
        "temperature": 98.6,
        "sbp": 120,
        "dbp": 80,
        "fio2": 21
    }

    for label, vitals in [("CRITICAL CASE", critical_vitals), ("STABLE CASE", stable_vitals)]:
        result = compute_risk_score(vitals)
        print(f"\n[ {label} ]")
        print(f"Risk Score: {result.risk_score} / 100")
        print(f"Severity:   {result.severity}")
        print(f"Alerts:     {', '.join(result.alerts) if result.alerts else 'None'}")
        print(f"Vital Sub-Scores: {result.vital_scores}")

    print("\n" + "="*50)
