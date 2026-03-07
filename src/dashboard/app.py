"""
Dynamic Clinical Risk Forecasting — Streamlit Dashboard
Main application entry point. Run with:
    streamlit run src/dashboard/app.py
"""

import sys
import logging
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.utils import (
    load_predictor, load_xgb_metrics, load_lstm_metrics,
    get_stay_vitals, get_vitals_at_hour, get_stay_info,
    VITAL_DISPLAY_NAMES, VITAL_UNITS, VITAL_COLORS,
    VITAL_NORMAL_RANGES, SEVERITY_COLORS,
    format_value, severity_badge_html, delta_arrow, metrics_to_dataframe,
)
from src.dashboard.charts import (
    vital_trend_chart, forecast_comparison_chart, risk_gauge,
    risk_history_chart, feature_importance_chart, model_comparison_chart,
    vital_sparkline,
)
from src.config import DATA_PROCESSED

logging.basicConfig(level=logging.WARNING)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Risk Forecasting",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid rgba(100,116,139,0.3);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(100,116,139,0.3);
    border-radius: 12px;
    padding: 12px 16px;
    backdrop-filter: blur(8px);
}

/* Header */
.dashboard-header {
    background: linear-gradient(90deg, rgba(139,92,246,0.2), rgba(236,72,153,0.1));
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 24px;
}

/* Vital cards */
.vital-card {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(100,116,139,0.25);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s;
}
.vital-card:hover { border-color: rgba(167,139,250,0.5); }

/* Alert banner */
.alert-banner {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
}
.alert-banner-moderate {
    background: rgba(245,158,11,0.15);
    border: 1px solid rgba(245,158,11,0.4);
}
.alert-banner-low {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
}

/* Info box */
.info-box {
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(100,116,139,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-size: 0.9rem;
    font-weight: 600;
}

/* Vasopressor badge */
.vaso-active {
    background: rgba(239,68,68,0.2);
    border: 1px solid #ef4444;
    color: #fca5a5;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.vaso-inactive {
    background: rgba(34,197,94,0.15);
    border: 1px solid #22c55e;
    color: #86efac;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 Clinical Risk Forecasting")
    st.markdown("---")

    # Model selection
    model_type = st.selectbox(
        "🤖 Model",
        options=["xgboost", "lstm"],
        format_func=lambda x: "XGBoost Baseline" if x == "xgboost" else "LSTM (Sequential)",
        help="Choose which trained model to use for forecasting.",
    )

    # Load predictor (cached)
    try:
        predictor = load_predictor(model_type, str(DATA_PROCESSED))
    except FileNotFoundError as e:
        st.error(f"**Model not found:** {e}")
        st.stop()
    except Exception as e:
        st.error(f"**Error loading model:** {e}")
        st.stop()

    # Stay selector
    stays = predictor.list_stays()
    if not stays:
        st.error("No stays found in features.parquet.")
        st.stop()

    selected_stay = st.selectbox(
        "🛏️ ICU Stay",
        options=stays,
        format_func=lambda x: f"Stay {x}",
        help="Select a patient ICU stay to inspect.",
    )

    # Load stay info
    stay_info = get_stay_info(predictor, selected_stay)
    max_hour  = stay_info["max_hour"]

    st.markdown(f"""
    <div class="info-box">
    <b>Stay ID:</b> {stay_info['stay_id']}<br>
    <b>Subject ID:</b> {stay_info['subject_id']}<br>
    <b>LOS:</b> {stay_info['los_hours']} hrs
    </div>
    """, unsafe_allow_html=True)

    # Hour slider
    default_hour = min(12, max_hour)
    as_of_hour = st.slider(
        "⏱️ Display up to hour",
        min_value=1,
        max_value=int(max_hour),
        value=int(default_hour),
        step=1,
        help="Slide to move the 'current time' forward/backward through the stay.",
    )

    st.markdown("---")
    st.markdown(
        "<span style='color:#94a3b8;font-size:0.75rem'>"
        "MIMIC-IV · 6h forecast · XGBoost / LSTM"
        "</span>",
        unsafe_allow_html=True,
    )


# ─── Main header ──────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="dashboard-header">
  <h1 style="margin:0;font-size:1.6rem;font-weight:700;color:#e2e8f0;">
    🔬 Dynamic Clinical Risk Forecasting
  </h1>
  <p style="margin:4px 0 0 0;color:#94a3b8;font-size:0.9rem;">
    Stay {selected_stay} &nbsp;·&nbsp; Hour 0 → {as_of_hour}
    &nbsp;·&nbsp; Model: <b>{'XGBoost' if model_type=='xgboost' else 'LSTM'}</b>
  </p>
</div>
""", unsafe_allow_html=True)


# ─── Run prediction ───────────────────────────────────────────────────────────

with st.spinner("Computing forecast …"):
    try:
        result = predictor.predict(stay_id=selected_stay, as_of_hour=as_of_hour)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

forecasts      = result["forecasts"]
risk_score     = result["risk_score"]
severity       = result["severity"]
vital_scores_d = result["vital_scores"]
alerts         = result["alerts"]
current_vitals = result["current_vitals"]

# Full history for this stay
df_stay   = predictor.get_stay_history(selected_stay)
df_window = get_vitals_at_hour(df_stay, as_of_hour)

# MAE look-up (XGBoost only)
xgb_metrics = load_xgb_metrics(str(DATA_PROCESSED / "baseline_xgb_metrics.json"))
mae_by_vital = {}
if xgb_metrics:
    for m in xgb_metrics:
        if m["split"] == "Test":
            v = m["target"].replace("_target", "")
            mae_by_vital[v] = m["mae"]


# ─── Three tabs ───────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "📊 Patient Overview",
    "🔮 6-Hour Forecast",
    "📈 Risk History",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Patient Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    # Stay metadata row
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("ICU Stay", f"{selected_stay}")
    with col_b:
        st.metric("Hours Displayed", f"{as_of_hour}")
    with col_c:
        st.metric("LOS (total)", f"{stay_info['los_hours']} hrs")
    with col_d:
        # Vasopressor status
        if "vasopressor_on" in df_window.columns:
            vaso_now = df_window["vasopressor_on"].iloc[-1] if not df_window.empty else 0
        else:
            vaso_now = 0
        if vaso_now:
            st.markdown('<span class="vaso-active">💉 Vasopressor ON</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="vaso-inactive">✅ No Vasopressor</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Multi-vital trend chart
    vitals_to_show = [v for v in ["heart_rate", "map", "sbp", "spo2", "resp_rate"]
                      if v in df_window.columns]
    fig_trend = vital_trend_chart(df_window, vitals_to_show,
                                  title=f"Vital Trends — Hours 0 to {as_of_hour}")
    st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")

    # Current vital snapshot grid
    st.markdown("#### 📍 Current Snapshot")
    vital_list = list(VITAL_DISPLAY_NAMES.keys())
    cols = st.columns(4)
    for i, vital in enumerate(vital_list):
        cur = current_vitals.get(vital)
        fore = forecasts.get(vital)
        disp = VITAL_DISPLAY_NAMES[vital]
        unit = VITAL_UNITS[vital]
        lo_n, hi_n = VITAL_NORMAL_RANGES.get(vital, (None, None))

        val_str = f"{cur:.1f} {unit}" if cur is not None and not np.isnan(cur) else "—"
        arrow   = delta_arrow(cur, fore)

        # indicator color if out of normal range
        if cur is not None and lo_n and hi_n:
            flag = "🟢" if lo_n <= cur <= hi_n else "🔴"
        else:
            flag = "⚪"

        with cols[i % 4]:
            st.markdown(f"""
            <div class="vital-card">
              <div style="font-size:0.7rem;color:#94a3b8;letter-spacing:0.05em;text-transform:uppercase;">{flag} {disp}</div>
              <div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-top:4px;">{val_str}</div>
              <div style="font-size:0.75rem;color:#a78bfa;margin-top:2px;">6h → {arrow} {format_value(vital, fore)}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 6-Hour Forecast
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    # Risk score + alerts row
    col_gauge, col_alerts = st.columns([1, 2])

    with col_gauge:
        fig_gauge = risk_gauge(risk_score, severity)
        st.plotly_chart(fig_gauge, use_container_width=True, key="risk_gauge")

    with col_alerts:
        st.markdown("#### 🚨 Active Alerts")
        if not alerts:
            st.markdown(
                '<div class="alert-banner alert-banner-low" style="margin-top:12px;">'
                '✅ <b>No alerts</b> — All forecasted vitals within safe ranges.</div>',
                unsafe_allow_html=True,
            )
        else:
            banner_class = "alert-banner" if severity == "High" else "alert-banner-moderate"
            for alert_msg in alerts:
                st.markdown(
                    f'<div class="{banner_class}">⚠️ {alert_msg}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Forecast comparison bar chart
    fig_fore = forecast_comparison_chart(current_vitals, forecasts, mae_by_vital)
    st.plotly_chart(fig_fore, use_container_width=True, key="forecast_bar")

    # Per-vital sparkline grid
    st.markdown("#### 🔍 Per-Vital Sparklines")
    vital_list = list(VITAL_DISPLAY_NAMES.keys())
    cols2 = st.columns(4)
    for i, vital in enumerate(vital_list):
        disp  = VITAL_DISPLAY_NAMES[vital]
        unit  = VITAL_UNITS[vital]
        cur   = current_vitals.get(vital)
        fore  = forecasts.get(vital)
        mae   = mae_by_vital.get(vital)
        vscore = vital_scores_d.get(vital, 0)

        # Sub-score color
        if vscore < 30:
            score_color = "#22c55e"
        elif vscore < 60:
            score_color = "#f59e0b"
        else:
            score_color = "#ef4444"

        with cols2[i % 4]:
            st.markdown(
                f'<div style="font-size:0.72rem;font-weight:600;color:#94a3b8;'
                f'text-transform:uppercase;letter-spacing:0.05em;">{disp}</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">'
                f'{format_value(vital, fore)}</div>'
                f'<div style="font-size:0.7rem;color:{score_color};">'
                f'Risk sub-score: {vscore:.0f}/100</div>',
                unsafe_allow_html=True,
            )
            fig_sp = vital_sparkline(df_window, vital, forecast_val=fore, mae=mae)
            st.plotly_chart(fig_sp, use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"spark_{vital}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Risk History
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("#### 📈 Full-Stay Risk Score Timeline")

    # Compute risk history lazily (can be slow — cache per stay+model)
    @st.cache_data(show_spinner="Computing risk history …")
    def get_cached_risk_history(stay_id, model_type_key, max_h):
        """Compute and cache risk history for a stay."""
        history = []
        # Sample every 2 hours for performance (full stays can be 100+ hrs)
        df_s = predictor.get_stay_history(stay_id)
        for hour in range(1, max_h + 1, 2):
            try:
                r = predictor.predict(stay_id=stay_id, as_of_hour=hour)
                row = df_s[df_s["hour_idx"] == hour]
                ts  = row["timestamp"].iloc[0] if not row.empty else None
                history.append({
                    "hour_idx":   hour,
                    "timestamp":  ts,
                    "risk_score": r["risk_score"],
                    "severity":   r["severity"],
                    "alerts":     r["alerts"],
                })
            except Exception:
                continue
        return history

    risk_history = get_cached_risk_history(selected_stay, model_type, max_hour)

    if risk_history:
        fig_hist = risk_history_chart(risk_history)
        st.plotly_chart(fig_hist, use_container_width=True, key="risk_history")

        # Summary stats
        scores = [h["risk_score"] for h in risk_history]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min Risk", f"{min(scores):.1f}")
        c2.metric("Max Risk", f"{max(scores):.1f}")
        c3.metric("Avg Risk", f"{np.mean(scores):.1f}")
        high_pct = 100 * sum(1 for h in risk_history if h["severity"] == "High") / len(risk_history)
        c4.metric("Time in High Risk", f"{high_pct:.0f}%")
    else:
        st.info("Risk history unavailable for this stay / hour range.")

    st.markdown("---")

    # Feature importance
    st.markdown("#### 🧠 Feature Importance")
    fi_vital = st.selectbox(
        "Select vital to inspect",
        options=list(VITAL_DISPLAY_NAMES.keys()),
        format_func=lambda v: VITAL_DISPLAY_NAMES[v],
        key="fi_vital_select",
    )

    if model_type == "xgboost":
        try:
            fi_df = predictor.get_feature_importance(fi_vital, top_n=10)
            if not fi_df.empty:
                fig_fi = feature_importance_chart(fi_df, fi_vital)
                st.plotly_chart(fig_fi, use_container_width=True, key="feat_imp")
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")
    else:
        st.info("Global feature importance is not available for the LSTM model.")

    st.markdown("---")

    # Model performance comparison
    st.markdown("#### 📊 Model Performance Comparison")
    if xgb_metrics:
        lstm_metrics_data = load_lstm_metrics(str(DATA_PROCESSED / "lstm_metrics.json"))
        fig_comp = model_comparison_chart(xgb_metrics, lstm_metrics_data)
        st.plotly_chart(fig_comp, use_container_width=True, key="model_comparison")

        # XGBoost metrics table
        with st.expander("XGBoost test metrics"):
            df_metrics = metrics_to_dataframe(xgb_metrics, split="Test")
            if not df_metrics.empty:
                st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python src/models/baseline_xgboost.py` to generate metrics.")
