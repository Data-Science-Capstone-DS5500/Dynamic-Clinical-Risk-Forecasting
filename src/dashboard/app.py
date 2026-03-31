"""
Dynamic Clinical Risk Forecasting — Streamlit Dashboard

"""

import sys
import logging
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from src.dashboard.utils import (
    load_predictor, load_xgb_metrics, load_lstm_metrics,
    get_vitals_at_hour, get_stay_info,
    VITAL_DISPLAY_NAMES, VITAL_UNITS, VITAL_COLORS,
    VITAL_NORMAL_RANGES, SEVERITY_COLORS,
    format_value, severity_badge_html, delta_arrow, metrics_to_dataframe,
    build_risk_history
)
from src.dashboard.charts import (
    vital_trend_chart, forecast_comparison_chart, risk_gauge,
    risk_history_chart, feature_importance_chart, model_comparison_chart,
    vital_sparkline, risk_trajectory_comparison_chart, risk_drivers_chart,
    what_if_trajectory_chart
)
from src.config import DATA_PROCESSED

logging.basicConfig(level=logging.WARNING)

# Page config
st.set_page_config(
    page_title="Clinical Risk Forecasting",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6, .dashboard-header h1 {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.01em;
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
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 16px;
    padding: 16px 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
div[data-testid="metric-container"] label {
    font-family: 'Outfit', sans-serif !important;
    color: #cbd5e1 !important;
    letter-spacing: 0.02em;
}

@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Header */
.dashboard-header {
    background: linear-gradient(270deg, rgba(139,92,246,0.25), rgba(236,72,153,0.15), rgba(56,189,248,0.2));
    background-size: 200% 200%;
    animation: gradient-shift 10s ease infinite;
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 20px;
    padding: 24px 32px;
    margin-bottom: 28px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
}

/* Vital cards */
.vital-card {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.vital-card:hover { 
    border-color: rgba(167,139,250,0.6); 
    box-shadow: 0 0 20px rgba(167,139,250, 0.2);
    transform: translateY(-2px);
}

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

/* Streamlit Native Component Overrides */
[data-testid="stMetricLabel"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}
.stButton > button {
    font-family: 'Outfit', sans-serif !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(139,92,246, 0.25) !important;
    border-color: #8b5cf6 !important;
    color: #e2e8f0 !important;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-size: 0.9rem;
    font-family: 'Outfit', sans-serif !important;
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


# Sidebar

with st.sidebar:
    st.markdown("## Clinical Risk Forecasting")
    st.markdown("---")

    # Model selection
    model_type = st.selectbox(
        "Prediction Mode",
        options=["lstm", "xgboost", "gru"],
        format_func=lambda x: {
            "lstm": "LSTM (Sequential)",
            "xgboost": "XGBoost (Baseline)",
            "gru": "GRU (Recurrent)"
        }.get(x, x),
        help="Select a model architecture for forecasting.",
    )

    # Load predictor
    try:
        predictor = load_predictor(model_type, str(DATA_PROCESSED))
    except FileNotFoundError as e:
        st.error(f"**Model not found:** {e}")
        st.stop()
    except Exception as e:
        st.error(f"**Error loading model:** {e}")
        st.info("💡 **Tip:** This might be an environment issue. Ensure you are running Streamlit from your virtual environment: "
                "`./.venv/bin/python3 -m streamlit run src/dashboard/app.py`")
        st.stop()

    # Stay selector
    stays = predictor.list_stays()
    if not stays:
        st.error("No stays found in features.parquet.")
        st.stop()

    selected_stay = st.selectbox(
        "ICU Stay",
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
        "Display up to hour",
        min_value=1,
        max_value=int(max_hour),
        value=int(default_hour),
        step=1,
        help="Slide to move the 'current time' forward/backward through the stay.",
    )

    st.markdown(f"---")
    st.caption(f"Predictor Version: `{predictor.VERSION}`")
    st.markdown(f"MIMIC-IV · 6h Forecast · XGBoost / LSTM")


# Main header

st.markdown(f"""
<div class="dashboard-header">
  <h1 style="margin:0;font-size:1.6rem;font-weight:700;color:#e2e8f0;">
    Clinical Risk Decision Support
  </h1>
  <p style="margin:4px 0 0 0;color:#94a3b8;font-size:0.9rem;">
    Stay {selected_stay} &nbsp;·&nbsp; Hour 0 → {as_of_hour}
    &nbsp;·&nbsp; Mode: <b>{model_type.upper()}</b>
  </p>
</div>
""", unsafe_allow_html=True)


# Run prediction

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

xgb_metrics = load_xgb_metrics(str(DATA_PROCESSED / "baseline_xgb_metrics.json"))
mae_by_vital = {}
if xgb_metrics:
    for m in xgb_metrics:
        if m["split"] == "Test":
            v = m["target"].replace("_target", "")
            mae_by_vital[v] = m["mae"]

# Risk history
risk_history = build_risk_history(predictor, selected_stay)


# Three tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "Patient Overview",
    "6-Hour Forecast",
    "Risk History",
    "Intervention Simulator",
])


# TAB 1 — Patient Overview

with tab1:
    # Stay metadata row
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("ICU Stay", f"{selected_stay}")
        
        # CSV Export integration
        csv_data = df_window.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Stay Data",
            data=csv_data,
            file_name=f"stay_{selected_stay}_report.csv",
            mime="text/csv"
        )
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
    show_vaso = st.toggle("Show Vasopressor Interventions", value=True)
    vitals_to_show = [v for v in ["heart_rate", "map", "sbp", "spo2", "resp_rate"]
                      if v in df_window.columns]
    fig_trend = vital_trend_chart(df_window, vitals_to_show,
                                  title=f"Vital Trends — Hours 0 to {as_of_hour}",
                                  show_vasopressor=show_vaso)
    st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")

    # Current vital snapshot grid
    st.markdown("#### Current Snapshot")
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



# TAB 2 — 6-Hour Forecast

with tab2:
    # Risk score + alerts row
    col_gauge, col_alerts = st.columns([1, 2])

    with col_gauge:
        fig_gauge = risk_gauge(risk_score, severity)
        st.plotly_chart(fig_gauge, use_container_width=True, key="risk_gauge")
        
        # Risk Drivers
        fig_drivers = risk_drivers_chart(vital_scores_d)
        st.plotly_chart(fig_drivers, use_container_width=True, key="risk_drivers")

    with col_alerts:
        st.markdown("#### Active Alerts")
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

    # Per vital sparkline grid
    st.markdown("#### Per-Vital Sparklines")
    vital_list = list(VITAL_DISPLAY_NAMES.keys())
    cols2 = st.columns(4)
    for i, vital in enumerate(vital_list):
        disp  = VITAL_DISPLAY_NAMES[vital]
        unit  = VITAL_UNITS[vital]
        cur   = current_vitals.get(vital)
        fore  = forecasts.get(vital)
        mae   = mae_by_vital.get(vital)
        vscore = vital_scores_d.get(vital, 0)

        # Sub score color
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


# TAB 3 — Risk History

with tab3:
    st.markdown("#### Full-Stay Risk Score Timeline")

    # Compute risk history
    @st.cache_data(show_spinner="Computing risk history …")
    def get_cached_risk_history(stay_id, model_type_key, max_h):
        
        history = []
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
        fig_hist = risk_trajectory_comparison_chart(risk_history)
        st.plotly_chart(fig_hist, use_container_width=True, key="risk_comparison_history")

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






# TAB 4 — Intervention Simulator

with tab4:
    st.markdown("### Clinical Intervention Simulation")
    st.info("Simulate how maintaining specific vital sign targets (e.g., via vasopressors or O₂ therapy) would evolve the patient's risk trajectory over the next 6 hours.")

    # 1. Initialize Session State for Sliders
    sim_keys = {
        "map": 70, "sbp": 120, "dbp": 80, "hr": 80, 
        "spo2": 95, "fio2": 21, "rr": 16, "temp": 98.6
    }
    for k, default_val in sim_keys.items():
        if f"sim_{k}" not in st.session_state:
            # Use current patient baseline if available, else clinical default
            st.session_state[f"sim_{k}"] = int(current_vitals.get(k, default_val)) if k in current_vitals else default_val

    # Vasopressor state
    if "sim_vasopressor_on" not in st.session_state:
        st.session_state.sim_vasopressor_on = False
    if "sim_vasopressor_rate" not in st.session_state:
        st.session_state.sim_vasopressor_rate = 0.0

    # 2. Simulation Controls
    with st.expander("Simulation Controls", expanded=True):
        st.markdown("#### Clinical Scenario Presets")
        cols_pre = st.columns(3)
        if cols_pre[0].button("Sepsis Bundle (MAP 65+)"):
            st.session_state.sim_map = 65
            st.session_state.sim_sbp = 95
            st.session_state.sim_fio2 = 35
            st.session_state.sim_vasopressor_on = True
            st.session_state.sim_vasopressor_rate = 0.1  # Standard initial dose
            st.rerun()
        if cols_pre[1].button("ARDS Protection (FiO2 50%+)"):
            st.session_state.sim_spo2 = 90
            st.session_state.sim_fio2 = 50
            st.session_state.sim_rr   = 28
            st.rerun()
        if cols_pre[2].button("Normotension Target"):
            st.session_state.sim_map = 85
            st.session_state.sim_sbp = 120
            st.session_state.sim_dbp = 80
            st.session_state.sim_hr  = 75
            st.session_state.sim_spo2 = 98
            st.session_state.sim_fio2 = 21
            st.session_state.sim_rr   = 16
            st.session_state.sim_temp = 98.6
            st.rerun()

        st.markdown("---")
        st.markdown("#### Physiological Targets")
        c1, c2, c3, c4 = st.columns(4)
        sim_map = c1.slider("Target MAP", 40, 150, key="sim_map")
        sim_sbp = c2.slider("Target SBP", 70, 200, key="sim_sbp")
        sim_dbp = c3.slider("Target DBP", 40, 120, key="sim_dbp")
        sim_hr  = c4.slider("Target HR", 40, 180, key="sim_hr")
        
        c5, c6, c7, c8 = st.columns(4)
        sim_spo2 = c5.slider("Target SpO₂", 70, 100, key="sim_spo2")
        sim_fio2 = c6.slider("Target FiO₂", 21, 100, key="sim_fio2")
        sim_rr   = c7.slider("Target Resp Rate", 8, 40, key="sim_rr")
        sim_temp = c8.slider("Target Temp (°F)", 94, 106, key="sim_temp")

        st.markdown("---")
        st.markdown("#### Vasopressor Administration")
        vcol1, vcol2 = st.columns([1, 2])
        sim_vasopressor_on = vcol1.toggle(
            "Vasopressor Active",
            key="sim_vasopressor_on",
            help="Toggle vasopressor administration (e.g., norepinephrine, dopamine)."
        )
        if sim_vasopressor_on:
            sim_vasopressor_rate = vcol2.slider(
                "Infusion Rate (mcg/kg/min)",
                min_value=0.01, max_value=1.0, step=0.01,
                key="sim_vasopressor_rate",
                help="Typical norepinephrine: 0.01–0.5 mcg/kg/min. Escalating >0.25 indicates refractory shock."
            )
        else:
            sim_vasopressor_rate = 0.0
            st.session_state.sim_vasopressor_rate = 0.0

    # Physiological Consistency Check
    if sim_sbp <= sim_map or sim_map <= sim_dbp:
        st.warning("⚠️ **Physiological Inconsistency**: Typically, SBP > MAP > DBP. Verify these blood pressure targets.")

    if st.button("Run Evolution Simulation", type="primary"):
        with st.status("Simulating clinical path …", expanded=True) as status:
            st.write("Applying physiological perturbations...")
            perturbations = {
                "map": float(sim_map),
                "sbp": float(sim_sbp),
                "dbp": float(sim_dbp),
                "heart_rate": float(sim_hr),
                "spo2": float(sim_spo2),
                "fio2": float(sim_fio2),
                "resp_rate": float(sim_rr),
                "temperature": float(sim_temp),
                "vasopressor_on": float(st.session_state.sim_vasopressor_on),
                "vasopressor_rate": float(st.session_state.sim_vasopressor_rate),
            }
            st.write("Running forward risk trajectory...")
            sim_data = predictor.simulate_intervention_evolution(selected_stay, as_of_hour, perturbations)
            
            status.update(label="Simulation Complete!", state="complete", expanded=False)
            st.toast("Simulation executed successfully", icon="✅")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Risk (Hour T)", f"{sim_data['risk_t0']:.1f}")
            m2.metric("Natural Evolution (T+6)", f"{sim_data['baseline_t6']:.1f}")
            m3.metric("Simulated Evolution (T+6)", f"{sim_data['simulated_t6']:.1f}")
            
            delta = sim_data["simulated_t6"] - sim_data["baseline_t6"]
            delta_color = "normal" if delta == 0 else "inverse" # lower is better
            m4.metric("Change vs. Baseline", f"{delta:+.1f}", delta=f"{delta:+.1f}", delta_color=delta_color)
            
            if delta < -5:
                st.success(f"**Clinical Impact**: This intervention is projected to reduce patient risk by **{abs(delta):.1f} points** compared to the baseline.")
            elif delta > 5:
                st.markdown("**Clinical Warning**: The simulated state increases the physiological risk score compared to the baseline trajectory.")

            # Trajectory Plot
            fig_sim = what_if_trajectory_chart(risk_history, sim_data)
            st.plotly_chart(fig_sim, use_container_width=True, key="sim_trajectory")
    else:
        st.caption("Adjust sliders above and click 'Run Evolution Simulation' to visualize the projected impact.")

