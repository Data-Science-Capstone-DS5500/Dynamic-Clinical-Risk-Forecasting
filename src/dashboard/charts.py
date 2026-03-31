"""
Dashboard Chart Builders using Plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

from src.dashboard.utils import (
    VITAL_DISPLAY_NAMES, VITAL_UNITS, VITAL_COLORS,
    VITAL_NORMAL_RANGES, SEVERITY_COLORS,
)

# Common layout theme
_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(30,41,59,0.8)",
        bordercolor="rgba(100,116,139,0.4)",
        borderwidth=1,
    ),
    xaxis=dict(gridcolor="rgba(100,116,139,0.2)", zerolinecolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="rgba(100,116,139,0.2)", zerolinecolor="rgba(0,0,0,0)"),
)


def _apply_base(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT)
    return fig


# 1. Multi-vital trend chart

def vital_trend_chart(df_window: pd.DataFrame, vitals: List[str],
                      title: str = "Vital Sign Trends",
                      show_vasopressor: bool = True) -> go.Figure:
    
    fig = go.Figure()

    # 1. Add Vasopressor markers
    if show_vasopressor and "vasopressor_on" in df_window.columns:
        vaso_hours = df_window[df_window["vasopressor_on"] == 1]["hour_idx"].unique()
        for vh in vaso_hours:
            fig.add_vline(x=vh, line_width=1, line_dash="dash", line_color="rgba(239,68,68,0.4)")
            
            if vh == vaso_hours[0]:
                fig.add_annotation(x=vh, y=1, yref="paper", text="💉 Vaso", showarrow=False, 
                                   font=dict(size=9, color="#ef4444"), xanchor="left")

    for vital in vitals:
        if vital not in df_window.columns:
            continue
        series = df_window[vital].dropna()
        if series.empty:
            continue

        color       = VITAL_COLORS.get(vital, "#94a3b8")
        disp_name   = VITAL_DISPLAY_NAMES.get(vital, vital)
        unit        = VITAL_UNITS.get(vital, "")

        # Glow Effect
        fig.add_trace(go.Scatter(
            x=df_window.loc[series.index, "hour_idx"],
            y=series,
            mode="lines",
            name=f"{disp_name} Glow",
            line=dict(color=color, width=8, shape='spline'),
            opacity=0.15,
            hoverinfo="skip",
            showlegend=False,
        ))

        # Main Line
        fig.add_trace(go.Scatter(
            x=df_window.loc[series.index, "hour_idx"],
            y=series,
            mode="lines+markers",
            name=f"{disp_name} ({unit})",
            line=dict(color=color, width=2.5, shape='spline'),
            marker=dict(size=4, color=color),
            hovertemplate=f"<b>{disp_name}</b><br>Hour: %{{x}}<br>Value: %{{y:.1f}} {unit}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Hours since ICU admission",
        yaxis_title="Value",
        hovermode="x unified",
    )
    return _apply_base(fig)


# 2. Forecast bar

def forecast_comparison_chart(current_vitals: Dict[str, Optional[float]],
                               forecast_vitals: Dict[str, Optional[float]],
                               mae_by_vital: Optional[Dict[str, float]] = None
                               ) -> go.Figure:
    
    vitals = [v for v in VITAL_DISPLAY_NAMES
              if current_vitals.get(v) is not None or forecast_vitals.get(v) is not None]

    cur_vals  = [current_vitals.get(v)  for v in vitals]
    fore_vals = [forecast_vitals.get(v) for v in vitals]
    labels    = [VITAL_DISPLAY_NAMES[v] for v in vitals]
    mae_vals  = [mae_by_vital.get(v, 0) if mae_by_vital else 0 for v in vitals]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Current",
        x=labels,
        y=cur_vals,
        marker_color="#60a5fa",
        opacity=0.75,
    ))

    fig.add_trace(go.Bar(
        name="6h Forecast",
        x=labels,
        y=fore_vals,
        marker_color="#f472b6",
        opacity=0.85,
        error_y=dict(
            type="data",
            array=mae_vals,
            visible=True,
            color="#f1f5f9",
            thickness=1.5,
        ) if any(m > 0 for m in mae_vals) else None,
    ))

    fig.update_layout(
        title="Current vs. 6h Forecast",
        barmode="group",
        xaxis_title="Vital Sign",
        yaxis_title="Value",
    )
    return _apply_base(fig)


# 3. Risk score gauge

def risk_gauge(score: float, severity: str) -> go.Figure:
    
    needle_color = SEVERITY_COLORS.get(severity, "#94a3b8")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain=dict(x=[0, 1], y=[0, 1]),
        title=dict(
            text=f"<b>Risk Score</b><br><span style='font-size:0.8em'>{severity}</span>",
            font=dict(size=16, color="#e2e8f0"),
        ),
        number=dict(
            suffix="",
            font=dict(size=36, color=needle_color),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="#94a3b8",
                tickfont=dict(color="#94a3b8"),
            ),
            bar=dict(color=needle_color, thickness=0.15),
            bgcolor="rgba(30,41,59,0.3)",
            borderwidth=0,
            steps=[
                dict(range=[0,  30], color="rgba(34,197,94,0.25)"),
                dict(range=[30, 60], color="rgba(245,158,11,0.25)"),
                dict(range=[60,100], color="rgba(239,68,68,0.25)"),
            ],
            threshold=dict(
                line=dict(color=needle_color, width=3),
                thickness=0.8,
                value=score,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=260,
    )
    return fig


# 4. Risk history timeline

def risk_history_chart(history: list) -> go.Figure:
    
    if not history:
        return go.Figure()

    hours  = [h["hour_idx"]   for h in history]
    scores = [h["risk_score"] for h in history]
    sevs   = [h["severity"]   for h in history]

    # Map severity to color per point
    pt_colors = [SEVERITY_COLORS.get(s, "#94a3b8") for s in sevs]

    fig = go.Figure()

    # Background severity bands
    max_hour = max(hours) if hours else 1
    for band_range, band_color in [
        ([0, 30],  "rgba(34,197,94,0.08)"),
        ([30, 60], "rgba(245,158,11,0.08)"),
        ([60,100], "rgba(239,68,68,0.08)"),
    ]:
        fig.add_shape(type="rect",
                      x0=0, x1=max_hour,
                      y0=band_range[0], y1=band_range[1],
                      fillcolor=band_color, line_width=0, layer="below")

    # Main risk score line
    fig.add_trace(go.Scatter(
        x=hours,
        y=scores,
        mode="lines+markers",
        name="Risk Score",
        line=dict(color="#a78bfa", width=2.5),
        marker=dict(color=pt_colors, size=6, line=dict(color="#1e293b", width=1)),
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.12)",
        hovertemplate="Hour %{x}<br>Risk: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title="Predictive Risk Timeline",
        xaxis_title="Hours since ICU admission",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 105]),
        hovermode="x unified",
    )
    return _apply_base(fig)


# 5. Feature importance chart

def feature_importance_chart(fi_df: pd.DataFrame, vital_name: str) -> go.Figure:
    
    if fi_df.empty:
        return go.Figure()

    fi_sorted = fi_df.sort_values("importance")

    fig = go.Figure(go.Bar(
        x=fi_sorted["importance"],
        y=fi_sorted["feature"],
        orientation="h",
        marker=dict(
            color=fi_sorted["importance"],
            colorscale=[[0, "#1e3a5f"], [0.5, "#7c3aed"], [1, "#ec4899"]],
            showscale=False,
        ),
        hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Feature Importance — {VITAL_DISPLAY_NAMES.get(vital_name, vital_name)}",
        xaxis_title="Importance Score",
        yaxis_title="",
        height=350,
    )
    return _apply_base(fig)


# 6. Model comparison table chart

def model_comparison_chart(all_metrics: dict) -> go.Figure:
    
    if not all_metrics:
        return go.Figure()

    # Determine vitals from the first available model
    first_model = list(all_metrics.keys())[0]
    vitals = list(all_metrics[first_model].keys())
    labels = [VITAL_DISPLAY_NAMES.get(v, v) for v in vitals]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Mean Absolute Error (Lower = Better)",
                        "Root Mean Squared Error (Lower = Better)",
                        "R² (Higher = Better)")
    )

    palette = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#c084fc"]

    for i, (model_name, metrics) in enumerate(all_metrics.items()):
        color = palette[i % len(palette)]
        
        x_mae = [metrics.get(v, {}).get("mae", 0) for v in vitals]
        x_rmse = [metrics.get(v, {}).get("rmse", 0) for v in vitals]
        x_r2 = [metrics.get(v, {}).get("r2", 0) for v in vitals]

        fig.add_trace(go.Bar(
            name=model_name, x=labels, y=x_mae,
            marker_color=color, opacity=0.85,
            legendgroup=model_name, showlegend=True
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name=model_name, x=labels, y=x_rmse,
            marker_color=color, opacity=0.85,
            legendgroup=model_name, showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            name=model_name, x=labels, y=x_r2,
            marker_color=color, opacity=0.85,
            legendgroup=model_name, showlegend=False
        ), row=3, col=1)

    fig = _apply_base(fig)
    fig.update_layout(
        title="",
        barmode="group",
        height=700,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
    )
    
    fig.update_yaxes(gridcolor="rgba(100,116,139,0.2)", zeroline=False)
    fig.update_xaxes(gridcolor="rgba(100,116,139,0.2)", zeroline=False)

    return fig


# 7. Sparklines for individual vitals

def vital_sparkline(df_window: pd.DataFrame, vital: str,
                    forecast_val: Optional[float] = None,
                    mae: Optional[float] = None) -> go.Figure:
    
    if vital not in df_window.columns:
        return go.Figure()

    series = df_window[vital].dropna()
    if series.empty:
        return go.Figure()

    hours = df_window.loc[series.index, "hour_idx"].tolist()
    vals  = series.tolist()
    color = VITAL_COLORS.get(vital, "#94a3b8")
    unit  = VITAL_UNITS.get(vital, "")
    lo_n, hi_n = VITAL_NORMAL_RANGES.get(vital, (None, None))

    fig = go.Figure()

    if lo_n and hi_n:
        max_h = max(hours) + FORECAST_HORIZON_OFFSET if hours else 6
        fig.add_shape(type="rect",
                      x0=min(hours), x1=max_h + 7,
                      y0=lo_n, y1=hi_n,
                      fillcolor="rgba(100,116,139,0.1)", line_width=0, layer="below")

    # History line
    fig.add_trace(go.Scatter(
        x=hours, y=vals,
        mode="lines",
        line=dict(color=color, width=2, shape='spline'),
        name="Observed",
        hovertemplate=f"%{{y:.1f}} {unit}<extra></extra>",
        showlegend=False,
    ))

    # Forecast point + confidence band
    if forecast_val is not None and hours:
        next_hour = max(hours) + 6
        fig.add_trace(go.Scatter(
            x=[next_hour], y=[forecast_val],
            mode="markers",
            marker=dict(color="#f472b6", size=10, symbol="diamond"),
            name="6h Forecast",
            showlegend=False,
            hovertemplate=f"Forecast: {forecast_val:.1f} {unit}<extra></extra>",
        ))
        if mae:
            fig.add_shape(type="rect",
                          x0=next_hour - 0.5, x1=next_hour + 0.5,
                          y0=forecast_val - mae, y1=forecast_val + mae,
                          fillcolor="rgba(244,114,182,0.2)",
                          line=dict(color="#f472b6", width=1, dash="dot"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5),
        font=dict(family="Inter, sans-serif", color="#e2e8f0", size=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(gridcolor="rgba(100,116,139,0.15)", zeroline=False),
        height=100,
    )
    return fig


def risk_trajectory_comparison_chart(history: list) -> go.Figure:
    
    fig = risk_history_chart(history)
    if not history:
        return fig
    
    hours = [h["hour_idx"] for h in history]
    max_h = max(hours) + 12 
    x_ref = np.linspace(0, max_h, 50)
    
    # 1. Population Median
    y_median = np.full_like(x_ref, 40.0)
    fig.add_trace(go.Scatter(
        x=x_ref, y=y_median, mode="lines",
        name="Population Median",
        line=dict(color="rgba(148,163,184,0.5)", width=2, dash="dot"),
        hovertemplate="Avg Hospital Patient: 40.0<extra></extra>"
    ))
    
    # 2. Typical Deteriorating Pathway
    y_det = 30 + 50 / (1 + np.exp(-0.1 * (x_ref - 24)))
    fig.add_trace(go.Scatter(
        x=x_ref, y=y_det, mode="lines",
        name="Typical Deterioration Pathway",
        line=dict(color="rgba(239,68,68,0.35)", width=2, dash="dash"),
        hovertemplate="Pathway: Clinical Decline<extra></extra>"
    ))
    
    # 3. Typical Recovery Pathway
    y_rec = 60 * np.exp(-0.02 * x_ref) + 10
    fig.add_trace(go.Scatter(
        x=x_ref, y=y_rec, mode="lines",
        name="Typical Recovery Pathway",
        line=dict(color="rgba(34,197,94,0.35)", width=2, dash="dash"),
        hovertemplate="Pathway: Recovery<extra></extra>"
    ))

    fig.update_layout(
        title="Risk Trajectory: Patient vs. Clinical Benchmarks",
        xaxis_title="Hours since ICU admission",
        yaxis_title="Risk Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def risk_drivers_chart(vital_scores: Dict[str, float]) -> go.Figure:
    
    weights = {
        "map": 2.0, "heart_rate": 1.5, "spo2": 1.5, "resp_rate": 1.2,
        "sbp": 1.0, "dbp": 0.8, "temperature": 0.8, "fio2": 0.7
    }
    
    drivers = []
    total_weighted_sum = 0
    
    for vital, score in vital_scores.items():
        if score > 0:
            w = weights.get(vital, 1.0)
            impact = w * score
            drivers.append({"vital": vital, "score": score, "impact": impact})
            total_weighted_sum += impact
            
    if not drivers or total_weighted_sum == 0:
        return go.Figure().update_layout(title="No active risk drivers detected.")
        
    for d in drivers:
        d["pct"] = (d["impact"] / total_weighted_sum) * 100
        d["name"] = VITAL_DISPLAY_NAMES.get(d["vital"], d["vital"])
        d["color"] = VITAL_COLORS.get(d["vital"], "#94a3b8")
        
    df = pd.DataFrame(drivers).sort_values("pct", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df["pct"],
        y=df["name"],
        orientation="h",
        marker=dict(color=df["color"]),
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="🔍 Risk Drivers (Relative Impact)",
        xaxis_title="Contribution to Total Risk (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 105]),
        height=min(150 + len(df) * 30, 400)
    )
    return _apply_base(fig)


def what_if_trajectory_chart(history: list, sim_data: dict) -> go.Figure:
    
    fig = go.Figure()
    
    # 1. History
    h_idx = [h["hour_idx"] for h in history]
    h_risk = [h["risk_score"] for h in history]
    fig.add_trace(go.Scatter(
        x=h_idx, y=h_risk, mode="lines",
        name="Patient History",
        line=dict(color="#94a3b8", width=2)
    ))
    
    t0 = sim_data["hour_t0"]
    t6 = sim_data["hour_t6"]
    r0 = sim_data["risk_t0"]
    rb6 = sim_data["baseline_t6"]
    rs6 = sim_data["simulated_t6"]
    
    # 2. Baseline Projection 
    fig.add_trace(go.Scatter(
        x=[t0, t6], y=[r0, rb6],
        mode="lines+markers",
        name="Baseline Forecast",
        line=dict(color="#64748b", width=2, dash="dash"),
        marker=dict(size=8, symbol="circle-open")
    ))
    
    # 3. Intervention Projection 
    fig.add_trace(go.Scatter(
        x=[t0, t6], y=[r0, rs6],
        mode="lines+markers",
        name="Intervention Path",
        line=dict(color="#a78bfa", width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond")
    ))
    
    # Background bands
    max_x = t6 + 2
    for band_range, band_color in [
        ([0, 30],  "rgba(34,197,94,0.05)"),
        ([30, 60], "rgba(245,158,11,0.05)"),
        ([60,100], "rgba(239,68,68,0.05)"),
    ]:
        fig.add_shape(type="rect", x0=min(h_idx), x1=max_x,
                      y0=band_range[0], y1=band_range[1],
                      fillcolor=band_color, line_width=0, layer="below")
                      
    fig.update_layout(
        title="🌀 Clinically Evolved Risk Projection",
        xaxis_title="Hours since ICU admission",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _apply_base(fig)


FORECAST_HORIZON_OFFSET = 6
