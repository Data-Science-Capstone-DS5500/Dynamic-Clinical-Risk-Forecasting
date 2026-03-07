"""
Dashboard Chart Builders using Plotly.
All functions return plotly.graph_objects.Figure objects ready for st.plotly_chart().
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

# ─── Common layout theme ───────────────────────────────────────────────────────
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


# ─── 1. Multi-vital trend chart ────────────────────────────────────────────────

def vital_trend_chart(df_window: pd.DataFrame, vitals: List[str],
                      title: str = "Vital Sign Trends") -> go.Figure:
    """
    Line chart of up to N vital signs over the displayed time window.
    One Y-axis per vital family to accommodate different scales.
    """
    fig = go.Figure()

    for vital in vitals:
        if vital not in df_window.columns:
            continue
        series = df_window[vital].dropna()
        if series.empty:
            continue

        color       = VITAL_COLORS.get(vital, "#94a3b8")
        disp_name   = VITAL_DISPLAY_NAMES.get(vital, vital)
        unit        = VITAL_UNITS.get(vital, "")
        lo, hi      = VITAL_NORMAL_RANGES.get(vital, (None, None))

        fig.add_trace(go.Scatter(
            x=df_window.loc[series.index, "hour_idx"],
            y=series,
            mode="lines+markers",
            name=f"{disp_name} ({unit})",
            line=dict(color=color, width=2),
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


# ─── 2. Forecast bar / comparison chart ───────────────────────────────────────

def forecast_comparison_chart(current_vitals: Dict[str, Optional[float]],
                               forecast_vitals: Dict[str, Optional[float]],
                               mae_by_vital: Optional[Dict[str, float]] = None
                               ) -> go.Figure:
    """
    Grouped bar chart: current vs. 6h-ahead forecast for each vital.
    Adds ± MAE error bars on the forecast bars when available.
    """
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


# ─── 3. Risk score gauge ──────────────────────────────────────────────────────

def risk_gauge(score: float, severity: str) -> go.Figure:
    """
    Semicircular gauge dial showing composite risk score 0–100.
    Color transitions: green (Low) → amber (Moderate) → red (High).
    """
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
            bar=dict(color=needle_color, thickness=0.25),
            bgcolor="rgba(30,41,59,0.6)",
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


# ─── 4. Risk history timeline ──────────────────────────────────────────────────

def risk_history_chart(history: list) -> go.Figure:
    """
    Line + filled area chart of risk score over the full ICU stay.
    Color-coded background bands for severity zones.
    """
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
        title="Risk Score Over ICU Stay",
        xaxis_title="Hours since ICU admission",
        yaxis_title="Composite Risk Score",
        yaxis=dict(range=[0, 105]),
        hovermode="x unified",
    )
    return _apply_base(fig)


# ─── 5. Feature importance chart ──────────────────────────────────────────────

def feature_importance_chart(fi_df: pd.DataFrame, vital_name: str) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
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


# ─── 6. Model comparison table chart ──────────────────────────────────────────

def model_comparison_chart(xgb_metrics: list, lstm_metrics: Optional[dict]) -> go.Figure:
    """
    Side-by-side grouped bar chart comparing XGBoost vs LSTM MAE per vital.
    """
    xgb_test = {m["target"].replace("_target", ""): m["mae"]
                for m in xgb_metrics if m["split"] == "Test"}

    vitals = list(xgb_test.keys())
    xgb_maes = [xgb_test.get(v, 0) for v in vitals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="XGBoost", x=[VITAL_DISPLAY_NAMES.get(v, v) for v in vitals],
        y=xgb_maes, marker_color="#60a5fa", opacity=0.85,
    ))

    if lstm_metrics:
        lstm_test = {m["target"]: m["mae"] for m in lstm_metrics.get("test", [])}
        lstm_maes = [lstm_test.get(v, None) for v in vitals]
        fig.add_trace(go.Bar(
            name="LSTM", x=[VITAL_DISPLAY_NAMES.get(v, v) for v in vitals],
            y=lstm_maes, marker_color="#f472b6", opacity=0.85,
        ))

    fig.update_layout(
        title="Model Comparison — Test MAE (lower = better)",
        barmode="group",
        xaxis_title="Vital Sign",
        yaxis_title="MAE",
    )
    return _apply_base(fig)


# ─── 7. Sparklines for individual vitals ──────────────────────────────────────

def vital_sparkline(df_window: pd.DataFrame, vital: str,
                    forecast_val: Optional[float] = None,
                    mae: Optional[float] = None) -> go.Figure:
    """
    Compact sparkline for a single vital with optional forecast point + band.
    """
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

    # Normal range shading
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
        line=dict(color=color, width=2),
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


# Helper so vital_sparkline doesn't need to import config
FORECAST_HORIZON_OFFSET = 6
