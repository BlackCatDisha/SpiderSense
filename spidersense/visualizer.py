"""
visualizer.py — All Plotly visualisations for SpiderSense Lite.

Keeps chart-building logic separate from the Streamlit app layer.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from services import (
    ServiceSystem, Service,
    STATUS_HEALTHY, STATUS_DEGRADED, STATUS_FAILED,
)

# ---------------------------------------------------------------------------
# Colour palette — matches the app's pastel design language
# ---------------------------------------------------------------------------
PAL = {
    "healthy":          "#A8D5B5",   # sage green
    "healthy_border":   "#6FAF83",
    "degraded":         "#F5DFA0",   # pale amber
    "degraded_border":  "#C9A83C",
    "failed":           "#EBCFD2",   # dusty rose
    "failed_border":    "#C4727A",
    "origin":           "#C5B4E3",   # lavender — predicted/actual origin
    "origin_border":    "#7A5FAF",
    "edge_normal":      "#C8C6D0",
    "edge_active":      "#7A9CC4",   # powder blue
    "edge_text":        "#77788A",
    "node_text":        "#303344",
    "bg":               "#F7F6F2",
    "card":             "#FFFFFF",
    "lavender":         "#DCD6F7",
    "sage":             "#D7E7D0",
    "blue":             "#D5E7F4",
    "rose":             "#EBCFD2",
    "peach":            "#F4DDC5",
    "primary_text":     "#303344",
    "secondary_text":   "#77788A",
}

# Fixed node positions for a clean, legible layout
# (x, y) in [0, 1] space
NODE_POS: Dict[str, Tuple[float, float]] = {
    "Database":    (0.50, 0.08),
    "Payment":     (0.22, 0.35),
    "Inventory":   (0.78, 0.35),
    "Orders":      (0.50, 0.58),
    "API Gateway": (0.50, 0.76),
    "Frontend":    (0.50, 0.95),
}


def _service_colour(svc: Service, is_predicted_root: bool = False) -> Tuple[str, str]:
    """Return (fill, border) colour for a service node."""
    if is_predicted_root:
        return PAL["origin"], PAL["origin_border"]
    if svc.status == STATUS_FAILED:
        return PAL["failed"], PAL["failed_border"]
    if svc.status == STATUS_DEGRADED:
        return PAL["degraded"], PAL["degraded_border"]
    return PAL["healthy"], PAL["healthy_border"]


# ---------------------------------------------------------------------------
# Main service-dependency graph
# ---------------------------------------------------------------------------
def service_graph(
    system: ServiceSystem,
    graph: nx.DiGraph,
    *,
    highlight_edges: Optional[List[Tuple[str, str]]] = None,
    predicted_root: Optional[str] = None,
) -> go.Figure:
    """
    Build the interactive Plotly figure showing the service topology.

    Parameters
    ----------
    highlight_edges : edges that carried the propagation signal (highlighted).
    predicted_root  : name of the service predicted as the root cause.
    """

    fig = go.Figure()

    # --- Draw edges --------------------------------------------------------
    for src, dst in graph.edges():
        x0, y0 = NODE_POS[src]
        x1, y1 = NODE_POS[dst]

        is_active = (
            highlight_edges is not None and (src, dst) in highlight_edges
        )
        colour = PAL["edge_active"] if is_active else PAL["edge_normal"]
        width  = 2.5 if is_active else 1.5

        # Arrowhead via annotation
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.3,
            arrowwidth=width,
            arrowcolor=colour,
        )

        # Edge line
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=colour, width=width),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Signal intensity label on active edges
        if is_active:
            src_svc = system.get(src)
            dst_svc = system.get(dst)
            if dst_svc.signal_intensity > 0:
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                fig.add_annotation(
                    x=mid_x, y=mid_y,
                    text=f"<b>{dst_svc.signal_intensity:.1f}</b>",
                    showarrow=False,
                    font=dict(size=10, color=PAL["edge_active"]),
                    bgcolor="rgba(255,255,255,0.75)",
                    borderpad=2,
                )

    # --- Draw nodes --------------------------------------------------------
    for svc in system.all():
        is_root = (predicted_root == svc.name)
        fill, border = _service_colour(svc, is_predicted_root=is_root)
        x, y = NODE_POS[svc.name]

        # Node circle
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers",
            marker=dict(
                size=48,
                color=fill,
                line=dict(color=border, width=2.5 if is_root else 1.5),
                symbol="circle",
            ),
            hovertemplate=(
                f"<b>{svc.name}</b><br>"
                f"Status: {svc.status.upper()}<br>"
                f"Latency: {svc.latency_ms:.0f} ms<br>"
                f"Error rate: {svc.error_rate:.1f}%<br>"
                f"Anomaly: {svc.anomaly:.1f}<br>"
                f"Signal: {svc.signal_intensity:.1f}<br>"
                "<extra></extra>"
            ),
            name=svc.name,
            showlegend=False,
        ))

        # Service label
        label_y_offset = -0.075
        fig.add_annotation(
            x=x, y=y + label_y_offset,
            text=f"<b>{svc.name}</b>",
            showarrow=False,
            font=dict(size=12, color=PAL["node_text"]),
            bgcolor="rgba(247,246,242,0.85)",
            borderpad=3,
        )

        # Status badge (small dot above node)
        if svc.status != STATUS_HEALTHY:
            badge_col = PAL["failed_border"] if svc.status == STATUS_FAILED else PAL["degraded_border"]
            fig.add_trace(go.Scatter(
                x=[x + 0.045], y=[y + 0.045],
                mode="markers",
                marker=dict(size=10, color=badge_col, symbol="circle"),
                hoverinfo="skip",
                showlegend=False,
            ))

    # --- Legend -----------------------------------------------------------
    legend_items = [
        ("Healthy",           PAL["healthy"],   PAL["healthy_border"]),
        ("Degraded",          PAL["degraded"],  PAL["degraded_border"]),
        ("Failed",            PAL["failed"],    PAL["failed_border"]),
        ("Predicted Origin",  PAL["origin"],    PAL["origin_border"]),
    ]
    for label, fill, border in legend_items:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=14, color=fill, line=dict(color=border, width=2)),
            name=label,
            showlegend=True,
        ))

    # --- Layout -----------------------------------------------------------
    fig.update_layout(
        paper_bgcolor=PAL["bg"],
        plot_bgcolor=PAL["bg"],
        margin=dict(l=20, r=20, t=20, b=20),
        height=500,
        xaxis=dict(
            range=[-0.08, 1.08], visible=False,
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            range=[-0.08, 1.12], visible=False,
            showgrid=False, zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.08,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=PAL["edge_normal"],
            borderwidth=1,
            font=dict(size=12, color=PAL["primary_text"]),
        ),
        font=dict(family="'Inter', 'Helvetica Neue', Arial, sans-serif"),
    )

    return fig


# ---------------------------------------------------------------------------
# Signal intensity bar chart
# ---------------------------------------------------------------------------
def signal_intensity_chart(system: ServiceSystem) -> go.Figure:
    """Horizontal bar chart showing anomaly intensity per service."""
    services = system.all()
    names     = [s.name for s in services]
    anomalies = [s.anomaly for s in services]
    colours   = []
    for s in services:
        if s.status == STATUS_FAILED:
            colours.append(PAL["failed_border"])
        elif s.status == STATUS_DEGRADED:
            colours.append(PAL["degraded_border"])
        else:
            colours.append(PAL["healthy_border"])

    fig = go.Figure(go.Bar(
        x=anomalies,
        y=names,
        orientation="h",
        marker=dict(color=colours, line=dict(color="white", width=0.5)),
        text=[f"{a:.1f}" for a in anomalies],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PAL["card"],
        plot_bgcolor=PAL["card"],
        xaxis=dict(
            title="Anomaly Intensity",
            range=[0, 110],
            gridcolor="#E6E3E8",
            tickfont=dict(color=PAL["secondary_text"]),
        ),
        yaxis=dict(
            tickfont=dict(color=PAL["primary_text"]),
            autorange="reversed",
        ),
        margin=dict(l=10, r=60, t=10, b=40),
        height=260,
        font=dict(family="'Inter', Arial, sans-serif", color=PAL["primary_text"]),
    )
    return fig


# ---------------------------------------------------------------------------
# Latency comparison chart
# ---------------------------------------------------------------------------
def latency_chart(system: ServiceSystem) -> go.Figure:
    """Bar chart comparing current vs baseline latency."""
    from services import BASELINES
    services = system.all()
    names   = [s.name for s in services]
    current = [s.latency_ms for s in services]
    baseline= [BASELINES[s.name]["latency_ms"] for s in services]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline", x=names, y=baseline,
        marker_color=PAL["sage"],
        marker_line=dict(color=PAL["healthy_border"], width=1),
    ))
    fig.add_trace(go.Bar(
        name="Current", x=names, y=current,
        marker_color=PAL["rose"],
        marker_line=dict(color=PAL["failed_border"], width=1),
    ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor=PAL["card"],
        plot_bgcolor=PAL["card"],
        xaxis=dict(
            tickfont=dict(color=PAL["primary_text"]),
            gridcolor="#E6E3E8",
        ),
        yaxis=dict(
            title="Latency (ms)",
            gridcolor="#E6E3E8",
            tickfont=dict(color=PAL["secondary_text"]),
        ),
        legend=dict(font=dict(size=11)),
        margin=dict(l=10, r=10, t=10, b=40),
        height=260,
        font=dict(family="'Inter', Arial, sans-serif", color=PAL["primary_text"]),
    )
    return fig


# ---------------------------------------------------------------------------
# Candidate residual ranking chart
# ---------------------------------------------------------------------------
def candidate_ranking_chart(candidates: list) -> go.Figure:
    """Horizontal bar chart of root-cause candidate residual scores."""
    names     = [c["service"] for c in candidates]
    residuals = [c["residual"] for c in candidates]
    max_res   = max(residuals) if residuals else 1.0

    colours = []
    for i, name in enumerate(names):
        if i == 0:
            colours.append(PAL["origin"])
        elif residuals[i] < max_res * 0.6:
            colours.append(PAL["lavender"])
        else:
            colours.append(PAL["blue"])

    fig = go.Figure(go.Bar(
        x=residuals,
        y=names,
        orientation="h",
        marker=dict(
            color=colours,
            line=dict(color=PAL["origin_border"], width=0.5),
        ),
        text=[f"{r:.1f}" for r in residuals],
        textposition="outside",
        hovertemplate="<b>%{y}</b> residual: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PAL["card"],
        plot_bgcolor=PAL["card"],
        xaxis=dict(
            title="Residual Score (lower = more likely root cause)",
            gridcolor="#E6E3E8",
            tickfont=dict(color=PAL["secondary_text"]),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color=PAL["primary_text"]),
        ),
        margin=dict(l=10, r=70, t=10, b=40),
        height=260,
        font=dict(family="'Inter', Arial, sans-serif", color=PAL["primary_text"]),
    )
    return fig


# ---------------------------------------------------------------------------
# Experimental results charts
# ---------------------------------------------------------------------------
def experiment_summary_chart(results: list, metrics: dict) -> go.Figure:
    """
    Two-panel figure:
      Left  — per-scenario correct/incorrect bars.
      Right — overall metric gauge.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=["Per-Scenario Results", "Overall Metrics"],
    )

    # Left — scenario bars
    scenario_names = [r.scenario_name for r in results]
    correct_vals   = [1 if r.correct else 0 for r in results]
    colours_sc     = [PAL["healthy_border"] if c else PAL["failed_border"]
                      for c in correct_vals]

    fig.add_trace(go.Bar(
        x=scenario_names,
        y=correct_vals,
        marker_color=colours_sc,
        text=["✓" if c else "✗" for c in correct_vals],
        textposition="inside",
        textfont=dict(size=16, color="white"),
        hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Right — metric bars
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    metric_vals  = [
        metrics["accuracy"], metrics["precision"],
        metrics["recall"],   metrics["f1"],
    ]
    metric_cols  = [PAL["lavender"], PAL["sage"], PAL["blue"], PAL["peach"]]

    fig.add_trace(go.Bar(
        x=metric_names,
        y=metric_vals,
        marker_color=metric_cols,
        marker_line=dict(color=PAL["secondary_text"], width=0.5),
        text=[f"{v:.1%}" for v in metric_vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b>: %{y:.1%}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.update_yaxes(range=[0, 1.15], row=1, col=2)
    fig.update_layout(
        paper_bgcolor=PAL["bg"],
        plot_bgcolor=PAL["bg"],
        margin=dict(l=20, r=20, t=50, b=60),
        height=320,
        font=dict(family="'Inter', Arial, sans-serif", color=PAL["primary_text"]),
    )
    return fig


def propagation_timeline_chart(events: list) -> go.Figure:
    """Scatter-line chart showing signal arrival time vs intensity per service."""
    if not events:
        fig = go.Figure()
        fig.add_annotation(text="No propagation data yet.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14))
        fig.update_layout(height=200, paper_bgcolor=PAL["card"])
        return fig

    names     = [e["service"] for e in events]
    arrivals  = [e["arrival_t"] for e in events]
    signals   = [e["signal"] for e in events]

    fig = go.Figure(go.Scatter(
        x=arrivals,
        y=signals,
        mode="lines+markers+text",
        text=names,
        textposition="top center",
        marker=dict(size=10, color=PAL["origin"], line=dict(color=PAL["origin_border"], width=1.5)),
        line=dict(color=PAL["origin"], width=1.5, dash="dot"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Arrival: t=%{x:.1f}<br>"
            "Signal: %{y:.1f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        paper_bgcolor=PAL["card"],
        plot_bgcolor=PAL["card"],
        xaxis=dict(title="Simulated arrival time (t)", gridcolor="#E6E3E8"),
        yaxis=dict(title="Signal intensity", gridcolor="#E6E3E8"),
        margin=dict(l=10, r=10, t=10, b=40),
        height=240,
        font=dict(family="'Inter', Arial, sans-serif", color=PAL["primary_text"]),
    )
    return fig

