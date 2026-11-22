"""
app.py — SpiderSense Lite main Streamlit application.

Bio-inspired fault localisation dashboard for a simulated e-commerce
service mesh.  Biological inspiration: spider vibration sensing.

Run with:
    streamlit run app.py
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import networkx as nx
from typing import Optional

# Local modules
from services import ServiceSystem, STATUS_HEALTHY, STATUS_DEGRADED, STATUS_FAILED
from fault_injector import (
    inject_fault, FAULT_TYPES,
    FAULT_LATENCY_SPIKE, FAULT_HIGH_ERROR_RATE,
    FAULT_SERVICE_TIMEOUT, FAULT_COMPLETE_FAILURE,
    SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_LEVELS,
)
from propagation import build_graph, propagate
from root_cause import run_analysis
from scenarios import DISPLAY_SCENARIOS, run_all_experiments, compute_metrics
from visualizer import (
    service_graph, signal_intensity_chart, latency_chart,
    candidate_ranking_chart, experiment_summary_chart,
    propagation_timeline_chart, PAL,
)

# ============================================================================
# Page configuration
# ============================================================================
st.set_page_config(
    page_title="SpiderSense Lite",
    page_icon="🕷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS — pastel engineering dashboard aesthetic
# ============================================================================
st.markdown("""
<style>
/* ---- Base typography & background ---------------------------------------- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F7F6F2;
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    color: #303344;
}

/* Engineering grid paper effect on the main background */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(180,175,195,0.13) 1px, transparent 1px),
        linear-gradient(90deg, rgba(180,175,195,0.13) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] { background: transparent; }

/* ---- Sidebar ------------------------------------------------------------- */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E6E3E8;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
}

/* All text inside sidebar */
[data-testid="stSidebar"] * {
    color: #303344 !important;
}

/* Section headers (## markdown) */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #77788A !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 0.4rem !important;
}

/* Radio button container */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    background: transparent !important;
}

/* Radio button labels — this is what was invisible */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] span,
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio > div > label > div > p {
    color: #303344 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* Radio button selected state */
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p,
[data-testid="stSidebar"] .stRadio label[aria-checked="true"] p {
    color: #7A5FAF !important;
    font-weight: 700 !important;
}

/* Radio circle dots */
[data-testid="stSidebar"] input[type="radio"] + div {
    border-color: #DCD6F7 !important;
}
[data-testid="stSidebar"] input[type="radio"]:checked + div {
    background-color: #7A5FAF !important;
    border-color: #7A5FAF !important;
}

/* Nav radio wrapper — give each option a hover card effect */
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent;
    border-radius: 6px;
    padding: 5px 8px !important;
    transition: background 0.12s;
    cursor: pointer;
    width: 100%;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: #F0EEF8 !important;
}

/* Dividers */
[data-testid="stSidebar"] hr {
    border-color: #E6E3E8 !important;
    margin: 0.6rem 0 !important;
}

/* Sidebar markdown plain text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown p {
    color: #77788A !important;
    font-size: 0.82rem !important;
}

/* ---- Cards --------------------------------------------------------------- */
.ss-card {
    background: #FFFFFF;
    border: 1px solid #E6E3E8;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ---- Section header ------------------------------------------------------ */
.ss-section-title {
    font-size: 1.0rem;
    font-weight: 600;
    color: #303344;
    letter-spacing: 0.03em;
    border-bottom: 2px solid #DCD6F7;
    padding-bottom: 0.35rem;
    margin-bottom: 1rem;
}

/* ---- Status chips -------------------------------------------------------- */
.chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.chip-healthy  { background:#D7E7D0; color:#3A6B4A; border:1px solid #A8D5B5; }
.chip-degraded { background:#FFF3CD; color:#856404; border:1px solid #F5DFA0; }
.chip-failed   { background:#EBCFD2; color:#8B2D34; border:1px solid #D6939B; }

/* ---- Metric tiles -------------------------------------------------------- */
.metric-tile {
    text-align: center;
    background: #FFFFFF;
    border: 1px solid #E6E3E8;
    border-radius: 8px;
    padding: 0.8rem 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.metric-tile .value {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-tile .label {
    font-size: 0.75rem;
    color: #77788A;
    margin-top: 0.2rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ---- Table --------------------------------------------------------------- */
.ss-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.ss-table th {
    text-align: left; padding: 0.5rem 0.8rem;
    background: #F7F6F2; border-bottom: 2px solid #E6E3E8;
    color: #77788A; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.ss-table td {
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid #F0EEF4;
    color: #303344;
}
.ss-table tr:last-child td { border-bottom: none; }
.ss-table tr:hover td { background: #FAF9FC; }

/* ---- Buttons ------------------------------------------------------------- */
.stButton > button {
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.03em;
    transition: all 0.15s ease;
}
.stButton > button:hover { opacity: 0.88; }

/* ---- Callout boxes ------------------------------------------------------- */
.callout {
    padding: 0.75rem 1rem;
    border-left: 3px solid;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
}
.callout-info   { background:#D5E7F4; border-color:#4A86B8; color:#1A3A58; }
.callout-warn   { background:#FFF3CD; border-color:#C9A83C; color:#5C4A0A; }
.callout-error  { background:#EBCFD2; border-color:#C4727A; color:#5C1A20; }
.callout-ok     { background:#D7E7D0; border-color:#6FAF83; color:#1D4A2C; }

/* ---- Confidence bar ------------------------------------------------------ */
.conf-bar-wrap { background:#E6E3E8; border-radius:4px; height:10px; }
.conf-bar-fill { height:10px; border-radius:4px; }

/* ---- Page title ---------------------------------------------------------- */
.page-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #303344;
    letter-spacing: -0.01em;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #77788A;
    margin-top: -0.3rem;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session-state initialisation
# ============================================================================
def _init_state() -> None:
    if "system" not in st.session_state:
        st.session_state.system = ServiceSystem()
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()
    if "propagation_events" not in st.session_state:
        st.session_state.propagation_events = []
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "fault_applied" not in st.session_state:
        st.session_state.fault_applied = False
    if "fault_detail" not in st.session_state:
        st.session_state.fault_detail = None
    if "injected_service" not in st.session_state:
        st.session_state.injected_service = None
    if "propagated" not in st.session_state:
        st.session_state.propagated = False
    if "experiment_results" not in st.session_state:
        st.session_state.experiment_results = None
    if "experiment_metrics" not in st.session_state:
        st.session_state.experiment_metrics = None
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Overview"


_init_state()
sys: ServiceSystem   = st.session_state.system
G:   nx.DiGraph      = st.session_state.graph


# ============================================================================
# Helpers
# ============================================================================
def _reset() -> None:
    sys.reset()
    st.session_state.propagation_events = []
    st.session_state.analysis_result    = None
    st.session_state.fault_applied      = False
    st.session_state.fault_detail       = None
    st.session_state.injected_service   = None
    st.session_state.propagated         = False


def _chip(status: str) -> str:
    css = {"healthy": "chip-healthy", "degraded": "chip-degraded", "failed": "chip-failed"}
    cls = css.get(status, "chip-healthy")
    return f'<span class="chip {cls}">{status.upper()}</span>'


def _status_colour(status: str) -> str:
    return {"healthy": "#6FAF83", "degraded": "#C9A83C", "failed": "#C4727A"}.get(status, "#77788A")


def _conf_bar(conf: float) -> str:
    pct = int(conf * 100)
    if pct >= 80:
        col = "#6FAF83"
    elif pct >= 60:
        col = "#C9A83C"
    else:
        col = "#C4727A"
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill" style="width:{pct}%;background:{col};"></div>'
        f'</div><div style="font-size:0.78rem;color:#77788A;margin-top:3px;">'
        f'Confidence: {pct}%</div>'
    )


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown('<div style="padding:0.5rem 0 1rem;">'
                '<span style="font-size:1.6rem;">🕷</span>'
                '<span style="font-size:1.1rem;font-weight:700;color:#303344;'
                'margin-left:0.5rem;">SpiderSense Lite</span></div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Fault Injection", "Signal Propagation",
         "Root-Cause Analysis", "Baseline Comparison", "Experiment Lab"],
        index=["Overview", "Fault Injection", "Signal Propagation",
               "Root-Cause Analysis", "Baseline Comparison",
               "Experiment Lab"].index(st.session_state.active_page),
        label_visibility="collapsed",
    )
    st.session_state.active_page = page

    st.markdown("---")
    st.markdown("## Quick Scenarios")
    for sc in DISPLAY_SCENARIOS:
        if st.button(sc.name, key=f"scenario_{sc.name}", use_container_width=True):
            _reset()
            inject_fault(sys, sc.service, sc.fault_type, sc.severity)
            st.session_state.fault_applied    = True
            st.session_state.injected_service = sc.service
            st.session_state.fault_detail     = {
                "service": sc.service, "fault_type": sc.fault_type,
                "severity": sc.severity
            }
            events = propagate(sys, G, sc.service)
            st.session_state.propagation_events = events
            st.session_state.propagated = True
            result = run_analysis(sys, G)
            result["actual"]  = sc.service
            result["correct"] = result["predicted"] == sc.service
            st.session_state.analysis_result = result
            st.session_state.active_page = "Root-Cause Analysis"
            st.rerun()

    st.markdown("---")
    st.markdown("## System Actions")
    if st.button("Reset System", use_container_width=True, type="secondary"):
        _reset()
        st.rerun()

    # Status summary in sidebar
    counts = sys.counts()
    st.markdown("---")
    st.markdown("## System Status")
    for label, key, col in [
        ("Total Services", "total", "#303344"),
        ("Healthy",        "healthy",  "#6FAF83"),
        ("Degraded",       "degraded", "#C9A83C"),
        ("Failed",         "failed",   "#C4727A"),
    ]:
        v = counts[key]
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:3px 0;font-size:0.88rem;">'
            f'<span style="color:#77788A;">{label}</span>'
            f'<span style="color:{col};font-weight:700;">{v}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:#AEADBF;text-align:center;">'
                'Bio-inspired Engineering Project<br>RVCE · Sem 3</div>',
                unsafe_allow_html=True)


# ============================================================================
# Page: Overview
# ============================================================================
if page == "Overview":
    st.markdown('<div class="page-title">System Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Service dependency topology and real-time health status.</div>',
        unsafe_allow_html=True,
    )

    # Introduction callout
    st.markdown("""
    <div class="callout callout-info">
    <b>SpiderSense Lite</b> is a bio-inspired fault localisation system for simulated software services.
    Just as a spider senses the location of a trapped insect from vibrations in its web,
    SpiderSense detects the <i>origin</i> of a cascading failure by analysing how a disturbance
    signal propagates through the service dependency graph.
    </div>
    """, unsafe_allow_html=True)

    # Status metric tiles
    counts = sys.counts()
    c1, c2, c3, c4 = st.columns(4)
    for col, label, key, colour in [
        (c1, "Total Services", "total",   "#303344"),
        (c2, "Healthy",        "healthy", "#6FAF83"),
        (c3, "Degraded",       "degraded","#C9A83C"),
        (c4, "Failed",         "failed",  "#C4727A"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="value" style="color:{colour};">{counts[key]}</div>'
                f'<div class="label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Service dependency graph
    st.markdown('<div class="ss-section-title">Service Dependency Graph</div>',
                unsafe_allow_html=True)

    # Determine which edges to highlight
    prop_edges = None
    predicted_root = None
    if st.session_state.propagated:
        prop_edges = [(e["edge"][0], e["edge"][1])
                      for e in st.session_state.propagation_events
                      if e.get("edge") and e["edge"][0]]
    if st.session_state.analysis_result:
        predicted_root = st.session_state.analysis_result.get("predicted")

    fig_graph = service_graph(
        sys, G,
        highlight_edges=prop_edges,
        predicted_root=predicted_root,
    )
    st.plotly_chart(fig_graph, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        '<div style="font-size:0.78rem;color:#77788A;text-align:center;margin-top:-0.5rem;">'
        'Arrows show the direction of <b>failure-impact propagation</b> (not normal request flow). '
        'Hover over nodes for details.</div>',
        unsafe_allow_html=True,
    )

    # Service status table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ss-section-title">Service Health Table</div>',
                unsafe_allow_html=True)

    rows_html = ""
    for svc in sys.all():
        rows_html += (
            f"<tr>"
            f"<td><b>{svc.name}</b></td>"
            f"<td>{_chip(svc.status)}</td>"
            f"<td>{svc.latency_ms:.0f} ms</td>"
            f"<td>{svc.error_rate:.1f}%</td>"
            f"<td>{svc.anomaly:.1f}</td>"
            f"<td>{'Yes' if svc.threshold_exceeded else '—'}</td>"
            f"</tr>"
        )

    st.markdown(f"""
    <div class="ss-card">
    <table class="ss-table">
      <thead>
        <tr>
          <th>Service</th><th>Status</th><th>Latency</th>
          <th>Error Rate</th><th>Anomaly Intensity</th><th>Signal Exceeded</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ss-section-title">How It Works</div>', unsafe_allow_html=True)

    steps = [
        ("1. Service Network",
         "Six e-commerce services are modelled as nodes in a directed dependency graph. "
         "Edges represent failure-impact pathways."),
        ("2. Fault Injection",
         "A fault (latency spike, high error rate, timeout, or complete failure) is applied "
         "to one service, modifying its simulated metrics."),
        ("3. Signal Propagation",
         "A disturbance signal travels outward along the dependency edges, attenuating at "
         "each hop — analogous to vibrations decaying across a spider's web."),
        ("4. Root-Cause Inference",
         "The algorithm tests each service as a hypothetical origin, computes the expected "
         "propagation pattern, and selects the candidate whose prediction best matches "
         "the observed anomaly distribution."),
        ("5. Diagnostic Results",
         "The predicted root cause, candidate rankings, propagation path, and confidence "
         "are displayed for review and comparison with conventional threshold monitoring."),
    ]

    cols = st.columns(len(steps))
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="ss-card" style="height:100%;">'
                f'<div style="font-size:0.82rem;font-weight:700;color:#7A5FAF;'
                f'margin-bottom:0.4rem;">{title}</div>'
                f'<div style="font-size:0.82rem;color:#303344;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ============================================================================
# Page: Fault Injection
# ============================================================================
elif page == "Fault Injection":
    st.markdown('<div class="page-title">Fault Injection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Select a service, choose a fault type and severity, then inject.'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.fault_applied:
        fd = st.session_state.fault_detail
        st.markdown(
            f'<div class="callout callout-warn">'
            f'Active fault: <b>{fd["fault_type"]}</b> on <b>{fd["service"]}</b> '
            f'[severity: {fd["severity"]}]. Reset the system before injecting a new fault.'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_form, col_info = st.columns([1.2, 1])

    with col_form:
        st.markdown('<div class="ss-section-title">Inject a Fault</div>',
                    unsafe_allow_html=True)

        with st.form("fault_form"):
            target_service = st.selectbox(
                "Target Service",
                sys.SERVICE_NAMES,
                help="The service that will be directly affected by the fault.",
            )
            fault_type = st.selectbox(
                "Fault Type",
                FAULT_TYPES,
                help=(
                    "Latency Spike: increased response time. "
                    "High Error Rate: elevated failure responses. "
                    "Service Timeout: connections hang and fail. "
                    "Complete Failure: service becomes entirely unavailable."
                ),
            )
            severity = st.select_slider(
                "Severity",
                options=[SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH],
                value=SEVERITY_MEDIUM,
            )

            col_a, col_b = st.columns(2)
            inject_btn = col_a.form_submit_button(
                "Inject Fault", use_container_width=True, type="primary"
            )
            reset_btn  = col_b.form_submit_button(
                "Reset System", use_container_width=True
            )

        if reset_btn:
            _reset()
            st.rerun()

        if inject_btn:
            if st.session_state.fault_applied:
                st.warning("Reset the system before injecting a new fault.")
            else:
                detail = inject_fault(sys, target_service, fault_type, severity)
                st.session_state.fault_applied    = True
                st.session_state.injected_service = target_service
                st.session_state.fault_detail     = {
                    "service": target_service,
                    "fault_type": fault_type,
                    "severity": severity,
                }
                st.success(
                    f"Fault injected: **{fault_type}** on **{target_service}** "
                    f"({severity} severity). "
                    f"Latency +{detail['delta_latency']:.0f} ms, "
                    f"Error rate +{detail['delta_error']:.1f}%."
                )
                st.rerun()

    with col_info:
        st.markdown('<div class="ss-section-title">Fault Type Reference</div>',
                    unsafe_allow_html=True)
        fault_info = {
            FAULT_LATENCY_SPIKE: (
                "Response time climbs dramatically. Downstream services begin "
                "experiencing timeouts waiting for this service's reply."
            ),
            FAULT_HIGH_ERROR_RATE: (
                "A large proportion of requests fail. Downstream services that "
                "depend on successful responses degrade rapidly."
            ),
            FAULT_SERVICE_TIMEOUT: (
                "Connections hang for a long time before failing. "
                "Thread pools and connection queues in dependent services fill up."
            ),
            FAULT_COMPLETE_FAILURE: (
                "The service is entirely unavailable. All dependent services "
                "immediately experience errors and anomalies."
            ),
        }
        for ft, desc in fault_info.items():
            st.markdown(
                f'<div class="ss-card" style="margin-bottom:0.6rem;">'
                f'<div style="font-size:0.85rem;font-weight:600;color:#303344;">{ft}</div>'
                f'<div style="font-size:0.82rem;color:#77788A;margin-top:0.25rem;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Current service state summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ss-section-title">Current Service State</div>',
                unsafe_allow_html=True)

    cols = st.columns(3)
    for i, svc in enumerate(sys.all()):
        with cols[i % 3]:
            border_col = _status_colour(svc.status)
            st.markdown(
                f'<div class="ss-card" style="border-left:4px solid {border_col};">'
                f'<div style="font-weight:700;font-size:0.95rem;">{svc.name}</div>'
                f'{_chip(svc.status)}'
                f'<div style="margin-top:0.5rem;font-size:0.82rem;color:#77788A;">'
                f'Latency: <b style="color:#303344;">{svc.latency_ms:.0f} ms</b><br>'
                f'Error rate: <b style="color:#303344;">{svc.error_rate:.1f}%</b><br>'
                f'Anomaly: <b style="color:#303344;">{svc.anomaly:.1f}</b>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ============================================================================
# Page: Signal Propagation
# ============================================================================
elif page == "Signal Propagation":
    st.markdown('<div class="page-title">Signal Propagation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Bio-inspired disturbance propagation through the service mesh.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Biological inspiration callout
    st.markdown("""
    <div class="callout callout-info">
    <b>Biological Inspiration:</b> When a spider's prey touches its web, vibrations travel
    outward through the silk threads, attenuating with distance and at junctions.
    The spider integrates signals from its legs to triangulate the disturbance source.
    SpiderSense models this computationally: a disturbance signal propagates through
    the service dependency graph, losing intensity at each hop according to the
    transmission coefficient of the connection.
    <br><br>
    <b>Propagation model:</b> S<sub>next</sub> = S<sub>current</sub> × T<sub>ij</sub> &nbsp;|&nbsp;
    t<sub>next</sub> = t<sub>current</sub> + d<sub>ij</sub>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.fault_applied:
        st.markdown("""
        <div class="callout callout-warn">
        No fault has been injected yet. Go to <b>Fault Injection</b> or use a
        Quick Scenario in the sidebar to set up a fault first.
        </div>
        """, unsafe_allow_html=True)
    else:
        fd = st.session_state.fault_detail
        col_run, col_reset = st.columns([1, 4])
        with col_run:
            run_btn = st.button("Run Propagation", type="primary", use_container_width=True)

        if run_btn:
            if st.session_state.propagated:
                st.info("Propagation already run. Reset to run again.")
            else:
                events = propagate(sys, G, st.session_state.injected_service)
                st.session_state.propagation_events = events
                st.session_state.propagated = True
                st.rerun()

        if st.session_state.propagated:
            events = st.session_state.propagation_events

            st.markdown('<div class="ss-section-title">Propagation Graph</div>',
                        unsafe_allow_html=True)

            prop_edges = [(e["edge"][0], e["edge"][1])
                          for e in events if e.get("edge") and e["edge"][0]]
            fig_prop = service_graph(sys, G, highlight_edges=prop_edges)
            st.plotly_chart(fig_prop, use_container_width=True,
                            config={"displayModeBar": False})

            # Propagation event table
            st.markdown('<div class="ss-section-title">Propagation Event Log</div>',
                        unsafe_allow_html=True)

            if events:
                rows = ""
                for e in events:
                    svc = sys.get(e["service"])
                    rows += (
                        f"<tr>"
                        f"<td>{e['arrival_t']:.2f}</td>"
                        f"<td><b>{e['service']}</b></td>"
                        f"<td>{e.get('parent', '—')}</td>"
                        f"<td>{e['signal']:.1f}</td>"
                        f"<td>{e.get('transmission', 0):.2f}</td>"
                        f"<td>{_chip(svc.status)}</td>"
                        f"</tr>"
                    )
                st.markdown(f"""
                <div class="ss-card">
                <table class="ss-table">
                  <thead>
                    <tr>
                      <th>Arrival Time (t)</th><th>Service</th><th>From</th>
                      <th>Signal Intensity</th><th>Transmission Coeff.</th><th>Status</th>
                    </tr>
                  </thead>
                  <tbody>{rows}</tbody>
                </table>
                </div>
                """, unsafe_allow_html=True)

                # Timeline chart
                st.markdown('<div class="ss-section-title">Signal Propagation Timeline</div>',
                            unsafe_allow_html=True)
                fig_timeline = propagation_timeline_chart(events)
                st.plotly_chart(fig_timeline, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.info("The fault did not propagate beyond the origin service "
                        "above the minimum signal threshold.")

            # Signal intensity chart
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown('<div class="ss-section-title">Anomaly Intensity</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(signal_intensity_chart(sys),
                                use_container_width=True, config={"displayModeBar": False})
            with col_r:
                st.markdown('<div class="ss-section-title">Latency Comparison</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(latency_chart(sys),
                                use_container_width=True, config={"displayModeBar": False})

            # Transmission coefficient table
            st.markdown('<div class="ss-section-title">Edge Transmission Coefficients</div>',
                        unsafe_allow_html=True)
            from propagation import TRANSMISSION, DELAYS
            edge_rows = ""
            for (src, dst), t in TRANSMISSION.items():
                d = DELAYS[(src, dst)]
                is_active = (src, dst) in prop_edges
                row_style = 'style="background:#EEF4FB;"' if is_active else ""
                edge_rows += (
                    f"<tr {row_style}>"
                    f"<td><b>{src}</b></td><td><b>{dst}</b></td>"
                    f"<td>{t:.2f}</td><td>{d:.1f}</td>"
                    f"<td>{'Active' if is_active else '—'}</td>"
                    f"</tr>"
                )
            st.markdown(f"""
            <div class="ss-card">
            <table class="ss-table">
              <thead>
                <tr>
                  <th>From</th><th>To</th>
                  <th>Transmission (T<sub>ij</sub>)</th>
                  <th>Delay (d<sub>ij</sub>)</th><th>In Propagation Path</th>
                </tr>
              </thead>
              <tbody>{edge_rows}</tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# Page: Root-Cause Analysis
# ============================================================================
elif page == "Root-Cause Analysis":
    st.markdown('<div class="page-title">Root-Cause Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Independent inference of the originating fault using residual minimisation.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.propagated:
        st.markdown("""
        <div class="callout callout-warn">
        Signal propagation has not been run yet. Inject a fault and run propagation first,
        or use a Quick Scenario from the sidebar.
        </div>
        """, unsafe_allow_html=True)
    else:
        col_run, _ = st.columns([1, 4])
        with col_run:
            analyse_btn = st.button("Run Analysis", type="primary", use_container_width=True)

        if analyse_btn or st.session_state.analysis_result is None:
            result = run_analysis(sys, G)
            if st.session_state.injected_service:
                result["actual"]  = st.session_state.injected_service
                result["correct"] = result["predicted"] == st.session_state.injected_service
            st.session_state.analysis_result = result
            st.rerun()

        result = st.session_state.analysis_result
        if result is None:
            st.info("Click 'Run Analysis' to start.")
        else:
            # Algorithm explanation callout
            st.markdown("""
            <div class="callout callout-info">
            <b>Algorithm:</b> The engine evaluates every service as a hypothetical fault origin.
            For each candidate <i>c</i>, it predicts the expected propagation pattern and computes
            the weighted residual R(c) = Σ wᵢ × (Oᵢ − Pᵢ(c))².
            The candidate with the <b>lowest residual</b> is selected as the predicted root cause.
            The engine does <b>not</b> access the injected-fault label.
            </div>
            """, unsafe_allow_html=True)

            # --- A. Root Cause prediction ---
            st.markdown('<div class="ss-section-title">A. Root Cause Prediction</div>',
                        unsafe_allow_html=True)

            col_pred, col_actual = st.columns(2)
            with col_pred:
                pred = result["predicted"]
                pred_svc = sys.get(pred)
                pred_color = "#7A5FAF"
                st.markdown(
                    f'<div class="ss-card" style="border-top:4px solid {pred_color};">'
                    f'<div style="font-size:0.78rem;font-weight:600;color:#77788A;'
                    f'text-transform:uppercase;letter-spacing:0.06em;">Predicted Root Cause</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:{pred_color};'
                    f'margin:0.4rem 0;">{pred}</div>'
                    f'{_chip(pred_svc.status)}'
                    f'<div style="margin-top:0.6rem;">'
                    f'{_conf_bar(result["confidence"])}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col_actual:
                actual = result.get("actual")
                correct = result.get("correct")
                if actual:
                    if correct:
                        verdict_col  = "#6FAF83"
                        verdict_text = "Correct prediction"
                        verdict_cls  = "callout-ok"
                    else:
                        verdict_col  = "#C4727A"
                        verdict_text = "Incorrect prediction"
                        verdict_cls  = "callout-error"
                    st.markdown(
                        f'<div class="ss-card" style="border-top:4px solid {verdict_col};">'
                        f'<div style="font-size:0.78rem;font-weight:600;color:#77788A;'
                        f'text-transform:uppercase;letter-spacing:0.06em;">Actual Injected Fault</div>'
                        f'<div style="font-size:1.5rem;font-weight:700;color:{verdict_col};'
                        f'margin:0.4rem 0;">{actual}</div>'
                        f'<div class="callout {verdict_cls}" style="margin-top:0.5rem;">'
                        f'{verdict_text}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="ss-card">'
                        '<div style="font-size:0.78rem;color:#77788A;">Actual fault not set.</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

            if result.get("uncertain"):
                st.markdown(
                    '<div class="callout callout-warn">⚠ Uncertainty: multiple candidates '
                    'have similar residual scores. The prediction may not be reliable.</div>',
                    unsafe_allow_html=True,
                )

            # --- B. Affected services ---
            st.markdown('<div class="ss-section-title">B. Affected Services</div>',
                        unsafe_allow_html=True)
            affected = sys.affected_services()
            if affected:
                chip_row = " &nbsp; ".join(
                    f'<b>{s.name}</b> {_chip(s.status)}' for s in affected
                )
                st.markdown(
                    f'<div class="ss-card">{chip_row}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ss-card">No services exceeded the anomaly threshold.</div>',
                    unsafe_allow_html=True,
                )

            # --- C. Propagation details table ---
            st.markdown('<div class="ss-section-title">C. Propagation Details</div>',
                        unsafe_allow_html=True)
            rows = ""
            for svc in sys.all():
                t_str = f"{svc.signal_arrival_t:.2f}" if svc.signal_arrival_t is not None else "—"
                rows += (
                    f"<tr>"
                    f"<td><b>{svc.name}</b></td>"
                    f"<td>{svc.signal_intensity:.1f}</td>"
                    f"<td>{t_str}</td>"
                    f"<td>{_chip(svc.status)}</td>"
                    f"</tr>"
                )
            st.markdown(f"""
            <div class="ss-card">
            <table class="ss-table">
              <thead>
                <tr>
                  <th>Service</th><th>Signal Intensity</th>
                  <th>Arrival Time (t)</th><th>Status</th>
                </tr>
              </thead>
              <tbody>{rows}</tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)

            # --- D. Candidate ranking ---
            st.markdown('<div class="ss-section-title">D. Candidate Ranking</div>',
                        unsafe_allow_html=True)
            col_chart, col_tbl = st.columns([1.2, 1])
            with col_chart:
                fig_rank = candidate_ranking_chart(result["candidates"])
                st.plotly_chart(fig_rank, use_container_width=True,
                                config={"displayModeBar": False})
            with col_tbl:
                rows = ""
                for c in result["candidates"]:
                    bold_start = "<b>" if c["rank"] == 1 else ""
                    bold_end   = "</b>" if c["rank"] == 1 else ""
                    rows += (
                        f"<tr>"
                        f"<td>{c['rank']}</td>"
                        f"<td>{bold_start}{c['service']}{bold_end}</td>"
                        f"<td>{c['residual']:.2f}</td>"
                        f"</tr>"
                    )
                st.markdown(f"""
                <div class="ss-card">
                <table class="ss-table">
                  <thead><tr><th>Rank</th><th>Service</th><th>Residual R(c)</th></tr></thead>
                  <tbody>{rows}</tbody>
                </table>
                <div style="font-size:0.78rem;color:#77788A;margin-top:0.5rem;">
                Lower residual = more likely root cause.</div>
                </div>
                """, unsafe_allow_html=True)

            # --- E. Propagation path graph ---
            st.markdown('<div class="ss-section-title">E. Predicted Propagation Path</div>',
                        unsafe_allow_html=True)
            path = result.get("propagation_path", [])
            if path:
                path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                fig_path = service_graph(
                    sys, G,
                    highlight_edges=path_edges,
                    predicted_root=result["predicted"],
                )
                st.plotly_chart(fig_path, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(
                    f'<div style="text-align:center;font-size:0.85rem;color:#77788A;">'
                    f'Predicted cascade: <b>{"  →  ".join(path)}</b></div>',
                    unsafe_allow_html=True,
                )

            # --- F. Explanation ---
            st.markdown('<div class="ss-section-title">F. Explanation</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="ss-card">{result["explanation"]}</div>',
                unsafe_allow_html=True,
            )


# ============================================================================
# Page: Baseline Comparison
# ============================================================================
elif page == "Baseline Comparison":
    st.markdown('<div class="page-title">Baseline Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Conventional threshold monitoring vs. SpiderSense root-cause inference.'
        '</div>',
        unsafe_allow_html=True,
    )

    from services import LATENCY_THRESHOLD, ERROR_RATE_THRESHOLD

    # Threshold monitoring output
    flagged_latency = [s for s in sys.all() if s.latency_ms > LATENCY_THRESHOLD]
    flagged_error   = [s for s in sys.all() if s.error_rate > ERROR_RATE_THRESHOLD]
    all_flagged     = list({s.name: s for s in flagged_latency + flagged_error}.values())

    col_base, col_spider = st.columns(2)

    with col_base:
        st.markdown('<div class="ss-section-title">Threshold Monitoring</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.8rem;color:#77788A;margin-bottom:0.5rem;">'
            f'Latency threshold: <b>{LATENCY_THRESHOLD:.0f} ms</b> &nbsp;|&nbsp; '
            f'Error rate threshold: <b>{ERROR_RATE_THRESHOLD:.0f}%</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if all_flagged:
            rows = ""
            for s in all_flagged:
                lat_flag = "⚑" if s.latency_ms > LATENCY_THRESHOLD else ""
                err_flag = "⚑" if s.error_rate > ERROR_RATE_THRESHOLD else ""
                rows += (
                    f"<tr>"
                    f"<td><b>{s.name}</b></td>"
                    f"<td>{s.latency_ms:.0f} ms {lat_flag}</td>"
                    f"<td>{s.error_rate:.1f}% {err_flag}</td>"
                    f"</tr>"
                )
            st.markdown(f"""
            <div class="ss-card">
            <div style="font-weight:600;color:#C4727A;margin-bottom:0.5rem;">
              Abnormal services detected: {len(all_flagged)}</div>
            <table class="ss-table">
              <thead><tr><th>Service</th><th>Latency</th><th>Error Rate</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
            <div style="font-size:0.82rem;color:#77788A;margin-top:0.75rem;">
              <b>Conclusion:</b> The following services appear abnormal:<br>
              <b>{', '.join(s.name for s in all_flagged)}</b><br>
              <i>No originating fault is identified.</i>
            </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="ss-card callout callout-ok">All services are within normal thresholds.</div>',
                unsafe_allow_html=True,
            )

    with col_spider:
        st.markdown('<div class="ss-section-title">SpiderSense Analysis</div>',
                    unsafe_allow_html=True)

        result = st.session_state.analysis_result
        if result:
            pred     = result["predicted"]
            conf     = result["confidence"]
            path     = result.get("propagation_path", [])
            path_str = " → ".join(path) if path else pred

            st.markdown(
                f'<div class="ss-card">'
                f'<div style="font-weight:600;color:#7A5FAF;margin-bottom:0.5rem;">'
                f'Root Cause Identified</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:#303344;margin-bottom:0.5rem;">'
                f'{pred}</div>'
                f'<div style="margin-bottom:0.75rem;">{_conf_bar(conf)}</div>'
                f'<div style="font-size:0.82rem;color:#77788A;">'
                f'<b>Predicted cascade:</b><br>'
                f'<span style="color:#303344;">{path_str}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ss-card callout callout-info">'
                'Run Root-Cause Analysis first to see SpiderSense results.'
                '</div>',
                unsafe_allow_html=True,
            )

    # Comparison narrative
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ss-section-title">Comparative Summary</div>',
                unsafe_allow_html=True)

    base_diag = (
        f"Abnormal services: **{', '.join(s.name for s in all_flagged)}**. "
        "No root cause identified."
        if all_flagged else
        "No abnormal services detected."
    )
    spider_diag = (
        f"Predicted root cause: **{result['predicted']}** "
        f"(confidence {result['confidence']*100:.0f}%). "
        f"Predicted cascade: {' → '.join(result.get('propagation_path', [result['predicted']]))}"
        if result else
        "Analysis not yet run."
    )

    st.markdown(f"""
    <div class="ss-card">
    <table class="ss-table">
      <thead><tr><th>Dimension</th><th>Threshold Monitoring</th><th>SpiderSense</th></tr></thead>
      <tbody>
        <tr>
          <td>Diagnostic output</td>
          <td>{base_diag}</td>
          <td>{spider_diag}</td>
        </tr>
        <tr>
          <td>Root-cause identification</td>
          <td>❌ Not provided</td>
          <td>✅ Provided</td>
        </tr>
        <tr>
          <td>Propagation path</td>
          <td>❌ Not provided</td>
          <td>✅ Provided</td>
        </tr>
        <tr>
          <td>Confidence indicator</td>
          <td>❌ Not provided</td>
          <td>✅ Provided</td>
        </tr>
        <tr>
          <td>May generate false positives</td>
          <td>Yes (flags secondary effects)</td>
          <td>Mitigated by residual ranking</td>
        </tr>
      </tbody>
    </table>
    <div style="font-size:0.8rem;color:#77788A;margin-top:0.75rem;">
    Both methods use the same simulated service observations.
    Threshold monitoring lists all abnormal services equally;
    SpiderSense uses propagation modelling to infer which service caused the others to become abnormal.
    </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Page: Experiment Lab
# ============================================================================
elif page == "Experiment Lab":
    st.markdown('<div class="page-title">Experiment Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Run all pre-defined test scenarios and evaluate localisation performance.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="callout callout-info">
    This section runs every defined scenario on a fresh system state,
    executes the full propagation and analysis pipeline, and measures
    root-cause localisation accuracy. All metrics are computed from actual
    algorithm outputs — no values are hardcoded.
    </div>
    """, unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_exp = st.button("Run All Experiments", type="primary", use_container_width=True)

    if run_exp:
        with st.spinner("Running experiments…"):
            exp_results = run_all_experiments(G)
            exp_metrics = compute_metrics(exp_results)
            st.session_state.experiment_results = exp_results
            st.session_state.experiment_metrics = exp_metrics
        st.rerun()

    exp_results = st.session_state.experiment_results
    exp_metrics = st.session_state.experiment_metrics

    if exp_results:
        # Metric tiles
        m_cols = st.columns(4)
        for col, (label, key) in zip(m_cols, [
            ("Accuracy",  "accuracy"),
            ("Precision", "precision"),
            ("Recall",    "recall"),
            ("F1 Score",  "f1"),
        ]):
            val = exp_metrics[key]
            pct = f"{val*100:.1f}%"
            colour = "#6FAF83" if val >= 0.8 else "#C9A83C" if val >= 0.6 else "#C4727A"
            with col:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="value" style="color:{colour};">{pct}</div>'
                    f'<div class="label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            f'<div style="font-size:0.82rem;color:#77788A;margin-top:0.5rem;margin-bottom:1rem;">'
            f'{exp_metrics["correct"]} correct / {exp_metrics["total"]} total scenarios</div>',
            unsafe_allow_html=True,
        )

        # Summary chart
        st.markdown('<div class="ss-section-title">Experiment Results Chart</div>',
                    unsafe_allow_html=True)
        fig_exp = experiment_summary_chart(exp_results, exp_metrics)
        st.plotly_chart(fig_exp, use_container_width=True, config={"displayModeBar": False})

        # Detailed results table
        st.markdown('<div class="ss-section-title">Per-Scenario Results</div>',
                    unsafe_allow_html=True)
        rows = ""
        for r in exp_results:
            verdict_col  = "#6FAF83" if r.correct else "#C4727A"
            verdict_text = "✓ Correct" if r.correct else "✗ Incorrect"
            rows += (
                f"<tr>"
                f"<td><b>{r.scenario_name}</b></td>"
                f"<td>{r.actual_fault}</td>"
                f"<td><b>{r.predicted_fault}</b></td>"
                f"<td style='color:{verdict_col};font-weight:700;'>{verdict_text}</td>"
                f"<td>{r.affected_count}</td>"
                f"<td>{r.analysis_time_ms:.1f} ms</td>"
                f"<td>{r.residual_score:.2f}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <div class="ss-card">
        <table class="ss-table">
          <thead>
            <tr>
              <th>Scenario</th><th>Actual Fault</th><th>Predicted</th>
              <th>Verdict</th><th>Affected Services</th>
              <th>Analysis Time</th><th>Best Residual</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        # SpiderSense vs threshold monitoring comparison
        st.markdown('<div class="ss-section-title">SpiderSense vs Threshold Monitoring</div>',
                    unsafe_allow_html=True)

        from services import LATENCY_THRESHOLD, ERROR_RATE_THRESHOLD

        # For each experiment, compute what threshold monitoring would have reported
        from scenarios import SCENARIOS
        from propagation import build_graph as _bg, propagate as _prop
        from fault_injector import inject_fault as _inj

        compare_rows = ""
        for scenario, exp_res in zip(SCENARIOS, exp_results):
            tmp_sys = ServiceSystem()
            _inj(tmp_sys, scenario.service, scenario.fault_type, scenario.severity)
            _prop(tmp_sys, G, scenario.service)

            baseline_flagged = [
                s.name for s in tmp_sys.all()
                if s.latency_ms > LATENCY_THRESHOLD or s.error_rate > ERROR_RATE_THRESHOLD
            ]
            baseline_str = ", ".join(baseline_flagged) if baseline_flagged else "None"
            spider_str   = exp_res.predicted_fault
            verdict_col  = "#6FAF83" if exp_res.correct else "#C4727A"

            compare_rows += (
                f"<tr>"
                f"<td><b>{scenario.name}</b></td>"
                f"<td>{scenario.service}</td>"
                f"<td><i>{baseline_str} (no root cause)</i></td>"
                f"<td style='color:{verdict_col};font-weight:700;'>{spider_str}</td>"
                f"</tr>"
            )

        st.markdown(f"""
        <div class="ss-card">
        <table class="ss-table">
          <thead>
            <tr>
              <th>Scenario</th><th>Actual Fault</th>
              <th>Threshold Monitoring Output</th>
              <th>SpiderSense Prediction</th>
            </tr>
          </thead>
          <tbody>{compare_rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="callout callout-info">'
            'Click "Run All Experiments" to execute the test suite.'
            '</div>',
            unsafe_allow_html=True,
        )

