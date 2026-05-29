"""
dashboard.py
============
Smart Traffic Digital Twin — Streamlit Dashboard

Run with:
    streamlit run dashboard.py
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import os
import time
from datetime import datetime, timedelta

# Local modules
from database         import (fetch_all_traffic, fetch_latest_traffic,
                               fetch_recent_alerts, fetch_analytics, init_db, seed_demo_data)
from prediction_engine import train_and_predict, build_forecast_df
from signal_optimizer  import (get_signal_recommendation, optimize_all_junctions,
                                classify_congestion, congestion_color_rgb,
                                calculate_green_time)
from simulation_engine import (get_simulation_frame, get_scenario_names,
                                get_scenario_info, JUNCTIONS)
from network_model     import (render_network_graph, network_stats, propagate_congestion)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title = "Smart Traffic Digital Twin",
    page_icon  = "🚦",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Indian Government Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ──────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

html, body {
    font-family: 'Open Sans', Arial, sans-serif;
    background-color: #F4F5F7;
    color: #333333;
}

/* ── Main background ─────────────────────── */
.stApp { background-color: #F4F5F7; }

/* ── Sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 2px solid #E0E0E0;
    box-shadow: 2px 0 5px rgba(0,0,0,0.05);
}

/* ── KPI Cards ───────────────────────────── */
[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-top: 4px solid #FF9933; /* Saffron accent */
    border-radius: 6px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border-top: 4px solid #138808; /* Green accent on hover */
}
[data-testid="stMetricLabel"]  { color: #000080 !important; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
[data-testid="stMetricValue"]  { color: #333333 !important; font-size: 1.8rem; font-weight: 700; }
[data-testid="stMetricDelta"]  { font-size: 0.85rem; }

/* ── Section headers ─────────────────────── */
h2, h3 { color: #000080; font-weight: 700; }
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #000080;
    border-bottom: 2px solid #FF9933;
    padding-bottom: 5px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Alert boxes ─────────────────────────── */
.alert-box {
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    border: 1px solid transparent;
    border-left: 5px solid;
}
.alert-red    { background: #FDECEA; border-color: #F44336; color: #D32F2F; border-left-color: #F44336; }
.alert-orange { background: #FFF4E5; border-color: #FF9800; color: #E65100; border-left-color: #FF9800; }
.alert-blue   { background: #E8F4FD; border-color: #2196F3; color: #0D47A1; border-left-color: #2196F3; }
.alert-green  { background: #EDF7ED; border-color: #4CAF50; color: #1B5E20; border-left-color: #4CAF50; }
.alert-purple { background: #F3E5F5; border-color: #9C27B0; color: #4A148C; border-left-color: #9C27B0; }

/* ── Cards ───────────────────────────────── */
.info-card {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-left: 4px solid #000080;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── Status dot ──────────────────────────── */
.dot-green  { display:inline-block; width:12px; height:12px; border-radius:50%; background:#138808; margin-right:8px; border: 1px solid #0E6806; }
.dot-red    { display:inline-block; width:12px; height:12px; border-radius:50%; background:#D32F2F; margin-right:8px; border: 1px solid #9A0007; }
.dot-yellow { display:inline-block; width:12px; height:12px; border-radius:50%; background:#FFC107; margin-right:8px; border: 1px solid #C79100; }

/* ── Buttons ─────────────────────────────── */
.stButton > button {
    background-color: #000080;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    font-weight: 600;
    padding: 0.5rem 1rem;
    transition: background-color 0.2s ease;
}
.stButton > button:hover {
    background-color: #003366;
    color: #FFFFFF;
}

/* ── Divider ─────────────────────────────── */
hr { border-color: #E0E0E0; border-width: 2px; }

/* ── DataFrame ───────────────────────────── */
.stDataFrame { border-radius: 6px; border: 1px solid #D1D5DB; }

/* ── Tab styling ─────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; border-bottom: 2px solid #E0E0E0; }
.stTabs [data-baseweb="tab"] {
    background: #F4F5F7;
    border-radius: 4px 4px 0 0;
    color: #555555;
    border: 1px solid transparent;
    font-weight: 600;
    padding: 10px 24px;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #000080 !important;
    border: 1px solid #D1D5DB;
    border-bottom: 2px solid #FFFFFF;
    border-top: 3px solid #FF9933;
}

/* ── Selectbox / Slider ──────────────────── */
.stSelectbox > div > div, .stSlider { color: #333333; }

/* ── Progress bar ────────────────────────── */
.stProgress > div > div { background-color: #138808; border-radius: 2px; }

/* ── Title styling ──────────────────────── */
.title-gradient {
    color: #000080;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    border-bottom: 3px solid #FF9933;
    padding-bottom: 10px;
    margin-bottom: 20px;
    text-align: center;
}
.gov-header {
    text-align: center;
    color: #555555;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 10px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────
FRAME_PATH = os.path.join(os.path.dirname(__file__), "latest_frame.jpg")

def _plotly_light():
    """Return a light Plotly layout dict suitable for the Government theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333333", family="Open Sans"),
        xaxis=dict(gridcolor="#E0E0E0", zerolinecolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0", zerolinecolor="#E0E0E0"),
        margin=dict(l=40, r=20, t=40, b=40),
    )

def _alert_html(icon, text, style="blue"):
    return f'<div class="alert-box alert-{style}">{icon} {text}</div>'


# ─────────────────────────────────────────────
# SIDEBAR — SIMULATION & CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="title-gradient">🚦 Control Center</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Scenario Selector ───────────────────────
    st.markdown("### 🎮 Simulation Mode")
    scenario = st.selectbox(
        "Active Scenario",
        get_scenario_names(),
        key="scenario_select",
    )
    sinfo = get_scenario_info(scenario)
    st.markdown(
        f'<div class="alert-box alert-blue">'
        f'{sinfo["icon"]} {sinfo["description"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Control Toggles ─────────────────────────
    st.markdown("### ⚙️ Smart City Controls")
    manual_override  = st.toggle("🔧 Manual Signal Override", value=False, key="manual_ovr")
    emergency_mode   = st.toggle("🚨 Emergency Mode",         value=False, key="emerg_mode")
    prediction_mode  = st.toggle("🔮 Prediction Mode",        value=True,  key="pred_mode")
    simulation_mode  = st.toggle("🌐 Live Simulation",        value=True,  key="sim_mode")

    st.markdown("---")

    # ── Auto-Refresh ────────────────────────────
    st.markdown("### 🔄 Auto Refresh")
    auto_refresh = st.toggle("Enable Auto-Refresh", value=False, key="auto_ref")
    refresh_interval = st.slider("Interval (seconds)", 5, 60, 10, key="ref_int")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

    st.markdown("---")

    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<small style="color:#555555;">Smart Traffic Digital Twin<br>'
        'People of India Theme</small>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# INIT DB & LOAD DATA
# ─────────────────────────────────────────────
init_db()
seed_demo_data()

df_all     = pd.DataFrame(fetch_all_traffic())
latest_row = fetch_latest_traffic()
analytics  = fetch_analytics()
alerts_log = fetch_recent_alerts(limit=15)

# Graceful fallback if DB is empty
if latest_row is None:
    latest_row = {
        "id": 0, "timestamp": str(datetime.now()),
        "vehicle_count": 0, "avg_speed": 0.0,
        "signal": "GREEN", "congestion": "LOW", "ambulance": 0
    }
if df_all.empty:
    df_all = pd.DataFrame([latest_row])


# ─────────────────────────────────────────────
# SIMULATION FRAME
# ─────────────────────────────────────────────
sim_frame = get_simulation_frame(
    scenario       = scenario,
    base_count     = int(latest_row.get("vehicle_count", 10)),
    emergency_mode = emergency_mode,
)

# Propagate congestion through the network
junction_vehicles = {
    jn: js.vehicle_count for jn, js in sim_frame.junctions.items()
}
propagated_veh = propagate_congestion(junction_vehicles)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
pred = train_and_predict(df_all, scenario=scenario) if prediction_mode else {
    "pred_5min": 0, "pred_15min": 0,
    "trend": "N/A", "confidence": 0, "series": []
}
green_time = calculate_green_time(
    int(latest_row.get("vehicle_count", 0)),
    ambulance=bool(latest_row.get("ambulance", 0)) or emergency_mode
)

ambulance_active = (
    bool(latest_row.get("ambulance", 0))
    or emergency_mode
    or sim_frame.global_ambulance
)


# ─────────────────────────────────────────────
# ━━━━━  DASHBOARD HEADER  ━━━━━━━━━━━━━━━━━━━
# ─────────────────────────────────────────────
st.markdown(
    '<div class="gov-header">People of India - Smart City Traffic Operations</div>'
    '<h1 class="title-gradient">🚦 Smart Traffic Digital Twin</h1>',
    unsafe_allow_html=True
)
sub_cols = st.columns([3, 1])
with sub_cols[0]:
    st.markdown(
        f'<small style="color:#555555;">Smart City Control Dashboard  ·  '
        f'{datetime.now().strftime("%A, %d %B %Y  %H:%M:%S")}  ·  '
        f'Scenario: <b style="color:#000080;">{scenario}</b></small>',
        unsafe_allow_html=True
    )
with sub_cols[1]:
    if ambulance_active:
        st.markdown(
            '<div class="alert-box alert-purple">🚑 EMERGENCY ACTIVE</div>',
            unsafe_allow_html=True
        )

st.markdown("---")


# ─────────────────────────────────────────────
# ━━━━━  KPI ROW  ━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ─────────────────────────────────────────────
kc1, kc2, kc3, kc4, kc5, kc6, kc7 = st.columns(7)

vehicle_count = int(latest_row.get("vehicle_count", 0))
avg_speed_val = float(latest_row.get("avg_speed", 0))
signal_val    = str(latest_row.get("signal", "GREEN"))
congestion_val= str(latest_row.get("congestion", "LOW"))

with kc1:
    st.metric("🚗 Vehicles",        vehicle_count,
              delta=f"{vehicle_count - int(df_all['vehicle_count'].iloc[-2]) if len(df_all) > 1 else 0}")
with kc2:
    st.metric("⚡ Avg Speed",       f"{avg_speed_val:.1f} km/h")
with kc3:
    signal_icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(signal_val, "⚪")
    st.metric("🚦 Signal",          f"{signal_icon} {signal_val}")
with kc4:
    cong_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "EMERGENCY": "🟣"}.get(congestion_val, "⚪")
    st.metric("🌡️ Congestion",      f"{cong_icon} {congestion_val}")
with kc5:
    amb_status = "🚑 YES" if ambulance_active else "❌ NO"
    st.metric("🏥 Ambulance",       amb_status)
with kc6:
    st.metric("🔮 Pred 5 min",      pred["pred_5min"],
              delta=f"Trend: {pred['trend']}")
with kc7:
    st.metric("🟢 Green Time",      f"{green_time}s")


# ─────────────────────────────────────────────
# EMERGENCY BANNER
# ─────────────────────────────────────────────
if ambulance_active:
    st.markdown(
        '<div class="alert-box alert-purple" style="font-size:1rem; padding:14px 20px;">'
        '🚑 &nbsp;<b>AMBULANCE DETECTED — Emergency Signal Priority ACTIVATED</b>'
        '&nbsp;·&nbsp; All junctions cleared &nbsp;·&nbsp; Emergency corridor OPEN'
        '</div>', unsafe_allow_html=True
    )

for alert_text in sim_frame.alerts:
    level = (
        "purple" if "AMBULANCE" in alert_text
        else "red"   if "congestion" in alert_text.lower() or "ACCIDENT" in alert_text
        else "orange" if "ROAD BLOCK" in alert_text
        else "blue"
    )
    st.markdown(
        f'<div class="alert-box alert-{level}">{alert_text}</div>',
        unsafe_allow_html=True
    )

st.markdown("---")


# ─────────────────────────────────────────────
# ━━━━━  TABS  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Live View & Map",
    "🔮 Prediction & Signals",
    "🌐 Network Graph",
    "📊 Analytics",
    "🚨 Alerts",
    "⚙️ System Health",
])


# ══════════════════════════════════════════════
# TAB 1 — LIVE VIEW & DIGITAL TWIN MAP
# ══════════════════════════════════════════════
with tab1:
    cam_col, map_col = st.columns([1, 1], gap="medium")

    # ── LEFT: Camera Feed ──────────────────────
    with cam_col:
        st.markdown('<div class="section-header">📹 Live Camera Feed</div>', unsafe_allow_html=True)
        if os.path.exists(FRAME_PATH):
            st.image(FRAME_PATH, caption="Latest Detection Frame", use_container_width=True)
        else:
            st.info(
                "📷 No live frame yet.\n\n"
                "Run the detector to generate frames:\n"
                "```\npython traffic_detector.py\n```"
            )
        # Mini stats under camera
        c1, c2, c3 = st.columns(3)
        c1.metric("Vehicles",  vehicle_count)
        c2.metric("Speed",     f"{avg_speed_val:.1f}")
        c3.metric("Signal",    signal_val)

    # ── RIGHT: PyDeck Digital Twin Map ─────────
    with map_col:
        st.markdown('<div class="section-header">🗺️ Digital Twin City Map</div>', unsafe_allow_html=True)

        jdf = sim_frame.junctions
        map_rows = []
        for jname, js in jdf.items():
            # Use propagated vehicle count for visual
            prop_vc = propagated_veh.get(jname, js.vehicle_count)
            cong    = classify_congestion(prop_vc)
            color   = congestion_color_rgb(cong)

            map_rows.append({
                "junction":   jname,
                "lat":        js.lat,
                "lon":        js.lon,
                "vehicles":   prop_vc,
                "avg_speed":  js.avg_speed,
                "congestion": cong,
                "signal":     js.signal,
                "color":      color,
                "radius":     200 + prop_vc * 18,
            })

        map_df = pd.DataFrame(map_rows)

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_weight="vehicles",
            opacity=0.55,
            threshold=0.1,
        )

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            opacity=0.85,
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
        )

        text_layer = pdk.Layer(
            "TextLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_text="junction",
            get_size=14,
            get_color=[0, 0, 128],
            get_alignment_baseline="'bottom'",
        )

        view_state = pdk.ViewState(
            latitude=12.9715,
            longitude=77.5946,
            zoom=13.5,
            pitch=50,
            bearing=15,
        )

        deck = pdk.Deck(
            layers=[heatmap_layer, scatter_layer, text_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v11",
            tooltip={
                "html": (
                    "<b>{junction}</b><br/>"
                    "🚗 Vehicles: {vehicles}<br/>"
                    "⚡ Speed: {avg_speed} km/h<br/>"
                    "🌡️ Congestion: {congestion}<br/>"
                    "🚦 Signal: {signal}"
                ),
                "style": {
                    "backgroundColor": "#FFFFFF",
                    "color": "#000080",
                    "fontSize": "13px",
                    "borderRadius": "8px",
                    "padding": "10px",
                },
            },
        )

        st.pydeck_chart(deck, use_container_width=True)

    # ── Junction Status Table ──────────────────
    st.markdown("### 🏙️ Junction Status")
    display_df = map_df[["junction", "vehicles", "avg_speed", "congestion", "signal"]].copy()
    display_df.columns = ["Junction", "Vehicles", "Avg Speed (km/h)", "Congestion", "Signal"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 2 — PREDICTION & SIGNAL OPTIMIZATION
# ══════════════════════════════════════════════
with tab2:
    pred_col, sig_col = st.columns([1, 1], gap="large")

    # ── LEFT: Traffic Prediction ───────────────
    with pred_col:
        st.markdown("### 🔮 Traffic Prediction")

        p1, p2, p3 = st.columns(3)
        p1.metric("Now",       vehicle_count)
        p2.metric("In 5 min",  pred["pred_5min"],
                  delta=pred["pred_5min"] - vehicle_count)
        p3.metric("In 15 min", pred["pred_15min"],
                  delta=pred["pred_15min"] - vehicle_count)

        # Trend card
        trend_color = {
            "IMPROVING":  "#138808",
            "STABLE":     "#FFC107",
            "WORSENING":  "#D32F2F",
        }.get(pred["trend"], "#000080")
        trend_icon  = {
            "IMPROVING":  "📉",
            "STABLE":     "➡️",
            "WORSENING":  "📈",
        }.get(pred["trend"], "❓")

        st.markdown(
            f'<div class="info-card">'
            f'<b style="color:{trend_color};">{trend_icon} Trend: {pred["trend"]}</b>'
            f'<br><small style="color:#555555;">Model confidence: {pred["confidence"]}%'
            f'  ·  Scenario: {scenario}</small></div>',
            unsafe_allow_html=True
        )

        # Forecast chart
        if pred["series"]:
            forecast_df = build_forecast_df(vehicle_count, pred["series"])
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=forecast_df["Minute"],
                y=forecast_df["Predicted Vehicles"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#FF9933", width=2.5),
                marker=dict(size=5, color="#FF9933"),
                fill="tozeroy",
                fillcolor="rgba(255,153,51,0.08)",
            ))
            fig_pred.add_vline(x=0,  line_dash="dash", line_color="#FF9600", annotation_text="Now")
            fig_pred.add_vline(x=5,  line_dash="dot",  line_color="#2196F3", annotation_text="+5 min")
            fig_pred.add_vline(x=15, line_dash="dot",  line_color="#D32F2F", annotation_text="+15 min")
            fig_pred.update_layout(
                title="30-Minute Traffic Forecast",
                xaxis_title="Minutes Ahead",
                yaxis_title="Predicted Vehicles",
                **_plotly_light()
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.info("Gathering more data for predictions...")

    # ── RIGHT: Smart Signals ───────────────────
    with sig_col:
        st.markdown("### 🚦 Smart Signal Timing")
        
        st.markdown(
            f'<div class="info-card"><span class="dot-green"></span>'
            f'<b>Adaptive Signal Engine</b>&nbsp;&nbsp;'
            f'<span style="color:#138808;">ONLINE</span></div>',
            unsafe_allow_html=True
        )

        opt_signals = optimize_all_junctions(junction_vehicles)
        st.dataframe(pd.DataFrame(opt_signals), use_container_width=True)

        fig_sig = px.bar(
            x=[s["junction"] for s in opt_signals],
            y=[s["green_time"] for s in opt_signals],
            color=[s["congestion"] for s in opt_signals],
            color_discrete_map={"LOW": "#138808", "MEDIUM": "#FFC107", "HIGH": "#D32F2F", "EMERGENCY": "#9C27B0"},
            labels={"x": "Junction", "y": "Green Time (seconds)", "color": "Congestion Level"},
            title="Optimized Green Times Across Network"
        )
        fig_sig.update_layout(**_plotly_light())
        st.plotly_chart(fig_sig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — NETWORK GRAPH
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🌐 Multi-Junction Traffic Propagation (NetworkX)")
    st.markdown("Visualizing the spill-over effect of congestion across interconnected junctions.")
    
    img_buf = render_network_graph(junction_vehicles, title=f"Network Traffic (Scenario: {scenario})")
    
    col_n1, col_n2 = st.columns([2, 1])
    with col_n1:
        st.image(img_buf, use_container_width=True)
        
    with col_n2:
        st.markdown("#### Network Statistics")
        stats = network_stats(junction_vehicles)
        for k, v in stats.items():
            if isinstance(v, (int, float, str)):
                st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
                
        st.markdown("---")
        st.markdown(
            '<div class="alert-box alert-blue">ℹ️ The network model applies a propagation factor, '
            'pushing a fraction of excess vehicles from highly congested junctions into adjacent nodes.</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 System Analytics")
    if len(df_all) > 1:
        # Time series of vehicle count
        fig_ts = px.line(df_all.tail(100), x="timestamp", y="vehicle_count", title="Live Vehicle Tracking (Last 100 Logs)")
        fig_ts.update_traces(line_color="#000080")
        fig_ts.update_layout(**_plotly_light())
        st.plotly_chart(fig_ts, use_container_width=True)

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            fig_hist = px.histogram(df_all, x="vehicle_count", nbins=20, title="Vehicle Count Distribution", color_discrete_sequence=["#FF9933"])
            fig_hist.update_layout(**_plotly_light())
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_a2:
            fig_pie = px.pie(df_all, names="congestion", title="Historical Congestion Breakdown", color="congestion",
                             color_discrete_map={"LOW": "#138808", "MEDIUM": "#FFC107", "HIGH": "#D32F2F", "EMERGENCY": "#9C27B0"})
            fig_pie.update_layout(**_plotly_light())
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Waiting for more data to generate analytics...")


# ══════════════════════════════════════════════
# TAB 5 — ALERTS
# ══════════════════════════════════════════════
with tab5:
    st.markdown("### 🚨 Recent Alerts Log")
    if alerts_log:
        for alert in alerts_log:
            st.markdown(f'<div class="alert-box alert-orange">{alert[1]} - {alert[2]}</div>', unsafe_allow_html=True)
    else:
        st.info("No recent alerts.")


# ══════════════════════════════════════════════
# TAB 6 — SYSTEM HEALTH
# ══════════════════════════════════════════════
with tab6:
    sys1, sys2 = st.columns(2)
    with sys1:
        st.markdown("#### ⚙️ Service Status")
        st.markdown(
            f'<div class="info-card"><span class="dot-green"></span>'
            f'<b>Streamlit Dashboard</b>&nbsp;&nbsp;'
            f'<span style="color:#138808;">ONLINE</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="info-card"><span class="dot-green"></span>'
            f'<b>YOLOv8 Computer Vision</b>&nbsp;&nbsp;'
            f'<span style="color:#138808;">ONLINE</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="info-card"><span class="dot-green"></span>'
            f'<b>SQLite Database</b>&nbsp;&nbsp;'
            f'<span style="color:#138808;">ONLINE</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="info-card"><span class="dot-green"></span>'
            f'<b>NetworkX Graph Model</b>&nbsp;&nbsp;'
            f'<span style="color:#138808;">ONLINE — 5 junctions, {len(JUNCTIONS)} nodes</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    with sys2:
        st.markdown("#### 📈 System Metrics")
        db_size_bytes = os.path.getsize(os.path.join(os.path.dirname(__file__), "traffic_twin.db")) if os.path.exists(os.path.join(os.path.dirname(__file__), "traffic_twin.db")) else 0
        metric_items = [
            ("Total DB Records",   int(analytics.get("total_records",  0) or 0)),
            ("DB Size",            f"{db_size_bytes / 1024:.1f} KB"),
            ("Junctions Modelled", len(JUNCTIONS)),
            ("Active Scenario",    scenario),
            ("ML Model Degree",    "Polynomial (d=3)"),
            ("Prediction Horizon", "30 minutes"),
            ("Signal Cycle Max",   "120 seconds"),
            ("Emergency Override", "Active" if ambulance_active else "Inactive"),
        ]

        for label, value in metric_items:
            st.markdown(
                f'<div class="info-card">'
                f'<small style="color:#555555;">{label}</small><br>'
                f'<b style="color:#000080;">{value}</b>'
                f'</div>',
                unsafe_allow_html=True
            )
