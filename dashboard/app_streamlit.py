import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import time
import plotly.graph_objects as go

from tensorflow import keras

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="NIDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    
    .metric-card {
        background-color: #1c2333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3748;
        height: 110px;                  
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #e2e8f0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        margin-top: 4px;
    }
    .alert-container {
        background-color: #1c2333;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border: 1px solid #2d3748;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .alert-label {
        font-size: 0.9rem;
        font-weight: bold;
        color: #e2e8f0;
    }
    .alert-count {
        font-size: 1.1rem;
        font-weight: bold;
        padding: 2px 10px;
        border-radius: 12px;
        background-color: #2d3748;
    }
    .alert-detail {
        font-size: 0.75rem;
        color: #718096;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

CLASS_COLORS = {
    "BENIGN":                    "#48bb78",
    "DDoS":                      "#fc8181",
    "DoS Hulk":                  "#f56565",
    "DoS GoldenEye":             "#e53e3e",
    "DoS slowloris":             "#c53030",
    "DoS Slowhttptest":          "#9b2c2c",
    "PortScan":                  "#76e4f7",
    "FTP-Patator":               "#b794f4",
    "SSH-Patator":               "#9f7aea",
    "Bot":                       "#f6ad55",
    "Web Attack - Brute Force":  "#ed8936",
    "Web Attack - XSS":          "#dd6b20",
    "Web Attack - Sql Injection":"#c05621",
    "Infiltration":              "#ff63c3",
    "Heartbleed":                "#ff0000",
    "Unknown Anomaly":           "#ffd700",
}

ALERT_CONFIDENCE_THRESHOLD = 0.85
MAX_HISTORY = 5000  # cap memory usage

# -----------------------------------------------------------------------
# Load models once
# -----------------------------------------------------------------------
@st.cache_resource
def load_models():
    scaler        = joblib.load("models/scaler.joblib")
    autoencoder   = keras.models.load_model("models/autoencoder.keras")
    xgb_model     = joblib.load("models/xgboost.joblib")
    label_encoder = joblib.load("models/label_encoder.joblib")
    with open("models/threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return scaler, autoencoder, xgb_model, label_encoder, threshold

@st.cache_data
def load_simulation_data():
    df = pd.read_csv("processed/dashboard_simulation.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# -----------------------------------------------------------------------
# Inference pipeline — one flow at a time
# -----------------------------------------------------------------------
def run_pipeline(row, scaler, autoencoder, xgb_model, label_encoder, threshold):
    features = row.drop("Label").values.reshape(1, -1)
    scaled   = scaler.transform(features)

    # autoencoder
    recon      = autoencoder.predict(scaled, verbose=0)
    mse        = float(np.mean(np.square(scaled - recon)))
    is_anomaly = mse > threshold

    # xgboost
    xgb_pred        = xgb_model.predict(scaled)[0]
    xgb_probs       = xgb_model.predict_proba(scaled)[0]
    xgb_confidence  = float(xgb_probs.max())
    predicted_class = label_encoder.inverse_transform([xgb_pred])[0]

    # decision logic
    benign_label = label_encoder.transform(["BENIGN"])[0]
    if is_anomaly and xgb_pred == benign_label:
        final_label = "Unknown Anomaly"
    else:
        final_label = predicted_class

    return {
        "label":          final_label,
        "xgb_confidence": xgb_confidence,  # XGBoost confidence (only meaningful for named classes)
        "mse":            mse,
        "mse_ratio":      mse / threshold,  # how many times above threshold (for Unknown Anomaly display)
        "is_anomaly":     is_anomaly,
        "true_label":     row["Label"],
        "dst_port":       row.get("Dst Port", "—"),
        "protocol":       row.get("Protocol", "—"),
    }

# -----------------------------------------------------------------------
# Session state init
# -----------------------------------------------------------------------
if "running"     not in st.session_state: st.session_state.running     = False
if "current_idx" not in st.session_state: st.session_state.current_idx = 0
if "history"     not in st.session_state: st.session_state.history     = []
if "alerts"      not in st.session_state: st.session_state.alerts      = []

# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ NIDS Controls")
    st.markdown("---")

    delay_ms = st.slider(
        "Speed (delay between flows)",
        min_value=0, max_value=1000, value=200, step=50,
        format="%d ms"
    )

    st.markdown("---")

    # single toggle button
    button_label = "⏸ Pause" if st.session_state.running else "▶ Start"
    if st.button(button_label, use_container_width=True):
        st.session_state.running = not st.session_state.running

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.running     = False
        st.session_state.current_idx = 0
        st.session_state.history     = []
        st.session_state.alerts      = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Model Info**")
    st.caption("Stage 1: Autoencoder (anomaly)")
    st.caption("Stage 2: XGBoost (15-class)")
    st.caption("Dataset: CIC-IDS2017 (Engelen)")

# -----------------------------------------------------------------------
# Load data and models
# -----------------------------------------------------------------------
scaler, autoencoder, xgb_model, label_encoder, threshold = load_models()
sim_data   = load_simulation_data()
total_rows = len(sim_data)

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("🛡️ Network Intrusion Detection System")
st.caption("Real-time flow analysis — CIC-IDS2017 simulation")
st.markdown("---")

# -----------------------------------------------------------------------
# Stat cards
# -----------------------------------------------------------------------
history       = st.session_state.history
total_flows   = len(history)
attack_count  = sum(1 for h in history if h["label"] not in ("BENIGN",))
attack_rate   = (attack_count / total_flows * 100) if total_flows > 0 else 0
unknown_count = sum(1 for h in history if h["label"] == "Unknown Anomaly")

named_attacks = [h["label"] for h in history
                 if h["label"] not in ("BENIGN", "Unknown Anomaly")]
top_attack = max(set(named_attacks), key=named_attacks.count) if named_attacks else "—"

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_flows:,}</div>
        <div class="metric-label">Total Flows</div>
    </div>""", unsafe_allow_html=True)

with c2:
    color = "#fc8181" if attack_rate > 5 else "#48bb78"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{attack_rate:.1f}%</div>
        <div class="metric-label">Attack Rate</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size:1.2rem">{top_attack}</div>
        <div class="metric-label">Top Attack Type</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#ffd700">{unknown_count:,}</div>
        <div class="metric-label">Unknown Anomalies</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Panel placeholders
# -----------------------------------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.subheader("Live Flow Feed")
    flow_table_placeholder = st.empty()
with right:
    st.subheader("Attack Distribution")
    bar_chart_placeholder = st.empty()

st.markdown("---")
bottom_left, bottom_right = st.columns([3, 2])
with bottom_left:
    st.subheader("Anomaly Score Timeline")
    line_chart_placeholder = st.empty()
with bottom_right:
    st.subheader("🚨 Alert Feed")
    alert_placeholder = st.empty()

# -----------------------------------------------------------------------
# Render all panels from current history
# -----------------------------------------------------------------------
def render_panels():
    if not history:
        return

    # -- flow table --
    recent = history[-50:][::-1]
    table_rows = []
    for h in recent:
        if h["label"] == "Unknown Anomaly":
            conf_display = f"AE: {h['mse_ratio']:.1f}x threshold"
        else:
            conf_display = f"{h['xgb_confidence']:.2%}"

        table_rows.append({
            "Label":      h["label"],
            "Confidence": conf_display,
            "MSE":        f"{h['mse']:.6f}",
            "Dst Port":   h["dst_port"],
            "Protocol":   h["protocol"],
        })

    flow_table_placeholder.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        height=350,
        hide_index=True
    )

    # -- attack bar chart (BENIGN excluded) --
    label_counts = {}
    for h in history:
        if h["label"] != "BENIGN":   # exclude BENIGN from attack chart
            label_counts[h["label"]] = label_counts.get(h["label"], 0) + 1

    if label_counts:
        bar_df = pd.DataFrame(
            list(label_counts.items()),
            columns=["Label", "Count"]
        ).sort_values("Count", ascending=True)

        bar_fig = go.Figure(go.Bar(
            x=bar_df["Count"],
            y=bar_df["Label"],
            orientation="h",
            marker_color=[CLASS_COLORS.get(l, "#718096") for l in bar_df["Label"]]
        ))
        bar_fig.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#1c2333",
            font_color="#e2e8f0",
            margin=dict(l=10, r=10, t=10, b=10),
            height=350,
            xaxis=dict(gridcolor="#2d3748"),
            yaxis=dict(gridcolor="#2d3748")
        )
        bar_chart_placeholder.plotly_chart(bar_fig, use_container_width=True)

    # -- anomaly score timeline --
    recent_200 = history[-200:]
    line_fig = go.Figure()

    for label in set(h["label"] for h in recent_200):
        flows_with_label = [h for h in recent_200 if h["label"] == label]
        indices = [recent_200.index(h) for h in flows_with_label]
        line_fig.add_trace(go.Scatter(
            x=indices,
            y=[h["mse"] for h in flows_with_label],
            mode="markers",
            name=label,
            marker=dict(color=CLASS_COLORS.get(label, "#718096"), size=4)
        ))

    line_fig.add_hline(
        y=threshold, line_dash="dash", line_color="#ffd700",
        annotation_text=f"Threshold ({threshold:.6f})",
        annotation_font_color="#ffd700"
    )
    line_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1c2333",
        font_color="#e2e8f0",
        margin=dict(l=10, r=10, t=10, b=10),
        height=300, showlegend=False,
        xaxis=dict(gridcolor="#2d3748", title="Recent flows"),
        yaxis=dict(gridcolor="#2d3748", title="Reconstruction MSE")
    )
    line_chart_placeholder.plotly_chart(line_fig, use_container_width=True)

    # -- alert feed — grouped by attack type --
    # count how many times each attack type has been alerted
    alert_counts = {}
    alert_last_port = {}
    for a in st.session_state.alerts:
        label = a["label"]
        alert_counts[label] = alert_counts.get(label, 0) + 1
        alert_last_port[label] = a["dst_port"]

    if alert_counts:
        alert_html = ""
        # sort by count descending so most frequent attack is on top
        for label, count in sorted(alert_counts.items(), key=lambda x: -x[1]):
            color = CLASS_COLORS.get(label, "#718096")
            last_port = alert_last_port[label]
            if label == "Unknown Anomaly":
                detail = "AE flagged — XGBoost unsure"
            else:
                detail = f"Last seen on port {last_port}"

            alert_html += f"""
            <div class="alert-container" style="border-left: 4px solid {color};">
                <div>
                    <div class="alert-label" style="color:{color}">{label}</div>
                    <div class="alert-detail">{detail}</div>
                </div>
                <div class="alert-count" style="color:{color}">{count}x</div>
            </div>"""
        alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
    else:
        alert_placeholder.caption("No alerts yet.")
# -----------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------
render_panels()

if st.session_state.running:
    idx = st.session_state.current_idx

    if idx < total_rows:
        row    = sim_data.iloc[idx]
        result = run_pipeline(row, scaler, autoencoder, xgb_model, label_encoder, threshold)

        # append to history, cap at MAX_HISTORY
        st.session_state.history.append(result)
        if len(st.session_state.history) > MAX_HISTORY:
            st.session_state.history = st.session_state.history[-MAX_HISTORY:]

        # alert feed — named attacks use XGBoost confidence threshold
        # Unknown Anomaly always goes to alert feed (AE flagged it)
        if result["label"] == "Unknown Anomaly":
            st.session_state.alerts.append(result)
        elif (result["label"] != "BENIGN"
              and result["xgb_confidence"] >= ALERT_CONFIDENCE_THRESHOLD):
            st.session_state.alerts.append(result)

        st.session_state.current_idx += 1
        time.sleep(delay_ms / 1000)
        st.rerun()

    else:
        st.session_state.running = False
        st.success("✅ Simulation complete — all flows processed.")