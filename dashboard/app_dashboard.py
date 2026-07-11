import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import joblib
import time
import numpy as np
import pandas as pd
from tensorflow import keras

from dash import Dash, html, dcc, Output, Input, State
import plotly.graph_objects as go

# -----------------------------------------------------------------------
# Load models once at server startup
# -----------------------------------------------------------------------
print("Loading models...")
scaler        = joblib.load("models/scaler.joblib")
autoencoder   = keras.models.load_model("models/autoencoder.keras")
xgb_model     = joblib.load("models/xgboost.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

with open("models/threshold.json") as f:
    threshold = json.load(f)["threshold"]

print("Loading simulation data...")
sim_data = pd.read_csv("processed/dashboard_simulation.csv")
sim_data = sim_data.sample(frac=1, random_state=42).reset_index(drop=True)
TOTAL_ROWS = len(sim_data)

BENIGN_LABEL = label_encoder.transform(["BENIGN"])[0]
ALERT_CONFIDENCE_THRESHOLD = 0.85
CONFIDENCE_GATE = 0.90   # final decision logic: AE override only trusted below this XGBoost BENIGN confidence
INTERVAL_MS = 800

print(f"Ready — {TOTAL_ROWS} flows loaded.")

CLASS_COLORS = {
    "BENIGN":                    "#48bb78",
    "DDoS":                      "#fc8181",
    "DoS Hulk":                  "#c53030",
    "DoS GoldenEye":             "#e53e3e",
    "DoS slowloris":             "#f56565",
    "DoS Slowhttptest":          "#9b2c2c",
    "PortScan":                  "#76e4f7",
    "FTP-Patator":               "#b794f4",
    "SSH-Patator":               "#9f7aea",
    "Bot":                       "#2f87da",
    "Web Attack - Brute Force":  "#ed8936",
    "Web Attack - XSS":          "#dd6b20",
    "Web Attack - Sql Injection":"#c05621",
    "Infiltration":              "#ff63c3",
    "Heartbleed":                "#ce1791",
    "Unknown Anomaly":           "#ffd700",
}

# -----------------------------------------------------------------------
# Inference pipeline — final confidence-gated decision logic
# -----------------------------------------------------------------------
def run_pipeline(row):
    features = row.drop("Label").values.reshape(1, -1)
    scaled   = scaler.transform(features)

    recon      = autoencoder.predict(scaled, verbose=0)
    mse        = float(np.mean(np.square(scaled - recon)))
    is_anomaly = mse > threshold

    xgb_pred          = xgb_model.predict(scaled)[0]
    xgb_probs         = xgb_model.predict_proba(scaled)[0]
    xgb_confidence     = float(xgb_probs.max())
    benign_confidence  = float(xgb_probs[BENIGN_LABEL])
    predicted_class    = label_encoder.inverse_transform([xgb_pred])[0]

    # final decision logic — confidence-gated
    # only override to Unknown Anomaly when XGBoost itself is unsure
    # about BENIGN (benign_confidence < CONFIDENCE_GATE)
    if is_anomaly and xgb_pred == BENIGN_LABEL and benign_confidence < CONFIDENCE_GATE:
        final_label = "Unknown Anomaly"
    else:
        final_label = predicted_class

    return {
        "label":          final_label,
        "xgb_confidence": xgb_confidence,
        "mse":            mse,
        "mse_ratio":      mse / threshold,
        "dst_port":       row.get("Dst Port", "—"),
        "protocol":       row.get("Protocol", "—"),
    }

# -----------------------------------------------------------------------
# App layout
# -----------------------------------------------------------------------
app = Dash(__name__)

def empty_fig():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1c2333",
        font_color="#e2e8f0", height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#2d3748", showticklabels=False),
        yaxis=dict(gridcolor="#2d3748", showticklabels=False),
        annotations=[dict(
            text="Press ▶ Start to begin",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color="#718096", size=14)
        )]
    )
    return fig

app.layout = html.Div(
    style={"backgroundColor": "#0e1117", "minHeight": "100vh", "padding": "20px", "fontFamily": "sans-serif"},
    children=[

        dcc.Store(id="state", data={
            "running":        False,
            "current_idx":    0,
            "history":        [],
            "total_flows":    0,
            "attack_count":   0,
            "unknown_count":  0,
            "label_counts":   {},
            "alerts":         {},
            "alert_ports":    {},
            "last_tick_time": None,
            "flow_rate":      0.0,
        }),

        dcc.Interval(id="interval", interval=INTERVAL_MS, disabled=True),

        html.Div([
            html.H1("🛡️ Network Intrusion Detection System",
                    style={"color": "#e2e8f0", "margin": "0"}),
            html.P("Real-time flow analysis — CIC-IDS2017 simulation",
                   style={"color": "#718096", "margin": "4px 0 0 0"}),
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Button("▶ Start", id="toggle-btn", style={
                "backgroundColor": "#2d3748", "color": "#e2e8f0",
                "border": "none", "borderRadius": "8px",
                "padding": "10px 24px", "fontSize": "1rem",
                "cursor": "pointer", "marginRight": "10px"
            }),
            html.Button("🔄 Reset", id="reset-btn", style={
                "backgroundColor": "#2d3748", "color": "#e2e8f0",
                "border": "none", "borderRadius": "8px",
                "padding": "10px 24px", "fontSize": "1rem",
                "cursor": "pointer"
            }),
            html.Span("Press ▶ Start to begin", id="status-text",
                      style={"color": "#718096", "marginLeft": "16px", "fontSize": "0.9rem"}),
        ], style={"marginBottom": "24px"}),

        html.Div([
            html.Div([
                html.Div("0", style={"fontSize": "2rem", "fontWeight": "bold", "color": "#e2e8f0"}),
                html.Div(label, style={"fontSize": "0.85rem", "color": "#718096", "marginTop": "4px"})
            ], style={
                "backgroundColor": "#1c2333", "borderRadius": "10px", "padding": "20px",
                "textAlign": "center", "border": "1px solid #2d3748", "height": "110px",
                "display": "flex", "flexDirection": "column", "justifyContent": "center"
            })
            for label in ["Total Flows", "Attack Rate", "Top Attack Type", "Unknown Anomalies"]
        ], id="stat-cards", style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "16px",
            "marginBottom": "24px"
        }),

        html.Div([
            html.Div([
                html.H3("Live Flow Feed", style={"color": "#e2e8f0", "marginBottom": "10px"}),
                html.Div(id="flow-table",
                         children=html.P("Press ▶ Start to begin", style={"color": "#718096"}))
            ], style={"flex": "3", "marginRight": "20px"}),

            html.Div([
                html.H3("Attack Distribution", style={"color": "#e2e8f0", "marginBottom": "10px"}),
                dcc.Graph(id="bar-chart", figure=empty_fig(), config={"displayModeBar": False})
            ], style={"flex": "2"}),

        ], style={"display": "flex", "marginBottom": "24px"}),

        html.Div([
            html.Div([
                html.H3("Anomaly Score Timeline", style={"color": "#e2e8f0", "marginBottom": "10px"}),
                dcc.Graph(id="line-chart", figure=empty_fig(), config={"displayModeBar": False})
            ], style={"flex": "3", "marginRight": "20px"}),

            html.Div([
                html.H3("🚨 Alert Feed", style={"color": "#e2e8f0", "marginBottom": "10px"}),
                html.Div(id="alert-feed",
                         children=html.P("No alerts yet.", style={"color": "#718096"}))
            ], style={"flex": "2"}),

        ], style={"display": "flex"}),
    ]
)

# -----------------------------------------------------------------------
# Callback 1 — toggle Start/Pause
# -----------------------------------------------------------------------
@app.callback(
    Output("interval", "disabled"),
    Output("toggle-btn", "children"),
    Output("state", "data", allow_duplicate=True),
    Input("toggle-btn", "n_clicks"),
    State("state", "data"),
    prevent_initial_call=True
)
def toggle_simulation(n_clicks, state):
    state["running"] = not state["running"]
    label    = "⏸ Pause" if state["running"] else "▶ Start"
    disabled = not state["running"]
    return disabled, label, state

# -----------------------------------------------------------------------
# Callback 2 — reset
# -----------------------------------------------------------------------
@app.callback(
    Output("interval", "disabled", allow_duplicate=True),
    Output("toggle-btn", "children", allow_duplicate=True),
    Output("state", "data", allow_duplicate=True),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True
)
def reset_simulation(n_clicks):
    fresh_state = {
        "running":        False,
        "current_idx":    0,
        "history":        [],
        "total_flows":    0,
        "attack_count":   0,
        "unknown_count":  0,
        "label_counts":   {},
        "alerts":         {},
        "alert_ports":    {},
        "last_tick_time": None,
        "flow_rate":      0.0,
    }
    return True, "▶ Start", fresh_state

# -----------------------------------------------------------------------
# Callback 3 — main simulation tick
# -----------------------------------------------------------------------
@app.callback(
    Output("state", "data"),
    Output("stat-cards", "children"),
    Output("flow-table", "children"),
    Output("bar-chart", "figure"),
    Output("line-chart", "figure"),
    Output("alert-feed", "children"),
    Output("status-text", "children"),
    Input("interval", "n_intervals"),
    State("state", "data"),
    prevent_initial_call=True
)
def simulation_tick(n_intervals, state):
    idx = state["current_idx"]

    if idx < TOTAL_ROWS and state["running"]:
        row    = sim_data.iloc[idx]
        result = run_pipeline(row)
        label  = result["label"]

        state["history"].append(result)
        if len(state["history"]) > 200:
            state["history"] = state["history"][-200:]

        state["total_flows"]  += 1
        state["label_counts"][label] = state["label_counts"].get(label, 0) + 1

        if label != "BENIGN":
            state["attack_count"] += 1
        if label == "Unknown Anomaly":
            state["unknown_count"] += 1

        if label == "Unknown Anomaly" or (
            label != "BENIGN"
            and result["xgb_confidence"] >= ALERT_CONFIDENCE_THRESHOLD
        ):
            state["alerts"][label] = state["alerts"].get(label, 0) + 1
            state["alert_ports"][label] = result["dst_port"]

        state["current_idx"] += 1

    # ---- stat cards ----
    total_flows   = state["total_flows"]
    attack_count  = state["attack_count"]
    attack_rate   = (attack_count / total_flows * 100) if total_flows > 0 else 0
    unknown_count = state["unknown_count"]

    named = {k: v for k, v in state["label_counts"].items()
             if k not in ("BENIGN", "Unknown Anomaly")}
    top_attack = max(named, key=named.get) if named else "—"

    def card(value, label, color="#e2e8f0"):
        return html.Div([
            html.Div(str(value), style={"fontSize": "2rem", "fontWeight": "bold", "color": color}),
            html.Div(label, style={"fontSize": "0.85rem", "color": "#718096", "marginTop": "4px"})
        ], style={
            "backgroundColor": "#1c2333", "borderRadius": "10px",
            "padding": "20px", "textAlign": "center",
            "border": "1px solid #2d3748", "height": "110px",
            "display": "flex", "flexDirection": "column", "justifyContent": "center"
        })

    rate_color = "#fc8181" if attack_rate > 5 else "#48bb78"
    cards = [
        card(f"{total_flows:,}", "Total Flows"),
        card(f"{attack_rate:.1f}%", "Attack Rate", rate_color),
        card(top_attack, "Top Attack Type"),
        card(f"{unknown_count:,}", "Unknown Anomalies", "#ffd700"),
    ]

    # ---- flow table — fixed 10 rows ----
    history = state["history"]
    if history:
        recent = history[-10:][::-1]
        rows = []
        for h in recent:
            conf = (f"AE: {h['mse_ratio']:.1f}x threshold"
                    if h["label"] == "Unknown Anomaly"
                    else f"{h['xgb_confidence']:.2%}")
            color = CLASS_COLORS.get(h["label"], "#e2e8f0")
            bg    = "#1a2420" if h["label"] == "BENIGN" else "#2a1e1e"
            rows.append(html.Tr([
                html.Td(h["label"],         style={"color": color, "fontWeight": "bold", "padding": "8px 12px"}),
                html.Td(conf,               style={"color": "#a0aec0", "padding": "8px 12px"}),
                html.Td(f"{h['mse']:.6f}",  style={"color": "#a0aec0", "padding": "8px 12px"}),
                html.Td(str(h["dst_port"]), style={"color": "#a0aec0", "padding": "8px 12px"}),
                html.Td(str(h["protocol"]), style={"color": "#a0aec0", "padding": "8px 12px"}),
            ], style={"backgroundColor": bg, "borderBottom": "1px solid #2d3748"}))

        table = html.Div(
            html.Table([
                html.Thead(html.Tr([
                    html.Th(col, style={
                        "color": "#718096", "padding": "8px 12px",
                        "textAlign": "left", "borderBottom": "2px solid #2d3748",
                        "position": "sticky", "top": "0", "backgroundColor": "#1c2333"
                    })
                    for col in ["Label", "Confidence", "MSE", "Dst Port", "Protocol"]
                ])),
                html.Tbody(rows)
            ], style={"width": "100%", "borderCollapse": "collapse"}),
            style={
                "backgroundColor": "#1c2333", "borderRadius": "8px",
                "border": "1px solid #2d3748",
                "height": "360px", "overflow": "hidden"
            }
        )
    else:
        table = html.P("Waiting for flows...", style={"color": "#718096"})

    # ---- bar chart ----
    attack_counts = {k: v for k, v in state["label_counts"].items() if k != "BENIGN"}
    if attack_counts:
        bar_df = pd.DataFrame(list(attack_counts.items()), columns=["Label", "Count"])
        bar_df = bar_df.sort_values("Count", ascending=True)
        bar_fig = go.Figure(go.Bar(
            x=bar_df["Count"], y=bar_df["Label"], orientation="h",
            marker_color=[CLASS_COLORS.get(l, "#718096") for l in bar_df["Label"]]
        ))
    else:
        bar_fig = go.Figure()

    bar_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1c2333",
        font_color="#e2e8f0", height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748")
    )

    # ---- anomaly timeline ----
    line_fig = go.Figure()
    if history:
        recent_200 = history[-200:]
        for label in set(h["label"] for h in recent_200):
            subset  = [h for h in recent_200 if h["label"] == label]
            indices = [recent_200.index(h) for h in subset]
            line_fig.add_trace(go.Scatter(
                x=indices,
                y=[h["mse"] for h in subset],
                mode="markers", name=label,
                marker=dict(color=CLASS_COLORS.get(label, "#718096"), size=4)
            ))

    line_fig.add_hline(
        y=threshold, line_dash="dash", line_color="#ffd700",
        annotation_text=f"Threshold ({threshold:.6f})",
        annotation_font_color="#ffd700"
    )
    line_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1c2333",
        font_color="#e2e8f0", height=300, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#2d3748", title="Recent flows"),
        yaxis=dict(gridcolor="#2d3748", title="Reconstruction MSE")
    )

    # ---- alert feed ----
    if state["alerts"]:
        alert_divs = []
        for label, count in sorted(state["alerts"].items(), key=lambda x: -x[1]):
            color     = CLASS_COLORS.get(label, "#718096")
            last_port = state["alert_ports"].get(label, "—")
            detail    = ("AE flagged — XGBoost unsure"
                         if label == "Unknown Anomaly"
                         else f"Last seen on port {last_port}")
            alert_divs.append(html.Div([
                html.Div([
                    html.Div(label, style={"color": color, "fontWeight": "bold", "fontSize": "0.9rem"}),
                    html.Div(detail, style={"color": "#718096", "fontSize": "0.75rem", "marginTop": "2px"})
                ]),
                html.Div(f"{count}x", style={
                    "color": color, "fontWeight": "bold",
                    "fontSize": "1.1rem", "backgroundColor": "#2d3748",
                    "padding": "2px 10px", "borderRadius": "12px"
                })
            ], style={
                "backgroundColor": "#1c2333", "borderRadius": "8px",
                "padding": "12px 16px", "marginBottom": "8px",
                "border": "1px solid #2d3748",
                "borderLeft": f"4px solid {color}",
                "display": "flex", "justifyContent": "space-between",
                "alignItems": "center"
            }))
        alerts = html.Div(alert_divs)
    else:
        alerts = html.P("No alerts yet.", style={"color": "#718096"})

    # ---- flow rate ----
    now = time.time()
    if state["last_tick_time"] is not None:
        elapsed = now - state["last_tick_time"]
        state["flow_rate"] = round(1.0 / elapsed, 1) if elapsed > 0 else 0.0
    state["last_tick_time"] = now

    rate = state.get("flow_rate", 0.0)
    status = f"Processing flows... {rate} flows/sec" if state["running"] else "Paused"

    return state, cards, table, bar_fig, line_fig, alerts, status


if __name__ == "__main__":
    app.run(debug=False, port=8050)