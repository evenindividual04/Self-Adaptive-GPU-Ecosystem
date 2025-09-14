# telemetry_dashboard.py (control UI + jobs + XAI, on top of robust base)

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt

import plotly.graph_objects as go

clusters = {
    'Cluster 1': [8000, 8001, 8002, 8003],  # Example ports for Cluster 1
    'Cluster 2': [8010, 8011, 8012, 8013] 
}

    
def plotly_clustered_topology(clusters, status_by_port):
    G = nx.Graph()
    cluster_offsets = {'Cluster 1': -1, 'Cluster 2': 1}  # Horizontal separation
    

    # Add nodes and connect within clusters
    for cluster, nodes in clusters.items():
        for node in nodes:
            G.add_node(node, cluster=cluster)
        # Chain connections within cluster (adjust if needed)
        for i in range(len(nodes) - 1):
            G.add_edge(nodes[i], nodes[i+1])

    # Layout for each cluster, shifted horizontally
    pos = {}
    for cluster, nodes in clusters.items():
        subgraph = G.subgraph(nodes)
        layout = nx.spring_layout(subgraph, seed=42)
        for node, (x, y) in layout.items():
            # Separate clusters along x
            offset = cluster_offsets.get(cluster, 0)
            pos[node] = (x + offset, y)

    # Edge plotting
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#888'), hoverinfo='none')

    # Node plotting
    node_x, node_y, node_color, node_text = [], [], [], []
    icons_map = {'temperature': '🔥', 'power': '⚡', 'memory': '💾', 'fan': '🌬️', 'utilization': '📈'}

    for node in G.nodes():
        x, y = pos[node]
        status = status_by_port.get(node, {})
        anomalous = status.get('is_anomalous', False)
        node_color.append('red' if anomalous else 'green')
        reasons = status.get('anomaly_reasons', [])
        icons = ''.join([icons_map.get(r.split()[1].lower(), '') for r in reasons if r])
        cluster_name = G.nodes[node]['cluster']
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Node: {node}<br>Cluster: {cluster_name}<br>Status: {'Anomalous' if anomalous else 'Normal'}<br>Reasons: {', '.join(reasons) or 'None'} {icons}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node) for node in G.nodes()],
        textposition='bottom center',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(color=node_color, size=20, line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Clustered Topology with Anomalies',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    st.plotly_chart(fig, use_container_width=True)


# --------------------
# Config & State
# --------------------
st.set_page_config(page_title="SAGE Telemetry Dashboard", layout="wide")
st.title("SAGE Telemetry Dashboard")

DATA_PATH = Path("training_data_2.json")
CONTROL_URL = "http://127.0.0.1:9000"
MONITOR_URL = "http://127.0.0.1:9100"

# Keep last file mtime in session for lightweight change detection
if "_last_mtime" not in st.session_state:
    st.session_state._last_mtime = 0.0
if "_last_refresh" not in st.session_state:
    st.session_state._last_refresh = time.time()

# --------------------
# Sidebar controls
# --------------------
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_sec = st.sidebar.slider("Refresh interval (s)", 5, 60, 10)
window_label = st.sidebar.selectbox("View window", ["All", "Last 50", "Last 200", "Last 1000"], index=1)
use_api = st.sidebar.checkbox("Use APIs for topology & jobs", value=True)

# Manual refresh
if st.sidebar.button("Refresh now"):
    st.rerun()  # Official rerun API

# --------------------
# Safe JSON load (file mode for time-series)
# --------------------
def load_json_safely(path: Path, retries: int = 3, delay: float = 0.2):
    for _ in range(retries):
        try:
            with path.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            time.sleep(delay)
    with path.open("r") as f:
        return json.load(f)

if not DATA_PATH.exists():
    st.warning("training_data_2.json not found. Start monitoring_service.py or the monitor API first.")
    st.stop()

try:
    data = load_json_safely(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load telemetry: {e}")
    st.stop()

# Window parsing (robust)
label_to_n = {"All": None, "Last 50": 50, "Last 200": 200, "Last 1000": 1000}
if window_label in label_to_n:
    n = label_to_n[window_label]
else:
    digits = "".join(filter(str.isdigit, window_label))
    n = int(digits) if digits else None

if n is not None and len(data) > 0:
    n = min(n, len(data))
    data = data[-n:]

if len(data) == 0:
    st.info("No telemetry records yet in the selected window.")
    st.stop()

# --------------------
# DataFrame prep
# --------------------
df = pd.DataFrame(data)

expected_cols = [
    "timestamp", "node_port", "utilization", "temperature",
    "power_usage", "memory_usage", "fan_speed", "is_anomalous", "anomaly_reasons"
]
for c in expected_cols:
    if c not in df.columns:
        df[c] = None

if df["timestamp"].notna().any():
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
else:
    df["ts"] = pd.Timestamp.now()

for c in ["utilization", "temperature", "power_usage", "memory_usage", "fan_speed"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --------------------
# KPIs
# --------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", len(df))
col2.metric("Online Nodes (seen)", int(df["node_port"].nunique() if "node_port" in df else 0))
avg_util = float(df["utilization"].mean()) if df["utilization"].notna().any() else 0.0
col3.metric("Avg Utilization", f"{avg_util:.2f}")
col4.metric("Anomalies (window)", int(df["is_anomalous"].fillna(False).sum()))

# --------------------
# Tabs: Util/Temp charts; Topology (controls); Job History; Explainability
# --------------------
tab_main, tab_topo, tab_jobs, tab_xai, tab_anomalies = st.tabs(["Charts", "Topology", "Job History", "Explainability", "Recent Anomalies"])

def plot_metric_with_anomalies(df, metric, y_label, title):
    fig = go.Figure()
    # Main metric line
    fig.add_trace(go.Scatter(
        x=df['ts'], y=df[metric],
        mode='lines+markers', name=metric.capitalize(),
        line=dict(color='royalblue')
    ))
    # Overlay anomalies
    if 'is_anomalous' in df.columns:
        anomalies = df[df['is_anomalous'] == True]
        fig.add_trace(go.Scatter(
            x=anomalies['ts'], y=anomalies[metric],
            mode='markers', name='Anomaly',
            marker=dict(color='red', size=10, symbol='x')
        ))
    # Layout with labels/title
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=y_label,
        legend_title='Legend',
        hovermode='x unified',
        template='plotly_dark',
        xaxis=dict(tickformat='%H:%M')  # Format time as HH:MM
    )
    return fig

def plot_anomaly_timeline(df):
    # Aggregate anomaly counts
    anomaly_counts = df[df['is_anomalous'] == True].groupby('ts').size().reset_index(name='count')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=anomaly_counts['ts'], y=anomaly_counts['count'],
        name='Anomaly Count', marker_color='red'
    ))
    fig.update_layout(
        title='Anomaly Timeline',
        xaxis_title='Time',
        yaxis_title='Anomaly Count',
        hovermode='x unified',
        template='plotly_dark',
        xaxis=dict(tickformat='%H:%M')
    )
    return fig




with tab_main:
    c1, c2 = st.columns(2)
    
    with c1:
        plot_df = df.dropna(subset=['ts', 'utilization'])
        if plot_df.empty:
            st.info("No utilization data available.")
        else:
            fig_util = plot_metric_with_anomalies(plot_df, 'utilization', 'Utilization (0-1)', 'GPU Utilization Over Time')
            st.plotly_chart(fig_util, use_container_width=True)
    
    with c2:
        plot_df_temp = df.dropna(subset=['ts', 'temperature'])
        if plot_df_temp.empty:
            st.info("No temperature data available.")
        else:
            fig_temp = plot_metric_with_anomalies(plot_df_temp, 'temperature', 'Temperature (°C)', 'GPU Temperature Over Time')
            st.plotly_chart(fig_temp, use_container_width=True)
    
    # Anomaly timeline (below the two charts)
    if not df.empty:
        fig_anom = plot_anomaly_timeline(df)
        st.plotly_chart(fig_anom, use_container_width=True)
    else:
        st.info("No anomaly data available.")


with tab_topo:
    try:
        if use_api:
            nodes_info = requests.get(f"{CONTROL_URL}/nodes", timeout=3).json()
            ports = nodes_info.get("ports", [])  # Define ports here
        else:
            ports = sorted(df["node_port"].dropna().astype(int).unique().tolist())  # Fallback to df

        latest = requests.get(f"{MONITOR_URL}/latest", timeout=3).json() if use_api \
            else df.groupby("node_port").tail(1).to_dict(orient="records")

        status_by_port = {}
        for x in latest:
            port = int(x.get("node_port") or x.get("port"))
            status_by_port[port] = x

        #G = nx.Graph()
        # Group ports into clusters for visualization (example: split in half)
        #mid = len(ports) // 2
        #clusters = {
         #   "Cluster 1": ports[:mid],
          #  "Cluster 2": ports[mid:]
        #}
        plotly_clustered_topology(clusters, status_by_port)  # Use the defined function
        
        # In plotly_clustered_topology function (after G = nx.Graph())
        
        
        
        temp_anom_nodes = [p for p in sum(clusters.values(), []) if any("temperature" in r.lower() for r in status_by_port.get(p, {}).get("anomaly_reasons", []))]
        if temp_anom_nodes:
            st.warning(f"Temperature anomaly on nodes: {temp_anom_nodes}. Consider boosting fan.")
            sel_node = st.selectbox("Select node to boost fan", temp_anom_nodes)
            if st.button(f"Boost Fan on {sel_node}"):
                if st.checkbox(f"Confirm boost on {sel_node}?"):
                    try:
                        requests.post(f"http://127.0.0.1:{sel_node}/fan", json={"speed": 4500}, timeout=3)
                        st.success(f"Fan boosted on {sel_node}.")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                else:
                    st.info("Action cancelled.")

        
        if ports:
            sel = st.selectbox("Select node", ports)
            c1, c2, c3 = st.columns(3)
            if c1.button("Drain", key=f"drain-{sel}"):
                requests.put(f"{CONTROL_URL}/nodes/{sel}/drain", timeout=3)
                st.rerun()
            if c2.button("Undrain", key=f"undrain-{sel}"):
                requests.put(f"{CONTROL_URL}/nodes/{sel}/undrain", timeout=3)
                st.rerun()
            if c3.button("Reset Node", key=f"reset-{sel}"):
                requests.post(f"http://127.0.0.1:{sel}/reset", timeout=3)
                st.rerun()

            c4, c5 = st.columns(2)
            if c4.button("Pause", key=f"pause-{sel}"):
                requests.post(f"http://127.0.0.1:{sel}/pause", timeout=3)
            if c5.button("Resume", key=f"resume-{sel}"):
                requests.post(f"http://127.0.0.1:{sel}/resume", timeout=3)

            c6, c7 = st.columns(2)
            if c6.button("Boost Fan", key=f"fan-boost-{sel}"):
                requests.post(f"http://127.0.0.1:{sel}/fan", json={"speed": 4500}, timeout=3)
            if c7.button("Normalize Fan", key=f"fan-norm-{sel}"):
                requests.post(f"http://127.0.0.1:{sel}/fan", json={"speed": 2000}, timeout=3)

            # NEW: show completed jobs...
            st.markdown("#### Recent jobs (completed)")
            sj = status_by_port.get(int(sel), {})
            rj = sj.get("recent_jobs", [])
            if rj:
                rj_df = pd.DataFrame(rj)
                if "start" in rj_df.columns:
                    rj_df["start"] = pd.to_datetime(rj_df["start"], unit="s", errors="coerce")
                if "end" in rj_df.columns:
                    rj_df["end"] = pd.to_datetime(rj_df["end"], unit="s", errors="coerce")
                st.dataframe(rj_df.sort_values("end", ascending=False), use_container_width=True, height=200)
            else:
                st.caption("No recent jobs reported by node.")

    except Exception as e:
        st.warning(f"Topology not available: {e}")


with tab_jobs:
    try:
        jobs = requests.get(f"{CONTROL_URL}/jobs?limit=500", timeout=3).json() if use_api else {"jobs": []}
        jdf = pd.DataFrame(jobs.get("jobs", []))
        if not jdf.empty:
            jdf["time"] = pd.to_datetime(jdf["time"], unit="s")
            node_filter = st.selectbox("Filter by node", ["All"] + sorted(df['node_port'].unique()))
            if node_filter != "All":
                jdf = jdf[jdf['node'] == node_filter]
            st.dataframe(jdf.sort_values("time", ascending=False), use_container_width=True)
            st.line_chart(jdf.set_index("time")["size"])
        else:
            st.info("No jobs yet.")
    except Exception as e:
        st.warning(f"Jobs API unavailable: {e}")

with tab_xai:
    st.caption("Model-based explanation of anomalies using SHAP over a RandomForest trained on your labeled telemetry (is_anomalous).")
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        import matplotlib.pyplot as plt
        import numpy as np

        features = ["utilization", "temperature", "power_usage", "memory_usage", "fan_speed"]
        use_df = df.dropna(subset=features + ["is_anomalous"]).copy()

        if use_df["is_anomalous"].nunique() < 2 or len(use_df) < 50:
            st.info("Not enough labeled variety yet to train an explainer. Collect more data or anomalies.")
        else:
            X = use_df[features]
            y = use_df["is_anomalous"].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            st.write(f"F1 (holdout): {f1_score(y_test, pred):.3f}")

            # Unified API: Explanation object
            explainer = shap.Explainer(clf, X_train, feature_names=features)  # primary SHAP interface [shap.Explainer] [7]
            ex = explainer(X_test)                                            # Explanation with .values/.base_values [13]

            vals = np.asarray(ex.values)   # (N,F) or (N,F,C) or (N,C,F)
            base = np.asarray(ex.base_values)

            # Reduce to 2D by selecting a class when needed (prefer positive class 1)
            def to_2d(values, base_values, prefer_cls=1):
                v = np.asarray(values)
                b = np.asarray(base_values)
                if v.ndim == 3:
                    # Try last axis as classes first (N,F,C); else (N,C,F)
                    if b.ndim == 2 and v.shape[-1] == b.shape[-1]:
                        ci = prefer_cls if v.shape[-1] > 1 else 0
                        v2 = v[..., ci]          # (N,F)
                        b2 = b[:, ci] if b.ndim == 2 else b
                    else:
                        ci = prefer_cls if v.shape[1] > 1 else 0
                        v2 = v[:, ci, :]         # (N,F)
                        b2 = b[:, ci] if b.ndim == 2 else b
                    return v2, b2
                return v, b

            sv2d, base_vec = to_2d(vals, base, prefer_cls=1)
            st.caption(f"SHAP shapes — sv: {sv2d.shape}, X_test: {X_test.shape}")

            # Global importances (static): shap.plots.bar on Explanation subset
            # Build a new Explanation with 2D values so bar plot sees a consistent object
            ex2d = shap.Explanation(values=sv2d, base_values=base_vec, data=X_test.values, feature_names=features)  # Explanation wrapper [13]
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            shap.plots.bar(ex2d, max_display=len(features), show=False, ax=ax_bar)  # Matplotlib Axes returned [3]
            st.pyplot(fig_bar)

            # Optional beeswarm summary (2D)
            fig_sw, _ = plt.subplots(figsize=(6, 4))
            shap.summary_plot(sv2d, X_test, feature_names=features, show=False)     # expects (N,F) + X matching shape [2]
            st.pyplot(fig_sw)
            
            st.markdown("""
            Explanation for the Latest Anomaly

            This section focuses on the most recent unusual event (anomaly) detected in the system. It breaks down how key factors—like how busy the system is (utilization), its temperature, power use, memory load, and fan speed—contributed to why the system flagged it as a problem. 

            - **Positive contributions** (in red) push the prediction toward "anomaly," meaning they made the event seem more unusual.
            - **Negative contributions** (in blue) pull it toward "normal," suggesting those factors were okay.
            - The "base value" is the average expected outcome, and the final "output value" shows the model's overall decision.

            This helps you quickly understand *why* the anomaly happened and what to check first, making troubleshooting easier without needing deep technical knowledge.
            """)

            # Local explanation code with checks
            #recent_anom = use_df[use_df["is_anomalous"] == 1].tail(1)[features]
            #if recent_anom.empty:
             #   st.info("No recent anomalies detected. Check back after more data is collected or induce one via node controls.")
            #else:
             #   try:
              #      ex1 = explainer(recent_anom)
               #     v1 = np.asarray(ex1.values)
                #    b1 = np.asarray(ex1.base_values)
                 #   v1_2d, b1_vec = to_2d(v1, b1, prefer_cls=1)
                  #  v1_vec = np.ravel(v1_2d)[-v1_2d.shape[-1]:]  # (F,)
                   # b1_scalar = float(np.ravel(b1_vec)) if np.size(b1_vec) else 0.0
            #        fig_force, _ = plt.subplots(figsize=(6, 2.8))
             #       shap.force_plot(b1_scalar, v1_vec, recent_anom.iloc[0, :], matplotlib=True, show=False)
              #      st.pyplot(fig_force)
               # except Exception as e:
                #    st.warning(f"Could not generate explanation: {e}. Ensure SHAP and Matplotlib are installed correctly.")
            
            
            # Local explanation for latest anomaly
            #recent_anom = use_df[use_df["is_anomalous"] == 1].tail(1)[features]
            #if not recent_anom.empty:
             #   ex1 = explainer(recent_anom)
              #  v1 = np.asarray(ex1.values)
               # b1 = np.asarray(ex1.base_values)
              #  v1_2d, b1_vec = to_2d(v1, b1, prefer_cls=1)
               # v1_vec = np.ravel(v1_2d)[-v1_2d.shape[-1]:]        # (F,)
              #  b1_scalar = float(np.ravel(b1_vec)) if np.size(b1_vec) else 0.0
               # fig_force, _ = plt.subplots(figsize=(6, 2.8))
                #shap.force_plot(b1_scalar, v1_vec, recent_anom.iloc[0, :], matplotlib=True, show=False)  # local plot [15]
                #t.pyplot(fig_force)
                
    except Exception as e:
        st.exception(e)
        st.warning("XAI not available. Ensure 'shap' and 'scikit-learn' are installed in the same Python environment.")
        

with tab_anomalies:
    # Filter for anomalies
    anomalies_df = df[df["is_anomalous"] == True].copy()
    if anomalies_df.empty:
        st.info("No recent anomalies detected yet.")
    else:
        # Format timestamp to readable datetime
        anomalies_df["datetime"] = pd.to_datetime(anomalies_df["timestamp"], unit="s", errors="coerce")
        
        # Select key columns
        columns = ["datetime", "node_port", "utilization", "temperature", "power_usage", "memory_usage", "fan_speed", "anomaly_reasons"]
        # Use only available columns
        available_columns = [col for col in columns if col in anomalies_df.columns]
        table_df = anomalies_df[available_columns].sort_values(by="datetime", ascending=False).head(20)  # Last 20 for performance
        
        # Display as interactive table
        st.dataframe(table_df, use_container_width=True, height=400)  # Adjustable height



# --------------------
# Smart auto-refresh (file-change or time-based)
# --------------------
try:
    mtime = DATA_PATH.stat().st_mtime
except Exception:
    mtime = 0.0

now = time.time()
should_refresh = False
if auto_refresh:
    if mtime > st.session_state._last_mtime:
        should_refresh = True
    elif now - st.session_state._last_refresh >= refresh_sec:
        should_refresh = True

if should_refresh:
    st.session_state._last_mtime = mtime
    st.session_state._last_refresh = now
    st.rerun()
