# monitoring_service.py (rolled logs + anomaly reasons + tiny API + SSE)
"""
Enhancements:
- Log rotation: writes hourly/day files under data_logs/ and keeps a trimmed 'training_data_2.json' for the dashboard.
- Persists anomaly_reasons and uses prev_sample per node for spike detection.
- Exposes a tiny read-only API: GET /cluster-stats, GET /latest.
- Optional SSE stream at GET /events providing near-real-time updates.

Run:  uvicorn monitoring_service:app --host 127.0.0.1 --port 9100
(Or python monitoring_service.py for a simple runner.)
"""
import os, json, time, threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gnn_model import EnhancedGNNDetector  # Adjust to your model file
import joblib
import numpy as np
from typing import List
import networkx as nx  # For topology features
from nx_topology import compute_nx_features, nx_to_edge_index  # Import from your nx_topology.py
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------- config ----------------
ALL_NODE_PORTS = [8000, 8001, 8002, 8003, 8010, 8011, 8012, 8013]
DATA_DIR = Path("data_logs")
DATA_DIR.mkdir(exist_ok=True)
DASH_FILE = Path("training_data_2.json")  # dashboard reads this
MAX_DASH_RECORDS = int(os.getenv("MAX_DASH_RECORDS", "100000"))
POLL_INTERVAL = int(os.getenv("POLL_SEC", "15"))
ROLL_MODE = os.getenv("ROLL_MODE", "hour")  # 'hour' or 'day'

TEMP_THRESHOLD = 85.0
COOL_HYST = float(os.getenv("COOL_HYST", "5.0"))  # NEW: undrain when below (TEMP_THRESHOLD - COOL_HYST)
AUTO_REMEDIATE = os.getenv("AUTO_REMEDIATE", "1") not in ("0","false","False")  # NEW
POWER_THRESHOLD = 250.0
MEMORY_THRESHOLD = 0.95
FAN_MAX = 5000
UTIL_SPIKE_DELTA = 0.3

# --------------- runtime state ---------------
prev_samples: Dict[int, Dict[str, Any]] = {}
last_cluster_stats: Dict[str, Any] = {"total_anomalies": 0, "avg_util": 0.0, "nodes_online": 0, "ts": time.time()}
last_samples_by_port: Dict[int, Dict[str, Any]] = {}

app = FastAPI(title="SAGE Monitor API")


# Load GNN model, scaler, threshold
cfg = {"model": {"input_dim": 9, "hidden_dim": 128, "output_dim": 2, "dropout": 0.3, "heads": 4, "dropedge_p": 0.10}}
gnn_model = EnhancedGNNDetector(cfg)
gnn_model.load_state_dict(torch.load("deploy/best_model.pth", map_location="cpu", weights_only=False))
gnn_model.eval()
scaler = joblib.load("deploy/scaler.pkl")
with open("deploy/threshold.json") as f:
    threshold_data = json.load(f)
GNN_THRESHOLD = threshold_data["threshold"]


# --------------- helpers ---------------
def roll_path(now: float) -> Path:
    dt = datetime.fromtimestamp(now)
    if ROLL_MODE == "day":
        fn = dt.strftime("%Y%m%d.json")
    else:
        fn = dt.strftime("%Y%m%d_%H.json")
    return DATA_DIR / fn

def load_dashboard() -> list:
    if DASH_FILE.exists():
        try:
            return json.loads(DASH_FILE.read_text())
        except json.JSONDecodeError:
            return []
    return []

def save_dashboard_append(sample: dict):
    data = load_dashboard()
    data.append(sample)
    # trim
    if len(data) > MAX_DASH_RECORDS:
        data = data[-MAX_DASH_RECORDS:]
    DASH_FILE.write_text(json.dumps(data, indent=2))

def append_rolled(sample: dict):
    p = roll_path(sample.get("timestamp", time.time()))
    if p.exists():
        try:
            arr = json.loads(p.read_text())
        except json.JSONDecodeError:
            arr = []
    else:
        arr = []
    arr.append(sample)
    p.write_text(json.dumps(arr, indent=2))

def fetch_metrics(port: int):
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
        if r.status_code != 200:
            return None
        r = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=3)
        if r.status_code == 200:
            d = r.json()
            d["timestamp"] = time.time()
            d["node_port"] = port
            return d
    except requests.RequestException:
        return None
    return None

# --- NEW: remediation helpers ---
def _fan_boost(port: int, speed: float = 4500.0):
    try:
        requests.post(f"http://127.0.0.1:{port}/fan", json={"speed": speed}, timeout=2)
    except requests.RequestException:
        pass

def _fan_normalize(port: int, base: float = 2000.0):
    try:
        requests.post(f"http://127.0.0.1:{port}/fan", json={"speed": base}, timeout=2)
    except requests.RequestException:
        pass

def _drain_node(port: int):
    try:
        requests.put(f"http://127.0.0.1:9000/nodes/{port}/drain", timeout=2)
    except requests.RequestException:
        pass

def _undrain_node(port: int):
    try:
        requests.put(f"http://127.0.0.1:9000/nodes/{port}/undrain", timeout=2)
    except requests.RequestException:
        pass

def build_graph(sample: dict) -> Data:
    # Telemetry features (5)
    telemetry_features = ["utilization", "memory_usage", "power_usage", "temperature", "fan_speed"]
    X_telemetry = np.array([[sample.get(f, 0.0) for f in telemetry_features]])  # Shape (1,5)

    # For single-node graph: create NX graph with 1 node
    G = nx.Graph()
    G.add_node(0)  # Single node

    # Compute NX features (will be zeros for single node: degree=0, etc.)
    nx_features = compute_nx_features(G)  # Shape (1,4)

    # Concatenate: telemetry + NX = 9 features
    X = np.concatenate([X_telemetry, nx_features], axis=1)  # Shape (1,9)

    # Scale
    X_scaled = scaler.transform(X)  # Matches expected (1,9)

    x = torch.tensor(X_scaled, dtype=torch.float32).squeeze(0)  # Shape (9,)

    # Edges: none for single node
    edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)



def detect_anomaly(sample: dict, prev_sample: dict | None):
    reasons = []
    if sample.get("temperature", 0) > TEMP_THRESHOLD:
        reasons.append("High temperature")
    if sample.get("power_usage", 0) > POWER_THRESHOLD:
        reasons.append("High power usage")
    if sample.get("memory_usage", 0) > MEMORY_THRESHOLD:
        reasons.append("High memory usage")
    if sample.get("fan_speed", 0) > FAN_MAX:
        reasons.append("High fan speed")
    if prev_sample:
        du = sample.get("utilization", 0) - prev_sample.get("utilization", 0)
        if du > UTIL_SPIKE_DELTA:
            reasons.append("Utilization spike")
            
    # GNN-based detection
    graph = build_graph([sample])  # Single-node graph; extend for multi-node if needed
    loader = DataLoader([graph], batch_size=1)
    with torch.no_grad():
        batch = next(iter(loader))
        logits = gnn_model(batch.x, batch.edge_index, batch.batch)
        prob = torch.softmax(logits, dim=1)[:, 1].item()

    is_anomalous = prob >= GNN_THRESHOLD
    if is_anomalous:
        reasons.append(f"GNN anomaly prob: {prob:.3f}")

    return is_anomalous or len(reasons) > 0, reasons


def build_graph(samples: List[Dict]) -> Data:
    # Telemetry features (5) - adjust order if needed
    telemetry_features = ["utilization", "temperature", "power_usage", "memory_usage", "fan_speed"]
    X_telemetry = np.array([[s.get(f, 0.0) for f in telemetry_features] for s in samples])  # Shape (N,5)

    # For single-node (N=1), create NX graph and compute features (defaults to zeros)
    num_nodes = len(samples)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))  # Add nodes
    # Add minimal edges if needed (e.g., self-loop or none for single node)
    if num_nodes == 1:
        # No edges for single node
        pass
    else:
        # Simple ring for multi-node (as in your original)
        for i in range(num_nodes):
            G.add_edge(i, (i + 1) % num_nodes)

    # Compute NX features (shape (N,4): degree, betweenness, eigenvector, clustering)
    nx_features = compute_nx_features(G)  # From nx_topology.py

    # Concatenate: telemetry + NX = 9 features, shape (N,9)
    X = np.concatenate([X_telemetry, nx_features], axis=1)

    # Scale
    X_scaled = scaler.transform(X)

    x = torch.tensor(X_scaled, dtype=torch.float32)

    # Edges: from NX graph (empty for single node)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])  # Bidirectional
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)



# --------------- polling loop ---------------
def poll_loop():
    global last_cluster_stats, last_samples_by_port
    print(f"[Monitoring] Started at {datetime.now().isoformat()} | interval={POLL_INTERVAL}s | roll={ROLL_MODE}")
    while True:
        try:
            stats = {"total_anomalies": 0, "avg_util": 0.0, "nodes_online": 0}
            for port in ALL_NODE_PORTS:
                sample = fetch_metrics(port)
                if sample:
                    prev = prev_samples.get(port)
                    is_anom, reasons = detect_anomaly(sample, prev)
                    sample["is_anomalous"] = is_anom
                    sample["anomaly_reasons"] = reasons
                    prev_samples[port] = sample
                    last_samples_by_port[port] = sample

                    stats["total_anomalies"] += 1 if is_anom else 0
                    stats["avg_util"] += sample.get("utilization", 0.0)
                    stats["nodes_online"] += 1

                    # NEW: auto-remediation for thermal events
                    if AUTO_REMEDIATE:
                        temp = float(sample.get("temperature", 0.0))
                        if temp >= TEMP_THRESHOLD:
                            # boost fan and drain node
                            _fan_boost(port, speed=4500.0)
                            _drain_node(port)
                            sample["remediation"] = ["fan_boost","drain"]
                        elif temp <= (TEMP_THRESHOLD - COOL_HYST):
                            # normalize fan and undrain node
                            _fan_normalize(port, base=2000.0)
                            _undrain_node(port)
                            sample["remediation"] = ["fan_normalize","undrain"]

                    # persist: rolled + dashboard
                    append_rolled(sample)
                    save_dashboard_append(sample)

            if stats["nodes_online"] > 0:
                stats["avg_util"] /= stats["nodes_online"]
            stats["ts"] = time.time()
            last_cluster_stats = stats
            print(f"[Cluster] nodes={stats['nodes_online']} avg_util={stats['avg_util']:.2f} anomalies={stats['total_anomalies']}")
        except Exception as e:
            print("[Monitoring] error:", e)
        time.sleep(POLL_INTERVAL)

# start background poller
threading.Thread(target=poll_loop, daemon=True).start()

# --------------- tiny API ---------------
@app.get("/cluster-stats")
def cluster_stats():
    return JSONResponse(last_cluster_stats)

@app.get("/latest")
def latest():
    # return last sample per port
    return JSONResponse(list(last_samples_by_port.values()))

@app.get("/events")
def events():
    # SSE: text/event-stream
    def gen():
        last_sent = 0.0
        while True:
            # send only when cluster ts changes
            ts = last_cluster_stats.get("ts", 0.0)
            if ts > last_sent:
                payload = json.dumps({
                    "cluster": last_cluster_stats,
                    "latest": list(last_samples_by_port.values()),
                })
                yield f" {payload}\n\n"
                last_sent = ts
            time.sleep(1)
    return StreamingResponse(gen(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("monitoring_service:app", host="127.0.0.1", port=9100, reload=False)
