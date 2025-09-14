# monitoring_service.py

import requests
import time
import json
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import global_tracker as gt  # Assuming this is your global_tracker.py

# List of all GPU node ports (sync with launch_nodes.py)
ALL_NODE_PORTS = [8000, 8001, 8002, 8003, 8010, 8011, 8012, 8013]

# File to store training data
DATA_FILE = Path("training_data_2.json")

# Anomaly thresholds (tweak as needed)
TEMP_THRESHOLD = 85.0  # °C
POWER_THRESHOLD = 250.0  # W
MEMORY_THRESHOLD = 0.95  # ~95% full
FAN_MAX = 5000  # arbitrary upper sanity check
UTIL_SPIKE_DELTA = 0.3  # sudden 30% jump

# Track previous samples per port for delta detection
prev_samples = defaultdict(dict)

def fetch_metrics(port):
    """Fetch metrics from a node, return dict or None if failed."""
    try:
        health = requests.get(f"http://localhost:{port}/health", timeout=3)
        if health.status_code != 200:
            return None
        metrics = requests.get(f"http://localhost:{port}/metrics", timeout=3)
        if metrics.status_code == 200:
            data = metrics.json()
            data["timestamp"] = time.time()
            data["node_port"] = port
            return data
    except requests.RequestException:
        return None
    return None

def detect_anomaly(sample, prev_sample=None):
    """Rule-based anomaly detector with delta checks."""
    reasons = []
    if sample["temperature"] > TEMP_THRESHOLD:
        reasons.append("High temperature")
    if sample["power_usage"] > POWER_THRESHOLD:
        reasons.append("High power usage")
    if sample["memory_usage"] > MEMORY_THRESHOLD:
        reasons.append("High memory usage")
    if sample["fan_speed"] > FAN_MAX:
        reasons.append("High fan speed")
    if prev_sample and (sample["utilization"] - prev_sample.get("utilization", 0)) > UTIL_SPIKE_DELTA:
        reasons.append("Utilization spike")
    return len(reasons) > 0, reasons

def load_existing_data():
    """Load old JSON dataset if exists, else return empty list."""
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_data(data):
    """Save dataset to JSON file (overwrite)."""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def monitor_loop(interval=15):
    """Main monitoring loop (poll every `interval` seconds)."""
    dataset = load_existing_data()
    print(f"[Monitoring] Started at {datetime.now().isoformat()}")
    print(f"[Monitoring] Polling every {interval}s... Press Ctrl+C to stop.")
    try:
        while True:
            cluster_stats = {"total_anomalies": 0, "avg_util": 0.0, "nodes_online": 0}
            for port in ALL_NODE_PORTS:
                sample = fetch_metrics(port)
                if sample:
                    prev_sample = prev_samples.get(port)
                    is_anom, reasons = detect_anomaly(sample, prev_sample)
                    sample["is_anomalous"] = is_anom
                    sample["anomaly_reasons"] = reasons
                    prev_samples[port] = sample  # Update previous for next cycle

                    dataset.append(sample)

                    gt.record_anomaly(is_anom)  # Integrate global tracker

                    cluster_stats["total_anomalies"] += 1 if is_anom else 0
                    cluster_stats["avg_util"] += sample["utilization"]
                    cluster_stats["nodes_online"] += 1

                    status = "ANOMALY!" if is_anom else "OK"
                    print(f"[{port}] util={sample['utilization']:.2f}, temp={sample['temperature']:.1f}°C, mem={sample['memory_usage']:.2f} => {status} (Reasons: {reasons})")

            if cluster_stats["nodes_online"] > 0:
                cluster_stats["avg_util"] /= cluster_stats["nodes_online"]
                print(f"\nCluster Stats: Avg Util={cluster_stats['avg_util']:.2f}, Total Anomalies={cluster_stats['total_anomalies']}, Online Nodes={cluster_stats['nodes_online']}\n")

            save_data(dataset)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[Monitoring] Stopped by user. Data saved.")

if __name__ == "__main__":
    monitor_loop(interval=20)  # Adjust polling interval as needed
