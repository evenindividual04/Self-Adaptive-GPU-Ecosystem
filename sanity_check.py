#!/usr/bin/env python3
# sanity_check.py — safe loader + summaries

import json
import time
import sys
from pathlib import Path
from collections import defaultdict
import statistics

DEFAULT_FILE = "training_data_2.json"

def load_json_safely(path: Path, retries: int = 5, delay: float = 0.2, default=None):
    """
    Robust JSON reader:
    - Returns `default` (e.g., []) if file is missing or empty.
    - Retries on JSONDecodeError to avoid mid-write reads.
    """
    default = [] if default is None else default
    for _ in range(retries):
        try:
            if not path.exists():
                return default
            s = path.read_text().strip()
            if not s:
                return default
            return json.loads(s)
        except json.JSONDecodeError:
            time.sleep(delay)
        except OSError:
            time.sleep(delay)
    # Final best-effort attempt; still return default on failure
    try:
        s = path.read_text().strip()
        return json.loads(s) if s else default
    except Exception:
        return default

def main():
    fname = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_FILE)
    data = load_json_safely(fname)  # safe even if file is empty or mid-write

    total_entries = len(data)
    print(f"Total entries: {total_entries}\n")

    # Node distribution
    node_counts = defaultdict(int)
    anomaly_counts = defaultdict(int)
    metrics = defaultdict(list)

    for entry in data:
        port = entry.get("node_port", "unknown")
        node_counts[port] += 1
        if entry.get("is_anomalous", False):
            anomaly_counts[port] += 1
        # Collect metrics
        for key in ["temperature", "power_usage", "memory_usage", "fan_speed", "utilization"]:
            if key in entry and entry[key] is not None:
                metrics[key].append(entry[key])

    print("Node port distribution:")
    for port, count in sorted(node_counts.items()):
        print(f"  Port {port}: {count}")

    print("\nAnomalous entries per node:")
    total_anomalies = sum(anomaly_counts.values())
    for port, count in sorted(anomaly_counts.items()):
        print(f"  Port {port}: {count}")
    pct = (total_anomalies / total_entries * 100.0) if total_entries else 0.0
    print(f"  Total anomalies: {total_anomalies} ({pct:.2f}%)\n")

    # Metrics summary
    print("Metrics summary (min / max / mean / stdev):")
    for key, values in metrics.items():
        if values and len(values) > 1:
            print(f"  {key}: min={min(values):.2f}, max={max(values):.2f}, mean={statistics.mean(values):.2f}, stdev={statistics.stdev(values):.2f}")
        elif values:
            # stdev undefined for n=1
            print(f"  {key}: min={min(values):.2f}, max={max(values):.2f}, mean={statistics.mean(values):.2f}, stdev=NA(n=1)")

    # Optional: check missing fields
    missing_fields = defaultdict(int)
    for entry in data:
        for key in ["utilization", "temperature", "power_usage", "memory_usage", "fan_speed", "is_anomalous"]:
            if key not in entry:
                missing_fields[key] += 1

    if missing_fields:
        print("\nMissing fields detected:")
        for key, count in missing_fields.items():
            print(f"  {key}: {count}")
    else:
        print("\nNo missing fields detected.")

if __name__ == "__main__":
    main()
