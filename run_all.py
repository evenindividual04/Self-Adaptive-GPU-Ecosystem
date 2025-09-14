# run_all.py (patched): launches nodes, control plane, monitoring API, dashboard, and alerting
import subprocess
import time
import requests
import sys
import os
import signal
from pathlib import Path
import argparse

# Cluster definitions (can scale easily)
CLUSTER_1_NODES = [8000, 8001, 8002, 8003]
CLUSTER_2_NODES = [8010, 8011, 8012, 8013]
ALL_CLUSTERS = {
    "Cluster-1": CLUSTER_1_NODES,
    "Cluster-2": CLUSTER_2_NODES,
}

CONTROL_PLANE_PORT = 9000
MONITOR_PORT = 9100

processes = []

# ---------------- helpers ----------------

def wait_for_health(url, retries=40, delay=0.5):
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(delay)
    return False

def launch(cmd, env=None, name=None):
    print(f"[RUN_ALL] Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    processes.append((name or cmd, proc))
    return proc

def launch_node(port: int):
    return launch([sys.executable, "gpu_node_service.py", "--port", str(port)], name=f"node-{port}")

def launch_all_nodes():
    print("\n[RUN_ALL] Launching GPU nodes...")
    for cluster_name, nodes in ALL_CLUSTERS.items():
        print(f"=== {cluster_name} ===")
        for port in nodes:
            proc = launch_node(port)
            ok = wait_for_health(f"http://127.0.0.1:{port}/health", retries=40, delay=0.25)
            status = "READY" if ok else "FAILED"
            print(f"  - port {port}: {status}")
            time.sleep(0.2)

def launch_control_plane():
    print(f"\n[RUN_ALL] Launching Control Plane on port {CONTROL_PLANE_PORT} ...")
    # control_plane.py runs uvicorn in __main__
    proc = launch([sys.executable, "control_plane.py"], name="control-plane")
    ok = wait_for_health(f"http://127.0.0.1:{CONTROL_PLANE_PORT}/health", retries=60, delay=0.5)
    print("  -> Control Plane:", "READY" if ok else "WARN")

def launch_monitoring_api(roll_mode: str, max_dash_records: int, poll_sec: int):
    print(f"\n[RUN_ALL] Launching Monitoring API on port {MONITOR_PORT} ...")
    env = os.environ.copy()
    env["ROLL_MODE"] = roll_mode
    env["MAX_DASH_RECORDS"] = str(max_dash_records)
    env["POLL_SEC"] = str(poll_sec)
    env["AUTO_REMEDIATE"] = env.get("AUTO_REMEDIATE", "1")   # NEW
    env["COOL_HYST"] = env.get("COOL_HYST", "5.0")           # NEW
    # uvicorn monitoring_service:app
    proc = launch([sys.executable, "-m", "uvicorn", "monitoring_service:app", "--host", "127.0.0.1", "--port", str(MONITOR_PORT)], env=env, name="monitor-api")
    # Probe cluster-stats endpoint
    ok = wait_for_health(f"http://127.0.0.1:{MONITOR_PORT}/cluster-stats", retries=60, delay=0.5)
    print("  -> Monitoring API:", "READY" if ok else "WARN")

def launch_submitter(rate:int=5, size_min:float=0.1, size_max:float=0.4, burst_prob:float=0.3, duration:int=0, max_jobs:int=0):
    print("\n[RUN_ALL] Launching steady submitter ...")
    cmd = [
        sys.executable, "submit_jobs.py",
        "--rate", str(rate),
        "--size-min", str(size_min),
        "--size-max", str(size_max),
        "--burst-prob", str(burst_prob),
        "--duration", str(duration),
        "--max-jobs", str(max_jobs),
    ]
    return launch(cmd, name="submitter")
def launch_dashboard():
    print("\n[RUN_ALL] Launching Streamlit Dashboard ...")
    # Prefer python -m streamlit for portability
    proc = launch([sys.executable, "-m", "streamlit", "run", "telemetry_dashboard.py"], name="dashboard")
    # no health endpoint; give it a short spin-up time
    time.sleep(2)

def launch_alerting():
    print("\n[RUN_ALL] Launching Alerting Service ...")
    proc = launch([sys.executable, "alerting_service.py"], name="alerting")
    time.sleep(0.5)
    
def launch_submitter(rate=5, size_min=0.1, size_max=0.4, burst_prob=0.3, duration=0, max_jobs=0):
    print("\n[RUN_ALL] Launching steady submitter ...")
    cmd = [
        sys.executable, "submit_jobs.py",
        "--rate", str(rate),
        "--size-min", str(size_min),
        "--size-max", str(size_max),
        "--burst-prob", str(burst_prob),
        "--duration", str(duration),
        "--max-jobs", str(max_jobs)
    ]
    return launch(cmd, name="submitter")


def shutdown_all():
    print("\n[RUN_ALL] Shutting down all services...")
    # Send terminate to all in reverse order
    for name, proc in reversed(processes):
        try:
            if proc.poll() is None:
                print(f"  - terminating {name}")
                proc.terminate()
        except Exception:
            pass
    # Grace period
    time.sleep(2)
    # Kill leftovers
    for name, proc in reversed(processes):
        try:
            if proc.poll() is None:
                print(f"  - killing {name}")
                proc.kill()
        except Exception:
            pass
    print("[RUN_ALL] Shutdown complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll", default=os.getenv("ROLL_MODE", "hour"), choices=["hour", "day"], help="Log rotation window")
    parser.add_argument("--max-dash-records", type=int, default=int(os.getenv("MAX_DASH_RECORDS", "5000")))
    parser.add_argument("--poll-sec", type=int, default=int(os.getenv("POLL_SEC", "5")))
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-alerting", action="store_true")
    parser.add_argument("--no-submit", action="store_true", help="Skip job submitter")
    # Submitter knobs
    parser.add_argument("--submit-rate", type=int, default=5)
    parser.add_argument("--submit-size-min", type=float, default=0.1)
    parser.add_argument("--submit-size-max", type=float, default=0.4)
    parser.add_argument("--submit-burst-prob", type=float, default=0.3)
    parser.add_argument("--submit-duration", type=int, default=0)
    parser.add_argument("--submit-max-jobs", type=int, default=0)
    args = parser.parse_args()

    try:
        launch_all_nodes()
        launch_control_plane()
        launch_monitoring_api(args.roll, args.max_dash_records, args.poll_sec)
        if not args.no_dashboard:
            launch_dashboard()
        if not args.no_alerting:
            # Only start if file exists
            if Path("alerting_service.py").exists():
                launch_alerting()
        if not args.no_submit:
            launch_submitter(
            rate=args.submit_rate,
            size_min=args.submit_size_min,
            size_max=args.submit_size_max,
            burst_prob=args.submit_burst_prob,
            duration=args.submit_duration,
            max_jobs=args.submit_max_jobs,
        )
        print("\n[RUN_ALL] All services launched. Press Ctrl+C to stop.\n")
        # Keep the parent alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_all()

if __name__ == "__main__":
    main()
