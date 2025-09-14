import subprocess
import time
import requests
import sys

# Cluster definitions (can scale easily)
CLUSTER_1_NODES = [8000, 8001, 8002, 8003]
CLUSTER_2_NODES = [8010, 8011, 8012, 8013]
ALL_CLUSTERS = {
    "Cluster-1": CLUSTER_1_NODES,
    "Cluster-2": CLUSTER_2_NODES,
}

# Node process handles
processes = []

def launch_node(port: int):
    """Launch a GPU node service on given port and return process handle."""
    process = subprocess.Popen(
        [sys.executable, "gpu_node_service.py", "--port", str(port)],
        stdout=sys.stdout,    # show logs in this terminal
        stderr=sys.stderr,
    )
    return process

def check_health(port: int, retries: int = 15, delay: float = 2.0) -> bool:
    """Check if a node is alive, with retries and delay between attempts."""
    url = f"http://127.0.0.1:{port}/health"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(delay)
    return False

def main(startup_delay: float = 0.5):
    print("\n🚀 Launching GPU nodes across clusters...\n")

    for cluster_name, nodes in ALL_CLUSTERS.items():
        print(f"=== {cluster_name} ===")
        for port in nodes:
            print(f" -> Starting node on port {port} ...", end="")
            proc = launch_node(port)
            processes.append(proc)

            if check_health(port):
                print(" ✅ [READY]")
            else:
                print(" ❌ [FAILED TO START]")

            # delay before starting next node
            time.sleep(startup_delay)

    print("\n✅ All nodes launched. Press Ctrl+C to shut down.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all nodes...")
        for proc in processes:
            proc.terminate()
        time.sleep(2)
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
