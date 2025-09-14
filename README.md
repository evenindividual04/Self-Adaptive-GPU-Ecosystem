# Self-Adaptive GPU Ecosystem for Anomaly Detection

This project simulates a distributed GPU ecosystem and uses a Graph Neural Network (GNN) to detect anomalies in real-time. It includes a monitoring service, a control plane for intelligent job scheduling, and a telemetry dashboard for visualization.

## Key Features

- **GPU Workload Simulation**: Simulates multiple GPU nodes, each with its own workload and telemetry data (utilization, temperature, power, etc.).
- **Real-time Monitoring**: A monitoring service polls telemetry data from all nodes, detects anomalies, and logs the data.
- **GNN-based Anomaly Detection**: A Graph Neural Network model is used to identify complex anomalous patterns from the collected telemetry data.
- **Intelligent Control Plane**: A control plane (scheduler) receives job requests and intelligently assigns them to the healthiest and most available GPU nodes based on a scoring policy.
- **Dynamic Node Management**: Nodes can be added, removed, or drained dynamically via the control plane's API.
- **Telemetry Dashboard**: A Streamlit-based dashboard visualizes the real-time status of the GPU cluster.
- **Automated Setup**: A single script (`run_all.py`) launches all the components of the ecosystem.

## Project Structure

```
.
├── gnn_anomaly_detector.py  # GNN model definition and training
├── monitoring_service.py    # Collects telemetry and detects anomalies
├── control_plane.py         # Schedules jobs and manages nodes
├── workload_simulator.py    # Simulates GPU workloads
├── gpu_node_service.py      # Simulates a single GPU node
├── telemetry_dashboard.py   # Streamlit dashboard for visualization
├── run_all.py               # Main script to launch the entire ecosystem
├── requirements.txt         # Python dependencies
└── ...
```

## Getting Started

### Prerequisites

- Python 3.8+
- An active virtual environment is recommended.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/evenindividual04/Self-Adaptive-GPU-Ecosystem.git
    cd Self-Adaptive-GPU-Ecosystem
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Ecosystem

The entire ecosystem can be launched using the `run_all.py` script. This will start the GPU node simulators, the control plane, the monitoring service, and the telemetry dashboard.

```bash
python run_all.py
```

After running the script, you can access the following services:

-   **Control Plane API**: `http://127.0.0.1:9000/docs`
-   **Monitoring Service API**: `http://127.0.0.1:9100/cluster-stats`
-   **Telemetry Dashboard**: `http://localhost:8501`

## How It Works

1.  **GPU Nodes (`gpu_node_service.py`)**: Multiple instances of this service are run to simulate a cluster of GPUs. Each node exposes endpoints to get metrics and submit jobs.
2.  **Monitoring Service (`monitoring_service.py`)**: This service periodically polls the GPU nodes for their telemetry data. It uses the GNN model to detect anomalies and logs the data.
3.  **Control Plane (`control_plane.py`)**: This is the "brain" of the ecosystem. It receives job requests and decides which GPU node to send them to. It considers factors like node utilization, health, and reliability.
4.  **Job Submitter (`submit_jobs.py`)**: This script (launched by `run_all.py`) continuously sends simulated jobs to the control plane, creating a realistic workload.
5.  **Dashboard (`telemetry_dashboard.py`)**: The dashboard reads the data logged by the monitoring service and provides a real-time view of the cluster's health and performance.
