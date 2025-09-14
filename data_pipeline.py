# data_pipeline.py

import json, math, os
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from nx_topology import build_ring_knn_graph, compute_nx_features, nx_to_edge_index  # NX graph + features

# 1) Canonical feature order (must match everywhere: training + inference)
FEATURES = ["utilization", "memory_usage", "power_usage", "temperature", "fan_speed"]  # 5 telemetry

# 2) Cluster map (keep in sync with launch_nodes.py)
CLUSTERS = {
    "Cluster-1": [8000, 8001, 8002, 8003],
    "Cluster-2": [8010, 8011, 8012, 8013],
}
PORT2CLUSTER = {p: cid for cid, ports in CLUSTERS.items() for p in ports}

def load_sim(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)

def build_snapshots(rows: List[Dict], window_s: int = 20) -> List[Dict]:
    # Group rows into snapshots by (cluster_id, window_start)
    buckets = defaultdict(lambda: {"nodes": {}})
    for r in rows:
        port = int(r["node_port"])
        cid = next((k for k, v in CLUSTERS.items() if port in v), "Cluster-1")
        win = int(math.floor(float(r["timestamp"]) / window_s) * window_s)
        buckets[(cid, win)]["nodes"][port] = r
    snaps = []
    for (cid, win), g in buckets.items():
        if len(g["nodes"]) < 3:  # skip tiny graphs
            continue
        snaps.append({"cluster_id": cid, "window_ts": win, "nodes": g["nodes"]})
    snaps.sort(key=lambda s: s["window_ts"])
    return snaps

def _topology_edges(ports: List[int]) -> List[List[int]]:
    # Ring per cluster: sparse O(n) and stable
    edges = []
    n = len(ports)
    for i in range(n):
        j = (i + 1) % n
        edges.append([i, j]); edges.append([j, i])
    return edges

def _knn_edges(feat_mat: np.ndarray, k: int = 3) -> List[List[int]]:
    import numpy as _np
    feat_mat = _np.asarray(feat_mat)
    if feat_mat.ndim != 2:
        raise ValueError(f"feat_mat must be 2D (N,F); got ndim={feat_mat.ndim}")
    n = feat_mat.shape[0]             # number of nodes
    if n <= 1 or k <= 0:
        return []
    from sklearn.metrics.pairwise import euclidean_distances
    D = euclidean_distances(feat_mat, feat_mat)
    edges = []
    for i in range(n):
        nbrs = _np.argsort(D[i])[1:min(k+1, n)]
        for j in nbrs:
            edges.append([i, j]); edges.append([j, i])
    return edges


def build_edges(ports: List[int], X: np.ndarray, k: int = 3) -> torch.Tensor:
    # Optional: ring + kNN builder (not used in make_graphs when NX is available)
    edges = _topology_edges(ports) + _knn_edges(X, k=k)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).T.contiguous()

def make_graphs(snaps: List[Dict]) -> Tuple[List[Data], np.ndarray]:
    graphs, times = [], []
    for s in snaps:
        ports = sorted(s["nodes"].keys())
        X = []
        node_flags = []
        for p in ports:
            row = s["nodes"][p]
            X.append([row.get(f, 0.0) for f in FEATURES])  # 5 telemetry feats
            node_flags.append(1 if row.get("is_anomalous", False) else 0)
        X = np.asarray(X, dtype=np.float32)

        # Build NetworkX ring+kNN graph once and compute structural features on the same graph
        G = build_ring_knn_graph(X, k=3)
        ei_np = nx_to_edge_index(G)             # (2,E) from the same G
        nxF = compute_nx_features(G)            # (N,4): degree, betweenness, eigenvector, clustering

        assert X.shape[0] == nxF.shape[0], (
            f"Row mismatch: X rows={X.shape[0]} NX rows={nxF.shape[0]}"
        )
        assert X.shape[1] == len(FEATURES), (
            f"Telemetry column mismatch: expected {len(FEATURES)}, got {X.shape[1]}"
        )
        assert nxF.shape[1] == 4, (
            f"NX features column mismatch: expected 4, got {nxF.shape[1]}"
        )

        # Concatenate telemetry + NX features → 9 columns total
        X_all = np.concatenate([X, nxF.astype(np.float32)], axis=1)  # (N, 5+4)
        edge_index = torch.tensor(ei_np, dtype=torch.long).contiguous()  # COO [2,E]
        y_graph = int(any(node_flags))  # graph-level label: 1 if any node anomalous

        graphs.append(Data(
            x=torch.tensor(X_all, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor([y_graph], dtype=torch.long),
        ))
        times.append(s["window_ts"])
    return graphs, np.asarray(times)

def split_and_scale(graphs: List[Data], times: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    # Time-based split to respect causality and avoid leakage
    idx = np.argsort(times)
    graphs = [graphs[i] for i in idx]
    n = len(graphs); n_train = int(n*train_ratio); n_val = int(n*val_ratio)
    train, val, test = graphs[:n_train], graphs[n_train:n_train+n_val], graphs[n_train+n_val:]

    # Fit scaler on train only, apply to val/test
    scaler = StandardScaler()
    if len(train) == 0:
        raise RuntimeError("No training graphs; collect more telemetry or widen window.")
    X_train = np.vstack([g.x.numpy() for g in train])
    scaler.fit(X_train)

    def _apply(gs):
        outs = []
        for g in gs:
            g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float32)
            outs.append(g)
        return outs

    return _apply(train), _apply(val), _apply(test), scaler

def save_datasets(train, val, test, scaler, out_dir="datasets", name="sim"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"graphs": train}, os.path.join(out_dir, f"{name}_train.pt"))
    torch.save({"graphs": val}, os.path.join(out_dir, f"{name}_val.pt"))
    torch.save({"graphs": test}, os.path.join(out_dir, f"{name}_test.pt"))
    import joblib; joblib.dump(scaler, os.path.join(out_dir, f"{name}_scaler.pkl"))
    print(f"Saved datasets to {out_dir}/ with {len(train)}/{len(val)}/{len(test)} graphs")