# gnn_model.py
from __future__ import annotations
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATv2Conv, GraphNorm, global_mean_pool
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Model: EnhancedGNNDetector
# ---------------------------
class EnhancedGNNDetector(nn.Module):
    """
    Graph-level classifier:
    - Input: batched graphs via PyG (x, edge_index, batch)
    - Output: logits per graph for CrossEntropyLoss
    """
    def __init__(self, config: Dict):
        super().__init__()
        assert 'model' in config and all(k in config['model'] for k in
               ['input_dim', 'hidden_dim', 'output_dim', 'dropout']), \
               "config['model'] must contain input_dim, hidden_dim, output_dim, dropout"
        d_in = int(config['model']['input_dim'])
        d_hid = int(config['model']['hidden_dim'])
        d_out = int(config['model']['output_dim'])
        p_drop = float(config['model']['dropout'])
        self.heads = int(config['model'].get('heads', 4))          # attention heads
        self.dropedge_p = float(config['model'].get('dropedge_p', 0.10))  # DropEdge prob.
        self.scaler = None # Optional StandardScaler, fit on train only elsewhere

        # Encoder: 3 GCN layers with residual and GraphNorm
        self.conv1 = GCNConv(d_in, d_hid);     self.norm1 = GraphNorm(d_hid)
        self.conv2 = GCNConv(d_hid, d_hid);    self.norm2 = GraphNorm(d_hid)
        self.conv3 = GCNConv(d_hid, d_hid // 2); self.norm3 = GraphNorm(d_hid // 2)

        # Attention: GATv2 over reduced channels
        out_per_head = max(d_hid // 8, 1)
        self.att = GATv2Conv(d_hid // 2, out_per_head, heads=self.heads, concat=True, dropout=p_drop)
        self.att_out_dim = out_per_head * self.heads
        self.att_norm = GraphNorm(self.att_out_dim)
        self.dropout = nn.Dropout(p_drop)

        # Classifier expects attention output dimension
        cls_in = self.att_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, d_hid // 2),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hid // 2, d_hid // 4),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hid // 4, d_out),
        )

    def _maybe_drop_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        """DropEdge-style edge dropout for regularization."""
        if (not self.training) or self.dropedge_p <= 0.0:
            return edge_index
        if edge_index is None or edge_index.numel() == 0:
            return edge_index
        E = edge_index.size(1)
        keep = torch.rand(E, device=edge_index.device) > self.dropedge_p
        if keep.sum() == 0:
            idx = torch.randint(E, (1,), device=edge_index.device)
            keep[idx] = True
        return edge_index[:, keep]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        edge_index = self._maybe_drop_edges(edge_index)
        # Block 1
        h1 = F.relu(self.conv1(x, edge_index)); h1 = self.norm1(h1, batch); h1 = self.dropout(h1)
        # Block 2 (+ residual)
        h2 = F.relu(self.conv2(h1, edge_index)); h2 = self.norm2(h2, batch); h2 = self.dropout(h2); h2 = h2 + h1
        # Block 3
        h3 = F.relu(self.conv3(h2, edge_index)); h3 = self.norm3(h3, batch); h3 = self.dropout(h3)
        # Attention
        hatt = self.att(h3, edge_index); hatt = F.relu(hatt); hatt = self.att_norm(hatt, batch)
        # Graph pooling
        h_pool = global_mean_pool(hatt, batch) if batch is not None else hatt.mean(dim=0, keepdim=True)
        # Classifier
        logits = self.classifier(h_pool)
        return logits

# ---------------------------
# Data: GraphDataProcessor (optional, if not using data_pipeline)
# ---------------------------
class GraphDataProcessor:
    """
    Build PyG graphs from simulator-style JSON rows.
    Assumes each row contains:
    - 'node_port', 'timestamp', and features: utilization, memory_usage, power_usage, temperature, fan_speed
    - 'is_anomalous' (bool) for weak supervision
    """
    FEATURES = ["utilization", "memory_usage", "power_usage", "temperature", "fan_speed"]

    CLUSTERS = {
        "Cluster-1": [8000, 8001, 8002, 8003],
        "Cluster-2": [8010, 8011, 8012, 8013],
    }

    def __init__(self, feature_names: Optional[List[str]] = None, k_nn: int = 3):
        self.feature_names = feature_names or self.FEATURES
        self.k_nn = k_nn
        self.scaler: Optional[StandardScaler] = None  # fit on train only elsewhere

    # ----- edge builders -----
    @staticmethod
    def _ring_edges(n: int) -> List[List[int]]:
        """Bidirectional ring for stability and sparsity."""
        if n <= 1:
            return []
        edges = []
        for i in range(n):
            j = (i + 1) % n
            edges.append([i, j]); edges.append([j, i])
        return edges

    @staticmethod
    def _knn_edges(X: np.ndarray, k: int) -> List[List[int]]:
        """Symmetric k-NN edges by Euclidean distance."""
        n = X.shape[0]  # FIX
        if n <= 1 or k <= 0:
            return []
        from sklearn.metrics.pairwise import euclidean_distances
        D = euclidean_distances(X, X)
        edges = []
        for i in range(n):
            nbrs = np.argsort(D[i])[1:min(k+1, n)]
            for j in nbrs:
                edges.append([i, j]); edges.append([j, i])
        return edges

    def build_edges(self, X: np.ndarray) -> torch.Tensor:
        """Combine ring + kNN (sparse)."""
        n = X.shape[0]  # FIX
        edges = self._ring_edges(n) + self._knn_edges(X, self.k_nn)
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ----- snapshot builder -----
    def _port_to_cluster(self, port: int) -> str:
        for cid, ports in self.CLUSTERS.items():
            if port in ports:
                return cid
        return "Cluster-1"

    def build_snapshots(self, rows: List[Dict], window_s: int = 20, min_nodes: int = 3) -> List[Dict]:
        """Group flat rows into per-timestamp snapshots across clusters."""
        buckets = defaultdict(lambda: {"nodes": {}})
        for r in rows:
            port = int(r["node_port"])
            cid = self._port_to_cluster(port)
            win = int(np.floor(float(r["timestamp"]) / window_s) * window_s)
            buckets[(cid, win)]["nodes"][port] = r
        snaps = []
        for (cid, win), g in buckets.items():
            if len(g["nodes"]) >= min_nodes:
                snaps.append({"cluster_id": cid, "window_ts": win, "nodes": g["nodes"]})
        snaps.sort(key=lambda s: s["window_ts"])
        return snaps

    def create_graph_from_snapshot(self, snapshot: Dict) -> Optional[Data]:
        """Build a graph Data from one snapshot dict (ports->rows); y=1 if any node anomalous."""
        ports = sorted(snapshot["nodes"].keys())
        X, node_flags = [], []
        for p in ports:
            row = snapshot["nodes"][p]
            X.append([row.get(f, 0.0) for f in self.feature_names])
            node_flags.append(1 if row.get("is_anomalous", False) else 0)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            return None  # Or raise an error for debugging

        # guard against degenerate cases
        if X.size == 0 or np.isnan(X).any() or np.isinf(X).any():
            return None
        edge_index = self.build_edges(X)
        if edge_index.numel() == 0:
            return None
        y_graph = int(any(node_flags))
        return Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor([y_graph], dtype=torch.long),
        )

    # ----- end-to-end helpers -----
    def graphs_from_rows(self, rows: List[Dict], window_s: int = 20) -> Tuple[List[Data], np.ndarray]:
        """Convert flat simulator rows into a list of graphs + times."""
        graphs, times = [], []
        snaps = self.build_snapshots(rows, window_s=window_s)
        for s in snaps:
            g = self.create_graph_from_snapshot(s)
            if g is not None:
                graphs.append(g)
                times.append(s["window_ts"])
        return graphs, np.asarray(times)

    def graphs_from_json(self, json_path: str, window_s: int = 20) -> Tuple[List[Data], np.ndarray]:
        """Load simulator JSON and build graphs."""
        with open(json_path, "r") as f:
            rows = json.load(f)
        return self.graphs_from_rows(rows, window_s=window_s)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
