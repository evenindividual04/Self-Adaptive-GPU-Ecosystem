import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, GraphNorm
from typing import Dict, List
import numpy as np
from sklearn.preprocessing import StandardScaler

class EnhancedGNNDetector(nn.Module):
    """Enhanced GNN with GCN and GAT layers and configurable architecture"""
    
    def __init__(self, config: Dict):
        super(EnhancedGNNDetector, self).__init__()
        
        input_dim = config['model']['input_dim']
        hidden_dim = config['model']['hidden_dim']
        output_dim = config['model']['output_dim']
        dropout = config['model']['dropout']
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        self.attention = GATConv(hidden_dim // 2, hidden_dim // 4, heads=4, concat=True, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.conv2(h1, edge_index)) + h1  # Residual
        h2 = self.dropout(h2)
        
        h3 = F.relu(self.conv3(h2, edge_index))
        h3 = self.dropout(h3)
        
        h_att = self.attention(h3, edge_index)
        h_att = F.relu(h_att)
        
        if batch is not None:
            h_pooled = global_mean_pool(h_att, batch)
        else:
            h_pooled = torch.mean(h_att, dim=0, keepdim=True)
        
        out = self.classifier(h_pooled)
        return out


class GraphDataProcessor:
    """Process raw telemetry data into graphs suitable for GNN training"""
    
    def __init__(self, feature_names: List[str]):
        self.scaler = StandardScaler()
        self.feature_names = feature_names
    
    def create_graph_from_snapshot(self, snapshot_data: Dict, labels: Dict = None) -> Data:
        node_features = []
        node_labels = []
        node_ports = sorted(snapshot_data.keys())
        
        for port in node_ports:
            metrics = snapshot_data[port]
            features = [metrics.get(f, 0) for f in self.feature_names]
            node_features.append(features)
            if labels:
                node_labels.append(1 if labels.get(port, False) else 0)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Fully connected edges except self-loops
        num_nodes = len(node_ports)
        edge_indices = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        if labels:
            y = torch.tensor(node_labels, dtype=torch.long)
            return Data(x=x, edge_index=edge_index, y=y)
        else:
            return Data(x=x, edge_index=edge_index)
    
    def prepare_training_data(self, training_file: str):
        with open(training_file, 'r') as f:
            raw_data = json.load(f)
        
        all_features = []
        for node_data in raw_data:
            for sample in node_data['samples']:
                feat_values = [sample.get(f, 0) for f in self.feature_names]
                all_features.append(feat_values)
        
        self.scaler.fit(all_features)
        
        # Group samples by timestamp for snapshots
        timestamp_groups = {}
        
        for node_data in raw_data:
            port = node_data['node_port']
            samples = node_data['samples']
            anomaly_labels = node_data['anomaly_labels']
            for sample, anom in zip(samples, anomaly_labels):
                ts = sample['timestamp']
                if ts not in timestamp_groups:
                    timestamp_groups[ts] = {'data': {}, 'labels': {}}
                # Normalize features
                norm_feats = self.scaler.transform([[sample.get(f, 0) for f in self.feature_names]])[0]
                for i, f in enumerate(self.feature_names):
                    sample[f] = norm_feats[i]
                timestamp_groups[ts]['data'][port] = sample
                timestamp_groups[ts]['labels'][port] = anom
        
        graphs = []
        labels = []
        for ts, group in timestamp_groups.items():
            if len(group['data']) >= 3:  # require at least 3 nodes
                graph = self.create_graph_from_snapshot(group['data'], group['labels'])
                graphs.append(graph)
                labels.append(1 if any(group['labels'].values()) else 0)
        
        return graphs, labels
