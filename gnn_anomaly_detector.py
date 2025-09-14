import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time

class GraphDataProcessor:
    """Process telemetry data into graph format for GNN"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = ['utilization', 'temperature', 'power_usage', 'memory_usage', 'fan_speed']
    
    def create_graph_from_snapshot(self, snapshot_data: Dict, labels: Dict = None) -> Data:
        """Create a PyTorch Geometric graph from a telemetry snapshot"""
        
        # Extract node features
        node_features = []
        node_labels = []
        node_ports = sorted(snapshot_data.keys())
        
        for port in node_ports:
            metrics = snapshot_data[port]
            features = [metrics.get(feature, 0) for feature in self.feature_names]
            node_features.append(features)
            
            if labels:
                node_labels.append(1 if labels.get(port, False) else 0)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edges - for simplicity, create a fully connected graph
        # In practice, you might use network topology or similarity-based edges
        num_nodes = len(node_ports)
        edge_indices = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edge_indices.append([i, j])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        # Create graph data object
        if labels:
            y = torch.tensor(node_labels, dtype=torch.long)
            return Data(x=x, edge_index=edge_index, y=y)
        else:
            return Data(x=x, edge_index=edge_index)
    
    def prepare_training_data(self, training_file: str) -> Tuple[List[Data], List[int]]:
        """Prepare graph data for training"""
        
        with open(training_file, 'r') as f:
            raw_data = json.load(f)
        
        graphs = []
        labels = []
        all_features = []
        
        # First pass: collect all features for normalization
        for node_data in raw_data:
            for sample in node_data['samples']:
                features = [sample.get(feature, 0) for feature in self.feature_names]
                all_features.append(features)
        
        # Fit scaler
        self.scaler.fit(all_features)
        
        # Create graphs from snapshots
        # Group samples by timestamp to create snapshots
        timestamp_groups = {}
        
        for node_data in raw_data:
            port = node_data['node_port']
            samples = node_data['samples']
            anomaly_labels = node_data['anomaly_labels']
            
            for sample, is_anomaly in zip(samples, anomaly_labels):
                timestamp = sample['timestamp']
                
                if timestamp not in timestamp_groups:
                    timestamp_groups[timestamp] = {'data': {}, 'labels': {}}
                
                timestamp_groups[timestamp]['data'][port] = sample
                timestamp_groups[timestamp]['labels'][port] = is_anomaly
        
        # Convert each timestamp snapshot to a graph
        for timestamp, group in timestamp_groups.items():
            if len(group['data']) >= 3:  # Need at least 3 nodes for meaningful graph
                # Normalize features
                for port, sample in group['data'].items():
                    features = [sample.get(feature, 0) for feature in self.feature_names]
                    normalized_features = self.scaler.transform([features])[0]
                    
                    for i, feature in enumerate(self.feature_names):
                        group['data'][port][feature] = normalized_features[i]
                
                graph = self.create_graph_from_snapshot(group['data'], group['labels'])
                graphs.append(graph)
                
                # Graph-level label (1 if any node is anomalous)
                has_anomaly = any(group['labels'].values())
                labels.append(1 if has_anomaly else 0)
        
        return graphs, labels

class GNNAnomalyDetector(nn.Module):
    """Graph Neural Network for anomaly detection"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 2):
        super(GNNAnomalyDetector, self).__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Attention layer for better feature learning
        self.attention = GATConv(hidden_dim // 2, hidden_dim // 4, heads=4, concat=True)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, batch=None):
        # Graph convolutions with residual connections
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = self.dropout(h2)
        
        h3 = F.relu(self.conv3(h2, edge_index))
        h3 = self.dropout(h3)
        
        # Attention mechanism
        h_att = self.attention(h3, edge_index)
        h_att = F.relu(h_att)
        
        # Global pooling for graph-level prediction
        if batch is not None:
            h_pooled = global_mean_pool(h_att, batch)
        else:
            h_pooled = torch.mean(h_att, dim=0, keepdim=True)
        
        # Final classification
        out = self.classifier(h_pooled)
        return out

class RuleBasedDetector:
    """Simple rule-based anomaly detector for comparison"""
    
    def __init__(self):
        self.thresholds = {
            'temperature': 80,
            'power_usage': 280,
            'memory_usage': 0.85,
            'fan_speed': 2800,
            'utilization': 0.95
        }
    
    def predict(self, graphs: List[Data]) -> List[int]:
        """Predict anomalies using simple thresholds"""
        predictions = []
        
        for graph in graphs:
            is_anomalous = False
            
            # Check each node in the graph
            for node_features in graph.x:
                features = node_features.numpy()
                
                # Check against thresholds (features are normalized, so we need to be careful)
                # For simplicity, we'll use a different approach - check for extreme values
                if (np.abs(features) > 2).any():  # More than 2 std devs from mean
                    is_anomalous = True
                    break
            
            predictions.append(1 if is_anomalous else 0)
        
        return predictions

class AnomalyDetectionBenchmark:
    """Benchmark GNN vs Rule-based anomaly detection"""
    
    def __init__(self, training_file: str = "training_data.json"):
        self.processor = GraphDataProcessor()
        self.gnn_model = None
        self.rule_detector = RuleBasedDetector()
        self.training_file = training_file
        
    def prepare_data(self):
        """Load and prepare training data"""
        print("📊 Preparing training data...")
        
        graphs, labels = self.processor.prepare_training_data(self.training_file)
        
        print(f"✅ Prepared {len(graphs)} graph samples")
        print(f"   - Normal samples: {labels.count(0)}")
        print(f"   - Anomalous samples: {labels.count(1)}")
        
        # Split data
        split_idx = int(0.8 * len(graphs))
        
        train_graphs = graphs[:split_idx]
        train_labels = labels[:split_idx]
        test_graphs = graphs[split_idx:]
        test_labels = labels[split_idx:]
        
        return (train_graphs, train_labels), (test_graphs, test_labels)
    
    def train_gnn(self, train_data: Tuple[List[Data], List[int]], epochs: int = 100):
        """Train the GNN model"""
        train_graphs, train_labels = train_data
        
        print(f"🤖 Training GNN model for {epochs} epochs...")
        
        # Initialize model
        self.gnn_model = GNNAnomalyDetector(input_dim=5)
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.gnn_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for graph, label in zip(train_graphs, train_labels):
                optimizer.zero_grad()
                
                # Forward pass
                out = self.gnn_model(graph.x, graph.edge_index)
                loss = criterion(out, torch.tensor([label], dtype=torch.long))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = out.argmax(dim=1).item()
                if pred == label:
                    correct += 1
                total += 1
            
            if (epoch + 1) % 20 == 0:
                acc = 100 * correct / total
                print(f"   Epoch {epoch + 1}/{epochs}: Loss = {total_loss/len(train_graphs):.4f}, "
                      f"Accuracy = {acc:.1f}%")
        
        print("✅ GNN training completed!")
    
    def evaluate_models(self, test_data: Tuple[List[Data], List[int]]) -> Dict:
        """Evaluate both GNN and rule-based models"""
        test_graphs, test_labels = test_data
        
        print("📈 Evaluating models...")
        
        # GNN predictions
        self.gnn_model.eval()
        gnn_predictions = []
        gnn_probabilities = []
        
        with torch.no_grad():
            for graph in test_graphs:
                out = self.gnn_model(graph.x, graph.edge_index)
                prob = F.softmax(out, dim=1)
                pred = out.argmax(dim=1).item()
                
                gnn_predictions.append(pred)
                gnn_probabilities.append(prob[0][1].item())  # Probability of anomaly
        
        # Rule-based predictions
        rule_predictions = self.rule_detector.predict(test_graphs)
        
        # Calculate metrics
        def calculate_metrics(predictions, labels, probabilities=None):
            acc = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
            
            metrics = {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            if probabilities:
                try:
                    auc = roc_auc_score(labels, probabilities)
                    metrics['auc'] = auc
                except ValueError:
                    metrics['auc'] = 0.0  # If only one class present
            
            return metrics
        
        gnn_metrics = calculate_metrics(gnn_predictions, test_labels, gnn_probabilities)
        rule_metrics = calculate_metrics(rule_predictions, test_labels)
        
        return {
            'gnn': gnn_metrics,
            'rule_based': rule_metrics,
            'test_size': len(test_labels),
            'anomaly_rate': sum(test_labels) / len(test_labels)
        }
    
    def visualize_results(self, results: Dict):
        """Create visualizations of the benchmark results"""
        
        # Prepare data for plotting
        methods = ['GNN', 'Rule-Based']
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        gnn_scores = [results['gnn'][m] for m in metrics]
        rule_scores = [results['rule_based'][m] for m in metrics]
        
        # Create comparison plot
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, gnn_scores, width, label='GNN', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, rule_scores, width, label='Rule-Based', color='#A23B72', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Anomaly Detection Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        plt.savefig('anomaly_detection_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Performance comparison plot saved as 'anomaly_detection_comparison.png'")
    
    def run_benchmark(self):
        """Run the complete benchmark"""
        print("🚀 Starting Anomaly Detection Benchmark")
        print("=" * 60)
        
        try:
            # Prepare data
            train_data, test_data = self.prepare_data()
            
            # Train GNN
            self.train_gnn(train_data, epochs=50)
            
            # Evaluate models
            results = self.evaluate_models(test_data)
            
            # Print results
            print("\n📋 BENCHMARK RESULTS")
            print("=" * 40)
            
            print(f"\nTest Set: {results['test_size']} samples")
            print(f"Anomaly Rate: {results['anomaly_rate']:.1%}")
            
            print(f"\n🤖 GNN Model:")
            for metric, score in results['gnn'].items():
                print(f"   {metric.title()}: {score:.4f}")
            
            print(f"\n📏 Rule-Based Model:")
            for metric, score in results['rule_based'].items():
                print(f"   {metric.title()}: {score:.4f}")
            
            # Determine winner
            gnn_f1 = results['gnn']['f1']
            rule_f1 = results['rule_based']['f1']
            
            print(f"\n🏆 WINNER:")
            if gnn_f1 > rule_f1:
                print(f"   GNN outperforms Rule-Based (F1: {gnn_f1:.4f} vs {rule_f1:.4f})")
            elif rule_f1 > gnn_f1:
                print(f"   Rule-Based outperforms GNN (F1: {rule_f1:.4f} vs {gnn_f1:.4f})")
            else:
                print(f"   It's a tie! (F1: {gnn_f1:.4f})")
            
            # Create visualization
            self.visualize_results(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Demo the anomaly detection system"""
    
    # Check if training data exists
    import os
    if not os.path.exists("training_data.json"):
        print("❌ No training data found!")
        print("   Please run 'python monitoring_service.py' first to collect data.")
        return
    
    # Run benchmark
    benchmark = AnomalyDetectionBenchmark()
    results = benchmark.run_benchmark()
    
    if results:
        print(f"\n🎉 Benchmark completed successfully!")
        print(f"   Check 'anomaly_detection_comparison.png' for visualization")
    else:
        print(f"\n💥 Benchmark failed!")

if __name__ == "__main__":
    main()