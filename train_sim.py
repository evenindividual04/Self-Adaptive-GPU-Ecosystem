# train_sim.py
# Standalone script to train the EnhancedGNNDetector on simulated graph data.
# - Loads pre-split datasets from .pt files.
# - Computes class weights for imbalance.
# - Supports custom configs (e.g., for Optuna tuning).
# - Prints device (MPS/CUDA/CPU) and enables MPS fallback.
# - Integrates Weights & Biases (W&B) for logging metrics, gradients, and hyperparameters.
# - Returns test F1 for hyperparameter optimization.

import os
from datetime import datetime
import numpy as np
import torch
import random
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from experimentlogger import ExperimentLogger  # Assuming this is your custom logger
from gnn_model import EnhancedGNNDetector
from gnn_trainer import GNNTrainer
import wandb  # Weights & Biases integration

# Print device info early
if torch.backends.mps.is_available():
    print("Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    print("Using device: CUDA (NVIDIA GPU)")
else:
    print("Using device: CPU")

# Enable MPS fallback for unimplemented ops
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from datetime import datetime
import os

def create_log_dir(base_dir: str, data_name: str) -> str:
    """Create a unique log directory with data name and timestamp."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    dir_name = f"{data_name}_{timestamp}"
    full_path = os.path.join(base_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def set_seed(s=42):
    """Set random seeds for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def load_split(path):
    """Load graph list from .pt file (PyTorch 2.6+ safe)."""
    return torch.load(path, weights_only=False)["graphs"]

def main(config=None, wandb_run_name=None):
    """Main training function; accepts optional config and W&B run name for tuning/logging."""
    set_seed(42)

    # Load pre-split graphs
    train_graphs = load_split("datasets/sim_train.pt")
    val_graphs = load_split("datasets/sim_val.pt")
    test_graphs = load_split("datasets/sim_test.pt")
    
    
    
    # Compute anomaly ratio (your check)
    anomaly_ratio = np.mean([g.y.item() for g in train_graphs])
    print(f"Anomaly ratio: {anomaly_ratio:.3f}")  # If <0.2, proceed with boost

    # Prepare labels for compute_class_weight
    y_train = np.array([int(g.y.item()) for g in train_graphs])

    # Compute base balanced weights
    cls_w = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    cls_w = torch.tensor(cls_w, dtype=torch.float32)

    # Auto-boost: Compute multiplier and apply only to anomaly class (cls_w[9])
    pos = sum(y_train)  # Anomalies (class 1)
    neg = len(y_train) - pos  # Normals (class 0)
    imbalance = neg / max(1, pos)  # Imbalance ratio (>=1)
    boost_factor = 1.1  # Tune 1.0-2.0; start with 1.5
    multiplier = max(1.0, imbalance * boost_factor)  # Auto-boosted multiplier
    cls_w[1] *= multiplier  # Apply to anomaly class only
    print(f"Class weights: {cls_w}")

    '''
    anomaly_ratio = np.mean([g.y.item() for g in train_graphs])
    print(f"Anomaly ratio: {anomaly_ratio:.3f}")  # If <0.2, proceed with boost

    # Compute dynamic class weights
    pos = sum(int(g.y.item()) for g in train_graphs)  # Count anomalies (class 1)
    neg = len(train_graphs) - pos                    # Count normals (class 0)

    imbalance = neg / max(1, pos)                    # Imbalance ratio (>=1)
    boost_factor = 1.5                               # Tune 1.0-2.0; start with 1.5
    multiplier = max(1.0, imbalance * boost_factor)  # Auto-boosted multiplier
    
    # Compute class weights from train labels
    y_train = np.array([int(g.y.item()) for g in train_graphs])
    cls_w = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    cls_w = torch.tensor([1.0,multiplier], dtype=torch.float32)
    cls_w[1] *= 1.5  # Boost anomaly weight (try 1.5-2.0; start with 1.5)
    '''

    # Use default config if none provided
    if config is None:
        config = {
            "model": {
                "input_dim": 9,       # Critical: 5 telemetry + 4 NX features
                "hidden_dim": 128,    # Good starting point; tune if needed
                "output_dim": 2,      # Binary classification (normal/anomaly)
                "dropout": 0.3,       # Helps with overfitting
                "heads": 4,           # For GATv2 attention
                "dropedge_p": 0.1     # Edge dropout for regularization
        },
            "training": {
                "learning_rate": 5e-4,  # Conservative LR for stability
                "batch_size": 32,       # Adjust based on memory (M3 should handle 64 if needed)
                "epochs": 80,           # Enough for convergence; early stopping will halt if plateaued
                "patience": 10          # For early stopping
        }
}
    
    # Initialize W&B run (project name can be changed)
    run = wandb.init(
        project="sage-project",  # Your W&B project name
        name=wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # Unique name per run
        config=config,  # Logs hyperparameters
        reinit=True  # Allows multiple runs in one script (for Optuna)
    )

    # Custom logger (your existing one)
    logger = ExperimentLogger("sim_only_gnn")
    logger.log_experiment_config(config)

    # Initialize model and watch it with W&B for gradient logging
    model = EnhancedGNNDetector(config)
    wandb.watch(model, log="all", log_freq=10)  # Logs gradients and parameters every 10 steps

    # Initialize trainer
    trainer = GNNTrainer(model, config, logger, class_weights=cls_w)

    # Train with per-epoch logging to W&B
    trainer.train(train_graphs, val_graphs)
    # If you want to log per-epoch metrics, implement here or remove the loop.
    # Example placeholder (remove if not needed):
    # for epoch in range(1, config["training"]["epochs"] + 1):
    #     wandb.log({"epoch": epoch})

    # Evaluate and log final test metrics to W&B
    metrics, _, _ = trainer.final_evaluation(test_graphs)
    wandb.log(metrics)  # Logs {'accuracy': ..., 'f1': ..., etc.}

    # Finish W&B run
    wandb.finish()

    # Return test F1 for Optuna (or other tuners)
    return metrics.get("f1", 0.0)

if __name__ == "__main__":
    main()


    '''config = {
        "model": {"input_dim": 5, "hidden_dim": 64, "output_dim": 2, "dropout": 0.3},
        "training": {"learning_rate": 1e-3, "batch_size": 32, "epochs": 60, "patience": 8},
    }'''
    