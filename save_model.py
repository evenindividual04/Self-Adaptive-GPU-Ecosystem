# scripts/save_model.py (run if needed)
import torch
from gnn_model import EnhancedGNNDetector

cfg = {
    "model": {
        "input_dim": 9,
        "hidden_dim": 128,
        "output_dim": 2,
        "dropout": 0.3,
        "heads": 4,
        "dropedge_p": 0.10
    }
}
model = EnhancedGNNDetector(cfg)
model.load_state_dict(torch.load("logs/sim_only_gnn/best_model.pth", map_location="cpu", weights_only=False))
torch.save(model.state_dict(), "deploy/best_model.pth")
print("Saved deploy/best_model.pth")
