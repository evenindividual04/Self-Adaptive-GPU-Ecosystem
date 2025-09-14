import json
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from gnn_model import EnhancedGNNDetector  # Adjust import if needed

def load_split(path):
    return torch.load(path, weights_only=False)["graphs"]

val = load_split("datasets/sim_val.pt")
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
m = EnhancedGNNDetector(cfg)
m.load_state_dict(torch.load("logs/sim_only_gnn/best_model.pth", map_location="cpu", weights_only=False))
m.eval()

y, p = [], []
with torch.no_grad():
    for b in DataLoader(val, batch_size=128, shuffle=False):
        logits = m(b.x, b.edge_index, b.batch)
        prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        y.extend(b.y.cpu().numpy().ravel())
        p.extend(prob)

out = {"y_val": list(map(int, y)), "p_val": list(map(float, p))}
json.dump(out, open("deploy/tmp_val_scores.json", "w"), indent=2)
print("Saved deploy/tmp_val_scores.json with", len(y), "examples")
print(f"Score stats: p_val min {np.min(p):.3f}, max {np.max(p):.3f}, mean {np.mean(p):.3f}, std {np.std(p):.3f}")