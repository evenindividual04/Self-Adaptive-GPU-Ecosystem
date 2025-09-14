# scripts/dry_run.py
import json, joblib, torch, numpy as np
from torch_geometric.loader import DataLoader
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
m = EnhancedGNNDetector(cfg)
m.load_state_dict(torch.load("deploy/best_model.pth", map_location="cpu", weights_only=False))
m.eval()
thr = json.load(open("deploy/threshold.json"))["threshold"]
val = torch.load("datasets/sim_val.pt", weights_only=False)["graphs"]
b = next(iter(DataLoader(val[:64], batch_size=64, shuffle=False)))
with torch.no_grad():
    pr = torch.softmax(m(b.x, b.edge_index, b.batch), dim=1)[:,1].cpu().numpy()
pred = (pr >= thr).astype(int)
print("thr=", thr, " prob[:5]=", np.round(pr[:5],3), " pred[:5]=", pred[:5])
