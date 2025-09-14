# threshold_tune.py
import os, json, numpy as np, torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
from gnn_model import EnhancedGNNDetector
import argparse


def load_split(path):
    return torch.load(path, weights_only=False)["graphs"]  # PyTorch 2.6+ safe for PyG Data [see docs]

'''
def pick_threshold(y, p):
    precision, recall, thr = precision_recall_curve(y, p)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    i = int(np.nanargmax(f1))
    best_thr = float(thr[max(i-1, 0)]) if len(thr) > 0 else 0.5
    ap = float(average_precision_score(y, p))
    return best_thr, float(f1[i]), ap
'''

def pick_threshold(y, p, min_precision=0.55):  # New param for precision floor
    precision, recall, thr = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mask = precision >= min_precision
    cand = np.flatnonzero(mask)
    if cand.size > 0:
        i = cand[np.argmax(f1[cand])]
    else:
        i = int(np.nanargmax(f1))  # Fallback to max F1
    best_thr = float(thr[max(i-1, 0)]) if len(thr) > 0 else 0.5
    ap = float(average_precision_score(y, p))
    return best_thr, float(f1[i]), ap

def main(ckpt_dir="logs/sim_only_gnn"):
    val_graphs = load_split("datasets/sim_val.pt")
    ckpt_dir = "logs/sim_only_gnn"
    config = {"model": {"input_dim": 9, "hidden_dim": 128, "output_dim": 2, "dropout": 0.3, "heads": 4, "dropedge_p": 0.10}}
    model = EnhancedGNNDetector(config)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location="cpu", weights_only=False))
    model.eval()

    y, p = [], []
    with torch.no_grad():
        for batch in DataLoader(val_graphs, batch_size=64, shuffle=False):
            logits = model(batch.x, batch.edge_index, batch.batch)   # pooled graph logits, per-graph [B,2] [12]
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # anomaly probability
            y.extend(batch.y.cpu().numpy().ravel()); p.extend(prob)

    y = np.array(y); p = np.array(p)
    #thr, f1, ap = pick_threshold(y, p)
    thr, f1, ap = pick_threshold(y, p, min_precision=0.55)  # New: min precision floor
    json.dump({"threshold": thr, "val_f1": f1, "val_ap": ap}, open("threshold.json", "w"), indent=2)
    print(f"Saved threshold.json with threshold={thr:.3f}, F1={f1:.3f}, AP={ap:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="logs/sim_only_gnn", help="Path to log directory")
    args = parser.parse_args()
    main(args.ckpt_dir)
