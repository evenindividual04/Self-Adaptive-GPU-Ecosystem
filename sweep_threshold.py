# sweep_threshold.py
import numpy as np, torch, json
from sklearn.metrics import precision_recall_curve
from torch_geometric.loader import DataLoader
from gnn_model import EnhancedGNNDetector
def load(p): return torch.load(p, weights_only=False)["graphs"]
val = load("datasets/sim_val.pt")
cfg = {"model":{"input_dim":9,"hidden_dim":128,"output_dim":2,"dropout":0.3,"heads":4,"dropedge_p":0.10}}
m = EnhancedGNNDetector(cfg); m.load_state_dict(torch.load("logs/sim_only_gnn/best_model.pth", map_location="cpu", weights_only=False)); m.eval()
y,p=[],[]
with torch.no_grad():
  for b in DataLoader(val, batch_size=256, shuffle=False):
    s = torch.softmax(m(b.x,b.edge_index,b.batch),dim=1)[:,1].cpu().numpy()
    p.extend(s); y.extend(b.y.cpu().numpy().ravel())
y=np.array(y); p=np.array(p)
'''prec, rec, thr = precision_recall_curve(y, p)
# pick threshold for precision >= 0.60 with best F1 among those
mask = prec >= 0.60
cand = np.where(mask)[:-1]  # drop last sentinel
f1 = 2*prec*rec/(prec+rec+1e-8)
i = int(cand[np.argmax(f1[cand])]) if cand.size>0 else int(np.nanargmax(f1))
thr_star = float(thr[max(i-1,0)]) if len(thr)>0 else 0.5
json.dump({"threshold":thr_star,"target_precision":0.60}, open("threshold.json","w"), indent=2)
print("saved threshold", thr_star)'''

prec, rec, thr = precision_recall_curve(y, p)
f1 = 2*prec*rec/(prec+rec+1e-8)

mask = prec >= 0.60
cand = np.where(mask)[0]

# Optionally, drop last sentinel if present (rare but sometimes needed).
if len(cand) > 0 and cand[-1] == len(f1):
    cand = cand[:-1]

# Use len(cand) instead of cand.size for robust indexing
i = int(cand[np.argmax(f1[cand])]) if len(cand) > 0 else int(np.nanargmax(f1))

thr_star = float(thr[max(i-1,0)]) if len(thr) > 0 else 0.5
json.dump({"threshold":thr_star,"target_precision":0.60}, open("threshold.json","w"), indent=2)
print("saved threshold", thr_star)
print(f"Precision at chosen threshold: {prec[i]:.3f}, Recall: {rec[i]:.3f}, F1: {f1[i]:.3f}")
