# eval_test.py
import os, json, numpy as np, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch_geometric.loader import DataLoader
from gnn_model import EnhancedGNNDetector

def load_split(p): return torch.load(p, weights_only=False)["graphs"]

test = load_split("datasets/sim_test.pt")
thr = json.load(open("threshold.json"))["threshold"]   # 0.699 from your run
cfg = {"model":{"input_dim":9,"hidden_dim":128,"output_dim":2,"dropout":0.3,"heads":4,"dropedge_p":0.10}}
m = EnhancedGNNDetector(cfg)
m.load_state_dict(torch.load("logs/sim_only_gnn/best_model.pth", map_location="cpu", weights_only=False))
m.eval()

y, p = [], []
with torch.no_grad():
    for b in DataLoader(test, batch_size=128, shuffle=False):
        logits = m(b.x, b.edge_index, b.batch)
        prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        y.extend(b.y.cpu().numpy().ravel()); p.extend(prob)

y = np.array(y); p = np.array(p)
pred = (p >= thr).astype(int)
print(dict(
  acc=float(accuracy_score(y,pred)),
  prec=float(precision_score(y,pred,zero_division=0)),
  rec=float(recall_score(y,pred,zero_division=0)),
  f1=float(f1_score(y,pred,zero_division=0)),
  auc=float(roc_auc_score(y,p)),
))
print("confusion_matrix:\n", confusion_matrix(y,pred))

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

# PR Curve
precision, recall, _ = precision_recall_curve(y, p)
plt.figure()
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('pr_curve.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y, p)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig('roc_curve.png')
plt.close()

print('Saved pr_curve.png and roc_curve.png')

