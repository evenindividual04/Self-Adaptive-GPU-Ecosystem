import argparse
import json
import numpy as np
from sklearn.metrics import precision_recall_curve  # PR curve API

ap = argparse.ArgumentParser()
ap.add_argument("--beta", type=float, default=1.0, help="F-beta weight (<1 favors precision, >1 favors recall)")
ap.add_argument("--min_precision", type=float, default=None, help="Optional precision floor")
args = ap.parse_args()

d = json.load(open("deploy/tmp_val_scores.json"))
y, p = np.array(d["y_val"]), np.array(d["p_val"])
prec, rec, thr = precision_recall_curve(y, p)  # thresholds align with prec[:-1], rec[:-1]

if len(thr) == 0:
    thr_star, fb = 0.5, 0.0
else:
    beta = args.beta
    f1b = (1 + beta * beta) * prec[:-1] * rec[:-1] / (beta * beta * prec[:-1] + rec[:-1] + 1e-8)
    if args.min_precision is not None:
        mask = prec[:-1] >= args.min_precision
        idx = np.argmax(f1b[mask]) if mask.any() else np.argmax(f1b)
        i = np.arange(len(thr))[mask][idx] if mask.any() else int(np.argmax(f1b))
    else:
        i = int(np.argmax(f1b))
    thr_star, fb = float(thr[i]), float(f1b[i])

meta = {"threshold": thr_star, "beta": beta, "val_fbeta": fb}
json.dump(meta, open("deploy/threshold.json", "w"), indent=2)
print("Saved deploy/threshold.json:", meta)
print(f"Selected threshold {thr_star:.3f} with F{beta:.1f}={fb:.3f}")
