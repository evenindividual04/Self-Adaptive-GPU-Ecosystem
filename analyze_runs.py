# analyze_runs.py
import argparse, csv, os
import pandas as pd

METRICS_OUT = "runs/summary.csv"
NODE_DIST_OUT = "runs/node_distribution.csv"

def summarize(df):
    out = {}
    out["policy"] = df["policy"].iloc if len(df)>0 else "unknown"
    out["jobs"] = len(df)
    out["success"] = int(df["success"].sum())
    out["rejects"] = int(((df["success"]==0) & (df["reason"].notna())).sum())
    out["unreachable"] = int(((df["reason"]=="unreachable") | (df["reason"]=="exception")).sum())
    out["p95_latency_s"] = float(df["latency_s"].astype(float).quantile(0.95)) if len(df)>0 else 0.0
    out["avg_latency_s"] = float(df["latency_s"].astype(float).mean()) if len(df)>0 else 0.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="CSV files from submit_burst runs")
    args = ap.parse_args()

    os.makedirs("runs", exist_ok=True)

    summaries = []
    node_rows = []

    for path in args.files:
        df = pd.read_csv(path)
        s = summarize(df)
        s["file"] = os.path.basename(path)
        summaries.append(s)
        # node distribution for successes
        dist = df[df["success"]==1].groupby(["policy","node"]).size().reset_index(name="count")
        dist["file"] = os.path.basename(path)
        node_rows.append(dist)

    summ_df = pd.DataFrame(summaries)
    summ_df.to_csv(METRICS_OUT, index=False)
    print("\nSummary:")
    print(summ_df)

    if node_rows:
        nd = pd.concat(node_rows, ignore_index=True)
        nd.to_csv(NODE_DIST_OUT, index=False)
        print("\nPer-node distribution (successes):")
        print(nd)

if __name__ == "__main__":
    main()
