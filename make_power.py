# make_power.py
import argparse, json, numpy as np, sys

def main():
    p = argparse.ArgumentParser(description="Emit per-second power.csv from 20s rows.jsonl")
    p.add_argument("rows_path", help="Path to rows.jsonl (20 s windows)")  # positional arg
    p.add_argument("--seconds-per-window", type=int, default=20)
    p.add_argument("--base-watts", type=float, default=50.0)
    p.add_argument("--k-util", type=float, default=1e-6, help="W per utilization unit")
    p.add_argument("--k-mem", type=float, default=1e-9, help="W per (bytes/s)")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.rows_path)]
    util = np.array([r.get("utilization", 0.0) for r in rows])            # proxy: inst/s or %
    memB = np.array([r.get("memory_usage", 0.0) for r in rows])           # bytes per 20 s
    mem_rate = memB / float(args.seconds_per_window)                      # bytes/s
    p_win = args.base_watts + args.k_util*util + args.k_mem*mem_rate      # toy Watts

    with open("power.csv","w") as f:
        for v in p_win:
            for _ in range(args.seconds_per_window):
                f.write(f"{float(v)}\n")
    print("wrote power.csv")

if __name__ == "__main__":
    main()
