# merge_thermal.py
import argparse, csv, json, numpy as np, sys

def read_jsonl(path):
    with open(path) as r:
        for line in r:
            s = line.strip()
            if s:
                yield json.loads(s)

def aggregate_thermal(csv_path, seconds_per_window=20):
    T, F, ct, cf = [], [], [], []
    with open(csv_path, newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        except Exception:
            dialect = csv.excel
        has_hdr = False
        try:
            has_hdr = csv.Sniffer().has_header(sample)
        except Exception:
            pass
        rd = csv.reader(f, dialect)
        if has_hdr:
            next(rd, None)
        for i, row in enumerate(rd):
            if not row or row.startswith("#"): 
                continue
            if len(row) < 3: 
                continue
            try:
                # Expect: time_s, temp_C, fan_pct
                # We ignore time_s here; it is implicit by row index
                temp = float(row[21])
                fan  = float(row)
            except ValueError:
                continue
            ct.append(temp); cf.append(fan)
            if (len(ct)) % seconds_per_window == 0:
                T.append(float(np.mean(ct))); F.append(float(np.mean(cf)))
                ct, cf = [], []
    return T, F

def main():
    ap = argparse.ArgumentParser(description="Merge SystemC thermal.csv into 20s JSONL rows")
    ap.add_argument("rows_jsonl", help="Input JSONL of 20s windows")
    ap.add_argument("--thermal_csv", default="thermal.csv", help="SystemC output CSV (time,temp,fan)")
    ap.add_argument("--out", default="systemc_rows.jsonl", help="Output JSONL path")
    ap.add_argument("--seconds-per-window", type=int, default=20)
    args = ap.parse_args()

    T, F = aggregate_thermal(args.thermal_csv, args.seconds_per_window)
    rows = list(read_jsonl(args.rows_jsonl))
    n = min(len(rows), len(T))
    if n == 0:
        print("No windows to merge; check inputs", file=sys.stderr); sys.exit(2)
    for i in range(n):
        rows[i]["temperature"] = float(T[i])
        rows[i]["fan_speed"]   = float(F[i])
    with open(args.out, "w") as w:
        for i in range(n):
            w.write(json.dumps(rows[i]) + "\n")
    print(f"Wrote {n} windows → {args.out}")

if __name__ == "__main__":
    main()
