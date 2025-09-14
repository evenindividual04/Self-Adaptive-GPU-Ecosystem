# submit_burst.py
import argparse, time, uuid, csv, random, requests, os

CONTROL_URL = os.getenv("CONTROL_URL", "http://127.0.0.1:9000")

JOB_TYPES = ["train", "infer", "etl"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=120, help="total jobs to submit")
    ap.add_argument("--rps", type=float, default=5.0, help="submit rate (req/sec)")
    ap.add_argument("--size-min", type=float, default=0.05)
    ap.add_argument("--size-max", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="runs/out.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # detect policy
    try:
        pol = requests.get(f"{CONTROL_URL}/health", timeout=2).json().get("policy", "unknown")
    except Exception:
        pol = "unknown"

    interval = 1.0 / max(args.rps, 1e-6)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id","submit_ts","resp_ts","latency_s","success","node","reason","size","type","policy"])    
        for i in range(args.jobs):
            job_id = str(uuid.uuid4())
            size = random.uniform(args.size_min, args.size_max)
            jtype = random.choice(JOB_TYPES)
            payload = {"job_id": job_id, "size": size, "type": jtype}
            t0 = time.time()
            try:
                r = requests.post(f"{CONTROL_URL}/schedule", json=payload, timeout=5)
                t1 = time.time()
                latency = t1 - t0
                ok = False
                node = None
                reason = None
                if r.status_code == 200:
                    resp = r.json()
                    ok = bool(resp.get("success"))
                    node = resp.get("node")
                    reason = None if ok else resp.get("reason")
                else:
                    reason = f"http_{r.status_code}"
                w.writerow([job_id, f"{t0:.6f}", f"{t1:.6f}", f"{latency:.6f}", int(ok), node, reason, f"{size:.3f}", jtype, pol])
            except Exception:
                t1 = time.time()
                w.writerow([job_id, f"{t0:.6f}", f"{t1:.6f}", f"{(t1-t0):.6f}", 0, None, "exception", f"{size:.3f}", jtype, pol])
            # rate pacing
            to_sleep = interval - (time.time() - t0)
            if to_sleep > 0:
                time.sleep(to_sleep)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
