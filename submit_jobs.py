# submit_jobs.py
import requests
import time
import random
import argparse
import uuid

CONTROL_PLANE_URL = "http://127.0.0.1:9000/schedule"
JOB_TYPES = ["training", "inference", "data-prep"]

def submit_job(job_id, size, job_type):
    payload = {"job_id": job_id, "size": size, "type": job_type}
    try:
        r = requests.post(CONTROL_PLANE_URL, json=payload, timeout=3)
        if r.status_code == 200 and r.json().get("success", False):
            print(f"[OK] Job {job_id} ({job_type}, size={size:.2f}) scheduled.")
            return True
        elif r.status_code == 429:
            ra = r.headers.get("Retry-After")
            try:
                backoff = float(ra) if ra is not None else 5.0
            except ValueError:
                backoff = 5.0
            print(f"[BACKOFF] 429 Retry-After={backoff}s; slowing submissions.")
            time.sleep(min(10.0, max(1.0, backoff)))
            return False
        else:
            print(f"[FAIL] Job {job_id} rejected: {r.text}")
            return False
    except requests.RequestException as e:
        print(f"[ERR] Failed to submit job {job_id}: {e}")
        return False

def main(args):
    job_count = 0
    start_time = time.time()
    print(f"[Submitter] Starting job submissions every {args.rate}s...")

    try:
        while True:
            # Check duration limit
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print("[Submitter] Duration reached, stopping.")
                break
            if args.max_jobs > 0 and job_count >= args.max_jobs:
                print("[Submitter] Max jobs reached, stopping.")
                break

            # Burst or single job
            if random.random() < args.burst_prob:
                burst_size = random.randint(2, 5)
                print(f"[Submitter] Burst workload: {burst_size} jobs")
                for _ in range(burst_size):
                    job_id = str(uuid.uuid4())[:8]
                    size = random.uniform(args.size_min, args.size_max)
                    job_type = random.choice(JOB_TYPES)
                    if submit_job(job_id, size, job_type):
                        job_count += 1
            else:
                job_id = str(uuid.uuid4())[:8]
                size = random.uniform(args.size_min, args.size_max)
                job_type = random.choice(JOB_TYPES)
                if submit_job(job_id, size, job_type):
                    job_count += 1

            time.sleep(args.rate)

    except KeyboardInterrupt:
        print("\n[Submitter] Stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=10, help="Seconds between job submissions")
    parser.add_argument("--size-min", type=float, default=0.1, help="Min job size (util fraction)")
    parser.add_argument("--size-max", type=float, default=0.5, help="Max job size (util fraction)")
    parser.add_argument("--max-jobs", type=int, default=0, help="Stop after N jobs (0 = infinite)")
    parser.add_argument("--burst-prob", type=float, default=0.2, help="Probability of burst per cycle")
    parser.add_argument("--duration", type=int, default=0, help="Stop after N seconds (0 = infinite)")
    args = parser.parse_args()
    main(args)
