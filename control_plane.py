# control_plane.py (Intelligent Scheduler: least/round-robin/auction-style score)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import threading
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

app = FastAPI(title="SAGE Control Plane (Intelligent Scheduler)")

# ---------------- persistence ----------------
NODES_FILE = Path("nodes.json")
JOBS_FILE = Path("jobs_log.json")
DRAINED_FILE = Path("drained_nodes.json")

MAX_JOBS = int(os.getenv("MAX_JOBS", "5000"))
HOT_TEMP = float(os.getenv("HOT_TEMP", "85.0"))  # NEW: thermal guard (°C)
DEFAULT_PORTS = [8000, 8001, 8002, 8003, 8010, 8011, 8012, 8013]
SCHED_POLICY = os.getenv("SCHED_POLICY", "score")  # "least" | "roundrobin" | "score"

_lock = threading.Lock()

# circuit breaker state: {port: {"fail": int, "open_until": float}}
_circuit: Dict[int, Dict[str, float]] = {}
FAIL_THRESHOLD = int(os.getenv("FAIL_THRESHOLD", "3"))
OPEN_SECONDS = int(os.getenv("OPEN_SECONDS", "30"))

# cache last status for /nodes
_last_status: Dict[int, Dict] = {}

# drained nodes (robust load)
def _json_load_or(p: Path, default):
    try:
        if p.exists():
            s = p.read_text().strip()
            if not s:
                return default
            return json.loads(s)
    except (OSError, json.JSONDecodeError, ValueError):
        return default
    return default

# Accept either [] or list of ints in file; coerce to ints and make a set
_drained_list = _json_load_or(DRAINED_FILE, [])
try:
    _drained = set(int(x) for x in _drained_list)
except Exception:
    _drained = set()

# Optional: write back a normalized empty list if file existed but was empty
if DRAINED_FILE.exists() and not _drained_list:
    DRAINED_FILE.write_text(json.dumps([], indent=2))

# reliability stats
_reliability: Dict[int, Dict[str, float]] = {}

# round-robin pointer
_rr_index = 0

def _save_nodes(ports: List[int]):
    NODES_FILE.write_text(json.dumps(sorted(list(set(ports))), indent=2))

def _load_nodes() -> List[int]:
    if NODES_FILE.exists():
        try:
            ports = json.loads(NODES_FILE.read_text())
            return [int(p) for p in ports]
        except Exception:
            return DEFAULT_PORTS.copy()
    return DEFAULT_PORTS.copy()

NODE_PORTS: List[int] = _load_nodes()

def _load_jobs() -> List[Dict]:
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text())
        except Exception:
            return []
    return []

def _save_jobs(jobs: List[Dict]):
    if len(jobs) > MAX_JOBS:
        jobs = jobs[-MAX_JOBS:]
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))

jobs_log: List[Dict] = _load_jobs()

# ---------------- models ----------------
class Job(BaseModel):
    job_id: str
    size: float
    type: str

class NodeAdd(BaseModel):
    port: int

# ---------------- helpers: circuit breaker & reliability ----------------
def _circuit_open(port: int) -> bool:
    entry = _circuit.get(port)
    return entry is not None and time.time() < entry.get("open_until", 0)

def _record_fail(port: int):
    entry = _circuit.setdefault(port, {"fail": 0, "open_until": 0})
    entry["fail"] = entry.get("fail", 0) + 1
    if entry["fail"] >= FAIL_THRESHOLD:
        entry["open_until"] = time.time() + OPEN_SECONDS

def _record_success(port: int):
    _circuit[port] = {"fail": 0, "open_until": 0}

def _rel(port: int) -> Dict[str, float]:
    return _reliability.setdefault(port, {"accepted": 0.0, "rejected": 0.0, "unreachable": 0.0, "last_ok": 0.0})

def _mark_accept(port: int):
    r = _rel(port)
    r["accepted"] += 1.0
    r["last_ok"] = time.time()

def _mark_reject(port: int, kind="rejected"):
    r = _rel(port)
    r[kind] = r.get(kind, 0.0) + 1.0

# ---------------- node I/O ----------------
def _get_node_status(port: int, retries: int = 2, timeout: float = 2.0) -> Optional[Dict]:
    if _circuit_open(port):
        return None
    backoff = 0.2
    for _ in range(retries + 1):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                util = float(data.get("utilization", 1.0))
                temp = float(data.get("temperature", 0.0))  
                anomalous = bool(data.get("is_anomalous", False))
                out = {"port": port, "util": util, "temperature": temp, "anomalous": anomalous}
                _last_status[port] = out
                _record_success(port)
                return out
        except requests.RequestException:
            pass
        time.sleep(backoff)
        backoff *= 2
    _record_fail(port)
    return None

def _submit_job(port: int, job: Job, retries: int = 2, timeout: float = 2.0) -> Tuple[bool, str]:
    backoff = 0.2
    payload = job.dict()
    for _ in range(retries + 1):
        try:
            r = requests.post(f"http://127.0.0.1:{port}/submit", json=payload, timeout=timeout)
            if r.status_code == 200:
                resp = r.json()
                if resp.get("success"):
                    _record_success(port)
                    _mark_accept(port)
                    return True, "accepted"
                else:
                    _mark_reject(port, "rejected")
                    return False, resp.get("reason", "rejected")
        except requests.RequestException:
            pass
        time.sleep(backoff)
        backoff *= 2
    _record_fail(port)
    _mark_reject(port, "unreachable")
    return False, "unreachable"

# ---------------- scoring policy ----------------
def _reliability_score(port: int) -> float:
    r = _rel(port)
    tot = r["accepted"] + r["rejected"] + r["unreachable"]
    if tot == 0:
        return 0.8  # optimistic prior
    ok = r["accepted"]
    fail_pen = 0.7 * r["rejected"] + 1.0 * r["unreachable"]
    base = max(0.05, (ok - fail_pen) / (tot + 1e-6))
    recency = 0.1 if (time.time() - r["last_ok"] < 120) else 0.0
    return max(0.05, min(1.0, base + recency))

def _policy_score(status: Dict, job: Job) -> float:
    util = status["util"]
    capacity_surplus = max(0.0, 1.0 - (util + job.size))
    trust = _reliability_score(status["port"])
    # Weighted sum: surplus (capability), trust (reliability), and a small bias to lower util
    return 0.6 * capacity_surplus + 0.35 * trust + 0.05 * (1.0 - util)

# ---------------- API ----------------
@app.get("/health")
def health():
    return {"status": "ok", "policy": SCHED_POLICY}

@app.get("/nodes")
def list_nodes():
    return {
        "ports": sorted(NODE_PORTS),
        "last_status": _last_status,
        "circuit": _circuit,
        "drained": sorted(list(_drained)),
        "reliability": _reliability,
    }

@app.post("/nodes")
def add_node(req: NodeAdd):
    port = int(req.port)
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail="Node health check failed")
    except requests.RequestException:
        raise HTTPException(status_code=400, detail="Node unreachable")
    with _lock:
        if port not in NODE_PORTS:
            NODE_PORTS.append(port)
            _save_nodes(NODE_PORTS)
    return {"added": port}

@app.delete("/nodes/{port}")
def remove_node(port: int):
    with _lock:
        if port in NODE_PORTS:
            NODE_PORTS.remove(port)
            _save_nodes(NODE_PORTS)
            _circuit.pop(port, None)
            _last_status.pop(port, None)
            _reliability.pop(port, None)
            _drained.discard(port)
            DRAINED_FILE.write_text(json.dumps(sorted(list(_drained)), indent=2))
            return {"removed": port}
    raise HTTPException(status_code=404, detail="Node not found")

@app.put("/nodes/{port}/drain")
def drain_node(port: int):
    with _lock:
        if port not in NODE_PORTS:
            raise HTTPException(status_code=404, detail="Node not registered")
        _drained.add(port)
        DRAINED_FILE.write_text(json.dumps(sorted(list(_drained)), indent=2))
    return {"drained": port}

@app.put("/nodes/{port}/undrain")
def undrain_node(port: int):
    with _lock:
        _drained.discard(port)
        DRAINED_FILE.write_text(json.dumps(sorted(list(_drained)), indent=2))
    return {"undrained": port}

@app.get("/jobs")
def get_jobs(limit: int = 200):
    data = jobs_log[-limit:]
    return {"count": len(jobs_log), "returned": len(data), "jobs": data}

@app.post("/schedule")
def schedule(job: Job):
    global _rr_index
    with _lock:
        # Gather statuses skipping circuit-open
        statuses = []
        for p in list(NODE_PORTS):
            st = _get_node_status(p)
            if st:
                statuses.append(st)
        if not statuses:
            return {"success": False, "reason": "No nodes reachable (circuit open or down)"}

        # Exclude drained and anomalous
        statuses = [s for s in statuses if s["port"] not in _drained]
        healthy = [s for s in statuses if not s["anomalous"]]  # baseline health
        cool = [s for s in healthy if float(s.get("temperature", 0.0)) < HOT_TEMP]
        if not cool:
            raise HTTPException(status_code=429, detail="Thermal backpressure: nodes are hot", headers={"Retry-After": "5"})
        if not healthy:
            return {"success": False, "reason": "No healthy nodes available"}

        # Capacity filter
        candidates = [s for s in cool if (s["util"] + job.size) <= 1.0]
        if not candidates:
            raise HTTPException(status_code=429, detail="No healthy capacity available; retry later", headers={"Retry-After": "5"})

        chosen = None
        if SCHED_POLICY == "least":
            candidates.sort(key=lambda s: s["util"])  # least utilized
            chosen = candidates
        elif SCHED_POLICY == "roundrobin":
            # rotate through ports while respecting candidates set
            cand_ports = [s["port"] for s in candidates]
            ordered = sorted(cand_ports)
            if ordered:
                _rr_index = (_rr_index + 1) % len(ordered)
                rr_port = ordered[_rr_index]
                chosen = [next(s for s in candidates if s["port"] == rr_port)]
            else:
                chosen = candidates
        else:  # "score"
            candidates.sort(key=lambda s: _policy_score(s, job), reverse=True)
            chosen = candidates

        # Try candidates in order
        for s in chosen:
            ok, reason = _submit_job(s["port"], job)
            if ok:
                rec = {"job_id": job.job_id, "node": s["port"], "time": time.time(), "size": job.size, "type": job.type}
                jobs_log.append(rec)
                _save_jobs(jobs_log)
                return {"success": True, "node": s["port"], "policy": SCHED_POLICY}
        return {"success": False, "reason": "All candidates rejected (likely full/paused)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("control_plane:app", host="127.0.0.1", port=9000, log_level="info", reload=False)
