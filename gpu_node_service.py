# gpu_node_service.py (with pause/resume/reset control actions)
from fastapi import FastAPI
from pydantic import BaseModel
import threading, time, random, argparse
import global_tracker as gt

app = FastAPI()

# Node base metrics
BASE_STATE = {
    "temperature": 65.0,
    "power_usage": 150.0,
    "fan_speed": 2000.0,
    "memory_usage": 0.0,
}

state = {
    "utilization": 0.0,
    "jobs": [],
    "job_count": 0,
    "temperature": BASE_STATE["temperature"],
    "power_usage": BASE_STATE["power_usage"],
    "memory_usage": BASE_STATE["memory_usage"],
    "fan_speed": BASE_STATE["fan_speed"],
    "is_anomalous": False,
    "paused": False,  # node-level acceptance switch
    # NEW: bounded buffer of recently completed jobs
    "recent_jobs": [],  # [{"job_id": str, "start": float, "end": float}], last 20
}

lock = threading.Lock()

class Job(BaseModel):
    job_id: str
    size: float
    type: str

# Hardware fluctuation + controlled anomaly injection
def hardware_fluctuation():
    with lock:
        state["is_anomalous"] = False
        # Small random fluctuations
        state["temperature"] += random.uniform(-0.2, 0.2)
        state["power_usage"] += random.uniform(-0.5, 0.5)
        state["fan_speed"] += random.uniform(-5, 5)
        state["memory_usage"] = min(max(state["memory_usage"], 0.0), 1.0)

        # Check global anomaly ratio and inject anomalies
        if gt.current_ratio() < 0.10:  # max 10% anomalies
            inject = random.random() < 0.07  # ~7% chance per cycle
            if inject:
                state["is_anomalous"] = True
                anomaly_type = random.choice(["util", "memory", "power"])
                if anomaly_type == "util":
                    delta = random.uniform(0.1, 0.3)
                    state["utilization"] = min(1.0, state["utilization"] + delta)
                    state["temperature"] += delta * 10
                    state["power_usage"] += delta * 20
                elif anomaly_type == "memory":
                    delta = random.uniform(0.1, 0.3)
                    state["memory_usage"] = min(1.0, state["memory_usage"] + delta)
                    state["fan_speed"] += delta * 150
                elif anomaly_type == "power":
                    state["power_usage"] += random.uniform(10, 25)
                    state["temperature"] += random.uniform(2, 5)
                    state["utilization"] = min(1.0, state["utilization"] + random.uniform(0.05, 0.1))

        # Record anomaly in global tracker
        gt.record_anomaly(state["is_anomalous"])

        # Decay metrics slowly back to base
        decay_factor = 0.01
        for key in ["temperature", "power_usage", "fan_speed", "memory_usage"]:
            state[key] += (BASE_STATE[key] - state[key]) * decay_factor
        # NEW: simple cooling effect from higher fan speed (more fan -> lower temp)
        # Models that higher fan speeds (above base) pull temperature down gradually.
        fan_boost = max(0.0, (state["fan_speed"] - BASE_STATE["fan_speed"])) / 3000.0
        if fan_boost > 0.0:
            state["temperature"] -= 0.4 * fan_boost  # gentle cooling per tick

    threading.Timer(2, hardware_fluctuation).start()

hardware_fluctuation()

# Job runner with automatic freeing
def job_runner():
    with lock:
        now = time.time()
        for job in state["jobs"][:]:
            if now >= job["end_time"]:
                state["utilization"] -= job["size"]
                state["utilization"] = max(0.0, state["utilization"])
                # NEW: record completion (keep last 20)
                state["recent_jobs"].append({
                    "job_id": job["job_id"],
                    "start": job["start_time"],
                    "end": now
                })
                if len(state["recent_jobs"]) > 20:
                    state["recent_jobs"] = state["recent_jobs"][-20:]
                state["jobs"].remove(job)
        # Free a job if utilization dangerously high
        if state["utilization"] > 0.95 and state["jobs"]:
            job_to_free = random.choice(state["jobs"])
            state["utilization"] -= job_to_free["size"]
            state["utilization"] = max(0.0, state["utilization"])
            state["jobs"].remove(job_to_free)
    threading.Timer(1, job_runner).start()

job_runner()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    with lock:
        metrics_data = {
            "utilization": state["utilization"],
            "jobs": [j["job_id"] for j in state["jobs"]],
            "job_count": len(state["jobs"]),
            "temperature": state["temperature"],
            "power_usage": state["power_usage"],
            "memory_usage": state["memory_usage"],
            "fan_speed": state["fan_speed"],
            "is_anomalous": state["is_anomalous"],
            "timestamp": time.time(),
            "note": "Direct JSON metrics; scalable for monitoring",
            # NEW: expose completed short jobs so bursts remain visible post-poll
            "recent_jobs": list(state["recent_jobs"]),
        }
        if state["is_anomalous"]:
            metrics_data["anomaly_reason"] = "High temperature"
        return metrics_data

# NEW: control actions
@app.post("/pause")
def pause():
    with lock:
        state["paused"] = True
    return {"paused": True}

@app.post("/resume")
def resume():
    with lock:
        state["paused"] = False
    return {"paused": False}

@app.post("/fan")
def set_fan(speed: float = BASE_STATE["fan_speed"]):
    # NEW: allow monitor/scheduler to bump or normalize fan speed (simulated)
    with lock:
        s = float(speed)
        s = max(1000.0, min(6000.0, s))  # clamp to a safe simulated range
        state["fan_speed"] = s
        return {"ok": True, "fan_speed": state["fan_speed"]}

@app.post("/reset")
def reset():
    with lock:
        state["utilization"] = 0.0
        state["jobs"] = []
        state["job_count"] = 0
        state["temperature"] = BASE_STATE["temperature"]
        state["power_usage"] = BASE_STATE["power_usage"]
        state["memory_usage"] = BASE_STATE["memory_usage"]
        state["fan_speed"] = BASE_STATE["fan_speed"]
        state["is_anomalous"] = False
    return {"reset": True}

@app.post("/submit")
def submit(job: Job):
    with lock:
        if state.get("paused"):
            return {"success": False, "reason": "paused"}
        if state["utilization"] + job.size > 1.0:
            return {"success": False, "reason": "Node full"}
        duration = random.uniform(15, 60)
        state["jobs"].append({
            "job_id": job.job_id,
            "size": job.size,
            "type": job.type,
            "start_time": time.time(),            # NEW
            "end_time": time.time() + duration
        })
        state["utilization"] += job.size
        state["job_count"] += 1
    return {"success": True, "duration": duration}

if __name__ == "__main__":
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args, _ = parser.parse_known_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
