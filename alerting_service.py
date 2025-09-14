# alerting_service.py
import json, time, os, requests
from pathlib import Path

DATA_FILE = Path("training_data_2.json")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", None)  # optional
POLL_SEC = int(os.getenv("ALERT_POLL", "10"))

print("[Alerting] Watching training_data_2.json for anomalies...")

last_len = 0
while True:
    try:
        if not DATA_FILE.exists():
            time.sleep(POLL_SEC)
            continue
        data = json.loads(DATA_FILE.read_text())
        if len(data) <= last_len:
            time.sleep(POLL_SEC)
            continue
        new = data[last_len:]
        last_len = len(data)
        # Emit alerts for new anomalies
        for rec in new:
            if rec.get("is_anomalous"):
                msg = (
                    f"ANOMALY node={rec.get('node_port','?')} util={rec.get('utilization',0):.2f} "
                    f"temp={rec.get('temperature',0):.1f} reasons={rec.get('anomaly_reasons',[])}"
                )
                print("[ALERT]", msg)
                if SLACK_WEBHOOK:
                    try:
                        requests.post(SLACK_WEBHOOK, json={"text": msg}, timeout=5)
                    except requests.RequestException:
                        pass
    except Exception as e:
        print("[Alerting] error:", e)
    time.sleep(POLL_SEC)
