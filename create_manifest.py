# scripts/create_manifest.py
import json

manifest = {
    "model_file": "best_model.pth",
    "scaler_file": "scaler.pkl",
    "threshold_file": "threshold.json",
    "feature_order": ["utilization", "memory_usage", "power_usage", "temperature", "fan_speed", "deg_c", "btw_c", "eig_c", "clus"],
    "window_seconds": 15,
    "model_version": "v1.0.0",
    "notes": "Threshold from validation PR curve (beta=1.0)"
}
json.dump(manifest, open("deploy/manifest.json", "w"), indent=2)
print("Saved deploy/manifest.json")
