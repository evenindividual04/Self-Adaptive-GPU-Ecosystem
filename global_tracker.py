# global_tracker.py

import threading

_lock = threading.Lock()
_total = 0
_anomalies = 0

def record_anomaly(is_anom):
    """Call this from each node to update anomaly stats."""
    global _total, _anomalies
    with _lock:
        _total += 1
        if is_anom:
            _anomalies += 1

def current_ratio():
    """Return the fraction of recent entries that were anomalous."""
    with _lock:
        if _total == 0:
            return 0.0
        return _anomalies / _total

def reset():
    """Reset anomaly tracking stats; call optionally if desired."""
    global _total, _anomalies
    with _lock:
        _total = 0
        _anomalies = 0
