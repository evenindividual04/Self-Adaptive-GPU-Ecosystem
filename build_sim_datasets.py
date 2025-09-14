# build_sim_datasets.py
from data_pipeline import load_sim, build_snapshots, make_graphs, split_and_scale, save_datasets

JSON_PATH = "training_data_2.json"   # produced by monitoring_service.py
WINDOW_S  = 15                       # match the monitor’s poll interval

rows = load_sim(JSON_PATH)
snaps = build_snapshots(rows, window_s=WINDOW_S)
graphs, times = make_graphs(snaps)
train, val, test, scaler = split_and_scale(graphs, times)
save_datasets(train, val, test, scaler, out_dir="datasets", name="sim")
print(f"graphs: train={len(train)}, val={len(val)}, test={len(test)}")
