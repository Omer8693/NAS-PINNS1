import os
import requests

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Omer8693/NAS-PINNS1/main/results/overnight_runs"
run_folder = "20260222_155103"
log_files = [
    "burgers_bayesian_multi.log",
    "burgers_bayesian_single.log",
    "burgers_naspinn_multi.log",
    "burgers_naspinn_single.log",
    "burgers_nsga2_multi.log",
    "burgers_nsga2_pso.log",
    "burgers_nsga2_single.log",
    "burgers_nsga3_multi.log",
    "burgers_nsga3_pso.log",
    "burgers_nsga3_single.log"
]

local_dir = f"results/overnight_runs/{run_folder}"
os.makedirs(local_dir, exist_ok=True)
for log_file in log_files:
    url = f"{GITHUB_RAW_BASE}/{run_folder}/{log_file}"
    local_path = os.path.join(local_dir, log_file)
    print(f"Downloading {url} -> {local_path}")
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"Saved: {local_path}")
    else:
        print(f"Not found: {url}")

print("Eksik log dosyaları geri yüklendi.")