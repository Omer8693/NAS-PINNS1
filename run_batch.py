#!/usr/bin/env python3
# This file is a renamed and updated version of run_poisson_batch.py for batch jobs
# All logic is identical, only filename and Poisson domain jobs are updated

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List

@dataclass
class Job:
    name: str
    cmd: List[str]

def build_jobs(include_pso: bool, quick: bool) -> List[Job]:
    python = sys.executable
    burgers_exact_nu = "0.01"
    short_flags = ["--epochs", "1", "--skip-lbfgs"] if quick else []
    jobs = [
        Job("burgers_naspinn_single", [python, "NAS_PINNs_burgers.py", "--nu", burgers_exact_nu, *short_flags]),
        Job("burgers_nsga2_single", [python, "NAS_PINNs_burgers_nsga2.py", "--nu", burgers_exact_nu, *short_flags]),
        Job("burgers_nsga3_single", [python, "NAS_PINNs_burgers_nsga3.py", "--nu", burgers_exact_nu, *short_flags]),
        Job("burgers_bayesian_single", [python, "NAS_PINNs_burgers_bayesian.py", "--nu", burgers_exact_nu, *short_flags]),
        Job("burgers_naspinn_multi", [python, "NAS_PINNs_burgers.py", "--multi-nu", "--nu-list", "0.01,0.04,0.07", *short_flags]),
        Job("burgers_nsga2_multi", [python, "NAS_PINNs_burgers_nsga2.py", "--multi-nu", "--nu-list", "0.01,0.04,0.07", *short_flags]),
        Job("burgers_nsga3_multi", [python, "NAS_PINNs_burgers_nsga3.py", "--multi-nu", "--nu-list", "0.01,0.04,0.07", *short_flags]),
        Job("burgers_bayesian_multi", [python, "NAS_PINNs_burgers_bayesian.py", "--multi-nu", "--nu-list", "0.01,0.04,0.07", *short_flags]),
        # New Poisson domain jobs
        Job("poisson_rectangular", [python, "poisson_domains/rectangular.py"]),
        Job("poisson_circle", [python, "poisson_domains/circle.py"]),
        Job("poisson_lshape", [python, "poisson_domains/lshape.py"]),
        Job("poisson_flower", [python, "poisson_domains/flower.py"]),
    ]
    return jobs

def run_job(job: Job, run_dir: str, timeout: int):
    started_at = time.time()
    log_path = os.path.join(run_dir, f"{job.name}.log")
    # Poisson domain jobs: print stdout live to terminal and save logs
    if job.name.startswith("poisson_"):
        import subprocess
        logf = open(log_path, "w", encoding="utf-8")
        logf.write(f"[START] {dt.datetime.now().isoformat()}\n")
        logf.write("COMMAND: " + " ".join(job.cmd) + "\n\n")
        logf.flush()
        try:
            proc = subprocess.Popen(job.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            code = None
            status = "ok"
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    print(line, end="")
                    logf.write(line)
                    logf.flush()
            code = proc.wait()
            if code != 0:
                status = "failed"
        except subprocess.TimeoutExpired:
            code = -9
            status = "timeout"
            logf.write("\n[TIMEOUT]\n")
        logf.close()
    else:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"[START] {dt.datetime.now().isoformat()}\n")
            logf.write("COMMAND: " + " ".join(job.cmd) + "\n\n")
            logf.flush()
            try:
                proc = subprocess.run(
                    job.cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout if timeout > 0 else None,
                    check=False,
                )
                code = proc.returncode
                status = "ok" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                code = -9
                status = "timeout"
                logf.write("\n[TIMEOUT]\n")
    elapsed = time.time() - started_at
    return {
        "job": job.name,
        "status": status,
        "return_code": code,
        "elapsed_sec": round(elapsed, 2),
        "log": log_path,
    }

def main():
    parser = argparse.ArgumentParser(description="Run all project scripts sequentially overnight.")
    parser.add_argument("--quick", action="store_true", help="quick smoke mode (very short epochs)")
    parser.add_argument("--timeout", type=int, default=0, help="per-job timeout in seconds (0 = unlimited)")
    parser.add_argument("--stop-on-error", action="store_true", help="stop if a job fails")
    parser.add_argument("--run-dir", type=str, default=None, help="existing run directory to resume from")
    args = parser.parse_args()
    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results", "overnight_runs", stamp)
        os.makedirs(run_dir, exist_ok=True)
    jobs = build_jobs(include_pso=False, quick=args.quick)
    summary = []
    print(f"Starting batch run with {len(jobs)} jobs")
    print(f"Logs: {run_dir}")
    summary_path = os.path.join(run_dir, "summary.csv")
    completed_jobs = set()
    summary_rows = []
    # Mevcut run_dir içindeki tamamlanmış işleri kontrol et
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            summary_rows = list(reader)
            seen = set()
            for row in summary_rows:
                job_name = row["job"]
                if job_name not in seen and row["status"] == "ok":
                    completed_jobs.add(job_name)
                    seen.add(job_name)
        summary = summary_rows.copy()

    # Ayrıca önceki overnight_runs/20260222_155103/summary.csv dosyasındaki tamamlanmış işleri kontrol et
    prev_summary_path = os.path.join("results", "overnight_runs", "20260222_155103", "summary.csv")
    if os.path.exists(prev_summary_path):
        with open(prev_summary_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                job_name = row["job"]
                if row["status"] == "ok":
                    completed_jobs.add(job_name)
    # Only run jobs that are not yet completed, in order
    # Eğer tüm burgers işleri tamamlanmışsa, sadece Poisson işleriyle başla
    burgers_jobs = [job for job in jobs if job.name.startswith("burgers_")]
    burgers_completed = all(job.name in completed_jobs for job in burgers_jobs)
    start_idx = 0
    if burgers_completed:
        # Find the index of the first Poisson job
        for i, job in enumerate(jobs):
            if job.name.startswith("poisson_"):
                start_idx = i
                break
    for idx, job in enumerate(jobs[start_idx:], start=start_idx+1):
        if job.name in completed_jobs:
            print(f"[{idx}/{len(jobs)}] {job.name} ... already completed, skipping.")
            continue
        print(f"[{idx}/{len(jobs)}] {job.name} ... running.")
        result = run_job(job, run_dir, timeout=args.timeout)
        summary.append(result)
        print(f"    -> {result['status']} ({result['elapsed_sec']}s)")
        if result["status"] != "ok":
            log_path = result["log"]
            print(f"Job failed: {job.name}")
            print(f"Log file: {log_path}")
            try:
                with open(log_path, "r", encoding="utf-8") as logf:
                    log_lines = logf.readlines()[-20:]
                    print("Last 20 lines:")
                    for line in log_lines:
                        print(line.strip())
            except Exception as e:
                print(f"Could not read log: {e}")
            input("After fixing, press Enter to continue.")
            print(f"Retrying: {job.name}")
            result_retry = run_job(job, run_dir, timeout=args.timeout)
            summary.append(result_retry)
            print(f"    -> {result_retry['status']} ({result_retry['elapsed_sec']}s)")
            if result_retry["status"] != "ok":
                print("Retry also failed. Stopping script.")
                break
    csv_path = os.path.join(run_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["job", "status", "return_code", "elapsed_sec", "log"])
        writer.writeheader()
        writer.writerows(summary)
    json_path = os.path.join(run_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    ok = sum(1 for s in summary if s["status"] == "ok")
    failed = sum(1 for s in summary if s["status"] != "ok")
    print("\nFinished.")
    print(f"Success: {ok}, Failed/Timeout: {failed}")
    print(f"Summary CSV: {csv_path}")

if __name__ == "__main__":
    main()
