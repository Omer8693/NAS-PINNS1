#!/usr/bin/env python3
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
        Job("poisson_naspinn_single", [python, "NAS_PINNs_poisson.py", *short_flags]),
        Job("poisson_nsga2_single", [python, "NAS_PINNs_poisson_nsga2.py", *short_flags]),
        Job("poisson_nsga3_single", [python, "NAS_PINNs_poisson_nsga3.py", *short_flags]),
        Job("poisson_bayesian_single", [python, "NAS_PINNs_poisson_bayesian.py", *short_flags]),
        Job("burgers_naspinn_multi", [python, "NAS_PINNs_burgers.py", "--multi-nu", "--nu-list", "0.01,0.04,0.07", *short_flags]),
        Job("burgers_nsga2_multi", [python, "NAS_PINNs_burgers_nsga2.py", "--multi-nu", "--nu-list", "0.01,0.04,0.07", *short_flags]),
        Job("burgers_nsga3_multi", [python, "NAS_PINNs_burgers_nsga3.py", "--multi-nu", "--nu-list", "0.01,0.04,0.07", *short_flags]),
        Job("burgers_bayesian_multi", [python, "NAS_PINNs_burgers_bayesian.py", "--multi-nu", "--nu-list", "0.01,0.04,0.07", *short_flags]),
        Job("poisson_naspinn_multi", [python, "NAS_PINNs_poisson.py", "--multi-seed", "--seed-list", "42,43,44", *short_flags]),
        Job("poisson_nsga2_multi", [python, "NAS_PINNs_poisson_nsga2.py", "--multi-seed", "--seed-list", "42,43,44", *short_flags]),
        Job("poisson_nsga3_multi", [python, "NAS_PINNs_poisson_nsga3.py", "--multi-seed", "--seed-list", "42,43,44", *short_flags]),
        Job("poisson_bayesian_multi", [python, "NAS_PINNs_poisson_bayesian.py", "--multi-seed", "--seed-list", "42,43,44", *short_flags]),
    ]

    if include_pso:
        pso_flags = ["--generations", "1", "--pop-size", "1"] if quick else []
        jobs.extend(
            [
                Job("burgers_naspinn_pso", [python, "NAS_PINNs_burgers_pso.py", *pso_flags]),
                Job("burgers_nsga2_pso", [python, "NAS_PINNs_burgers_nsga2_pso.py", *pso_flags]),
                Job("burgers_nsga3_pso", [python, "NAS_PINNs_burgers_nsga3_pso.py", *pso_flags]),
                Job("burgers_bayesian_pso", [python, "NAS_PINNs_burgers_bayesian_pso.py", *pso_flags]),
                Job("poisson_naspinn_pso", [python, "NAS_PINNs_poisson_pso.py", *pso_flags]),
                Job("poisson_nsga2_pso", [python, "NAS_PINNs_poisson_nsga2_pso.py", *pso_flags]),
                Job("poisson_nsga3_pso", [python, "NAS_PINNs_poisson_nsga3_pso.py", *pso_flags]),
                Job("poisson_bayesian_pso", [python, "NAS_PINNs_poisson_bayesian_pso.py", *pso_flags]),
            ]
        )

    return jobs


def run_job(job: Job, run_dir: str, timeout: int):
    started_at = time.time()
    log_path = os.path.join(run_dir, f"{job.name}.log")

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
    # Eğer summary.csv ve summary.json yoksa, run_dir içindeki log dosyalarından job durumlarını yeniden oluştur
    def repair_summary_from_logs(run_dir, jobs):
        summary = []
        for job in jobs:
            log_path = os.path.join(run_dir, f"{job.name}.log")
            status = "not-started"
            code = None
            elapsed = None
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as logf:
                    lines = logf.readlines()
                    if any("[START]" in l for l in lines):
                        if any("COMMAND:" in l for l in lines):
                            if any("[TIMEOUT]" in l for l in lines):
                                status = "timeout"
                            elif any("Traceback" in l for l in lines):
                                status = "failed"
                            else:
                                status = "ok"
                        else:
                            status = "failed"
                    else:
                        status = "not-started"
                # Elapsed time ve return code logdan alınamıyor, None bırakıyoruz
            summary.append({
                "job": job.name,
                "status": status,
                "return_code": code,
                "elapsed_sec": elapsed,
                "log": log_path,
            })
        return summary


    parser = argparse.ArgumentParser(description="Run all project scripts sequentially overnight.")
    parser.add_argument("--no-pso", action="store_true", help="exclude PSO runner scripts")
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


    jobs = build_jobs(include_pso=not args.no_pso, quick=args.quick)
    summary = []

    print(f"Starting overnight run with {len(jobs)} jobs")
    print(f"Logs: {run_dir}")

    # summary.csv varsa, sadece eksik/hatalı işleri tekrar çalıştır
    summary_path = os.path.join(run_dir, "summary.csv")
    completed_jobs = set()
    summary_rows = []
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            summary_rows = list(reader)
            # Sadece ilk 'ok' olan job'ı completed_jobs'a ekle
            seen = set()
            for row in summary_rows:
                job_name = row["job"]
                if job_name not in seen and row["status"] == "ok":
                    completed_jobs.add(job_name)
                    seen.add(job_name)
        summary = summary_rows.copy()
    elif not os.path.exists(summary_path):
        # summary.csv yoksa, loglardan onar
        summary = repair_summary_from_logs(run_dir, jobs)
        for row in summary:
            if row["status"] == "ok":
                completed_jobs.add(row["job"])

    # Sadece tamamlanmamış işleri sırayla çalıştır
    for idx, job in enumerate(jobs, 1):
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
            print(f"Hata logu: {log_path}")
            try:
                with open(log_path, "r", encoding="utf-8") as logf:
                    log_lines = logf.readlines()[-20:]
                    print("Son 20 satır:")
                    for line in log_lines:
                        print(line.strip())
            except Exception as e:
                print(f"Log okunamadı: {e}")
            input("Onarım yaptıktan sonra Enter'a basın ve devam edilecek.")
            print(f"Tekrar deneniyor: {job.name}")
            result_retry = run_job(job, run_dir, timeout=args.timeout)
            summary.append(result_retry)
            print(f"    -> {result_retry['status']} ({result_retry['elapsed_sec']}s)")
            if result_retry["status"] != "ok":
                print("Tekrar deneme de başarısız. Script durduruluyor.")
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
