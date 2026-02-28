#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Job:
    name: str
    cmd: List[str]
    log_path: Path


def build_jobs(repo_root: Path, run_dir: Path) -> List[Job]:
    py = sys.executable
    logs_root = run_dir / "logs"
    artifacts_root = run_dir / "artifacts"
    logs_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    jobs: List[Job] = []

    # Strict paper baseline: Burgers 1D
    jobs.append(
        Job(
            name="burgers_naspinn_paper",
            cmd=[
                py,
                "NAS_PINNs_burgers.py",
                "--paper-protocol",
                "--paper-nus",
                "0.1,0.07,0.04",
                "--repeats",
                "5",
                "--train-nt",
                "21",
                "--train-nx",
                "250",
                "--test-nt",
                "21",
                "--test-nx",
                "500",
                "--epochs",
                "15000",
                "--stage",
                "lbfgs",
                "--seed",
                "42",
                "--save-dir",
                str(artifacts_root / "burgers" / "naspinn"),
            ],
            log_path=logs_root / "burgers" / "naspinn.log",
        )
    )

    # Strict paper baseline: Advection 1D
    jobs.append(
        Job(
            name="advection_naspinn_paper",
            cmd=[
                py,
                "NAS_PINNs_advection.py",
                "--paper-protocol",
                "--paper-betas",
                "1.0,0.5,0.1",
                "--repeats",
                "5",
                "--epochs",
                "12000",
                "--layers",
                "4",
                "--base-neurons",
                "128",
                "--train-nt",
                "40",
                "--train-nx",
                "120",
                "--test-nt",
                "40",
                "--test-nx",
                "120",
                "--stage",
                "lbfgs",
                "--seed",
                "42",
                "--save-dir",
                str(artifacts_root / "advection" / "naspinn"),
            ],
            log_path=logs_root / "advection" / "naspinn.log",
        )
    )

    # Strict paper baseline: Burgers 2D
    jobs.append(
        Job(
            name="burgers2d_naspinn_paper",
            cmd=[
                py,
                "NAS_PINNs_burgers2d.py",
                "--paper-protocol",
                "--repeats",
                "5",
                "--epochs",
                "12000",
                "--layers",
                "5",
                "--base-neurons",
                "128",
                "--train-nt",
                "20",
                "--train-nx",
                "25",
                "--train-ny",
                "25",
                "--test-nt",
                "41",
                "--test-nx",
                "500",
                "--test-ny",
                "500",
                "--slice-times",
                "0,1,2",
                "--stage",
                "lbfgs",
                "--seed",
                "42",
                "--save-dir",
                str(artifacts_root / "burgers2d" / "naspinn"),
            ],
            log_path=logs_root / "burgers2d" / "naspinn.log",
        )
    )

    for job in jobs:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)

    return jobs


def run_job(job: Job, repo_root: Path) -> tuple[int, float]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    start = time.perf_counter()
    with job.log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n==== {dt.datetime.now().isoformat()} :: {job.name} ====\n")
        logf.write(f"CMD: {shlex.join(job.cmd)}\n\n")
        proc = subprocess.run(
            job.cmd,
            cwd=str(repo_root),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return proc.returncode, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict paper NAS-PINN baseline runner")
    parser.add_argument("--run-dir", type=str, default=None, help="optional output run directory")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--stop-on-error", action="store_true", help="stop at first failed job")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = repo_root / "results" / "strict_paper_baseline" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(repo_root, run_dir)
    print(f"Run directory: {run_dir}")
    print(f"Job count    : {len(jobs)}")

    if args.dry_run:
        for idx, job in enumerate(jobs, start=1):
            print(f"[{idx}/{len(jobs)}] {job.name}")
            print(f"  log: {job.log_path}")
            print(f"  cmd: {shlex.join(job.cmd)}")
        return

    summary_rows = []
    failed = 0
    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {job.name} ...")
        code, elapsed = run_job(job, repo_root)
        status = "ok" if code == 0 else "failed"
        print(f"  -> {status} ({elapsed:.1f}s) | log: {job.log_path}")
        summary_rows.append(
            {
                "job": job.name,
                "status": status,
                "exit_code": code,
                "elapsed_seconds": round(elapsed, 3),
                "log_path": str(job.log_path),
                "command": shlex.join(job.cmd),
            }
        )
        if code != 0:
            failed += 1
            if args.stop_on_error:
                print("Stopping due to --stop-on-error.")
                break

    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["job", "status", "exit_code", "elapsed_seconds", "log_path", "command"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    summary_json = run_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    ok = sum(1 for r in summary_rows if r["status"] == "ok")
    print("\nFinished.")
    print(f"Success: {ok}, Failed: {failed}")
    print(f"Summary CSV : {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
