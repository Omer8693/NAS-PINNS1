#!/usr/bin/env python3
"""Run a paper-parameter baseline suite in an isolated folder.

This runner does not modify any existing training code. It only executes
existing entry scripts with fixed baseline arguments and writes outputs/logs
under `results/fair_baseline_paper/...`.
"""

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


EQUATIONS = ("burgers", "poisson", "advection", "burgers2d")
METHODS = ("naspinn", "nsga2", "nsga3", "bayesian")


@dataclass
class Job:
    equation: str
    method: str
    script: str
    cmd: List[str]
    log_path: Path

    @property
    def name(self) -> str:
        return f"{self.equation}_{self.method}"


def parse_csv_arg(value: str, allowed: tuple[str, ...], flag_name: str) -> List[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    invalid = [x for x in items if x not in allowed]
    if invalid:
        raise ValueError(f"Invalid {flag_name} values: {invalid}. Allowed: {list(allowed)}")
    return items


def build_command(equation: str, method: str, out_dir: Path) -> tuple[str, List[str]]:
    # Fixed paper-baseline seed across the suite.
    seed = "42"
    python = sys.executable

    if equation == "burgers":
        if method == "naspinn":
            script = "NAS_PINNs_burgers.py"
            cmd = [
                python,
                script,
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
                seed,
                "--save-dir",
                str(out_dir),
            ]
        elif method == "nsga2":
            script = "NAS_PINNs_burgers_nsga2.py"
            cmd = [
                python,
                script,
                "--multi-nu",
                "--nu-list",
                "0.1,0.07,0.04",
                "--epochs",
                "15000",
                "--seed",
                seed,
                "--save-dir",
                str(out_dir),
            ]
        elif method == "nsga3":
            script = "NAS_PINNs_burgers_nsga3.py"
            cmd = [
                python,
                script,
                "--multi-nu",
                "--nu-list",
                "0.1,0.07,0.04",
                "--epochs",
                "15000",
                "--seed",
                seed,
                "--save-dir",
                str(out_dir),
            ]
        elif method == "bayesian":
            script = "NAS_PINNs_burgers_bayesian.py"
            cmd = [
                python,
                script,
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
                "--seed",
                seed,
                "--save-dir",
                str(out_dir),
            ]
        else:
            raise ValueError(f"Unsupported method for {equation}: {method}")

    elif equation == "poisson":
        domains = "rectangular,circle,lshape,flower,annulus"
        if method == "naspinn":
            script = "NAS_PINNs_poisson.py"
            cmd = [
                python,
                script,
                "--multi-domain",
                "--domain-list",
                domains,
                "--epochs",
                "12000",
                "--stage",
                "lbfgs",
                "--seed",
                seed,
                "--save-dir",
                str(out_dir),
            ]
        elif method == "nsga2":
            script = "NAS_PINNs_poisson_nsga2.py"
            cmd = [
                python,
                script,
                "--multi-domain",
                "--domain-list",
                domains,
                "--epochs",
                "12000",
                "--proxy-epochs",
                "600",
                "--skip-pso",
                "--seed",
                seed,
                "--save-dir",
                str(out_dir),
            ]
        elif method == "nsga3":
            script = "NAS_PINNs_poisson_nsga3.py"
            cmd = [
                python,
                script,
                "--multi-domain",
                "--domain-list",
                domains,
                "--epochs",
                "12000",
                "--proxy-epochs",
                "600",
                "--skip-pso",
                "--seed",
                seed,
                "--save-dir",
                str(out_dir),
            ]
        elif method == "bayesian":
            script = "NAS_PINNs_poisson_bayesian.py"
            cmd = [
                python,
                script,
                "--multi-domain",
                "--domain-list",
                domains,
                "--epochs",
                "12000",
                "--proxy-epochs",
                "700",
                "--bo-init-points",
                "4",
                "--bo-iters",
                "12",
                "--skip-pso",
                "--seed",
                seed,
                "--save-dir",
                str(out_dir),
            ]
        else:
            raise ValueError(f"Unsupported method for {equation}: {method}")

    elif equation == "advection":
        if method == "naspinn":
            script = "NAS_PINNs_advection.py"
            cmd = [
                python,
                script,
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
                seed,
                "--save-dir",
                str(out_dir),
            ]
        elif method == "nsga2":
            script = "NAS_PINNs_advection_nsga2.py"
            cmd = [python, script, "--profile", "paper_baseline", "--paper-protocol", "--save-dir", str(out_dir)]
        elif method == "nsga3":
            script = "NAS_PINNs_advection_nsga3.py"
            cmd = [python, script, "--profile", "paper_baseline", "--paper-protocol", "--save-dir", str(out_dir)]
        elif method == "bayesian":
            script = "NAS_PINNs_advection_bayesian.py"
            cmd = [python, script, "--profile", "paper_baseline", "--paper-protocol", "--save-dir", str(out_dir)]
        else:
            raise ValueError(f"Unsupported method for {equation}: {method}")

    elif equation == "burgers2d":
        if method == "naspinn":
            script = "NAS_PINNs_burgers2d.py"
            cmd = [
                python,
                script,
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
                seed,
                "--save-dir",
                str(out_dir),
            ]
        elif method == "nsga2":
            script = "NAS_PINNs_burgers2d_nsga2.py"
            cmd = [python, script, "--profile", "paper_baseline", "--paper-protocol", "--save-dir", str(out_dir)]
        elif method == "nsga3":
            script = "NAS_PINNs_burgers2d_nsga3.py"
            cmd = [python, script, "--profile", "paper_baseline", "--paper-protocol", "--save-dir", str(out_dir)]
        elif method == "bayesian":
            script = "NAS_PINNs_burgers2d_bayesian.py"
            cmd = [python, script, "--profile", "paper_baseline", "--paper-protocol", "--save-dir", str(out_dir)]
        else:
            raise ValueError(f"Unsupported method for {equation}: {method}")

    else:
        raise ValueError(f"Unsupported equation: {equation}")

    return script, cmd


def build_jobs(repo_root: Path, run_dir: Path, equations: List[str], methods: List[str]) -> List[Job]:
    jobs: List[Job] = []
    logs_root = run_dir / "logs"
    artifacts_root = run_dir / "artifacts"
    logs_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    for equation in equations:
        for method in methods:
            out_dir = artifacts_root / equation / method
            out_dir.mkdir(parents=True, exist_ok=True)
            script, cmd = build_command(equation, method, out_dir)
            script_path = repo_root / script
            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")
            log_path = logs_root / equation / f"{method}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            jobs.append(
                Job(
                    equation=equation,
                    method=method,
                    script=script,
                    cmd=cmd,
                    log_path=log_path,
                )
            )
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
    parser = argparse.ArgumentParser(description="Isolated paper-baseline runner")
    parser.add_argument(
        "--equations",
        type=str,
        default="burgers,poisson,advection,burgers2d",
        help="comma list: burgers,poisson,advection,burgers2d",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="naspinn,nsga2,nsga3,bayesian",
        help="comma list: naspinn,nsga2,nsga3,bayesian",
    )
    parser.add_argument("--run-dir", type=str, default=None, help="optional explicit run directory")
    parser.add_argument("--stop-on-error", action="store_true", help="stop at first failed job")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    equations = parse_csv_arg(args.equations, EQUATIONS, "--equations")
    methods = parse_csv_arg(args.methods, METHODS, "--methods")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = repo_root / "results" / "fair_baseline_paper" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(repo_root, run_dir, equations, methods)

    print(f"Run directory: {run_dir}")
    print(f"Equation set : {','.join(equations)}")
    print(f"Method set   : {','.join(methods)}")
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
                "equation": job.equation,
                "method": job.method,
                "script": job.script,
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
            fieldnames=[
                "equation",
                "method",
                "script",
                "status",
                "exit_code",
                "elapsed_seconds",
                "log_path",
                "command",
            ],
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
