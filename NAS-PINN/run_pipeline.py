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
    family: str


def build_jobs(args, seed, run_dir, rep_idx) -> List[Job]:
    python = sys.executable
    jobs: List[Job] = []

    burgers_flags = ["--multi-nu", "--nu-list", args.nu_list, "--seed", str(seed)]
    poisson_flags = ["--multi-domain", "--domain-list", args.domain_list, "--seed", str(seed)]
    burgers_stage_flags = ["--skip-lbfgs"] if args.burgers_stage == "adam" else []
    poisson_stage_lbfgs_flags = ["--skip-lbfgs"] if args.poisson_stage == "adam" else []
    poisson_stage_pso_flags = ["--skip-pso"] if args.poisson_stage != "pso" else []

    if args.quick:
        burgers_quick = ["--epochs", "1000", "--skip-lbfgs"]
        poisson_quick = ["--epochs", "1200"]
        nsga_quick = ["--proxy-epochs", "80", "--n-gen", "3"]
        nsga2_quick_extra = ["--pop-size", "8"]
        nsga3_quick_extra = ["--ref-partitions", "6"]
        bayes_quick = ["--bo-init-points", "2", "--bo-iters", "3"]
    else:
        burgers_quick = []
        poisson_quick = []
        nsga_quick = []
        nsga2_quick_extra = []
        nsga3_quick_extra = []
        bayes_quick = []

    rep_root = os.path.join(run_dir, "artifacts", f"rep_{rep_idx:02d}")

    jobs.extend(
        [
            Job(
                "burgers_naspinn",
                [
                    python,
                    "NAS_PINNs_burgers.py",
                    *burgers_flags,
                    "--stage",
                    args.burgers_stage,
                    *burgers_quick,
                    "--save-dir",
                    os.path.join(rep_root, "burgers", "naspinn"),
                ],
                "burgers",
            ),
            Job(
                "burgers_nsga2",
                [
                    python,
                    "NAS_PINNs_burgers_nsga2.py",
                    *burgers_flags,
                    *burgers_stage_flags,
                    *burgers_quick,
                    "--save-dir",
                    os.path.join(rep_root, "burgers", "nsga2"),
                ],
                "burgers",
            ),
            Job(
                "burgers_nsga3",
                [
                    python,
                    "NAS_PINNs_burgers_nsga3.py",
                    *burgers_flags,
                    *burgers_stage_flags,
                    *burgers_quick,
                    "--save-dir",
                    os.path.join(rep_root, "burgers", "nsga3"),
                ],
                "burgers",
            ),
            Job(
                "burgers_bayesian",
                [
                    python,
                    "NAS_PINNs_burgers_bayesian.py",
                    *burgers_flags,
                    *burgers_stage_flags,
                    *burgers_quick,
                    "--save-dir",
                    os.path.join(rep_root, "burgers", "bayesian"),
                ],
                "burgers",
            ),
            Job(
                "poisson_naspinn",
                [
                    python,
                    "NAS_PINNs_poisson.py",
                    *poisson_flags,
                    "--stage",
                    args.poisson_stage,
                    *poisson_quick,
                    "--save-dir",
                    os.path.join(rep_root, "poisson", "naspinn"),
                ],
                "poisson",
            ),
            Job(
                "poisson_nsga2",
                [
                    python,
                    "NAS_PINNs_poisson_nsga2.py",
                    *poisson_flags,
                    *poisson_quick,
                    *nsga_quick,
                    *nsga2_quick_extra,
                    *poisson_stage_lbfgs_flags,
                    *poisson_stage_pso_flags,
                    "--save-dir",
                    os.path.join(rep_root, "poisson", "nsga2"),
                ],
                "poisson",
            ),
            Job(
                "poisson_nsga3",
                [
                    python,
                    "NAS_PINNs_poisson_nsga3.py",
                    *poisson_flags,
                    *poisson_quick,
                    *nsga_quick,
                    *nsga3_quick_extra,
                    *poisson_stage_lbfgs_flags,
                    *poisson_stage_pso_flags,
                    "--save-dir",
                    os.path.join(rep_root, "poisson", "nsga3"),
                ],
                "poisson",
            ),
            Job(
                "poisson_bayesian",
                [
                    python,
                    "NAS_PINNs_poisson_bayesian.py",
                    *poisson_flags,
                    *poisson_quick,
                    *bayes_quick,
                    *poisson_stage_lbfgs_flags,
                    *poisson_stage_pso_flags,
                    "--save-dir",
                    os.path.join(rep_root, "poisson", "bayesian"),
                ],
                "poisson",
            ),
        ]
    )
    return jobs


def run_job(job: Job, log_path: str):
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"[START] {dt.datetime.now().isoformat()}\n")
        logf.write("COMMAND: " + " ".join(job.cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(job.cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = time.time() - started
    return proc.returncode, elapsed


def main():
    parser = argparse.ArgumentParser(description="Sequential experiment runner for NAS-PINNs project")
    parser.add_argument("--repeats", type=int, default=1, help="repeat full job list with incremented seeds")
    parser.add_argument("--base-seed", type=int, default=42, help="initial seed")
    parser.add_argument("--stage", choices=["adam", "lbfgs", "pso"], default="lbfgs", help="legacy: apply same stage to both equations")
    parser.add_argument(
        "--burgers-stage",
        choices=["adam", "lbfgs", "pso"],
        default=None,
        help="stage mode for Burgers only (default: follow --stage)",
    )
    parser.add_argument(
        "--poisson-stage",
        choices=["adam", "lbfgs", "pso"],
        default=None,
        help="stage mode for Poisson only (default: follow --stage)",
    )
    parser.add_argument("--nu-list", type=str, default="0.01,0.04,0.07", help="Burgers viscosity list")
    parser.add_argument(
        "--domain-list",
        type=str,
        default="rectangular,circle,lshape,flower,annulus",
        help="Poisson domain list",
    )
    parser.add_argument("--quick", action="store_true", help="quick smoke mode")
    parser.add_argument("--stop-on-error", action="store_true", help="stop on first failed job")
    parser.add_argument("--run-dir", type=str, default=None, help="optional existing run directory")
    args = parser.parse_args()
    args.burgers_stage = args.burgers_stage or args.stage
    args.poisson_stage = args.poisson_stage or args.stage
    if args.burgers_stage == "pso":
        print("Note: Burgers PSO runs only in NAS-PINN baseline; other Burgers methods use Adam/L-BFGS path.")

    if args.run_dir:
        run_dir = args.run_dir
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results", "pipeline_runs", stamp)
    os.makedirs(run_dir, exist_ok=True)

    nu_count = len([x for x in args.nu_list.split(",") if x.strip()])
    domain_count = len([x for x in args.domain_list.split(",") if x.strip()])
    top_level_runs = args.repeats * 8
    estimated_sub_runs = args.repeats * ((4 * nu_count) + (4 * domain_count))

    print(f"Run directory: {run_dir}")
    print(f"Burgers stage: {args.burgers_stage}")
    print(f"Poisson stage: {args.poisson_stage}")
    print(f"Top-level job count: {top_level_runs}")
    print(f"Estimated sub-experiment count: {estimated_sub_runs}")

    summary = []
    job_idx = 0
    total_jobs = args.repeats * 8

    for rep in range(args.repeats):
        seed = args.base_seed + rep
        jobs = build_jobs(args, seed=seed, run_dir=run_dir, rep_idx=rep + 1)
        for job in jobs:
            job_idx += 1
            log_path = os.path.join(run_dir, f"{job.name}_rep{rep+1}.log")
            print(f"[{job_idx}/{total_jobs}] {job.name} (seed={seed}) ...")
            code, elapsed = run_job(job, log_path)
            status = "ok" if code == 0 else "failed"
            print(f"  -> {status} ({elapsed:.1f}s)")
            summary.append(
                {
                    "repeat": rep + 1,
                    "seed": seed,
                    "job": job.name,
                    "family": job.family,
                    "status": status,
                    "return_code": code,
                    "elapsed_sec": round(elapsed, 2),
                    "log": log_path,
                }
            )
            if code != 0 and args.stop_on_error:
                print("Stopping due to --stop-on-error.")
                break
        if args.stop_on_error and summary and summary[-1]["status"] != "ok":
            break

    csv_path = os.path.join(run_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "repeat",
                "seed",
                "job",
                "family",
                "status",
                "return_code",
                "elapsed_sec",
                "log",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    json_path = os.path.join(run_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    ok = sum(1 for row in summary if row["status"] == "ok")
    fail = sum(1 for row in summary if row["status"] != "ok")
    print("\nFinished.")
    print(f"Success: {ok}, Failed: {fail}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
