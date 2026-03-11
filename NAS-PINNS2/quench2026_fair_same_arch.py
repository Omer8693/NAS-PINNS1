#!/usr/bin/env python3
"""Strict fair comparison on one fixed architecture across multiple seeds.

Flow per seed:
1) Train Adam-only (same architecture, same hyperparams)
2) Start from Adam checkpoint -> LBFGS5000 refinement
3) Start from Adam checkpoint -> PSO refinement
4) Write per-seed and aggregate (mean/std) tables
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: List[str], log_path: Path) -> None:
    ensure_dir(log_path.parent)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stage_map_from_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[str(row["stage"]).strip().lower()] = float(row["objective"])
            except (TypeError, ValueError, KeyError):
                continue
    return out


def parse_seeds(text: str) -> List[int]:
    out = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("No valid seeds provided")
    return out


def write_csv(rows: List[Dict], path: Path, fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description="Strict fair same-architecture comparison for quench2026")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--neurons", type=int, default=132)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lbfgs-max-iter", type=int, default=5000)
    parser.add_argument("--base-dir", type=str, default="results/quench2026/fair_same_arch")
    parser.add_argument("--refine-dir", type=str, default="results/quench2026/fair_same_arch_refine")
    parser.add_argument("--n-time-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--w-physics", type=float, default=50.0)
    parser.add_argument("--w-ic", type=float, default=1e-3)
    parser.add_argument("--w-bc", type=float, default=1e-18)
    parser.add_argument("--w-data", type=float, default=1e-5)
    parser.add_argument("--run", action="store_true", help="Execute training/refinement commands")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parent
    baseline_script = repo / "naspinn_baseline_with_quench_2026_data.py"
    refine_script = repo / "quench2026_baseline_adam_refine.py"
    base_dir = (repo / args.base_dir).resolve()
    refine_dir = (repo / args.refine_dir).resolve()
    seeds = parse_seeds(args.seeds)
    arch = f"L{int(args.layers)}_N{int(args.neurons)}"

    all_rows: List[Dict] = []

    for seed in seeds:
        seed_base = base_dir / f"seed_{seed}" / arch
        seed_refine_root = refine_dir / f"seed_{seed}"
        seed_log = seed_refine_root / "fair_run.log"
        ensure_dir(seed_refine_root)

        adam_cmd = [
            sys.executable,
            str(baseline_script),
            "--save-dir",
            str(seed_base),
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--layers",
            str(args.layers),
            "--base-neurons",
            str(args.neurons),
            "--n-time-steps",
            str(args.n_time_steps),
            "--log-every",
            str(args.log_every),
            "--adam-lr",
            str(args.adam_lr),
            "--w-physics",
            str(args.w_physics),
            "--w-ic",
            str(args.w_ic),
            "--w-bc",
            str(args.w_bc),
            "--w-data",
            str(args.w_data),
            "--skip-lbfgs",
            "--force-final",
        ]

        refine_cmd = [
            sys.executable,
            str(refine_script),
            "--source-dir",
            str(seed_base),
            "--output-root",
            str(seed_refine_root),
            "--seed",
            str(seed),
            "--lbfgs-max-iter",
            str(args.lbfgs_max_iter),
            "--run",
        ]

        print("")
        print(f"[seed={seed}] Adam command:   {' '.join(adam_cmd)}")
        print(f"[seed={seed}] Refine command: {' '.join(refine_cmd)}")

        if args.run:
            run_command(adam_cmd, seed_log)
            run_command(refine_cmd, seed_log)

        # Collect outputs (whether newly run or already present)
        adam_meta = read_json(seed_base / "run_meta.json")
        lbfgs_stage = stage_map_from_csv(seed_refine_root / "lbfgs" / arch / "stage_summary.csv")
        pso_stage = stage_map_from_csv(seed_refine_root / "pso" / arch / "stage_summary.csv")
        lbfgs_meta = read_json(seed_refine_root / "lbfgs" / arch / "run_meta.json")
        pso_meta = read_json(seed_refine_root / "pso" / arch / "run_meta.json")

        row = {
            "seed": seed,
            "arch": arch,
            "layers": int(args.layers),
            "neurons": int(args.neurons),
            "adam_obj": float(adam_meta.get("best_objective")),
            "lbfgs_obj": float(lbfgs_stage.get("lbfgs")),
            "pso_obj": float(pso_stage.get("pso")),
            "adam_runtime_s": float(adam_meta.get("run_time_seconds", np.nan)),
            "lbfgs_runtime_s": float(lbfgs_meta.get("run_time_seconds", np.nan)),
            "pso_runtime_s": float(pso_meta.get("run_time_seconds", np.nan)),
            "lbfgs_better_than_adam": int(float(lbfgs_stage.get("lbfgs")) < float(adam_meta.get("best_objective"))),
            "pso_better_than_adam": int(float(pso_stage.get("pso")) < float(adam_meta.get("best_objective"))),
        }
        all_rows.append(row)

    detail_csv = refine_dir / f"{arch}_per_seed.csv"
    detail_fields = [
        "seed",
        "arch",
        "layers",
        "neurons",
        "adam_obj",
        "lbfgs_obj",
        "pso_obj",
        "adam_runtime_s",
        "lbfgs_runtime_s",
        "pso_runtime_s",
        "lbfgs_better_than_adam",
        "pso_better_than_adam",
    ]
    write_csv(all_rows, detail_csv, detail_fields)

    # Aggregate table
    def mean_std(col: str):
        vals = np.array([float(r[col]) for r in all_rows], dtype=np.float64)
        return float(np.mean(vals)), float(np.std(vals))

    adam_m, adam_s = mean_std("adam_obj")
    lbfgs_m, lbfgs_s = mean_std("lbfgs_obj")
    pso_m, pso_s = mean_std("pso_obj")
    adam_rt_m, adam_rt_s = mean_std("adam_runtime_s")
    lbfgs_rt_m, lbfgs_rt_s = mean_std("lbfgs_runtime_s")
    pso_rt_m, pso_rt_s = mean_std("pso_runtime_s")

    agg_rows = [
        {
            "stage": "adam",
            "objective_mean": adam_m,
            "objective_std": adam_s,
            "runtime_mean_s": adam_rt_m,
            "runtime_std_s": adam_rt_s,
            "improve_vs_adam_pct": 0.0,
        },
        {
            "stage": "lbfgs",
            "objective_mean": lbfgs_m,
            "objective_std": lbfgs_s,
            "runtime_mean_s": lbfgs_rt_m,
            "runtime_std_s": lbfgs_rt_s,
            "improve_vs_adam_pct": (adam_m - lbfgs_m) / adam_m * 100.0,
        },
        {
            "stage": "pso",
            "objective_mean": pso_m,
            "objective_std": pso_s,
            "runtime_mean_s": pso_rt_m,
            "runtime_std_s": pso_rt_s,
            "improve_vs_adam_pct": (adam_m - pso_m) / adam_m * 100.0,
        },
    ]
    agg_csv = refine_dir / f"{arch}_aggregate_mean_std.csv"
    agg_fields = ["stage", "objective_mean", "objective_std", "runtime_mean_s", "runtime_std_s", "improve_vs_adam_pct"]
    write_csv(agg_rows, agg_csv, agg_fields)

    print("")
    print(f"Per-seed table:  {detail_csv}")
    print(f"Aggregate table: {agg_csv}")


if __name__ == "__main__":
    main()
