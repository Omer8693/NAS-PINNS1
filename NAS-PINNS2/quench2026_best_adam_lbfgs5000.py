#!/usr/bin/env python3
"""Refine each method's best-Adam architecture with separate LBFGS and PSO runs."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METHODS = ("nsga2", "nsga3", "bayesian")


def parse_arch_name(name: str) -> Optional[Tuple[int, int]]:
    # eval_00021_L6_N132 -> (6, 132)
    parts = name.split("_")
    for i, token in enumerate(parts):
        if token.startswith("L") and i + 1 < len(parts) and parts[i + 1].startswith("N"):
            try:
                return int(token[1:]), int(parts[i + 1][1:])
            except ValueError:
                return None
    return None


def find_best_adam_trial(method_dir: Path) -> Optional[Dict]:
    best = None
    for meta_path in sorted((method_dir / "search_trials").glob("eval_*_L*_N*/run_meta.json")):
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            obj = float(meta.get("best_objective"))
        except Exception:
            continue
        trial_dir = meta_path.parent
        arch = parse_arch_name(trial_dir.name)
        if arch is None:
            continue
        layers, neurons = arch
        row = {
            "trial_dir": str(trial_dir.resolve()),
            "layers": layers,
            "neurons": neurons,
            "adam_objective": obj,
        }
        if best is None or obj < best["adam_objective"]:
            best = row
    return best


def write_adam_stage_summary(target_dir: Path, adam_objective: float) -> None:
    path = target_dir / "stage_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "objective", "checkpoint", "selected"])
        writer.writeheader()
        writer.writerow(
            {
                "stage": "adam",
                "objective": float(adam_objective),
                "checkpoint": "baseline_model_adam.pth",
                "selected": 1,
            }
        )


def seed_from_adam_trial(row: Dict, target_dir: Path) -> None:
    trial_dir = Path(row["trial_dir"])
    src_adam = trial_dir / "baseline_model_adam.pth"
    if not src_adam.exists():
        raise FileNotFoundError(f"Missing Adam checkpoint: {src_adam}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_adam, target_dir / "baseline_model_adam.pth")
    for opt_name in ("metrics.csv", "loss_curves.png"):
        src = trial_dir / opt_name
        if src.exists():
            shutil.copy2(src, target_dir / opt_name)
    write_adam_stage_summary(target_dir, float(row["adam_objective"]))

    seed_meta = {
        "source_trial_dir": str(trial_dir.resolve()),
        "seeded_adam_checkpoint": str((target_dir / "baseline_model_adam.pth").resolve()),
        "adam_objective": float(row["adam_objective"]),
        "layers": int(row["layers"]),
        "neurons": int(row["neurons"]),
    }
    with (target_dir / "seed_info.json").open("w", encoding="utf-8") as f:
        json.dump(seed_meta, f, indent=2)


def base_cmd(
    python_exe: str,
    baseline_script: Path,
    out_dir: Path,
    layers: int,
    neurons: int,
    seed: int,
    lbfgs_max_iter: int,
    pso_iters: int,
    pso_swarm: int,
    pso_span: float,
) -> List[str]:
    return [
        python_exe,
        str(baseline_script),
        "--save-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--epochs",
        "1000",
        "--layers",
        str(layers),
        "--base-neurons",
        str(neurons),
        "--n-time-steps",
        "10",
        "--log-every",
        "100",
        "--adam-lr",
        "0.001",
        "--w-physics",
        "50.0",
        "--w-ic",
        "0.001",
        "--w-bc",
        "1e-18",
        "--w-data",
        "1e-05",
        "--lbfgs-max-iter",
        str(lbfgs_max_iter),
        "--lbfgs-col-points",
        "1024",
        "--lbfgs-ic-points",
        "256",
        "--lbfgs-bc-points",
        "256",
        "--lbfgs-time-steps",
        "4",
        "--lbfgs-history-size",
        "20",
        "--lbfgs-line-search",
        "strong_wolfe",
        "--pso-iters",
        str(pso_iters),
        "--pso-swarm",
        str(pso_swarm),
        "--pso-span",
        str(pso_span),
        "--force-final",
    ]


def run_command(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def write_selection_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "layers",
        "neurons",
        "adam_objective",
        "trial_dir",
        "lbfgs_target_dir",
        "pso_target_dir",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def parse_args():
    parser = argparse.ArgumentParser(description="Best-Adam architecture -> separate LBFGS(5000) and PSO runs")
    parser.add_argument("--results-root", type=str, default="results/quench2026/pipeline")
    parser.add_argument("--output-root", type=str, default="results/quench2026/best_adam_lbfgs5000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lbfgs-max-iter", type=int, default=5000)
    parser.add_argument("--pso-iters", type=int, default=8)
    parser.add_argument("--pso-swarm", type=int, default=16)
    parser.add_argument("--pso-span", type=float, default=0.25)
    parser.add_argument("--run", action="store_true", help="Execute commands (default: print only)")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parent
    baseline_script = repo / "naspinn_baseline_with_quench_2026_data.py"
    results_root = Path(args.results_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected = []
    for method in METHODS:
        method_dir = results_root / method
        best = find_best_adam_trial(method_dir)
        if best is None:
            print(f"[{method}] no valid search trial found, skipping")
            continue
        arch_key = f"L{best['layers']}_N{best['neurons']}"
        lbfgs_dir = output_root / "lbfgs" / method / arch_key
        pso_dir = output_root / "pso" / method / arch_key
        row = dict(best)
        row["method"] = method
        row["lbfgs_target_dir"] = str(lbfgs_dir.resolve())
        row["pso_target_dir"] = str(pso_dir.resolve())
        selected.append(row)

    selection_csv = output_root / "selected_best_adam_architectures.csv"
    write_selection_csv(selected, selection_csv)
    print(f"Selection table: {selection_csv}")

    for row in selected:
        method = row["method"]
        layers = int(row["layers"])
        neurons = int(row["neurons"])
        lbfgs_dir = Path(row["lbfgs_target_dir"])
        pso_dir = Path(row["pso_target_dir"])

        seed_from_adam_trial(row, lbfgs_dir)
        seed_from_adam_trial(row, pso_dir)

        lbfgs_cmd = base_cmd(
            python_exe=sys.executable,
            baseline_script=baseline_script,
            out_dir=lbfgs_dir,
            layers=layers,
            neurons=neurons,
            seed=int(args.seed),
            lbfgs_max_iter=int(args.lbfgs_max_iter),
            pso_iters=int(args.pso_iters),
            pso_swarm=int(args.pso_swarm),
            pso_span=float(args.pso_span),
        )
        pso_cmd = base_cmd(
            python_exe=sys.executable,
            baseline_script=baseline_script,
            out_dir=pso_dir,
            layers=layers,
            neurons=neurons,
            seed=int(args.seed),
            lbfgs_max_iter=int(args.lbfgs_max_iter),
            pso_iters=int(args.pso_iters),
            pso_swarm=int(args.pso_swarm),
            pso_span=float(args.pso_span),
        ) + ["--skip-lbfgs", "--use-pso"]

        print("")
        print(f"[{method}] lbfgs: {' '.join(lbfgs_cmd)}")
        print(f"[{method}] pso:   {' '.join(pso_cmd)}")

        if args.run:
            run_command(lbfgs_cmd, lbfgs_dir / "run.log")
            run_command(pso_cmd, pso_dir / "run.log")


if __name__ == "__main__":
    main()
