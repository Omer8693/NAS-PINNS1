#!/usr/bin/env python3
"""Refine baseline from saved Adam checkpoint with LBFGS(5000) and PSO."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_arch(arch_name: str) -> Tuple[int, int]:
    # L5_N96 -> (5, 96)
    left, right = arch_name.split("_")
    return int(left[1:]), int(right[1:])


def read_stage_objectives(stage_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with stage_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[str(row["stage"]).strip().lower()] = float(row["objective"])
            except (TypeError, ValueError, KeyError):
                continue
    return out


def write_seed_stage_summary(target_dir: Path, adam_objective: float) -> None:
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


def seed_dir_from_adam(source_dir: Path, target_dir: Path, adam_objective: float) -> None:
    src_ckpt = source_dir / "baseline_model_adam.pth"
    if not src_ckpt.exists():
        raise FileNotFoundError(f"Adam checkpoint not found: {src_ckpt}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_ckpt, target_dir / "baseline_model_adam.pth")
    for name in ("metrics.csv", "loss_curves.png"):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)

    write_seed_stage_summary(target_dir, adam_objective)
    seed_info = {
        "source_dir": str(source_dir.resolve()),
        "seed_checkpoint": str((target_dir / "baseline_model_adam.pth").resolve()),
        "seed_adam_objective": float(adam_objective),
    }
    with (target_dir / "seed_info.json").open("w", encoding="utf-8") as f:
        json.dump(seed_info, f, indent=2)


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


def stage_best(stage_map: Dict[str, float]) -> Tuple[Optional[str], Optional[float]]:
    if not stage_map:
        return None, None
    best_stage = min(stage_map, key=stage_map.get)
    return best_stage, float(stage_map[best_stage])


def write_comparison_csv(
    source_stage: Dict[str, float],
    lbfgs_stage: Dict[str, float],
    pso_stage: Dict[str, float],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, stages in (
        ("baseline_original", source_stage),
        ("baseline_from_adam_lbfgs5000", lbfgs_stage),
        ("baseline_from_adam_pso", pso_stage),
    ):
        best_stage, best_obj = stage_best(stages)
        rows.append(
            {
                "variant": variant,
                "adam_obj": stages.get("adam", ""),
                "lbfgs_obj": stages.get("lbfgs", ""),
                "pso_obj": stages.get("pso", ""),
                "best_stage": best_stage or "",
                "best_obj": best_obj if best_obj is not None else "",
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["variant", "adam_obj", "lbfgs_obj", "pso_obj", "best_stage", "best_obj"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_cmd(
    python_exe: str,
    script_path: Path,
    save_dir: Path,
    layers: int,
    neurons: int,
    seed: int,
    lbfgs_max_iter: int,
    use_pso: bool,
) -> List[str]:
    cmd = [
        python_exe,
        str(script_path),
        "--save-dir",
        str(save_dir),
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
        "5",
        "--pso-swarm",
        "8",
        "--pso-span",
        "0.25",
        "--force-final",
    ]
    if use_pso:
        cmd.extend(["--skip-lbfgs", "--use-pso"])
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Adam-seeded LBFGS5000/PSO runner")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="results/quench2026/baseline/L5_N96",
        help="Baseline source directory containing baseline_model_adam.pth",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/quench2026/baseline_adam_refine5000",
        help="Where to write lbfgs/pso refined runs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lbfgs-max-iter", type=int, default=5000)
    parser.add_argument("--run", action="store_true", help="Run training commands")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    source_dir = (repo_root / args.source_dir).resolve()
    output_root = (repo_root / args.output_root).resolve()
    train_script = repo_root / "naspinn_baseline_with_quench_2026_data.py"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    source_stage = read_stage_objectives(source_dir / "stage_summary.csv")
    if "adam" not in source_stage:
        raise RuntimeError(f"Adam objective not found in {source_dir / 'stage_summary.csv'}")

    layers, neurons = parse_arch(source_dir.name)
    arch_key = f"L{layers}_N{neurons}"
    lbfgs_dir = output_root / "lbfgs" / arch_key
    pso_dir = output_root / "pso" / arch_key

    seed_dir_from_adam(source_dir, lbfgs_dir, source_stage["adam"])
    seed_dir_from_adam(source_dir, pso_dir, source_stage["adam"])

    lbfgs_cmd = build_cmd(
        python_exe=sys.executable,
        script_path=train_script,
        save_dir=lbfgs_dir,
        layers=layers,
        neurons=neurons,
        seed=int(args.seed),
        lbfgs_max_iter=int(args.lbfgs_max_iter),
        use_pso=False,
    )
    pso_cmd = build_cmd(
        python_exe=sys.executable,
        script_path=train_script,
        save_dir=pso_dir,
        layers=layers,
        neurons=neurons,
        seed=int(args.seed),
        lbfgs_max_iter=int(args.lbfgs_max_iter),
        use_pso=True,
    )

    print(f"LBFGS command: {' '.join(lbfgs_cmd)}")
    print(f"PSO command:   {' '.join(pso_cmd)}")

    if args.run:
        run_command(lbfgs_cmd, lbfgs_dir / "run.log")
        run_command(pso_cmd, pso_dir / "run.log")

    lbfgs_stage = read_stage_objectives(lbfgs_dir / "stage_summary.csv")
    pso_stage = read_stage_objectives(pso_dir / "stage_summary.csv")
    out_csv = output_root / "baseline_refine_comparison.csv"
    write_comparison_csv(source_stage, lbfgs_stage, pso_stage, out_csv)
    print(f"Comparison table: {out_csv}")


if __name__ == "__main__":
    main()
