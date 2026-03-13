#!/usr/bin/env python3
"""Common helpers used by all optimization pipelines."""

import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from cfg import REPO_ROOT


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_log(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = f"[{stamp}] {message}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required for this pipeline, but torch.cuda.is_available() is False. "
            "Set CUDA correctly and rerun."
        )


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_params(baseline_mod, layers: int, neurons: int) -> int:
    model = baseline_mod.NAS_PINN(layers=layers, base_neurons=neurons)
    return int(sum(p.numel() for p in model.parameters()))


def run_command(cmd: List[str], log_file: Path, env: Dict[str, str]) -> None:
    ensure_dir(log_file.parent)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}). See {log_file}")


def baseline_cmd(
    baseline_script: Path,
    save_dir: Path,
    seed: int,
    epochs: int,
    layers: int,
    neurons: int,
    n_time_steps: int,
    log_every: int,
    adam_lr: float,
    w_physics: float,
    w_ic: float,
    w_bc: float,
    w_data: float,
    temp_ref_t0_mode: str,
    skip_lbfgs: bool,
    use_pso: bool,
    lbfgs_max_iter: int,
    lbfgs_col_points: int,
    lbfgs_ic_points: int,
    lbfgs_bc_points: int,
    lbfgs_time_steps: int,
    lbfgs_history_size: int,
    lbfgs_line_search: str,
    pso_iters: int,
    pso_swarm: int,
    pso_span: float,
    force_final: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        str(baseline_script),
        "--save-dir",
        str(save_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--layers",
        str(layers),
        "--base-neurons",
        str(neurons),
        "--n-time-steps",
        str(n_time_steps),
        "--log-every",
        str(log_every),
        "--adam-lr",
        str(adam_lr),
        "--w-physics",
        str(w_physics),
        "--w-ic",
        str(w_ic),
        "--w-bc",
        str(w_bc),
        "--w-data",
        str(w_data),
        "--temp-ref-t0-mode",
        str(temp_ref_t0_mode),
        "--lbfgs-max-iter",
        str(lbfgs_max_iter),
        "--lbfgs-col-points",
        str(lbfgs_col_points),
        "--lbfgs-ic-points",
        str(lbfgs_ic_points),
        "--lbfgs-bc-points",
        str(lbfgs_bc_points),
        "--lbfgs-time-steps",
        str(lbfgs_time_steps),
        "--lbfgs-history-size",
        str(lbfgs_history_size),
        "--lbfgs-line-search",
        str(lbfgs_line_search),
        "--pso-iters",
        str(pso_iters),
        "--pso-swarm",
        str(pso_swarm),
        "--pso-span",
        str(pso_span),
    ]
    if skip_lbfgs:
        cmd.append("--skip-lbfgs")
    if use_pso:
        cmd.append("--use-pso")
    if force_final:
        cmd.append("--force-final")
    return cmd


def read_run_meta(run_dir: Path) -> Dict:
    with (run_dir / "run_meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def read_stage_summary(run_dir: Path) -> List[Dict]:
    path = run_dir / "stage_summary.csv"
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def arch_key(layers: int, neurons: int) -> str:
    return f"L{int(layers)}_N{int(neurons)}"


def evaluate_architecture(
    args,
    baseline_mod,
    baseline_script: Path,
    method_dir: Path,
    layers: int,
    neurons: int,
    eval_idx: int,
    log_path: Path,
    env: Dict[str, str],
) -> Tuple[float, float, Path]:
    key = arch_key(layers, neurons)
    cache_path = method_dir / "search_cache.json"
    cache = load_json(cache_path, default={})
    if key in cache and cache[key].get("status") in {"ok", "failed"}:
        entry = cache[key]
        return float(entry["objective"]), float(entry["param_count"]), Path(entry["trial_dir"])

    trial_dir = method_dir / "search_trials" / f"eval_{eval_idx:05d}_{key}"
    ensure_dir(trial_dir)
    trial_log = trial_dir / "train.log"
    append_log(log_path, f"[search] eval={eval_idx} arch={key} -> {trial_dir}")

    cmd = baseline_cmd(
        baseline_script=baseline_script,
        save_dir=trial_dir,
        seed=int(args.seed + eval_idx),
        epochs=int(args.proxy_epochs),
        layers=int(layers),
        neurons=int(neurons),
        n_time_steps=int(args.n_time_steps),
        log_every=max(1, int(args.proxy_epochs // 4)),
        adam_lr=float(args.adam_lr),
        w_physics=float(args.w_physics),
        w_ic=float(args.w_ic),
        w_bc=float(args.w_bc),
        w_data=float(args.w_data),
        temp_ref_t0_mode=str(args.temp_ref_t0_mode),
        skip_lbfgs=True,
        use_pso=False,
        lbfgs_max_iter=int(args.lbfgs_max_iter),
        lbfgs_col_points=int(args.lbfgs_col_points),
        lbfgs_ic_points=int(args.lbfgs_ic_points),
        lbfgs_bc_points=int(args.lbfgs_bc_points),
        lbfgs_time_steps=int(args.lbfgs_time_steps),
        lbfgs_history_size=int(args.lbfgs_history_size),
        lbfgs_line_search=str(args.lbfgs_line_search),
        pso_iters=int(args.pso_iters),
        pso_swarm=int(args.pso_swarm),
        pso_span=float(args.pso_span),
        force_final=False,
    )

    try:
        run_command(cmd, trial_log, env=env)
        meta = read_run_meta(trial_dir)
        objective = float(meta["best_objective"])
        param_count = float(meta.get("param_count", count_params(baseline_mod, layers, neurons)))
        cache[key] = {
            "status": "ok",
            "objective": objective,
            "param_count": param_count,
            "trial_dir": str(trial_dir),
            "eval_idx": int(eval_idx),
        }
        save_json(cache_path, cache)
        return objective, param_count, trial_dir
    except Exception as exc:
        param_count = float(count_params(baseline_mod, layers, neurons))
        objective = float(args.proxy_fail_penalty + param_count)
        cache[key] = {
            "status": "failed",
            "objective": objective,
            "param_count": param_count,
            "trial_dir": str(trial_dir),
            "eval_idx": int(eval_idx),
            "error": str(exc),
        }
        save_json(cache_path, cache)
        append_log(log_path, f"[search] eval={eval_idx} arch={key} failed -> penalty objective={objective:.6e}")
        return objective, param_count, trial_dir


def run_final_training(
    args,
    baseline_script: Path,
    method_dir: Path,
    best_layers: int,
    best_neurons: int,
    log_path: Path,
    env: Dict[str, str],
) -> Dict:
    final_dir = method_dir / "final" / arch_key(best_layers, best_neurons)
    done_path = final_dir / "done.json"
    if done_path.exists() and (final_dir / "baseline_model.pth").exists() and not bool(args.force_final):
        return load_json(done_path, default={})

    ensure_dir(final_dir)
    append_log(log_path, f"[final] training {arch_key(best_layers, best_neurons)} -> {final_dir}")
    final_log = final_dir / "train.log"
    cmd = baseline_cmd(
        baseline_script=baseline_script,
        save_dir=final_dir,
        seed=int(args.seed),
        epochs=int(args.epochs),
        layers=int(best_layers),
        neurons=int(best_neurons),
        n_time_steps=int(args.n_time_steps),
        log_every=int(args.log_every),
        adam_lr=float(args.adam_lr),
        w_physics=float(args.w_physics),
        w_ic=float(args.w_ic),
        w_bc=float(args.w_bc),
        w_data=float(args.w_data),
        temp_ref_t0_mode=str(args.temp_ref_t0_mode),
        skip_lbfgs=bool(args.skip_lbfgs_final),
        use_pso=bool(args.use_pso_final),
        lbfgs_max_iter=int(args.lbfgs_max_iter),
        lbfgs_col_points=int(args.lbfgs_col_points),
        lbfgs_ic_points=int(args.lbfgs_ic_points),
        lbfgs_bc_points=int(args.lbfgs_bc_points),
        lbfgs_time_steps=int(args.lbfgs_time_steps),
        lbfgs_history_size=int(args.lbfgs_history_size),
        lbfgs_line_search=str(args.lbfgs_line_search),
        pso_iters=int(args.pso_iters),
        pso_swarm=int(args.pso_swarm),
        pso_span=float(args.pso_span),
        force_final=bool(args.force_final),
    )
    run_command(cmd, final_log, env=env)
    meta = read_run_meta(final_dir)
    stage_rows = read_stage_summary(final_dir)
    result = {
        "final_dir": str(final_dir),
        "best_stage": meta.get("best_stage"),
        "best_objective": float(meta.get("best_objective", np.nan)),
        "param_count": int(meta.get("param_count", 0)),
        "run_time_seconds": float(meta.get("run_time_seconds", np.nan)),
        "stage_rows": stage_rows,
    }
    save_json(done_path, result)
    return result


def write_search_population_csv(save_path: Path, X: np.ndarray, F: np.ndarray) -> None:
    ensure_dir(save_path.parent)
    with save_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layers", "neurons", "obj_proxy_loss", "obj_param_count"])
        for x, y in zip(X, F):
            writer.writerow([int(round(float(x[0]))), int(round(float(x[1]))), float(y[0]), float(y[1])])


def infer_checkpoint(run_dir: Path, best_stage: Optional[str]) -> Path:
    if best_stage:
        ckpt = run_dir / f"baseline_model_{best_stage}.pth"
        if ckpt.exists():
            return ckpt
    for name in ("baseline_model.pth", "baseline_model_lbfgs.pth", "baseline_model_adam.pth", "baseline_model_pso.pth"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")
