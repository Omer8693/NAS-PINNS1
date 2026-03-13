import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = REPO_ROOT / "naspinn_baseline_with_quench_2026_data.py"

_BASELINE_MODULE = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def append_log(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = f"[{stamp}] {message}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_baseline_module():
    global _BASELINE_MODULE
    if _BASELINE_MODULE is not None:
        return _BASELINE_MODULE

    spec = importlib.util.spec_from_file_location("quench2026_baseline_mod", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline script: {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _BASELINE_MODULE = module
    return module


def count_params(layers: int, neurons: int) -> int:
    module = load_baseline_module()
    model = module.NAS_PINN(layers=layers, base_neurons=neurons)
    return int(sum(p.numel() for p in model.parameters()))


def arch_key(layers: int, neurons: int) -> str:
    return f"L{int(layers)}_N{int(neurons)}"


def read_run_meta(run_dir: Path) -> Dict:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run_meta.json in {run_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_stage_summary(run_dir: Path) -> List[Dict]:
    stage_csv = run_dir / "stage_summary.csv"
    if not stage_csv.exists():
        return []
    rows: List[Dict] = []
    with open(stage_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def run_command(cmd: List[str], log_file: Path, cwd: Path = REPO_ROOT) -> None:
    ensure_dir(log_file.parent)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}). See log: {log_file}")


def baseline_cmd(
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
        str(BASELINE_SCRIPT),
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


def evaluate_architecture(
    args,
    method_dir: Path,
    layers: int,
    neurons: int,
    eval_idx: int,
    pipeline_log: Path,
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
    append_log(pipeline_log, f"[search] eval={eval_idx} arch={key} -> {trial_dir}")

    cmd = baseline_cmd(
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
        run_command(cmd, trial_log)
        meta = read_run_meta(trial_dir)
        objective = float(meta["best_objective"])
        param_count = float(meta.get("param_count", count_params(layers, neurons)))

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
        # Do not crash the whole search on one failed architecture (OOM, etc.).
        param_count = float(count_params(layers, neurons))
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
        append_log(
            pipeline_log,
            f"[search] eval={eval_idx} arch={key} failed, assigned penalty objective={objective:.4e}",
        )
        return objective, param_count, trial_dir


def run_final_training(args, method_dir: Path, best_layers: int, best_neurons: int, pipeline_log: Path) -> Dict:
    final_dir = method_dir / "final" / arch_key(best_layers, best_neurons)
    done_path = final_dir / "done.json"
    if done_path.exists() and (final_dir / "baseline_model.pth").exists() and not bool(args.force_final):
        return load_json(done_path, default={})

    ensure_dir(final_dir)
    append_log(
        pipeline_log,
        f"[final] training arch={arch_key(best_layers, best_neurons)} -> {final_dir}",
    )
    final_log = final_dir / "train.log"

    cmd = baseline_cmd(
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
    run_command(cmd, final_log)

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
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layers", "neurons", "obj_proxy_loss", "obj_param_count"])
        for x, y in zip(X, F):
            writer.writerow([int(round(float(x[0]))), int(round(float(x[1]))), float(y[0]), float(y[1])])


def run_method(
    method_name: str,
    args,
    search_fn: Callable[[argparse.Namespace, Path, Path], Dict],
) -> Dict:
    method_dir = Path(args.save_dir).resolve()
    ensure_dir(method_dir)
    log_path = method_dir / "logs" / f"{method_name}.log"
    state_path = method_dir / "run_state.json"
    state = load_json(state_path, default={"method": method_name, "search_done": False, "final_done": False})

    append_log(log_path, f"=== {method_name} run started ===")
    append_log(log_path, f"save_dir={method_dir}")

    if not state.get("search_done", False):
        search_result = search_fn(args, method_dir, log_path)
        state["search_done"] = True
        state["search"] = search_result
        save_json(state_path, state)
        append_log(log_path, f"search done | best_arch={search_result['best_layers']}x{search_result['best_neurons']}")
    else:
        append_log(log_path, "search already done; resuming from saved state")

    if not args.skip_final and (bool(args.force_final) or not state.get("final_done", False)):
        best_layers = int(state["search"]["best_layers"])
        best_neurons = int(state["search"]["best_neurons"])
        if bool(args.force_final) and state.get("final_done", False):
            append_log(log_path, "force-final enabled; rerunning final stage")
        final_result = run_final_training(args, method_dir, best_layers, best_neurons, log_path)
        state["final_done"] = True
        state["final"] = final_result
        save_json(state_path, state)
        append_log(log_path, f"final done | best_stage={final_result.get('best_stage')}")
    elif args.skip_final:
        append_log(log_path, "final stage skipped by --skip-final")
    else:
        append_log(log_path, "final already done; nothing to do")

    summary_csv = method_dir / "method_summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "best_layers",
                "best_neurons",
                "search_best_proxy_loss",
                "search_best_param_count",
                "final_best_stage",
                "final_best_objective",
                "final_run_time_seconds",
            ]
        )
        search = state.get("search", {})
        final = state.get("final", {})
        writer.writerow(
            [
                method_name,
                search.get("best_layers"),
                search.get("best_neurons"),
                search.get("best_proxy_loss"),
                search.get("best_param_count"),
                final.get("best_stage"),
                final.get("best_objective"),
                final.get("run_time_seconds"),
            ]
        )

    append_log(log_path, f"=== {method_name} run completed ===")
    return state


def add_shared_args(parser: argparse.ArgumentParser, default_save_dir: str) -> None:
    default_path = Path(default_save_dir)
    if not default_path.is_absolute():
        default_path = (REPO_ROOT / default_path).resolve()
    parser.add_argument("--save-dir", type=str, default=str(default_path))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5000, help="Final training Adam epochs")
    parser.add_argument("--proxy-epochs", type=int, default=300, help="Search-time proxy Adam epochs")
    parser.add_argument("--skip-final", action="store_true", help="Only run architecture search")

    parser.add_argument("--layers-min", type=int, default=3)
    parser.add_argument("--layers-max", type=int, default=6)
    parser.add_argument("--neurons-min", type=int, default=64)
    parser.add_argument("--neurons-max", type=int, default=160)
    parser.add_argument(
        "--proxy-fail-penalty",
        type=float,
        default=1e30,
        help="Objective assigned when a proxy evaluation fails (e.g., CUDA OOM)",
    )

    parser.add_argument("--n-time-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs-max-iter", type=int, default=500)
    parser.add_argument("--lbfgs-col-points", type=int, default=2048)
    parser.add_argument("--lbfgs-ic-points", type=int, default=512)
    parser.add_argument("--lbfgs-bc-points", type=int, default=512)
    parser.add_argument("--lbfgs-time-steps", type=int, default=4)
    parser.add_argument("--lbfgs-history-size", type=int, default=20)
    parser.add_argument("--lbfgs-line-search", type=str, default="strong_wolfe", choices=["none", "strong_wolfe"])
    parser.add_argument("--pso-iters", type=int, default=8)
    parser.add_argument("--pso-swarm", type=int, default=16)
    parser.add_argument("--pso-span", type=float, default=0.25)

    parser.add_argument("--skip-lbfgs-final", action="store_true", help="Disable L-BFGS in final training")
    parser.add_argument("--use-pso-final", action="store_true", help="Enable PSO in final training")
    parser.add_argument("--force-final", action="store_true", help="Re-run final stage even if it was completed")

    parser.add_argument("--w-physics", type=float, default=50.0)
    parser.add_argument("--w-ic", type=float, default=1e-3)
    parser.add_argument("--w-bc", type=float, default=1e-18)
    parser.add_argument("--w-data", type=float, default=1e-2)
    parser.add_argument("--temp-ref-t0-mode", type=str, default="align_ic", choices=["align_ic", "drop", "keep"])
