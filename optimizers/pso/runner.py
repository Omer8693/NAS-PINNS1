import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from fuzzy_pso import PSO


@dataclass
class ParamSpec:
    name: str
    low: float
    high: float
    kind: str = "float"
    log: bool = False


TARGET_CONFIGS = {
    "burgers-naspinn": {
        "script": "NAS_PINNs_burgers.py",
        "params": [
            ParamSpec("epochs", 500, 4000, kind="int"),
            ParamSpec("nu", 0.003, 0.10, kind="float", log=False),
        ],
        "defaults": ["--skip-lbfgs"],
    },
    "burgers-nsga2": {
        "script": "NAS_PINNs_burgers_nsga2.py",
        "params": [
            ParamSpec("epochs", 500, 4000, kind="int"),
            ParamSpec("lr", 5e-4, 2e-3, kind="float", log=True),
            ParamSpec("lambda-ic", 20.0, 200.0, kind="float"),
            ParamSpec("lambda-bc", 20.0, 200.0, kind="float"),
        ],
        "defaults": ["--skip-lbfgs", "--skip-nsga"],
    },
    "burgers-nsga3": {
        "script": "NAS_PINNs_burgers_nsga3.py",
        "params": [
            ParamSpec("epochs", 500, 4000, kind="int"),
            ParamSpec("lr", 5e-4, 2e-3, kind="float", log=True),
            ParamSpec("lambda-ic", 20.0, 200.0, kind="float"),
            ParamSpec("lambda-bc", 20.0, 200.0, kind="float"),
        ],
        "defaults": ["--skip-lbfgs", "--skip-nsga"],
    },
    "burgers-bayesian": {
        "script": "NAS_PINNs_burgers_bayesian.py",
        "params": [
            ParamSpec("epochs", 500, 4000, kind="int"),
            ParamSpec("bo-init-points", 1, 6, kind="int"),
            ParamSpec("bo-iters", 2, 12, kind="int"),
            ParamSpec("bo-epochs", 100, 1200, kind="int"),
        ],
        "defaults": ["--skip-lbfgs"],
    },
    "poisson-naspinn": {
        "script": "NAS_PINNs_poisson.py",
        "params": [
            ParamSpec("epochs", 500, 4000, kind="int"),
        ],
        "defaults": ["--skip-lbfgs"],
    },
    "poisson-nsga2": {
        "script": "NAS_PINNs_poisson_nsga2.py",
        "params": [
            ParamSpec("epochs", 500, 4000, kind="int"),
            ParamSpec("search-pop", 8, 24, kind="int"),
            ParamSpec("search-gen", 3, 12, kind="int"),
            ParamSpec("search-eval-epochs", 100, 1200, kind="int"),
        ],
        "defaults": ["--skip-lbfgs"],
    },
    "poisson-nsga3": {
        "script": "NAS_PINNs_poisson_nsga3.py",
        "params": [
            ParamSpec("epochs", 500, 4000, kind="int"),
            ParamSpec("search-pop", 8, 24, kind="int"),
            ParamSpec("search-gen", 3, 12, kind="int"),
            ParamSpec("search-eval-epochs", 100, 1200, kind="int"),
        ],
        "defaults": ["--skip-lbfgs"],
    },
    "poisson-bayesian": {
        "script": "NAS_PINNs_poisson_bayesian.py",
        "params": [
            ParamSpec("epochs", 500, 4000, kind="int"),
            ParamSpec("bo-init-points", 1, 6, kind="int"),
            ParamSpec("bo-iters", 2, 12, kind="int"),
            ParamSpec("bo-epochs", 100, 1200, kind="int"),
        ],
        "defaults": ["--skip-lbfgs"],
    },
}


REL_PATTERNS = [
    re.compile(r"Relative L2 error \(full grid\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    re.compile(r"Final relative L2 error:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
]


def _decode_value(spec: ParamSpec, x_val: float):
    val = float(x_val)
    if spec.log:
        val = 10 ** val
    if spec.kind == "int":
        return int(round(val))
    return float(val)


def _candidate_to_cli(params, specs):
    cli = []
    for spec, val in zip(specs, params):
        decoded = _decode_value(spec, val)
        cli.extend([f"--{spec.name}", str(decoded)])
    return cli


def _extract_rel_l2(stdout_text: str):
    found = []
    for patt in REL_PATTERNS:
        found.extend([float(x) for x in patt.findall(stdout_text)])
    if not found:
        return None
    return found[-1]


def _read_rel_l2_from_file(run_dir: str):
    l2_file = os.path.join(run_dir, "l2_error.txt")
    if not os.path.exists(l2_file):
        return None

    try:
        with open(l2_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("rel_l2,"):
                    return float(line.strip().split(",")[-1])
    except Exception:
        return None

    return None


def _read_runtime_seconds(run_dir: str):
    rt_file = os.path.join(run_dir, "run_time.txt")
    if not os.path.exists(rt_file):
        return math.nan
    with open(rt_file, "r", encoding="utf-8") as f:
        line = f.read().strip()
    try:
        return float(line.split(",")[-1])
    except Exception:
        return math.nan


def _run_target(script, save_dir, base_args, candidate_args, timeout):
    os.makedirs(save_dir, exist_ok=True)
    cmd = [sys.executable, script, "--save-dir", save_dir] + base_args + candidate_args
    started = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    rel_l2 = _read_rel_l2_from_file(save_dir)
    if rel_l2 is None:
        rel_l2 = _extract_rel_l2(proc.stdout)
    run_time = _read_runtime_seconds(save_dir)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "rel_l2": rel_l2,
        "wall_time": elapsed,
        "run_time": run_time,
    }


class ScriptTuningProblem(ElementwiseProblem):
    def __init__(self, script, specs, base_args, runs_root, timeout=0):
        xl = []
        xu = []
        for s in specs:
            if s.log:
                xl.append(np.log10(s.low))
                xu.append(np.log10(s.high))
            else:
                xl.append(s.low)
                xu.append(s.high)

        super().__init__(n_var=len(specs), n_obj=1, n_ieq_constr=0, xl=np.array(xl), xu=np.array(xu))
        self.script = script
        self.specs = specs
        self.base_args = base_args
        self.runs_root = runs_root
        self.timeout = timeout
        self.eval_counter = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_counter += 1
        run_dir = os.path.join(self.runs_root, f"eval_{self.eval_counter:03d}")
        candidate_args = _candidate_to_cli(x, self.specs)

        result = _run_target(self.script, run_dir, self.base_args, candidate_args, timeout=self.timeout)

        penalty = 10.0
        if result["returncode"] != 0 or result["rel_l2"] is None:
            score = penalty
        else:
            score = float(result["rel_l2"])

        out["F"] = [score]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "rel_l2", "run_time_seconds", "wall_time_seconds", "save_dir", "params"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    p = argparse.ArgumentParser(description="PSO-based separate tuning/comparison runner (non-invasive)")
    p.add_argument("--target", required=True, choices=sorted(TARGET_CONFIGS.keys()))
    p.add_argument("--output-dir", type=str, default="results/pso_compare")
    p.add_argument("--base-args", type=str, default="", help="extra args passed to each target run")

    p.add_argument("--pop-size", type=int, default=8)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--w", type=float, default=0.9)
    p.add_argument("--c1", type=float, default=2.0)
    p.add_argument("--c2", type=float, default=2.0)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--initial-velocity", type=str, default="random", choices=["random", "zero"])
    p.add_argument("--max-velocity-rate", type=float, default=0.20)
    p.add_argument("--pertube-best", action="store_true")

    p.add_argument("--timeout", type=int, default=0, help="timeout in seconds per evaluation (0 disables)")
    return p


def run_for_target(target: str, args):
    cfg = TARGET_CONFIGS[target]
    script = cfg["script"]
    specs = cfg["params"]
    base_args = list(cfg.get("defaults", [])) + shlex.split(args.base_args)

    root_out = os.path.join(args.output_dir, target)
    os.makedirs(root_out, exist_ok=True)

    baseline_dir = os.path.join(root_out, "baseline")
    baseline = _run_target(script, baseline_dir, base_args, [], timeout=args.timeout if args.timeout > 0 else None)

    problem = ScriptTuningProblem(
        script=script,
        specs=specs,
        base_args=base_args,
        runs_root=os.path.join(root_out, "pso_evals"),
        timeout=args.timeout if args.timeout > 0 else None,
    )

    algorithm = PSO(
        pop_size=args.pop_size,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        adaptive=args.adaptive,
        initial_velocity=args.initial_velocity,
        max_velocity_rate=args.max_velocity_rate,
        pertube_best=args.pertube_best,
    )

    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", args.generations),
        seed=args.seed,
        save_history=False,
        verbose=True,
    )

    best_x = np.atleast_1d(res.X)
    best_params_dict = {s.name: _decode_value(s, v) for s, v in zip(specs, best_x)}
    best_cli = _candidate_to_cli(best_x, specs)

    best_dir = os.path.join(root_out, "pso_best")
    best = _run_target(script, best_dir, base_args, best_cli, timeout=args.timeout if args.timeout > 0 else None)

    summary_rows = [
        {
            "mode": "baseline",
            "rel_l2": baseline["rel_l2"],
            "run_time_seconds": baseline["run_time"],
            "wall_time_seconds": baseline["wall_time"],
            "save_dir": baseline_dir,
            "params": json.dumps({}, ensure_ascii=False),
        },
        {
            "mode": "pso_best",
            "rel_l2": best["rel_l2"],
            "run_time_seconds": best["run_time"],
            "wall_time_seconds": best["wall_time"],
            "save_dir": best_dir,
            "params": json.dumps(best_params_dict, ensure_ascii=False),
        },
    ]

    summary_csv = os.path.join(root_out, "pso_comparison.csv")
    _write_csv(summary_csv, summary_rows)

    with open(os.path.join(root_out, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params_dict, f, indent=2, ensure_ascii=False)

    with open(os.path.join(root_out, "baseline_stdout.txt"), "w", encoding="utf-8") as f:
        f.write(baseline["stdout"])
    with open(os.path.join(root_out, "pso_best_stdout.txt"), "w", encoding="utf-8") as f:
        f.write(best["stdout"])

    print(f"Saved PSO comparison: {summary_csv}")
    print(f"Best PSO params: {best_params_dict}")


def main(default_target=None):
    parser = build_parser()
    args = parser.parse_args()
    target = default_target or args.target
    run_for_target(target, args)


if __name__ == "__main__":
    main()
