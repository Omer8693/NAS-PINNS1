#!/usr/bin/env python3
"""Shared config and CLI arguments for the quench pipeline."""

import argparse
from pathlib import Path


WORK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WORK_ROOT.parent
BASELINE_SCRIPT = REPO_ROOT / "NAS-PINNS2" / "naspinn_baseline_with_quench_2026_data.py"
PYMOO_DIR = WORK_ROOT / "Pymoo"


def parse_args():
    parser = argparse.ArgumentParser(description="GPU-only paper-baseline quench pipeline")
    parser.add_argument("--save-dir", type=str, default=str((WORK_ROOT / "results" / "paper_gpu_baseline").resolve()))
    parser.add_argument("--methods", type=str, default="nsga2,nsga3,bayesian")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-device", type=str, default="0")
    parser.add_argument("--force-final", action="store_true")
    parser.add_argument("--skip-final", action="store_true")

    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--proxy-epochs", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--n-time-steps", type=int, default=10)
    parser.add_argument("--temp-ref-t0-mode", type=str, default="align_ic", choices=["align_ic", "drop", "keep"])

    parser.add_argument("--layers-min", type=int, default=3)
    parser.add_argument("--layers-max", type=int, default=6)
    parser.add_argument("--neurons-min", type=int, default=64)
    parser.add_argument("--neurons-max", type=int, default=160)
    parser.add_argument("--proxy-fail-penalty", type=float, default=1e30)

    parser.add_argument("--pop-size", type=int, default=24)
    parser.add_argument("--n-gen", type=int, default=16)
    parser.add_argument("--ref-partitions", type=int, default=10)
    parser.add_argument("--bo-init-points", type=int, default=4)
    parser.add_argument("--bo-iters", type=int, default=12)

    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--w-physics", type=float, default=50.0)
    parser.add_argument("--w-ic", type=float, default=1e-3)
    parser.add_argument("--w-bc", type=float, default=1e-18)
    parser.add_argument("--w-data", type=float, default=1e-2)

    parser.add_argument("--lbfgs-max-iter", type=int, default=5000)
    parser.add_argument("--lbfgs-col-points", type=int, default=2048)
    parser.add_argument("--lbfgs-ic-points", type=int, default=512)
    parser.add_argument("--lbfgs-bc-points", type=int, default=512)
    parser.add_argument("--lbfgs-time-steps", type=int, default=4)
    parser.add_argument("--lbfgs-history-size", type=int, default=20)
    parser.add_argument("--lbfgs-line-search", type=str, default="strong_wolfe", choices=["none", "strong_wolfe"])
    parser.add_argument("--skip-lbfgs-final", action="store_true")

    parser.add_argument("--use-pso-final", action="store_true", default=True)
    parser.add_argument("--no-pso-final", action="store_false", dest="use_pso_final")
    parser.add_argument("--pso-iters", type=int, default=8)
    parser.add_argument("--pso-swarm", type=int, default=16)
    parser.add_argument("--pso-span", type=float, default=0.25)
    return parser.parse_args()
