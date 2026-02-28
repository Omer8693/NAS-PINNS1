#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from paper_strict_scratch.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict-paper NSGA-III NAS-PINN runner")
    parser.add_argument("--equation", type=str, choices=["burgers1d", "advection1d", "burgers2d"], required=True)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cases", type=str, default=None, help="CSV values: nu list for burgers1d, beta list for advection1d")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--proxy-epochs", type=int, default=None)
    parser.add_argument("--lbfgs-max-iter", type=int, default=None)
    parser.add_argument("--pso-iters", type=int, default=None)
    parser.add_argument("--pso-swarm", type=int, default=None)
    parser.add_argument("--pso-span", type=float, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--n-gen", type=int, default=None)
    parser.add_argument("--ref-partitions", type=int, default=None)
    parser.add_argument("--bo-init-points", type=int, default=None)
    parser.add_argument("--bo-iters", type=int, default=None)
    parser.add_argument("--skip-lbfgs", action="store_true")
    parser.add_argument("--skip-pso", action="store_true")
    args = parser.parse_args()

    if args.save_dir:
        out_dir = Path(args.save_dir)
    else:
        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results") / "paper_strict_scratch" / args.equation / "nsga3" / stamp

    run_experiment(
        method="nsga3",
        equation_name=args.equation,
        save_dir=out_dir,
        repeats=args.repeats,
        base_seed=args.seed,
        cases_csv=args.cases,
        skip_lbfgs=args.skip_lbfgs,
        skip_pso=args.skip_pso,
        epochs=args.epochs,
        proxy_epochs=args.proxy_epochs,
        lbfgs_max_iter=args.lbfgs_max_iter,
        pso_iters=args.pso_iters,
        pso_swarm=args.pso_swarm,
        pso_span=args.pso_span,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        ref_partitions=args.ref_partitions,
        bo_init_points=args.bo_init_points,
        bo_iters=args.bo_iters,
    )


if __name__ == "__main__":
    main()
