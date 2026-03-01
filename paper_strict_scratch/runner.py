from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .config import (
    ADVECTION1D_BETA_LIST,
    BURGERS1D_NU_LIST,
    EQUATION_CONFIGS,
    MASK_LEVELS,
)
from .equations import Advection1DEquation, Burgers1DEquation, Burgers2DEquation
from .model import SearchPINN, load_model_state
from .search import search_architecture
from .trainer import evaluate_stage, run_lbfgs, run_pso, set_seed, train_adam
from .visualization import plot_loss_curve, save_equation_plots


def parse_cases(equation_name: str, cases_csv: Optional[str]) -> List[float]:
    if equation_name == "burgers1d":
        default = list(BURGERS1D_NU_LIST)
    elif equation_name == "advection1d":
        default = list(ADVECTION1D_BETA_LIST)
    elif equation_name == "burgers2d":
        default = [0.0]
    else:
        raise ValueError(f"Unsupported equation: {equation_name}")

    if not cases_csv:
        return default

    values = [float(v.strip()) for v in cases_csv.split(",") if v.strip()]
    if equation_name == "burgers2d":
        # Single-case equation.
        return [0.0]
    return values


def build_equation(equation_name: str, cfg, case_value: float):
    if equation_name == "burgers1d":
        return Burgers1DEquation(cfg, nu=float(case_value))
    if equation_name == "advection1d":
        return Advection1DEquation(cfg, beta=float(case_value))
    if equation_name == "burgers2d":
        return Burgers2DEquation(cfg)
    raise ValueError(f"Unsupported equation: {equation_name}")


def _write_stage_history(path: Path, history: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        for i, loss in enumerate(history):
            w.writerow([i, float(loss)])


def _write_stage_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "train_loss", "rel_l2", "elapsed_sec"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_stage_metrics(
    path: Path,
    method: str,
    equation_name: str,
    case_label: str,
    seed: int,
    stage_name: str,
    train_loss: float,
    rel_l2: float,
    elapsed_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "equation",
                "case",
                "seed",
                "stage",
                "train_loss",
                "rel_l2",
                "elapsed_sec",
            ]
        )
        writer.writerow(
            [
                method,
                equation_name,
                case_label,
                int(seed),
                stage_name,
                f"{float(train_loss):.8e}",
                f"{float(rel_l2):.8e}",
                f"{float(elapsed_sec):.6f}",
            ]
        )


def _write_l2_file(path: Path, stage_name: str, rel_l2: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"stage,{stage_name}\n")
        f.write(f"rel_l2,{float(rel_l2):.8e}\n")


def _copy_stage_to_root(stage_dir: Path, root_dir: Path) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    for src in stage_dir.iterdir():
        if not src.is_file():
            continue
        shutil.copy2(src, root_dir / src.name)


def run_single_case(
    method: str,
    equation_name: str,
    cfg,
    equation,
    seed: int,
    run_dir: Path,
    device: torch.device,
    skip_lbfgs: bool,
    skip_pso: bool,
) -> Dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    model = SearchPINN(
        input_dim=cfg.input_dim,
        hidden_layers=cfg.hidden_layers,
        base_neurons=cfg.base_neurons,
        mask_levels=MASK_LEVELS,
    ).to(device)
    train_data = equation.sample_train(device)

    search_info: Dict[str, object] = {}
    t_start = time.perf_counter()

    if method == "naspinn":
        print(f"Starting NAS-PINN baseline: case={equation.case_label()}")
        adam_out = train_adam(
            model=model,
            equation=equation,
            train_data=train_data,
            stage_cfg=cfg.stage,
            mask_indices=None,
            optimize_arch=True,
        )
        load_model_state(model, adam_out.best_state)
        best_masks = model.infer_best_masks()
    else:
        search_out = search_architecture(method, cfg, equation, device=device, seed=seed)
        best_masks = list(search_out.masks)
        search_info = {
            "proxy_loss": float(search_out.proxy_loss),
            "effective_params": int(search_out.effective_params),
        }
        print(f"Best mask selections: {best_masks}")

        # Re-initialize for final stage training after architecture search.
        set_seed(seed)
        model = SearchPINN(
            input_dim=cfg.input_dim,
            hidden_layers=cfg.hidden_layers,
            base_neurons=cfg.base_neurons,
            mask_levels=MASK_LEVELS,
        ).to(device)
        train_data = equation.sample_train(device)

        adam_out = train_adam(
            model=model,
            equation=equation,
            train_data=train_data,
            stage_cfg=cfg.stage,
            mask_indices=best_masks,
            optimize_arch=False,
        )
        load_model_state(model, adam_out.best_state)

    print(f"Best mask selections: {best_masks}")

    stages: Dict[str, Dict[str, object]] = {}
    case_label = equation.case_label()

    adam_stage_dir = run_dir / "stage_adam"
    adam_stage_dir.mkdir(parents=True, exist_ok=True)
    adam_eval = evaluate_stage(
        model=model,
        equation=equation,
        train_data=train_data,
        stage_name="adam",
        mask_indices=best_masks,
        device=device,
        train_loss=adam_out.best_loss,
        history=adam_out.history,
        elapsed_sec=adam_out.elapsed_sec,
    )
    _write_stage_history(adam_stage_dir / "loss_history.csv", adam_out.history)
    plot_loss_curve(
        adam_out.history,
        adam_stage_dir / "loss_curve.png",
        title=f"{equation_name} {method} loss (ADAM, {case_label})",
    )
    _write_l2_file(adam_stage_dir / "l2_error.txt", "adam", adam_eval.rel_l2)
    _write_stage_metrics(
        adam_stage_dir / "metrics.csv",
        method=method,
        equation_name=equation_name,
        case_label=case_label,
        seed=seed,
        stage_name="adam",
        train_loss=adam_eval.train_loss,
        rel_l2=adam_eval.rel_l2,
        elapsed_sec=adam_eval.elapsed_sec,
    )
    save_equation_plots(
        equation=equation,
        model=model,
        mask_indices=best_masks,
        device=device,
        stage_dir=adam_stage_dir,
        rel_l2=adam_eval.rel_l2,
    )
    stages["adam"] = {
        "train_loss": adam_eval.train_loss,
        "rel_l2": adam_eval.rel_l2,
        "elapsed_sec": adam_eval.elapsed_sec,
    }

    adam_best_state = adam_out.best_state

    if not skip_lbfgs:
        print("L-BFGS refinement...")
        load_model_state(model, adam_best_state)
        lb_loss, lb_hist, lb_elapsed = run_lbfgs(
            model=model,
            equation=equation,
            train_data=train_data,
            stage_cfg=cfg.stage,
            mask_indices=best_masks,
            device=device,
        )
        lb_eval = evaluate_stage(
            model=model,
            equation=equation,
            train_data=train_data,
            stage_name="lbfgs",
            mask_indices=best_masks,
            device=device,
            train_loss=lb_loss,
            history=lb_hist,
            elapsed_sec=lb_elapsed,
        )
        lb_dir = run_dir / "stage_lbfgs"
        _write_stage_history(lb_dir / "loss_history.csv", lb_hist)
        plot_loss_curve(
            lb_hist if lb_hist else adam_out.history,
            lb_dir / "loss_curve.png",
            title=f"{equation_name} {method} loss (LBFGS, {case_label})",
        )
        _write_l2_file(lb_dir / "l2_error.txt", "lbfgs", lb_eval.rel_l2)
        _write_stage_metrics(
            lb_dir / "metrics.csv",
            method=method,
            equation_name=equation_name,
            case_label=case_label,
            seed=seed,
            stage_name="lbfgs",
            train_loss=lb_eval.train_loss,
            rel_l2=lb_eval.rel_l2,
            elapsed_sec=lb_eval.elapsed_sec,
        )
        save_equation_plots(
            equation=equation,
            model=model,
            mask_indices=best_masks,
            device=device,
            stage_dir=lb_dir,
            rel_l2=lb_eval.rel_l2,
        )
        stages["lbfgs"] = {
            "train_loss": lb_eval.train_loss,
            "rel_l2": lb_eval.rel_l2,
            "elapsed_sec": lb_eval.elapsed_sec,
        }

    if not skip_pso:
        print("PSO refinement...")
        load_model_state(model, adam_best_state)
        pso_loss, pso_hist, pso_elapsed = run_pso(
            model=model,
            equation=equation,
            train_data=train_data,
            stage_cfg=cfg.stage,
            mask_indices=best_masks,
            device=device,
        )
        print(f"PSO best objective: {pso_loss:.4e}")
        pso_eval = evaluate_stage(
            model=model,
            equation=equation,
            train_data=train_data,
            stage_name="pso",
            mask_indices=best_masks,
            device=device,
            train_loss=pso_loss,
            history=pso_hist,
            elapsed_sec=pso_elapsed,
        )
        pso_dir = run_dir / "stage_pso"
        _write_stage_history(pso_dir / "loss_history.csv", pso_hist)
        plot_loss_curve(
            pso_hist if pso_hist else adam_out.history,
            pso_dir / "loss_curve.png",
            title=f"{equation_name} {method} loss (PSO, {case_label})",
        )
        _write_l2_file(pso_dir / "l2_error.txt", "pso", pso_eval.rel_l2)
        _write_stage_metrics(
            pso_dir / "metrics.csv",
            method=method,
            equation_name=equation_name,
            case_label=case_label,
            seed=seed,
            stage_name="pso",
            train_loss=pso_eval.train_loss,
            rel_l2=pso_eval.rel_l2,
            elapsed_sec=pso_eval.elapsed_sec,
        )
        save_equation_plots(
            equation=equation,
            model=model,
            mask_indices=best_masks,
            device=device,
            stage_dir=pso_dir,
            rel_l2=pso_eval.rel_l2,
        )
        stages["pso"] = {
            "train_loss": pso_eval.train_loss,
            "rel_l2": pso_eval.rel_l2,
            "elapsed_sec": pso_eval.elapsed_sec,
        }

    best_stage, best_stats = min(stages.items(), key=lambda kv: kv[1]["rel_l2"])
    print(f"Selected best stage: {best_stage} (rel_l2={best_stats['rel_l2']:.4e})")

    summary_rows = [
        {
            "stage": stage_name,
            "train_loss": stage_stats["train_loss"],
            "rel_l2": stage_stats["rel_l2"],
            "elapsed_sec": stage_stats["elapsed_sec"],
        }
        for stage_name, stage_stats in stages.items()
    ]
    _write_stage_summary(run_dir / "results_summary.csv", summary_rows)

    best_stage_dir = run_dir / f"stage_{best_stage}"
    stage_best_dir = run_dir / "stage_best"
    if stage_best_dir.exists():
        shutil.rmtree(stage_best_dir)
    shutil.copytree(best_stage_dir, stage_best_dir)
    with (stage_best_dir / "selected_stage.txt").open("w", encoding="utf-8") as f:
        f.write(f"{best_stage}\n")
    _copy_stage_to_root(best_stage_dir, run_dir)

    total_time = time.perf_counter() - t_start
    out = {
        "equation": equation_name,
        "case": case_label,
        "method": method,
        "seed": int(seed),
        "best_masks": list(best_masks),
        "stages": stages,
        "best_stage": best_stage,
        "best_rel_l2": float(best_stats["rel_l2"]),
        "run_time_sec": float(total_time),
        "search": search_info,
    }

    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Run time: {total_time:.2f} s")
    return out


def run_experiment(
    method: str,
    equation_name: str,
    save_dir: Path,
    repeats: Optional[int] = None,
    base_seed: Optional[int] = None,
    cases_csv: Optional[str] = None,
    skip_lbfgs: bool = False,
    skip_pso: bool = False,
    epochs: Optional[int] = None,
    proxy_epochs: Optional[int] = None,
    lbfgs_max_iter: Optional[int] = None,
    pso_iters: Optional[int] = None,
    pso_swarm: Optional[int] = None,
    pso_span: Optional[float] = None,
    pop_size: Optional[int] = None,
    n_gen: Optional[int] = None,
    ref_partitions: Optional[int] = None,
    bo_init_points: Optional[int] = None,
    bo_iters: Optional[int] = None,
) -> Path:
    method = method.lower()
    if method not in {"naspinn", "nsga2", "nsga3", "bayesian"}:
        raise ValueError(f"Unsupported method: {method}")

    if equation_name not in EQUATION_CONFIGS:
        raise ValueError(f"Unsupported equation: {equation_name}")

    cfg = EQUATION_CONFIGS[equation_name]
    if repeats is not None:
        cfg = replace(cfg, repeats=int(repeats))
    if base_seed is not None:
        cfg = replace(cfg, base_seed=int(base_seed))
    stage_cfg = cfg.stage
    if epochs is not None:
        stage_cfg = replace(stage_cfg, epochs=int(epochs))
    if lbfgs_max_iter is not None:
        stage_cfg = replace(stage_cfg, lbfgs_max_iter=int(lbfgs_max_iter))
    if pso_iters is not None:
        stage_cfg = replace(stage_cfg, pso_iters=int(pso_iters))
    if pso_swarm is not None:
        stage_cfg = replace(stage_cfg, pso_swarm=int(pso_swarm))
    if pso_span is not None:
        stage_cfg = replace(stage_cfg, pso_span=float(pso_span))
    cfg = replace(cfg, stage=stage_cfg)

    search_cfg = cfg.search
    if proxy_epochs is not None:
        search_cfg = replace(search_cfg, proxy_epochs=int(proxy_epochs))
    if pop_size is not None:
        search_cfg = replace(search_cfg, pop_size=int(pop_size))
    if n_gen is not None:
        search_cfg = replace(search_cfg, n_gen=int(n_gen))
    if ref_partitions is not None:
        search_cfg = replace(search_cfg, ref_partitions=int(ref_partitions))
    if bo_init_points is not None:
        search_cfg = replace(search_cfg, bo_init_points=int(bo_init_points))
    if bo_iters is not None:
        search_cfg = replace(search_cfg, bo_iters=int(bo_iters))
    cfg = replace(cfg, search=search_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: {asdict(cfg)}")

    cases = parse_cases(equation_name, cases_csv)

    save_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows: List[Dict[str, object]] = []

    for case_value in cases:
        equation = build_equation(equation_name, cfg, case_value)
        case_label = equation.case_label()

        for rep in range(cfg.repeats):
            run_seed = int(cfg.base_seed + rep)
            run_dir = save_dir / f"rep_{rep + 1:02d}" / equation_name / method / case_label
            print("=" * 72)
            print(f"[RUN ] {method}/{equation_name}/{case_label} (seed={run_seed})")
            result = run_single_case(
                method=method,
                equation_name=equation_name,
                cfg=cfg,
                equation=equation,
                seed=run_seed,
                run_dir=run_dir,
                device=device,
                skip_lbfgs=skip_lbfgs,
                skip_pso=skip_pso,
            )
            aggregate_rows.append(result)
            print(f"[OK  ] {method}/{equation_name}/{case_label} (rep={rep + 1})")

    summary_csv = save_dir / f"summary_{equation_name}_{method}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "equation",
                "case",
                "method",
                "seed",
                "best_masks",
                "best_stage",
                "best_rel_l2",
                "run_time_sec",
            ],
        )
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(
                {
                    "equation": row["equation"],
                    "case": row["case"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "best_masks": ";".join(str(m) for m in row["best_masks"]),
                    "best_stage": row["best_stage"],
                    "best_rel_l2": row["best_rel_l2"],
                    "run_time_sec": row["run_time_sec"],
                }
            )

    summary_json = save_dir / f"summary_{equation_name}_{method}.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(aggregate_rows, f, indent=2)

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    return summary_csv
