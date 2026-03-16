"""Core experiment orchestration for NAS-PINN benchmarks."""

import json
import os
import time
from typing import Callable, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from problems import get_problem
from src.opt_nsga2 import run_nsga2
from src.opt_nsga3 import run_nsga3
from src.opt_bayesian import run_bayesian
from src.baseline_data import BaselinePaper
from src.config import DEVICE
from src.physics_model import TemporalSkipScheduler
from src.pinn_network import build_network_from_config
from src.trainers import full_training_pipeline

AVAILABLE_PROBLEMS = ("quenching", "poisson", "burgers", "allen_cahn")
DEFAULT_PROBLEM_NAME = "quenching"
_TEMPORAL_PROBLEMS = {"quenching", "burgers", "allen_cahn"}
TIME_MODES = ("full", "fixed", "adaptive")


def normalize_problem_name(requested_name: str) -> str:
    name = str(requested_name).strip().lower()
    if name not in AVAILABLE_PROBLEMS:
        allowed = ", ".join(AVAILABLE_PROBLEMS)
        raise ValueError(f"Unknown problem: {requested_name!r}. Choose from: {allowed}")
    return name


def normalize_time_mode(requested_mode: str) -> str:
    mode = str(requested_mode).strip().lower()
    if mode not in TIME_MODES:
        allowed = ", ".join(TIME_MODES)
        raise ValueError(f"Unknown time mode: {requested_mode!r}. Choose from: {allowed}")
    return mode


def stabilize_poisson_config(config: dict) -> dict:
    """Clamp Poisson architectures to a stable minimum capacity."""
    cfg = dict(config)
    neurons = [int(n) for n in cfg.get("neurons", [])]
    if not neurons:
        neurons = [96]

    target_layers = max(int(cfg.get("n_layers", len(neurons) or 1)), 3)
    mean_width = int(np.clip(round(float(np.mean(neurons))), 64, 128))
    stable_neurons = []
    for i in range(target_layers):
        src = neurons[i] if i < len(neurons) else mean_width
        stable_neurons.append(int(np.clip(src, 64, 128)))

    changed = (
        cfg.get("n_layers") != target_layers or
        cfg.get("neurons") != stable_neurons or
        cfg.get("activation") not in {"sin", "tanh"} or
        bool(cfg.get("residual", False))
    )

    cfg["n_layers"] = target_layers
    cfg["neurons"] = stable_neurons
    if cfg.get("activation") not in {"sin", "tanh"}:
        cfg["activation"] = "sin"
    cfg["residual"] = False
    cfg["_stabilized_for_poisson"] = changed
    return cfg


def make_quick_train_fn(problem) -> Callable:
    """Short Adam training loop used by NAS evaluators."""
    loss_fn = problem.make_loss_fn()
    val_fn  = problem.make_val_fn()

    def quick_train(config: dict, n_epochs: int = 300):
        cfg = dict(config)
        cfg["n_input"]  = problem.n_input
        cfg["n_output"] = problem.n_output
        model     = build_network_from_config(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        get_batch = problem.make_batch_fn()

        for epoch in range(1, n_epochs + 1):
            batch = get_batch(epoch)
            # domain tensörü requires_grad olmayabilir (bazı problemlerde)
            domain = batch.get("domain")
            if isinstance(domain, torch.Tensor) and not domain.requires_grad:
                batch["domain"] = domain.requires_grad_(True)

            optimizer.zero_grad()
            loss, _ = loss_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        return val_fn(model), model.count_params()

    return quick_train


_SKIP_THRESHOLDS = {
    "quenching":  1e-3,
    "burgers":    5e-2,
    "allen_cahn": 1e-2,
}


def _make_scheduler(problem_name: str, problem, time_mode: str) -> Optional[TemporalSkipScheduler]:
    if time_mode != "adaptive":
        return None
    if problem_name not in _TEMPORAL_PROBLEMS:
        return None
    if "t" not in problem.domain_bounds:
        return None
    t_start, t_end = problem.domain_bounds["t"]
    threshold = _SKIP_THRESHOLDS.get(problem_name, 1e-2)
    return TemporalSkipScheduler(
        t_start=float(t_start),
        t_end=float(t_end),
        n_initial_steps=20,
        skip_threshold=threshold,
    )


# ─────────────────────────────────────────────────────────────
# Görselleştirme
# ─────────────────────────────────────────────────────────────

def _plot_quenching_results(model, problem, train_res, skip_stats, paper_metrics, save_dir, label):
    """
    Quenching sonuçlarını görselleştir:
      Panel 1: Analitik vs PINN sıcaklık alanı (t_mid'de)
      Panel 2: Hata alanı
      Panel 3: Soğuma eğrisi karşılaştırması (PINN + analitik + Fig 7 su sıcaklığı)
    """
    os.makedirs(save_dir, exist_ok=True)
    from src.config import T_END, X_MAX, Y_MAX

    t_min, t_max = problem.domain_bounds["t"]
    x_min, x_max = problem.domain_bounds["x"]
    y_min, y_max = problem.domain_bounds["y"]

    # Sıcaklık alanı
    t_val = 0.5 * (t_min + t_max)
    nx, ny = 50, 30
    x_g = torch.linspace(x_min, x_max, nx, device=DEVICE)
    y_g = torch.linspace(y_min, y_max, ny, device=DEVICE)
    xx, yy = torch.meshgrid(x_g, y_g, indexing="ij")
    tt = torch.full_like(xx, t_val)
    coords_grid = torch.stack([tt.flatten(), xx.flatten(), yy.flatten()], dim=1)

    model.eval()
    with torch.no_grad():
        pred_T = model(coords_grid).cpu().numpy().reshape(nx, ny)
    exact_T = problem.analytical_solution(coords_grid).cpu().numpy().reshape(nx, ny)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    vmin = min(pred_T.min(), exact_T.min())
    vmax = max(pred_T.max(), exact_T.max())

    im0 = axes[0].imshow(exact_T, origin="lower", aspect="auto",
                          extent=[y_min, y_max, x_min, x_max],
                          cmap="hot", vmin=vmin, vmax=vmax)
    axes[0].set_title("Analytical Reference")
    axes[0].set_xlabel("y [m]"); axes[0].set_ylabel("x [m]")
    plt.colorbar(im0, ax=axes[0], label="T [°C]")

    im1 = axes[1].imshow(pred_T, origin="lower", aspect="auto",
                          extent=[y_min, y_max, x_min, x_max],
                          cmap="hot", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{label} Prediction (t={t_val:.1f}s)")
    axes[1].set_xlabel("y [m]")
    plt.colorbar(im1, ax=axes[1], label="T [°C]")

    err = np.abs(pred_T - exact_T)
    im2 = axes[2].imshow(err, origin="lower", aspect="auto",
                          extent=[y_min, y_max, x_min, x_max],
                          cmap="Reds")
    axes[2].set_title(f"|Error| MAE={err.mean():.2f} °C")
    axes[2].set_xlabel("y [m]")
    plt.colorbar(im2, ax=axes[2], label="|Error| [°C]")

    plt.suptitle(f"Temperature Field Comparison - {label}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "exact_vs_pred.png"), dpi=150, bbox_inches="tight")
    plt.close()

    _plot_training_curves(train_res, save_dir, label, filename="loss_l2_comparison.png")
    _plot_skip_stats(skip_stats, save_dir, label)

    # Soğuma eğrisi karşılaştırması (paper ile)
    if paper_metrics and "T_PINN" in paper_metrics:
        _plot_cooling_curve(paper_metrics, save_dir, label)


def _plot_cooling_curve(paper_metrics: dict, save_dir: str, label: str):
    """
    PLAN.md Grafik 1: Zaman - Sıcaklık karşılaştırması.
    Seriler:
      - PINN (bu model)
      - Analitik soğuma yaklaşımı
      - Paper Fig 7 su tankı sıcaklığı (Channel 1 — referans bağlam)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    t_pinn    = paper_metrics["time_s"]
    T_pinn    = paper_metrics["T_PINN"]
    T_anal    = paper_metrics["T_analytical"]
    t_fig7    = paper_metrics["fig7_time_s"]
    T_water_ch1 = paper_metrics["fig7_water_ch1"]

    ax = axes[0]
    ax.plot(t_pinn, T_anal, "k-",  lw=2.0, label="Analytical approx (reference)")
    ax.plot(t_pinn, T_pinn, "r--", lw=2.0, label=f"PINN ({label}) — part surface")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Part Cooling Curve: PINN vs Analytical")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Metrikleri göster
    metrics_text = (
        f"L2_rel = {paper_metrics['L2_rel']:.4f}\n"
        f"MAE    = {paper_metrics['MAE_C']:.2f} °C\n"
        f"RMSE   = {paper_metrics['RMSE_C']:.2f} °C\n"
        f"R²     = {paper_metrics['R2']:.4f}"
    )
    ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
            va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax2 = axes[1]
    ax2.plot(t_fig7, T_water_ch1, "b-o", markersize=4, lw=1.5,
             label="Paper Fig 7 – Water Ch1 (top)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Water Temperature [°C]")
    ax2.set_title("Paper Fig 7: Water Tank Temperature Rise\n(reference context)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, 0.05, "Note: water temp ≠ part temp\nShown as process context only",
             transform=ax2.transAxes, ha="center", fontsize=8,
             color="gray", style="italic")

    plt.suptitle(
        f"Paper Baseline Comparison — {label}\n"
        f"(Part cooling vs analytical proxy | Fig 7 water temp as context)",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "paper_cooling_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] paper_cooling_comparison.png kaydedildi")


def _plot_paper_distortion_reference(save_dir: str):
    """
    PLAN.md Grafik 2: Fig 17 distorsiyon verisi — PINN mekanik modeli olmadan
    sadece paper FEM ve ölçüm verisini göster; PINN sütunu eksik olarak işaretle.
    """
    bl = BaselinePaper()
    ds = bl.get("fig17_ht")
    points = ds["points"]
    n = len(points)
    x_idx = np.arange(n)

    measured  = np.array(ds["Measured_Bottom"])
    simulated = np.array(ds["Simulated_Bottom"])

    fig, ax = plt.subplots(figsize=(15, 5))
    width = 0.3
    ax.bar(x_idx - width / 2, measured,  width, label="Measured (Paper Fig 17)",
           color="#212121", alpha=0.85)
    ax.bar(x_idx + width / 2, simulated, width, label="FEM (baseline_paper Fig 18)",
           color="#1565C0", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_idx[::2])
    ax.set_xticklabels(points[::2], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Distortion [mm]")
    ax.set_title(
        "Fig 17/18 — HT Quenching Distortion: Bottom Layer\n"
        "(PINN distortion comparison requires mechanical model — not yet implemented)",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # PINN eksik notu
    ax.text(0.5, 0.02,
            "PINN currently predicts thermal field only. "
            "Distortion comparison requires coupling with mechanical model.",
            transform=ax.transAxes, ha="center", fontsize=8,
            color="#E53935", style="italic")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "paper_reference_fig17_bottom.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {save_path} kaydedildi")


def _plot_training_curves(train_res, save_dir, label, filename="loss_history.png"):
    colors = {"adam": "#1565C0", "lbfgs": "#E53935", "pso": "#2E7D32"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    for phase_key in ["adam", "lbfgs", "pso"]:
        if phase_key in train_res:
            hist = train_res[phase_key]["history"]["loss"]
            if hist:
                ax.semilogy(hist, label=phase_key.upper(),
                            color=colors[phase_key], linewidth=1.5)
    ax.set_xlabel("Step / Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss History")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    phases = [
        key for key in ["adam", "lbfgs", "pso"]
        if key in train_res and train_res[key].get("final_l2") is not None
        and np.isfinite(train_res[key]["final_l2"])
    ]
    l2_vals = [train_res[key]["final_l2"] for key in phases]
    if phases:
        bars = ax.bar(phases, l2_vals, color=[colors[key] for key in phases],
                      alpha=0.85, width=0.5)
        y_top = max(l2_vals) * 1.15 if l2_vals else 1.0
        for bar, val in zip(bars, l2_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + y_top * 0.02,
                    f"{val:.4e}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, y_top)
    else:
        ax.text(0.5, 0.5, "No valid L2 results", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel("L2 Error"); ax.set_title("Final L2 Error Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Optimizer Comparison - {label}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_skip_stats(skip_stats, save_dir, label):
    if not skip_stats:
        return

    visited = skip_stats.get("visited_times", [])
    skipped = skip_stats.get("skipped_times", [])
    n_vis   = skip_stats.get("visited",  len(visited))
    n_skip  = skip_stats.get("skipped",  len(skipped))
    total   = skip_stats.get("total_steps", 0)
    eff     = skip_stats.get("efficiency_pct", 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    if visited:
        ax.scatter(visited, [1]*len(visited), s=80, c="#1565C0",
                   label=f"Visited ({len(visited)})", zorder=5)
    if skipped:
        ax.scatter(skipped, [1]*len(skipped), s=80, c="#E53935", marker="x",
                   label=f"Skipped ({len(skipped)})", zorder=5, linewidths=2)
    if visited or skipped:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Temporal skip not active", ha="center", va="center",
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_yticks([])
    ax.set_title("Temporal Skip Visualization")
    ax.set_xlabel("Time [s]")

    ax = axes[1]
    if n_skip > 0:
        ax.pie([n_vis, n_skip],
               labels=[f"Visited\n({n_vis})", f"Skipped\n({n_skip})"],
               colors=["#1565C0", "#E53935"],
               autopct="%1.1f%%", startangle=90)
    else:
        ax.bar(["Visited"], [n_vis], color="#1565C0")
    ax.set_title(f"Temporal Efficiency\nSaved {eff}% of {total} steps")

    plt.suptitle(f"TemporalSkipScheduler - {label}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temporal_skip_stats.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_benchmark_results(model, problem, train_res, skip_stats, save_dir, label):
    os.makedirs(save_dir, exist_ok=True)
    _plot_training_curves(train_res, save_dir, label, filename="l2_comparison.png")

    try:
        n_vis = 200
        if problem.name in {"burgers", "allen_cahn"}:
            t_min, t_max = problem.domain_bounds["t"]
            x_min, x_max = problem.domain_bounds["x"]
            t_vis = 0.5 * (t_min + t_max)
            x_v = torch.linspace(x_min, x_max, n_vis, device=DEVICE)
            t_v = torch.full_like(x_v, t_vis)
            coords_vis = torch.stack([t_v, x_v], dim=1)
            x_axis = x_v.cpu().numpy()
            title = f"u(x, t={t_vis:.2f})"
        elif problem.name == "poisson":
            x_min, x_max = problem.domain_bounds["x"]
            y_min, y_max = problem.domain_bounds["y"]
            x_v = torch.linspace(x_min, x_max, n_vis, device=DEVICE)
            y_v = torch.full_like(x_v, 0.5 * (y_min + y_max))
            coords_vis = torch.stack([x_v, y_v], dim=1)
            x_axis = x_v.cpu().numpy()
            title = "u(x, y=0)"
        else:
            raise ValueError(f"Unsupported benchmark plotting: {problem.name}")

        model.eval()
        with torch.no_grad():
            pred_v = model(coords_vis).cpu().numpy().flatten()
        exact_v = problem.analytical_solution(coords_vis).cpu().numpy().flatten()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(x_axis, exact_v, "k-", lw=2, label="Reference")
        axes[0].plot(x_axis, pred_v,  "r--", lw=2, label="PINN")
        axes[0].set_title(title); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        abs_err = np.abs(pred_v - exact_v)
        axes[1].plot(x_axis, abs_err, "b-", lw=1.5)
        axes[1].set_title(f"|Error| MAE={abs_err.mean():.2e}")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Reference vs PINN - {label}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "exact_vs_pred.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"  [Plot] exact_vs_pred.png skipped: {exc}")

    _plot_skip_stats(skip_stats, save_dir, label)


# ─────────────────────────────────────────────────────────────
# Paper Baseline Export
# ─────────────────────────────────────────────────────────────

def export_quenching_baseline(model, problem, save_dir: str, time_mode: str,
                               fixed_skip: int) -> Dict:
    """
    Paper baseline ile karşılaştırma:
      1. Termal karşılaştırma: PINN soğuma eğrisi vs analitik (Fig 7 zamanları)
      2. Distorsiyon referansı: Fig 17/18 paper verisi (PINN mekanik model gerektirir)
      3. Paper referans grafikleri export

    Önceki: sadece "comparison_ready: False" yazıyordu.
    Şimdi: gerçek termal metrikleri hesaplar ve kaydeder.
    """
    bl = BaselinePaper()
    os.makedirs(save_dir, exist_ok=True)
    reference_dir = os.path.join(save_dir, "paper_reference")
    os.makedirs(reference_dir, exist_ok=True)

    # Paper referans görsellerini export et
    bl.export_reference_bundle(reference_dir)

    # Distorsiyon referansını görselleştir (PINN kısmı boş — mekanik model yok)
    _plot_paper_distortion_reference(save_dir)

    # Termal karşılaştırma: paper Fig 7 zamanlarında PINN soğuma eğrisi
    paper_metrics = {}
    if hasattr(problem, "make_paper_comparison_fn"):
        compare_fn = problem.make_paper_comparison_fn()
        try:
            paper_metrics = compare_fn(model)
            print(f"\n  [Paper Comparison] Termal metrikler:")
            print(f"    L2_rel  = {paper_metrics['L2_rel']:.4f}")
            print(f"    MAE     = {paper_metrics['MAE_C']:.2f} °C")
            print(f"    RMSE    = {paper_metrics['RMSE_C']:.2f} °C")
            print(f"    R²      = {paper_metrics['R2']:.4f}")
        except Exception as exc:
            print(f"  [Paper Comparison] Thermal comparison error: {exc}")

    # Fig 7 zaman serisi
    fig7_data = bl.get("fig7_water_temp")

    return {
        "status": "comparison_complete",
        "comparison_ready": True,
        "time_mode":   time_mode,
        "fixed_skip":  fixed_skip,
        "thermal_comparison": paper_metrics,
        "fig7_time_s": fig7_data["time_s"],
        "reference_dir": reference_dir,
        "distortion_note": (
            "FEM distortion data (Fig 17/18) exported as reference. "
            "PINN distortion comparison requires coupling with a mechanical model."
        ),
        "reference_datasets": ["fig7_water_temp", "fig17_ht", "fig19_optimized", "fig21_crossmember"],
        "metrics_summary": {
            "L2_rel":   paper_metrics.get("L2_rel"),
            "MAE_C":    paper_metrics.get("MAE_C"),
            "RMSE_C":   paper_metrics.get("RMSE_C"),
            "R2":       paper_metrics.get("R2"),
            "MaxErr_C": paper_metrics.get("MaxErr_C"),
        } if paper_metrics else {},
    }


# ─────────────────────────────────────────────────────────────
# Problem Hazırlık
# ─────────────────────────────────────────────────────────────

def _prepare_problem(problem_name: str,
                     time_mode: str = "full",
                     fixed_skip: int = 2) -> Dict:
    problem_kwargs = {}
    batch_kwargs   = {}
    if problem_name == "quenching":
        problem_kwargs = {"time_mode": time_mode, "fixed_stride": fixed_skip}
        batch_kwargs   = {"time_mode": time_mode, "fixed_stride": fixed_skip}
    problem   = get_problem(problem_name, **problem_kwargs)
    scheduler = _make_scheduler(problem_name, problem, time_mode)
    return {
        "problem":     problem,
        "scheduler":   scheduler,
        "loss_fn":     problem.make_loss_fn(),
        "val_fn":      problem.make_val_fn(),
        "get_batch":   problem.make_batch_fn(scheduler=scheduler, **batch_kwargs),
        "quick_train": make_quick_train_fn(problem),
    }


# ─────────────────────────────────────────────────────────────
# Ana Deney Fonksiyonu
# ─────────────────────────────────────────────────────────────

def run_experiment(
    optimizer:             str   = "nsga2",
    problem_name:          str   = DEFAULT_PROBLEM_NAME,
    time_mode:             str   = "full",
    fixed_skip:            int   = 2,
    adam_epochs:           int   = 10000,
    lbfgs_max_iter:        int   = 5000,
    lbfgs_history_size:    int   = 20,
    lbfgs_tolerance_grad:  float = 1e-7,
    lbfgs_tolerance_change: float = 1e-9,
    pso_steps:             int   = 50,
    pso_particles:         int   = 20,
    run_lbfgs:             bool  = False,   # DÜZELTİLDİ: CLI varsayılanıyla tutarlı
    run_pso:               bool  = False,
    nas_pop:               int   = 24,
    nas_gen:               int   = 16,
    nas_calls:             int   = 16,
    nas_epochs:            int   = 300,
    output_dir:            str   = "results",
) -> dict:
    problem_name = normalize_problem_name(problem_name)
    time_mode    = normalize_time_mode(time_mode)

    if problem_name == "poisson" and (run_lbfgs or run_pso):
        print("[INFO] Poisson Adam-only mode: LBFGS and PSO disabled.")
        run_lbfgs = False
        run_pso   = False

    bundle     = _prepare_problem(problem_name, time_mode=time_mode, fixed_skip=fixed_skip)
    problem    = bundle["problem"]
    scheduler  = bundle["scheduler"]
    loss_fn    = bundle["loss_fn"]
    val_fn     = bundle["val_fn"]
    get_batch  = bundle["get_batch"]
    quick_train = bundle["quick_train"]

    save_dir = os.path.join(output_dir, problem_name, optimizer)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'#' * 55}")
    print(f"  Experiment: {problem_name.upper()} + {optimizer.upper()}")
    print(f"  Adam epochs: {adam_epochs}  |  L-BFGS: {run_lbfgs}  |  PSO: {run_pso}")
    if run_lbfgs:
        print(f"  L-BFGS max_iter: {lbfgs_max_iter}  |  history: {lbfgs_history_size}")
    if run_pso:
        print(f"  PSO steps: {pso_steps}  |  particles: {pso_particles}")
    if problem_name == "quenching":
        print(f"  Time mode: {time_mode.upper()}  |  Fixed skip stride: {fixed_skip}")
    print(f"{'#' * 55}")

    # ── Adım 1: NAS
    print(f"\n[Step 1/4] Architecture Search ({optimizer.upper()}) ...")
    t_nas = time.time()

    common_kwargs = {
        "n_epochs": nas_epochs,
        "n_input":  problem.n_input,
        "n_output": problem.n_output,
    }
    obj_mode = "l2_only" if problem_name == "poisson" else "balanced"

    if optimizer == "nsga2":
        pareto     = run_nsga2(quick_train, pop_size=nas_pop, n_gen=nas_gen,
                               objective_mode=obj_mode, **common_kwargs)
        best_config = pareto[0]
    elif optimizer == "nsga3":
        pareto      = run_nsga3(quick_train, pop_size=nas_pop, n_gen=nas_gen,
                                objective_mode=obj_mode, **common_kwargs)
        best_config = pareto[0]
    elif optimizer == "bayesian":
        best_config = run_bayesian(quick_train, n_calls=nas_calls,
                                   n_initial=min(4, max(1, nas_calls // 4)),
                                   **common_kwargs)
        pareto = [best_config]
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Choose: nsga2, nsga3, bayesian")

    nas_time = time.time() - t_nas
    best_config["n_input"]  = problem.n_input
    best_config["n_output"] = problem.n_output

    if problem_name == "poisson":
        original = dict(best_config)
        best_config = stabilize_poisson_config(best_config)
        if best_config.pop("_stabilized_for_poisson", False):
            print(f"  [INFO] Stabilized Poisson arch: "
                  f"{original.get('neurons')} → {best_config['neurons']}")

    print(f"\n  NAS done: {nas_time:.1f}s")
    print(f"  Best arch: layers={best_config['n_layers']}  "
          f"neurons={best_config['neurons']}  act={best_config['activation']}")

    with open(os.path.join(save_dir, "best_arch.json"), "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    # ── Adım 2: Tam Eğitim
    print(f"\n[Step 2/4] Full Training (Adam {adam_epochs} epochs) ...")
    model = build_network_from_config(best_config)
    train_res = full_training_pipeline(
        model=model,
        loss_fn=loss_fn,
        get_batch=get_batch,
        val_fn=val_fn,
        adam_epochs=adam_epochs,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_history_size=lbfgs_history_size,
        lbfgs_tolerance_grad=lbfgs_tolerance_grad,
        lbfgs_tolerance_change=lbfgs_tolerance_change,
        pso_steps=pso_steps,
        pso_particles=pso_particles,
        run_lbfgs=run_lbfgs,
        run_pso=run_pso,
    )

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # ── Adım 3: Görselleştirme ve Paper Karşılaştırması
    print(f"\n[Step 3/4] Generating plots ...")
    skip_stats = (get_batch.get_stats() if hasattr(get_batch, "get_stats")
                  else (scheduler.get_stats() if scheduler is not None else {}))

    paper_metrics = {}
    if problem_name == "quenching":
        paper_metrics = export_quenching_baseline(
            model, problem, save_dir, time_mode, fixed_skip)
        _plot_quenching_results(
            model, problem, train_res, skip_stats,
            paper_metrics.get("thermal_comparison", {}),
            save_dir, optimizer.upper()
        )
    else:
        _plot_benchmark_results(
            model, problem, train_res, skip_stats, save_dir,
            f"{problem_name.upper()}_{optimizer.upper()}"
        )

    # ── Adım 4: Sonuçları kaydet
    print(f"\n[Step 4/4] Saving results ...")
    result_summary = {
        "problem":    problem_name,
        "optimizer":  optimizer,
        "time_mode":  time_mode  if problem_name == "quenching" else None,
        "fixed_skip": fixed_skip if problem_name == "quenching" else None,
        "best_config": best_config,
        "nas_time_s":  nas_time,
        "adam_epochs": adam_epochs,
        "lbfgs_max_iter":        lbfgs_max_iter    if run_lbfgs else None,
        "lbfgs_history_size":    lbfgs_history_size if run_lbfgs else None,
        "lbfgs_tolerance_grad":  lbfgs_tolerance_grad  if run_lbfgs else None,
        "lbfgs_tolerance_change": lbfgs_tolerance_change if run_lbfgs else None,
        "pso_steps":    pso_steps    if run_pso else None,
        "pso_particles": pso_particles if run_pso else None,
        "temporal_skip": skip_stats,
        "train_results": {
            phase: {k: v for k, v in values.items() if k != "history"}
            for phase, values in train_res.items()
            if isinstance(values, dict)
        },
        # Paper baseline metrikleri — artık gerçek termal değerler içeriyor
        "paper_baseline": {
            "status":       paper_metrics.get("status", "not_run"),
            "time_mode":    time_mode,
            "fixed_skip":   fixed_skip,
            "thermal_metrics": paper_metrics.get("metrics_summary", {}),
            "distortion_note": paper_metrics.get("distortion_note", ""),
        } if problem_name == "quenching" else {},
    }

    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)
    print(f"  Results saved: {os.path.join(save_dir, 'results.json')}")

    return result_summary


# ─────────────────────────────────────────────────────────────
# Tüm Optimizatörleri Karşılaştır
# ─────────────────────────────────────────────────────────────

def compare_all_optimizers(
    adam_epochs:           int   = 10000,
    lbfgs_max_iter:        int   = 5000,
    lbfgs_history_size:    int   = 20,
    lbfgs_tolerance_grad:  float = 1e-7,
    lbfgs_tolerance_change: float = 1e-9,
    pso_steps:             int   = 50,
    pso_particles:         int   = 20,
    run_lbfgs:             bool  = False,
    run_pso:               bool  = False,
    nas_pop:               int   = 24,
    nas_gen:               int   = 16,
    nas_calls:             int   = 16,
    nas_epochs:            int   = 300,
    output_dir:            str   = "results",
    problem_name:          str   = DEFAULT_PROBLEM_NAME,
    time_mode:             str   = "full",
    fixed_skip:            int   = 2,
) -> None:
    problem_name = normalize_problem_name(problem_name)
    time_mode    = normalize_time_mode(time_mode)
    all_results  = {}

    for optimizer in ("nsga2", "nsga3", "bayesian"):
        print(f"\n{'#' * 55}")
        print(f"  Running {optimizer.upper()} ...")
        print(f"{'#' * 55}")
        try:
            all_results[optimizer] = run_experiment(
                optimizer=optimizer,
                problem_name=problem_name,
                time_mode=time_mode,
                fixed_skip=fixed_skip,
                adam_epochs=adam_epochs,
                lbfgs_max_iter=lbfgs_max_iter,
                lbfgs_history_size=lbfgs_history_size,
                lbfgs_tolerance_grad=lbfgs_tolerance_grad,
                lbfgs_tolerance_change=lbfgs_tolerance_change,
                pso_steps=pso_steps,
                pso_particles=pso_particles,
                run_lbfgs=run_lbfgs,
                run_pso=run_pso,
                nas_pop=nas_pop,
                nas_gen=nas_gen,
                nas_calls=nas_calls,
                nas_epochs=nas_epochs,
                output_dir=output_dir,
            )
        except Exception as exc:
            print(f"  [ERROR] {optimizer}: {exc}")
            all_results[optimizer] = {"error": str(exc)}

    print(f"\n{'=' * 80}")
    print("  FULL COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(
        f"  {'Optimizer':<12} | {'Adam L2':>10} | {'LBFGS L2':>10} | "
        f"{'PSO L2':>10} | {'MAE(°C)':>10} | {'R2':>8} | {'Skip%':>6}"
    )
    print(f"  {'─' * 76}")

    for optimizer, result in all_results.items():
        if "error" in result:
            print(f"  {optimizer:<12} | ERROR: {result['error'][:50]}")
            continue
        train    = result.get("train_results", {})
        baseline = result.get("paper_baseline", {})
        skip     = result.get("temporal_skip", {})
        thermal  = baseline.get("thermal_metrics", {})

        adam_l2  = train.get("adam",  {}).get("final_l2", None)
        lbfgs_l2 = train.get("lbfgs", {}).get("final_l2", None) if "lbfgs" in train else None
        pso_l2   = train.get("pso",   {}).get("final_l2", None) if "pso"   in train else None
        mae_C    = thermal.get("MAE_C",  None)
        r2       = thermal.get("R2",     None)
        skip_pct = skip.get("efficiency_pct", 0)

        def _fmt_e(v): return f"{v:>10.4e}" if (v is not None and np.isfinite(float(v))) else "         -"
        def _fmt_f(v): return f"{v:>10.4f}" if (v is not None and np.isfinite(float(v))) else "       N/A"
        def _fmt_r(v): return f"{v:>8.4f}"  if (v is not None and np.isfinite(float(v))) else "     N/A"

        print(
            f"  {optimizer:<12} | {_fmt_e(adam_l2)} | {_fmt_e(lbfgs_l2)} | "
            f"{_fmt_e(pso_l2)} | {_fmt_f(mae_C)} | {_fmt_r(r2)} | {skip_pct:>5.1f}%"
        )

    print(f"{'=' * 80}")
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, "comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Comparison saved: {comparison_path}")
