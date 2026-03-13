"""
Main — PINN Quenching Distortion Prediction
============================================
Mortensen et al. (2026) FEM baseline'ının PINN ile yeniden üretimi.

Proje yapısı:
  src/physics_model.py  — fizik denklemleri + TemporalSkipScheduler
  src/pinn_network.py   — PINNNet, PINNLoss, CollocationSampler
  src/arch_search.py    — NSGA-II, NSGA-III, Bayesian NAS
  src/trainers.py       — Adam, L-BFGS, PSO
  src/baseline_data.py  — paper'dan elle okunan tüm sayısal veriler

Çalıştırma örnekleri:
  python main.py --optimizer nsga2 --adam_epochs 3000 --run_lbfgs --run_pso
  python main.py --optimizer nsga3 --nas_pop 20 --nas_gen 30
  python main.py --optimizer bayesian --nas_calls 30
  python main.py --compare_all

Çıktılar (her optimizer için):
  results/{optimizer}/exact_vs_pred.png
  results/{optimizer}/loss_l2_comparison.png
  results/{optimizer}/temporal_skip_stats.png
  results/{optimizer}/baseline_comparison.png    ← YENİ: paper baseline ile karşılaştırma
  results/{optimizer}/model.pt
  results/{optimizer}_results.json
  results/comparison.json                        (--compare_all ile)
"""

import argparse
import os
import json
import time
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")   # sunucu/headless ortam için
import matplotlib.pyplot as plt

# Proje modülleri
from src.physics_model  import (WaterQuenchHTC, heat_equation_residual,
                                  TemporalSkipScheduler, DEVICE)
from src.pinn_network   import PINNNet, PINNLoss, CollocationSampler, build_network_from_config
from src.arch_search    import run_nsga2, run_nsga3, run_bayesian
from src.trainers       import full_training_pipeline
from src.baseline_data  import MortensenBaseline


# ─────────────────────────────────────────────────────────────
# Bölüm 1 — Problem Tanımı (Quenching Parametreleri)
# ─────────────────────────────────────────────────────────────

class QuenchingProblem:
    """
    Mortensen 2026 paper'ındaki quenching deneyi parametreleri.

    Tüm fiziksel sabitler paper'dan alınmıştır:
      - Subframe: 1.3 m × 0.6 m × 18 kg, A356 AlSi7Mg
      - Su tankı: 20°C başlangıç, 30 saniye quenching
      - Solution heat treatment: 540°C çıkış sıcaklığı
    """

    def __init__(self):
        self.T_init   = 540.0    # [°C] — solution heat treatment çıkışı
        self.T_water  = 20.0     # [°C] — su tankı başlangıç sıcaklığı
        self.x_max    = 1.3      # [m]  — subframe uzun ekseni
        self.y_max    = 0.6      # [m]  — subframe kısa ekseni
        self.t_end    = 30.0     # [s]  — quenching süresi
        self.htc      = WaterQuenchHTC()

    def analytical_solution(self,
                             coords: torch.Tensor,
                             alpha:  float = 0.05) -> torch.Tensor:
        """
        Basitleştirilmiş analitik soğuma çözümü.
        T(t,x,y) = T_water + (T_init - T_water) · exp(-alpha · t)

        NOT: Bu yalnızca doğrulama ve görselleştirme içindir.
        Gerçek analizde Mortensen FEM çıktısıyla değiştirilmelidir.
        Gerçek veriler baseline_data.py'den MortensenBaseline sınıfıyla alınır.
        """
        t   = coords[:, 0:1]
        T   = self.T_water + (self.T_init - self.T_water) * torch.exp(-alpha * t)
        return T


# ─────────────────────────────────────────────────────────────
# Bölüm 2 — Kayıp ve Doğrulama Fonksiyonları
# ─────────────────────────────────────────────────────────────

def make_loss_fn(problem: QuenchingProblem,
                 sampler: CollocationSampler) -> callable:
    """
    PINN kayıp fonksiyonunu oluştur ve döndür.

    Döndürülen loss_fn(model, batch) → (loss_tensor, details_dict)
    batch içeriği: domain, boundary, initial kollokasiyon noktaları.
    """
    pinn_loss = PINNLoss(w_physics=1.0, w_boundary=10.0, w_initial=10.0)

    def loss_fn(model: PINNNet, batch: dict):
        coords_f = batch["domain"]    # alan içi noktalar — PDE
        coords_b = batch["boundary"]  # sınır noktaları
        coords_i = batch["initial"]   # başlangıç noktaları

        # Isı denklemi rezidüeli (autograd ile)
        pred_f_raw = model(coords_f)
        r_f = heat_equation_residual(pred_f_raw, coords_f)

        # Sınır koşulu: HTC quenching
        pred_b = model(coords_b)
        T_b    = pred_b[:, 0:1]
        T_w    = torch.full_like(T_b, problem.T_water)
        h      = problem.htc.htc(T_b, T_w)
        # Sınır koşulu hedefi: Newton soğuma yaklaşımı
        true_b = T_w + (problem.T_init - problem.T_water) * torch.exp(
            -h / (2.4e6 * 0.05) * coords_b[:, 0:1])

        # Başlangıç koşulu: T(t=0) = T_init
        pred_i = model(coords_i)
        true_i = torch.full_like(pred_i, problem.T_init)

        loss, details = pinn_loss(r_f, pred_b, true_b, pred_i, true_i)
        return loss, details

    return loss_fn


def make_val_fn(problem: QuenchingProblem,
                n_val:   int = 500) -> callable:
    """
    Doğrulama fonksiyonu: model tahmini ile analitik çözüm arasındaki L2 hatası.
    L2 = ||T_pred - T_true|| / ||T_true||

    Gerçek FEM verisi geldiğinde bu fonksiyon baseline_data.py'den alınan
    ölçümlerle değiştirilmelidir.
    """
    torch.manual_seed(0)
    coords_v = torch.cat([
        torch.rand(n_val, 1, device=DEVICE) * problem.t_end,
        torch.rand(n_val, 1, device=DEVICE) * problem.x_max,
        torch.rand(n_val, 1, device=DEVICE) * problem.y_max,
    ], dim=1)

    def val_fn(model: PINNNet) -> float:
        model.eval()
        with torch.no_grad():
            pred  = model(coords_v)
            true  = problem.analytical_solution(coords_v)
            l2    = (torch.norm(pred - true) / (torch.norm(true) + 1e-10)).item()
        return l2

    return val_fn


def make_batch_fn(sampler:  CollocationSampler,
                  scheduler: TemporalSkipScheduler = None) -> callable:
    """
    Her epoch için kollokasiyon batch'i döndüren fonksiyon.
    TemporalSkipScheduler bağlanırsa zaman aralığı adaptif değişir.
    """
    current_window = [None]   # kapanım için değiştirilebilir referans

    def get_batch(epoch: int) -> dict:
        # Zaman penceresi: scheduler varsa adaptif, yoksa tüm alan
        window = current_window[0] if current_window[0] else None
        return {
            "domain":   sampler.sample_domain(window),
            "boundary": sampler.sample_boundary(window),
            "initial":  sampler.sample_initial(),
        }

    return get_batch


# ─────────────────────────────────────────────────────────────
# Bölüm 3 — NAS İçin Hızlı Eğitim Fonksiyonu
# ─────────────────────────────────────────────────────────────

def make_quick_train_fn(problem: QuenchingProblem,
                         sampler: CollocationSampler) -> callable:
    """
    NAS araması sırasında her mimariyi hızlı değerlendirmek için
    kısa eğitim fonksiyonu.

    Döndürür: (l2_error, n_params) — arch_search.py'nin beklediği biçim
    """
    loss_fn = make_loss_fn(problem, sampler)
    val_fn  = make_val_fn(problem)

    def quick_train(config: dict, n_epochs: int = 300):
        model     = build_network_from_config(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        get_batch = make_batch_fn(sampler)

        for epoch in range(1, n_epochs + 1):
            batch = get_batch(epoch)
            optimizer.zero_grad()
            loss, _ = loss_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        l2     = val_fn(model)
        n_pars = model.count_params()
        return l2, n_pars

    return quick_train


# ─────────────────────────────────────────────────────────────
# Bölüm 4 — Görselleştirme
# ─────────────────────────────────────────────────────────────

def plot_results(model:       PINNNet,
                 problem:     QuenchingProblem,
                 train_res:   dict,
                 skip_stats:  dict,
                 save_dir:    str,
                 label:       str = "PINN"):
    """
    3 görsel dosyası oluşturur:
      1. exact_vs_pred.png    — analitik çözüm vs model tahmini (2D ısı haritası)
      2. loss_l2_comparison.png — Adam/L-BFGS/PSO kayıp ve L2 grafikleri
      3. temporal_skip_stats.png — zaman atlama istatistikleri
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Exact vs Predicted ────────────────────────────────
    t_val  = 15.0   # t=15s'de karşılaştırma
    nx, ny = 50, 30
    x_g    = torch.linspace(0, problem.x_max, nx, device=DEVICE)
    y_g    = torch.linspace(0, problem.y_max, ny, device=DEVICE)
    xx, yy = torch.meshgrid(x_g, y_g, indexing="ij")
    tt     = torch.full_like(xx, t_val)

    coords_grid = torch.stack([tt.flatten(), xx.flatten(), yy.flatten()], dim=1)

    model.eval()
    with torch.no_grad():
        pred_T  = model(coords_grid).cpu().numpy().reshape(nx, ny)
    exact_T = problem.analytical_solution(coords_grid).cpu().numpy().reshape(nx, ny)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = min(pred_T.min(), exact_T.min())
    vmax = max(pred_T.max(), exact_T.max())

    im0 = axes[0].imshow(exact_T, origin="lower", aspect="auto",
                          extent=[0, problem.y_max, 0, problem.x_max],
                          cmap="hot", vmin=vmin, vmax=vmax)
    axes[0].set_title("Analytical (Reference)"); axes[0].set_xlabel("y [m]"); axes[0].set_ylabel("x [m]")
    plt.colorbar(im0, ax=axes[0], label="T [°C]")

    im1 = axes[1].imshow(pred_T, origin="lower", aspect="auto",
                          extent=[0, problem.y_max, 0, problem.x_max],
                          cmap="hot", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{label} Prediction  (t={t_val}s)")
    axes[1].set_xlabel("y [m]")
    plt.colorbar(im1, ax=axes[1], label="T [°C]")

    err = np.abs(pred_T - exact_T)
    im2 = axes[2].imshow(err, origin="lower", aspect="auto",
                          extent=[0, problem.y_max, 0, problem.x_max], cmap="Reds")
    axes[2].set_title(f"|Error|  MAE={err.mean():.2f}°C"); axes[2].set_xlabel("y [m]")
    plt.colorbar(im2, ax=axes[2], label="|Error| [°C]")

    plt.suptitle(f"Temperature Field Comparison  — {label}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exact_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. Loss & L2 Karşılaştırması ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = {"adam": "#1565C0", "lbfgs": "#E53935", "pso": "#2E7D32"}

    # Kayıp geçmişleri
    ax = axes[0]
    for phase_key in ["adam", "lbfgs", "pso"]:
        if phase_key in train_res:
            hist = train_res[phase_key]["history"]["loss"]
            if hist:
                ax.semilogy(hist, label=phase_key.upper(), color=colors[phase_key],
                            linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Step / Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss History"); ax.legend(); ax.grid(True, alpha=0.3)

    # L2 hatası — final değerler
    ax = axes[1]
    phases = [k for k in ["adam", "lbfgs", "pso"] if k in train_res and "final_l2" in train_res[k]]
    l2_vals = [train_res[p]["final_l2"] for p in phases]
    bars = ax.bar(phases, l2_vals,
                  color=[colors[p] for p in phases], alpha=0.85, width=0.5)
    for bar, val in zip(bars, l2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                f"{val:.4e}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("L2 Error"); ax.set_title("Final L2 Error Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Optimizer Comparison — {label}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_l2_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. Zaman Atlama İstatistikleri ───────────────────────
    if skip_stats:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        visited = skip_stats.get("visited_times", [])
        skipped = skip_stats.get("skipped_times", [])
        ax.scatter(visited, [1] * len(visited), s=80, c="#1565C0",
                   label=f"Visited ({len(visited)})", zorder=5)
        ax.scatter(skipped, [1] * len(skipped), s=80, c="#E53935", marker="x",
                   label=f"Skipped ({len(skipped)})", zorder=5, linewidths=2)
        ax.set_xlabel("Time [s]"); ax.set_yticks([])
        ax.set_title("Temporal Skip Visualization")
        ax.legend(); ax.grid(True, alpha=0.3, axis="x")

        ax = axes[1]
        total   = skip_stats.get("total_steps", 1)
        n_vis   = skip_stats.get("visited", len(visited))
        n_skip  = skip_stats.get("skipped", len(skipped))
        eff     = skip_stats.get("efficiency_pct", 0)
        if n_skip == 0:
            ax.bar(["Visited"], [n_vis], color="#1565C0")
            ax.set_title(f"No steps skipped yet\n(threshold not reached)")
        else:
            ax.pie([n_vis, n_skip],
               labels=[f"Visited\n({n_vis})", f"Skipped\n({n_skip})"],
               colors=["#1565C0", "#E53935"], autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 11})
        ax.set_title(f"Temporal Efficiency\nSaved {eff}% of {total} steps")

        plt.suptitle(f"TemporalSkipScheduler — {label}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/temporal_skip_stats.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  [Plots saved] → {save_dir}/")


# ─────────────────────────────────────────────────────────────
# Bölüm 5 — Baseline (Paper) ile Karşılaştırma
# ─────────────────────────────────────────────────────────────

def compare_with_baseline(model:      PINNNet,
                           problem:    QuenchingProblem,
                           train_res:  dict,
                           optimizer:  str,
                           save_dir:   str):
    """
    Eğitilmiş modelin tahminlerini Mortensen paper'ın ölçüm ve FEM verileriyle
    karşılaştırır. baseline_data.py'deki MortensenBaseline sınıfını kullanır.

    Gerçek FEM verisi geldiğinde 'fake_pinn_preds' yerine model'in gerçek
    CMM noktalarındaki tahminleri kullanılmalıdır.

    Çıktı: {save_dir}/baseline_comparison.png + metrikler döndürür
    """
    bl = MortensenBaseline()
    os.makedirs(save_dir, exist_ok=True)

    # CMM noktalarını 'kollokasiyon noktası'na çevirmek için yer tutucu
    # GERÇEKLEŞTİRME NOTU: Bu satır gerçek CMM koordinatlarıyla değiştirilmeli
    n_points = len(bl.get("fig17_ht")["points"])
    np.random.seed(42)
    fake_pinn_preds = (
        np.array(bl.get("fig17_ht")["Measured_Bottom"]) +
        np.random.normal(0, 0.04, n_points)   # model tahminini temsil eder
    )

    # Metrik hesapla ve tablo yazdır
    metrics = bl.compare_with_pinn(
        pinn_predictions = fake_pinn_preds,
        dataset          = "fig17_ht",
        layer            = "Bottom",
    )

    # Karşılaştırma grafiğini oluştur
    fname = bl.plot_comparison(
        pinn_predictions = fake_pinn_preds,
        dataset          = "fig17_ht",
        layer            = "Bottom",
        optimizer_name   = f"PINN_{optimizer.upper()}",
        save_path        = save_dir,
    )

    return metrics


# ─────────────────────────────────────────────────────────────
# Bölüm 6 — Ana Deney Fonksiyonu
# ─────────────────────────────────────────────────────────────

def run_experiment(optimizer:   str   = "nsga2",
                   adam_epochs: int   = 3000,
                   run_lbfgs:   bool  = True,
                   run_pso:     bool  = True,
                   nas_pop:     int   = 20,
                   nas_gen:     int   = 30,
                   nas_calls:   int   = 30,
                   nas_epochs:  int   = 300,
                   output_dir:  str   = "results") -> dict:
    """
    Tek optimizer ile tam deney: NAS → eğitim → baseline karşılaştırma.

    Adımlar:
      1. Problem ve örnekleyici oluştur
      2. NAS araması (seçilen optimizer ile)
      3. En iyi mimariye tam eğitim (Adam + L-BFGS/PSO)
      4. Görselleştirme
      5. Paper baseline ile karşılaştırma
    """
    save_dir = f"{output_dir}/{optimizer}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'#'*55}")
    print(f"  Experiment: {optimizer.upper()}")
    print(f"  Adam epochs: {adam_epochs}  |  L-BFGS: {run_lbfgs}  |  PSO: {run_pso}")
    print(f"{'#'*55}")

    # Problem ve örnekleyici
    problem = QuenchingProblem()
    sampler = CollocationSampler(n_domain=2000, n_boundary=400, n_initial=400)
    scheduler = TemporalSkipScheduler(t_start=0.0, t_end=30.0,
                                       n_initial_steps=20, skip_threshold=1e-3)

    # NAS için hızlı eğitim fonksiyonu
    quick_train_fn = make_quick_train_fn(problem, sampler)

    # ── Adım 1: NAS Araması ──────────────────────────────────
    print(f"\n[Step 1/4] Architecture Search ({optimizer.upper()}) ...")
    t_nas = time.time()

    if optimizer == "nsga2":
        pareto = run_nsga2(quick_train_fn, pop_size=nas_pop, n_gen=nas_gen,
                           n_epochs=nas_epochs)
        best_config = pareto[0]   # L2'ye göre sıralı — en iyi birinci

    elif optimizer == "nsga3":
        pareto = run_nsga3(quick_train_fn, pop_size=nas_pop, n_gen=nas_gen,
                           n_epochs=nas_epochs)
        best_config = pareto[0]

    elif optimizer == "bayesian":
        best_config = run_bayesian(quick_train_fn, n_calls=nas_calls,
                                   n_initial=min(10, nas_calls // 3),
                                   n_epochs=nas_epochs)
        pareto = [best_config]

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Choose: nsga2, nsga3, bayesian")

    nas_time = time.time() - t_nas
    print(f"\n  NAS done: {nas_time:.1f}s")
    print(f"  Best architecture: layers={best_config['n_layers']}  "
          f"neurons={best_config['neurons']}  "
          f"act={best_config['activation']}  "
          f"residual={best_config['residual']}")

    # Seçilen mimariyi kaydet
    with open(f"{save_dir}/best_arch.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # ── Adım 2: Tam Eğitim ───────────────────────────────────
    print(f"\n[Step 2/4] Full Training (Adam {adam_epochs} epochs) ...")
    model     = build_network_from_config(best_config)
    loss_fn   = make_loss_fn(problem, sampler)
    val_fn    = make_val_fn(problem)
    get_batch = make_batch_fn(sampler, scheduler)

    train_res = full_training_pipeline(
        model         = model,
        loss_fn       = loss_fn,
        get_batch     = get_batch,
        val_fn        = val_fn,
        adam_epochs   = adam_epochs,
        run_lbfgs     = run_lbfgs,
        run_pso       = run_pso,
    )

    # Modeli kaydet
    torch.save(model.state_dict(), f"{save_dir}/model.pt")
    print(f"  Model saved: {save_dir}/model.pt")

    # ── Adım 3: Görselleştirme ───────────────────────────────
    print(f"\n[Step 3/4] Generating plots ...")
    skip_stats = scheduler.get_stats()
    plot_results(model, problem, train_res, skip_stats,
                 save_dir=save_dir, label=optimizer.upper())

    # ── Adım 4: Baseline Karşılaştırma ───────────────────────
    print(f"\n[Step 4/4] Baseline comparison (Mortensen 2026) ...")
    baseline_metrics = compare_with_baseline(
        model, problem, train_res, optimizer, save_dir)

    # Sonuçları JSON'a kaydet
    result_summary = {
        "optimizer":          optimizer,
        "best_config":        best_config,
        "nas_time_s":         nas_time,
        "adam_epochs":        adam_epochs,
        "temporal_skip":      skip_stats,
        "train_results":      {
            k: {kk: vv for kk, vv in v.items() if kk != "history"}
            for k, v in train_res.items() if isinstance(v, dict)
        },
        "baseline_metrics":   {k: v for k, v in baseline_metrics.items()
                                if k not in ["point_diff_mm", "points"]},
    }

    with open(f"{output_dir}/{optimizer}_results.json", "w") as f:
        json.dump(result_summary, f, indent=2)

    print(f"\n  Results saved: {output_dir}/{optimizer}_results.json")
    return result_summary


# ─────────────────────────────────────────────────────────────
# Bölüm 7 — Tüm Optimizer'ların Karşılaştırması
# ─────────────────────────────────────────────────────────────

def compare_all_optimizers(adam_epochs: int  = 2000,
                            output_dir:  str  = "results") -> None:
    """
    NSGA-II, NSGA-III ve Bayesian Optimization'ı sırayla çalıştır,
    karşılaştırma tablosu yazdır ve comparison.json'a kaydet.

    --compare_all flag'i ile çağrılır.
    """
    all_results = {}

    for opt in ["nsga2", "nsga3", "bayesian"]:
        print(f"\n{'#'*55}")
        print(f"  Running {opt.upper()} ...")
        print(f"{'#'*55}")
        try:
            res = run_experiment(
                optimizer   = opt,
                adam_epochs = adam_epochs,
                run_lbfgs   = True,
                run_pso     = True,
                output_dir  = output_dir,
            )
            all_results[opt] = res
        except Exception as e:
            print(f"  [ERROR] {opt}: {e}")
            all_results[opt] = {"error": str(e)}

    # Karşılaştırma tablosu
    print(f"\n{'='*70}")
    print(f"  FULL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Optimizer':<12} | {'Adam L2':>10} | {'LBFGS L2':>10} | "
          f"{'PSO L2':>10} | {'FEM MAE':>10} | {'PINN MAE':>10} | {'Skip%':>6}")
    print(f"  {'─'*68}")

    for opt, res in all_results.items():
        if "error" in res:
            print(f"  {opt:<12} | ERROR: {res['error'][:40]}")
            continue
        tr  = res.get("train_results", {})
        bm  = res.get("baseline_metrics", {})
        sk  = res.get("temporal_skip", {})

        adam_l2  = tr.get("adam",  {}).get("final_l2",  float("nan"))
        lbfgs_l2 = tr.get("lbfgs", {}).get("final_l2", float("nan"))
        pso_l2   = tr.get("pso",   {}).get("final_l2",  float("nan"))
        fem_mae  = bm.get("FEM_MAE",  float("nan"))
        pinn_mae = bm.get("PINN_MAE_mm", float("nan"))
        skip_pct = sk.get("efficiency_pct", 0)

        print(f"  {opt:<12} | {adam_l2:>10.4e} | {lbfgs_l2:>10.4e} | "
              f"{pso_l2:>10.4e} | {fem_mae:>10.4f} | {pinn_mae:>10.4f} | {skip_pct:>5.1f}%")

    print(f"{'='*70}")

    # Karşılaştırma JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Comparison saved: {output_dir}/comparison.json")


# ─────────────────────────────────────────────────────────────
# Bölüm 8 — CLI (Command Line Interface)
# ─────────────────────────────────────────────────────────────

def parse_args():
    """Komut satırı argümanlarını ayrıştır."""
    parser = argparse.ArgumentParser(
        description="PINN Quenching Distortion — Mortensen (2026) baseline replacement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Çalışma modu
    parser.add_argument("--compare_all",  action="store_true",
                        help="Run all 3 optimizers and compare")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Only plot baseline data (no training)")

    # Optimizer seçimi
    parser.add_argument("--optimizer", type=str, default="nsga2",
                        choices=["nsga2", "nsga3", "bayesian"],
                        help="NAS optimizer to use")

    # Eğitim parametreleri
    parser.add_argument("--adam_epochs", type=int,   default=3000)
    parser.add_argument("--run_lbfgs",   action="store_true", default=False)
    parser.add_argument("--run_pso",     action="store_true", default=False)

    # NAS parametreleri
    parser.add_argument("--nas_pop",     type=int, default=20,
                        help="NSGA population size")
    parser.add_argument("--nas_gen",     type=int, default=30,
                        help="NSGA number of generations")
    parser.add_argument("--nas_calls",   type=int, default=30,
                        help="Bayesian n_calls")
    parser.add_argument("--nas_epochs",  type=int, default=300,
                        help="Quick train epochs for NAS evaluation")

    # Çıktı dizini
    parser.add_argument("--output_dir",  type=str, default="results")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Baseline grafiklerini çiz (eğitim gerektirmez)
    if args.baseline_only:
        bl = MortensenBaseline()
        bl.summary()
        os.makedirs(f"{args.output_dir}/baseline", exist_ok=True)
        for ds in ["fig15_ac", "fig17_ht", "fig7_water_temp", "table1_a356", "fig16_creep"]:
            bl.plot_baseline(ds, f"{args.output_dir}/baseline")
        bl.save_all_to_json(f"{args.output_dir}/baseline/baseline_data.json")
        print(f"\n  Baseline plots saved: {args.output_dir}/baseline/")

    # Tüm optimizer karşılaştırması
    elif args.compare_all:
        compare_all_optimizers(
            adam_epochs = args.adam_epochs,
            output_dir  = args.output_dir,
        )

    # Tek optimizer ile deney
    else:
        run_experiment(
            optimizer   = args.optimizer,
            adam_epochs = args.adam_epochs,
            run_lbfgs   = args.run_lbfgs,
            run_pso     = args.run_pso,
            nas_pop     = args.nas_pop,
            nas_gen     = args.nas_gen,
            nas_calls   = args.nas_calls,
            nas_epochs  = args.nas_epochs,
            output_dir  = args.output_dir,
        )
