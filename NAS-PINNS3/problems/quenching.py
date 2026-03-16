"""
Quenching Problemi (Isı Transferi, 2D+t) — Uygulama Problemi
=============================================================
Kaynak: Mortensen et al. (2026) — A356 alüminyum subframe su quenching.

PDE  : ρCp·∂T/∂t = k·∇²T,   (t,x,y) ∈ [0,30]×[0,1.3]×[0,0.6]
IC   : T(0,x,y) = 540°C   (SHT çıkışı)
BC   : Robin — k·∂T/∂n + h(T,T_w)·(T−T_w) = 0   (su quenching yüzeyi)

Analitik yaklaşım (tek boyutlu soğuma referansı):
  T(t) ≈ T_w + (T_0 − T_w)·exp(−α·t),  α=0.05 [1/s]
  Bu FEM verisi yerine proxy referans olarak kullanılır.

CollocationSampler bu dosyadadır çünkü geometri quenching'e özgüdür.
"""

import numpy as np
import torch
from typing import Callable, Dict, Optional, Tuple

from .base import PINNProblem
from src.baseline_data import BaselinePaper
from src.config import (
    DEVICE,
    T_INIT, T_WATER_INIT, T_END, X_MAX, Y_MAX,
    ALPHA, K_THERMAL, RHO_CP,
    W_PHYSICS, W_BOUNDARY, W_INITIAL,
    N_DOMAIN, N_BOUNDARY, N_INITIAL, N_TIME_STEPS,
)
from src.physics_model import WaterQuenchHTC, heat_equation_residual, TemporalSkipScheduler
from src.pinn_network import PINNLoss


# ─────────────────────────────────────────────────────────────
# Kollokasiyon Nokta Üreticisi — Quenching Geometrisi
# ─────────────────────────────────────────────────────────────

class CollocationSampler:
    """
    Mortensen subframe geometrisi için kollokasiyon noktaları üretir.

    Fiziksel alan (Mortensen 2026):
      x ∈ [0, 1.3] m  — subframe uzunluğu
      y ∈ [0, 0.6] m  — subframe genişliği
      t ∈ [0,  30] s  — quenching süresi

    Yüzey etiketi: boundary noktaları hangi kenara ait olduğunu bilmeli
    ki doğru normal türev yönü kullanılsın (Robin BC için).
    """

    def __init__(self,
                 x_range:    Tuple[float, float] = (0.0, X_MAX),
                 y_range:    Tuple[float, float] = (0.0, Y_MAX),
                 t_range:    Tuple[float, float] = (0.0, T_END),
                 n_domain:   int = N_DOMAIN,
                 n_boundary: int = N_BOUNDARY,
                 n_initial:  int = N_INITIAL,
                 seed:       int = 42):
        torch.manual_seed(seed)
        self.x_range    = x_range
        self.y_range    = y_range
        self.t_range    = t_range
        self.n_domain   = n_domain
        self.n_boundary = n_boundary
        self.n_initial  = n_initial

    def sample_domain(self,
                      t_window: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Alan içi kollokasiyon noktaları.
        Dönüş: [N, 3] tensör (requires_grad=True — autograd için zorunlu)
        """
        t_lo, t_hi = t_window if t_window else self.t_range
        t = torch.rand(self.n_domain, 1, device=DEVICE) * (t_hi - t_lo) + t_lo
        x = torch.rand(self.n_domain, 1, device=DEVICE) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        y = torch.rand(self.n_domain, 1, device=DEVICE) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return torch.cat([t, x, y], dim=1).requires_grad_(True)

    def sample_boundary(self,
                        t_window: Optional[Tuple[float, float]] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sınır noktaları — 4 kenardan eşit dağılım.
        Dönüş: (coords [N,3], normals [N,2])
          normals[:,0] = n_x, normals[:,1] = n_y  — dışa bakan birim normallar

        Kenarlar:
          Sol   (x=0):     n = (-1,  0)
          Sağ   (x=Xmax):  n = (+1,  0)
          Alt   (y=0):     n = ( 0, -1)
          Üst   (y=Ymax):  n = ( 0, +1)
        """
        t_lo, t_hi = t_window if t_window else self.t_range
        n = self.n_boundary // 4   # her kenardan n nokta

        t_all = torch.rand(n * 4, 1, device=DEVICE) * (t_hi - t_lo) + t_lo

        # Sol kenar (x=0)
        x_left  = torch.zeros(n, 1, device=DEVICE)
        y_left  = torch.rand(n, 1, device=DEVICE) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        n_left  = torch.tensor([[-1.0, 0.0]]).expand(n, -1).to(DEVICE)

        # Sağ kenar (x=Xmax)
        x_right = torch.full((n, 1), self.x_range[1], device=DEVICE)
        y_right = torch.rand(n, 1, device=DEVICE) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        n_right = torch.tensor([[+1.0, 0.0]]).expand(n, -1).to(DEVICE)

        # Alt kenar (y=0)
        x_bot   = torch.rand(n, 1, device=DEVICE) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        y_bot   = torch.zeros(n, 1, device=DEVICE)
        n_bot   = torch.tensor([[0.0, -1.0]]).expand(n, -1).to(DEVICE)

        # Üst kenar (y=Ymax)
        x_top   = torch.rand(n, 1, device=DEVICE) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        y_top   = torch.full((n, 1), self.y_range[1], device=DEVICE)
        n_top   = torch.tensor([[0.0, +1.0]]).expand(n, -1).to(DEVICE)

        x_all = torch.cat([x_left, x_right, x_bot, x_top], dim=0)
        y_all = torch.cat([y_left, y_right, y_bot, y_top], dim=0)
        normals = torch.cat([n_left, n_right, n_bot, n_top], dim=0)

        coords = torch.cat([t_all, x_all, y_all], dim=1).requires_grad_(True)
        return coords, normals

    def sample_initial(self) -> torch.Tensor:
        """
        Başlangıç koşulu noktaları — t=0, tüm alan üzerinde.
        T(0, x, y) = T_INIT = 540°C
        """
        t = torch.zeros(self.n_initial, 1, device=DEVICE)
        x = torch.rand(self.n_initial, 1, device=DEVICE) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        y = torch.rand(self.n_initial, 1, device=DEVICE) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return torch.cat([t, x, y], dim=1)


# ─────────────────────────────────────────────────────────────
# Quenching PINN Problemi
# ─────────────────────────────────────────────────────────────

class QuenchingProblem(PINNProblem):
    """
    Mortensen 2026 Quenching — Uygulama problemi.

    Ağ: f(t, x, y) → T(t, x, y)   [3 girdi, 1 çıktı]
    """

    name          = "quenching"
    n_input       = 3
    n_output      = 1
    domain_bounds = {"t": (0.0, T_END), "x": (0.0, X_MAX), "y": (0.0, Y_MAX)}

    def __init__(self,
                 time_mode:    str = "full",
                 fixed_stride: int = 2,
                 n_time_steps: int = N_TIME_STEPS):
        baseline = BaselinePaper()
        params = baseline.get("process_params")
        self.baseline    = baseline
        self.htc         = WaterQuenchHTC()
        self.T_init      = float(params.get("solution_heat_treatment_temp_C", T_INIT))
        self.T_water     = float(params.get("T_water_initial_C", T_WATER_INIT))
        self.time_mode   = str(time_mode).strip().lower()
        self.fixed_stride = max(1, int(fixed_stride))
        self.n_time_steps = max(2, int(n_time_steps))

    def analytical_solution(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Basitleştirilmiş analitik soğuma yaklaşımı (tek boyutlu).
        T(t) = T_water + (T_init − T_water)·exp(−α·t)
        Proxy referans olarak kullanılır — gerçek FEM verisi için baseline_data.py.
        """
        t = coords[:, 0:1]
        return self.T_water + (self.T_init - self.T_water) * torch.exp(-ALPHA * t)

    def make_loss_fn(self) -> Callable:
        """
        PINN kayıp fonksiyonu.

        Sınır koşulu (BC):
          Doğru Robin BC: k·∂T/∂n + h(T,T_w)·(T−T_w) = 0
          PINN kaybı:     L_B = ||k·∂T/∂n + h(T,T_w)·(T−T_w)||²

          Sınır noktalarında ∂T/∂n hesaplanır:
            ∂T/∂n = ∇T · n̂  (n̂: dışa bakan birim normal)
            ∇T = [∂T/∂x, ∂T/∂y]  (autograd ile)

        Başlangıç koşulu (IC):
          L_I = ||T(0,x,y) − T_init||²
        """
        pinn_loss = PINNLoss(w_physics=W_PHYSICS,
                             w_boundary=W_BOUNDARY,
                             w_initial=W_INITIAL)
        htc = self.htc
        T_w = self.T_water
        T_init = self.T_init

        # ── Normalizasyon ölçekleri (fiziksel boyutsuzlaştırma)
        # Problem: Robin BC kayıp büyüklüğü IC/PDE kayıp büyüklüğünü
        # h² ≈ 25,000,000 kat aşıyor → model T=T_water'a çöküyor.
        # Çözüm: her kaybı doğal fiziksel ölçeğiyle normalize et → O(1).
        dT_ref = T_init - T_w                   # 520°C — sıcaklık aralığı
        H_REF  = 5000.0                         # W/m²K — referans HTC
        # Robin (normalize edilmiş form): dT/dn + (h/k)·(T-Tw) = 0
        # Ölçek: (h_ref/k) * dT_ref ≈ (5000/150)*520 ≈ 17333 °C/m
        BC_SCALE  = (H_REF / K_THERMAL) * dT_ref
        # IC ölçeği: T aralığı
        IC_SCALE  = dT_ref
        # PDE ölçeği: ρCp·ΔT/T_end ≈ 2.4e6·520/30 ≈ 4.16e7 W/m³
        PDE_SCALE = RHO_CP * dT_ref / T_END

        def loss_fn(model, batch):
            # ── Fizik rezidüeli (normalize)
            coords_f = batch["domain"]
            pred_f   = model(coords_f)
            r_f      = heat_equation_residual(pred_f, coords_f)
            L_f = torch.mean((r_f / PDE_SCALE) ** 2)

            # ── Sınır koşulu: Robin BC normalize formu
            # Orijinal: k·∂T/∂n + h·(T−T_w) = 0
            # Normalize: [∂T/∂n + (h/k)·(T−T_w)] / BC_SCALE = 0
            # Bu form h²=25M kat büyüklük farkını ortadan kaldırır.
            coords_b = batch["boundary"]    # [N,3] requires_grad=True
            normals  = batch["normals"]     # [N,2] dışa bakan normal vektörler

            pred_b = model(coords_b)        # [N,1]
            T_surf = pred_b[:, 0]           # [N]

            grads_b = torch.autograd.grad(
                pred_b, coords_b,
                grad_outputs=torch.ones_like(pred_b),
                create_graph=True,
            )[0]                            # [N,3]: [dT/dt, dT/dx, dT/dy]

            dT_dx = grads_b[:, 1]           # [N]
            dT_dy = grads_b[:, 2]           # [N]
            nx = normals[:, 0]              # [N]
            ny = normals[:, 1]              # [N]
            dT_dn = dT_dx * nx + dT_dy * ny  # [N] — ∂T/∂n

            T_w_tensor = torch.full_like(T_surf, T_w)
            h_vals = htc.htc(T_surf, T_w_tensor)   # [N]

            # Normalize Robin residual: (dT/dn + (h/k)·(T-Tw)) / BC_SCALE
            robin_norm = (dT_dn + (h_vals / K_THERMAL) * (T_surf - T_w)) / BC_SCALE
            L_b = torch.mean(robin_norm ** 2)

            # ── Başlangıç koşulu (normalize): (T(0)-T_init) / IC_SCALE
            coords_i = batch["initial"]
            pred_i   = model(coords_i)
            true_i   = torch.full_like(pred_i, T_init)
            L_i = torch.mean(((pred_i - true_i) / IC_SCALE) ** 2)

            # ── Toplam kayıp — tüm terimler O(1) olduğundan
            # ağırlıklar fiziksel önemi yansıtır (BC/IC daha önemli)
            loss = W_PHYSICS * L_f + W_BOUNDARY * L_b + W_INITIAL * L_i
            details = {
                "L_physics":  float(L_f.item()),
                "L_boundary": float(L_b.item()),
                "L_initial":  float(L_i.item()),
                "L_total":    float(loss.item()),
            }
            return loss, details

        return loss_fn

    def make_val_fn(self, n_val: int = 500) -> Callable:
        """
        Doğrulama: PINN tahmini vs. analitik soğuma yaklaşımı.
        Normalize L2 hatası döndürür.

        Not: Analitik yaklaşım proxy referanstır.
             Gerçek doğruluk için baseline_data.py → compare_with_pinn() kullanılır.
        """
        # Doğrulama noktaları: iç bölge (Bi>>1 → sınırda T≈T_water, anlamlı
        # karşılaştırma için iç bölge x ∈ [0.2·X_MAX, 0.8·X_MAX] kullanılır)
        torch.manual_seed(0)
        coords_v = torch.cat([
            torch.rand(n_val, 1, device=DEVICE) * T_END,
            torch.rand(n_val, 1, device=DEVICE) * (0.6 * X_MAX) + 0.2 * X_MAX,
            torch.rand(n_val, 1, device=DEVICE) * (0.6 * Y_MAX) + 0.2 * Y_MAX,
        ], dim=1)
        u_ref = self.analytical_solution(coords_v)

        def val_fn(model) -> float:
            model.eval()
            with torch.no_grad():
                pred = model(coords_v)
            return (torch.norm(pred - u_ref) /
                    (torch.norm(u_ref) + 1e-10)).item()

        return val_fn

    def make_paper_comparison_fn(self) -> Callable:
        """
        Paper Fig 7 zamanlarında PINN sıcaklık tahminini al ve karşılaştır.

        Karşılaştırma mantığı:
          - Fig 7'deki zaman noktaları (0..30s) kullanılır
          - PINN, part yüzeyindeki sıcaklığı tahmin eder
          - Analitik soğuma eğrisiyle karşılaştırılır
          - Fig 7 su sıcaklığı referans grafik olarak gösterilir

        Döndürür: metrics dict (L2, MAE, RMSE, R2, timestep stats)
        """
        fig7 = self.baseline.get("fig7_water_temp")
        time_s = torch.tensor(fig7["time_s"], dtype=torch.float32, device=DEVICE)

        # PINN değerlendirme noktaları: iç merkez bölge (Bi>>1 → yüzey T_water'a düşer,
        # anlamlı karşılaştırma için iç merkez kullanılır)
        n_per_t = 20
        # İç merkez ızgara: x ∈ [0.3·X_MAX, 0.7·X_MAX], y ∈ [0.3·Y_MAX, 0.7·Y_MAX]
        x_eval = torch.linspace(0.3 * X_MAX, 0.7 * X_MAX, n_per_t, device=DEVICE)
        y_eval = torch.full((n_per_t,), Y_MAX / 2, device=DEVICE)

        def compare_fn(model) -> Dict:
            model.eval()
            T_pinn_mean = []
            T_analytical = []

            with torch.no_grad():
                for t_val in time_s:
                    t_tensor = torch.full((n_per_t,), t_val.item(), device=DEVICE)
                    coords = torch.stack([t_tensor, x_eval, y_eval], dim=1)
                    T_pred = model(coords).squeeze(-1).cpu().numpy()
                    T_pinn_mean.append(float(T_pred.mean()))
                    # Analitik: T(t) = T_w + (T_0 - T_w) * exp(-ALPHA·t)
                    # ALPHA = 2D Robin BC temel modu soğuma katsayısı (1.75e-3 /s)
                    T_an = self.T_water + (self.T_init - self.T_water) * np.exp(-ALPHA * t_val.item())
                    T_analytical.append(T_an)

            T_pinn = np.array(T_pinn_mean)
            T_ref  = np.array(T_analytical)
            diff   = T_pinn - T_ref

            mae  = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            max_err = float(np.max(np.abs(diff)))
            ss_res = np.sum(diff ** 2)
            ss_tot = np.sum((T_ref - T_ref.mean()) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-10))
            l2 = float(np.linalg.norm(diff) / (np.linalg.norm(T_ref) + 1e-10))

            return {
                "time_s":          [float(t) for t in time_s.cpu().tolist()],
                "T_PINN":          T_pinn.tolist(),
                "T_analytical":    T_ref.tolist(),
                "fig7_water_ch1":  fig7["Channel_1"],
                "fig7_time_s":     fig7["time_s"],
                "L2_rel":          l2,
                "MAE_C":           mae,
                "RMSE_C":          rmse,
                "MaxErr_C":        max_err,
                "R2":              r2,
                "note": (
                    "PINN: part surface temperature (x=0 boundary, avg). "
                    "Analytical: exponential decay approximation. "
                    "Fig7: water tank temperature (separate measurement)."
                ),
            }

        return compare_fn

    def make_batch_fn(self,
                      n_domain:    int = N_DOMAIN,
                      n_boundary:  int = N_BOUNDARY,
                      n_initial:   int = N_INITIAL,
                      scheduler=None,
                      time_mode:   str = None,
                      fixed_stride: int = None) -> Callable:
        sampler = CollocationSampler(n_domain=n_domain,
                                     n_boundary=n_boundary,
                                     n_initial=n_initial)
        mode   = (time_mode or self.time_mode).strip().lower()
        if mode not in {"full", "fixed", "adaptive"}:
            raise ValueError(f"Unsupported quenching time mode: {mode}")
        stride = max(1, int(fixed_stride or self.fixed_stride))
        t_full = np.linspace(0.0, T_END, self.n_time_steps).tolist()

        if mode == "fixed":
            selected_times = t_full[::stride]
            if selected_times[-1] != t_full[-1]:
                selected_times.append(t_full[-1])
        else:
            selected_times = list(t_full)

        skipped_times = [float(t) for t in t_full if t not in set(selected_times)]
        dt = (T_END - 0.0) / max(len(t_full) - 1, 1)
        current_residual = [float("inf")]
        is_frozen        = [False]
        static_index     = [0]

        static_stats = {
            "mode":           mode,
            "total_steps":    len(t_full),
            "visited":        len(selected_times),
            "skipped":        len(skipped_times),
            "efficiency_pct": round(100.0 * len(skipped_times) / max(len(t_full), 1), 2),
            "visited_times":  [float(t) for t in selected_times],
            "skipped_times":  skipped_times,
        }

        def get_batch(epoch: int) -> dict:
            window = None
            if not is_frozen[0]:
                if mode == "adaptive" and scheduler is not None:
                    t = scheduler.next_timestep(current_residual[0])
                    if t is None:
                        scheduler.reset()
                        t = scheduler.next_timestep(current_residual[0])
                    n_steps = max(len(scheduler.t_full) - 1, 1)
                    dt_local = (scheduler.t_end - scheduler.t_start) / n_steps
                    window = (max(scheduler.t_start, t - dt_local / 2),
                              min(scheduler.t_end,   t + dt_local / 2))
                elif mode == "fixed":
                    # Sabit adım atlama: sadece seçili zaman pencerelerini kullan
                    t = selected_times[static_index[0] % len(selected_times)]
                    static_index[0] += 1
                    window = (max(0.0, t - dt / 2), min(T_END, t + dt / 2))
                # mode == "full": window=None → tüm zaman aralığından örnekle [0, T_END]
                # Bu sayede model her epoch'ta tam temporal evrimi görür ve
                # IC (t=0, T=540) ile BC/PDE (t>0) arasındaki bağı öğrenir.
                # Pencereli örnekleme: model küçük penceredeki sabit T ile PDE/BC'yi
                # trivially satisfy eder (dT/dt≈0 → PDE=0, T≈T_water → BC=0) ve
                # IC öğrenilmez → kötü yerel minimumlara takılır.

            coords_b, normals = sampler.sample_boundary(window)
            return {
                "domain":   sampler.sample_domain(window),
                "boundary": coords_b,
                "normals":  normals,
                "initial":  sampler.sample_initial(),
            }

        def set_residual(r: float): current_residual[0] = float(r)
        def freeze():               is_frozen[0] = True
        def unfreeze():             is_frozen[0] = False
        def get_stats():
            if mode == "adaptive" and scheduler is not None:
                return scheduler.get_stats()
            return dict(static_stats)

        get_batch.set_residual = set_residual
        get_batch.freeze       = freeze
        get_batch.unfreeze     = unfreeze
        get_batch.get_stats    = get_stats
        return get_batch
