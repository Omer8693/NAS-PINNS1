"""
Training Module — Adam + L-BFGS + PSO
=======================================
Eğitim sırası (3 faz):
  Phase 1 : Adam            — bulk eğitim
  Phase 2 : L-BFGS          — Adam ağırlıklarından başlar, Quasi-Newton ince ayarı
  Phase 3 : PSO             — Adam ağırlıklarından başlar (bağımsız), global arama

L-BFGS ve PSO BAĞIMSIZDIR:
  - İkisi de adam_state'ten başlar (PSO, L-BFGS sonrası değil)
  - Bu sayede hangisinin Adam'ı daha iyi iyileştirdiği test edilir
"""

import time
from copy import deepcopy
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.config import DEVICE


# ─────────────────────────────────────────────────────────────
# Bölüm 1 — Adam Eğitimi
# ─────────────────────────────────────────────────────────────

class AdamTrainer:
    """
    Adam optimizer ile PINN eğitimi.

    Özellikler:
      - StepLR öğrenme hızı düşüşü
      - Gradient clipping (max_norm=1.0)
      - En iyi model kaydı
    """

    def __init__(self,
                 model:           nn.Module,
                 loss_fn:         Callable,
                 lr:              float = 1e-3,
                 lr_decay_step:   int   = 1000,
                 lr_decay_gamma:  float = 0.9):
        self.model   = model
        self.loss_fn = loss_fn
        self.history = {"loss": [], "l2": [], "phase": "adam"}

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        loss, details = self.loss_fn(self.model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        return details

    def train(self,
              get_batch:   Callable,
              n_epochs:    int = 5000,
              print_every: int = 500,
              val_fn:      Optional[Callable] = None) -> Dict:
        print(f"\n{'─'*48}")
        print(f"  ADAM TRAINING  ({n_epochs} epochs)")
        print(f"{'─'*48}")

        t0         = time.time()
        best_loss  = float("inf")
        best_state = None

        for epoch in range(1, n_epochs + 1):
            batch   = get_batch(epoch)
            details = self.train_step(batch)
            self.history["loss"].append(details["L_total"])

            if hasattr(get_batch, "set_residual"):
                get_batch.set_residual(details.get("L_physics", float("inf")))

            if epoch % print_every == 0:
                l2 = val_fn(self.model) if val_fn else 0.0
                self.history["l2"].append({"epoch": epoch, "l2": l2})

                if details["L_total"] < best_loss:
                    best_loss  = details["L_total"]
                    best_state = deepcopy(self.model.state_dict())

                elapsed = time.time() - t0
                print(f"  Epoch {epoch:6d} | Loss: {details['L_total']:.4e} "
                      f"| L2: {l2:.4e} | LR: {self.scheduler.get_last_lr()[0]:.2e} "
                      f"| {elapsed:.0f}s")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  → Best model restored (loss={best_loss:.4e})")

        total_time = time.time() - t0
        print(f"  Adam complete: {total_time:.1f}s")
        return {
            "phase":      "adam",
            "epochs":     n_epochs,
            "final_loss": self.history["loss"][-1] if self.history["loss"] else None,
            "train_time": total_time,
            "history":    self.history,
        }


# ─────────────────────────────────────────────────────────────
# Bölüm 2 — L-BFGS İnce Ayarı
# ─────────────────────────────────────────────────────────────

class LBFGSFinetuner:
    """
    Adam sonrası L-BFGS ile hassas ince ayar — NAS-PINNS1 parametreleri.

    NAS-PINNS1 (cfg.py) referans değerleri:
      lr=1.0, max_iter=5000, history_size=20, line_search=strong_wolfe
    """

    def __init__(self,
                 model:            nn.Module,
                 loss_fn:          Callable,
                 max_iter:         int   = 5000,
                 history_size:     int   = 20,
                 tolerance_grad:   float = 1e-7,
                 tolerance_change: float = 1e-9):
        self.model   = model
        self.loss_fn = loss_fn
        self.history = {"loss": [], "phase": "lbfgs"}
        self._max_iter         = max_iter
        self._history_size     = history_size
        self._tolerance_grad   = tolerance_grad
        self._tolerance_change = tolerance_change

        # lr=1.0 — strong_wolfe line search adım boyutunu yönetiyor
        self.optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr               = 1.0,
            max_iter         = max_iter,
            history_size     = history_size,
            tolerance_grad   = tolerance_grad,
            tolerance_change = tolerance_change,
            line_search_fn   = "strong_wolfe",
        )

    def finetune(self,
                 get_batch: Callable,
                 n_steps:   int = 1,
                 val_fn:    Optional[Callable] = None) -> Dict:
        """
        Sabit batch üzerinde tek L-BFGS step() çağrısı.
        İçeride max_iter iterasyona kadar strong_wolfe ile optimize eder.
        """
        print(f"\n{'─'*48}")
        print(f"  L-BFGS FINE-TUNING  (max_iter={self._max_iter}, single step)")
        print(f"{'─'*48}")

        t0    = time.time()
        batch = get_batch(0)   # Sabit batch: closure hep aynı noktaları kullanır
        final_loss = None

        def closure():
            nonlocal final_loss
            self.optimizer.zero_grad()
            loss, _ = self.loss_fn(self.model, batch)
            loss.backward()
            val = loss.item()
            if np.isfinite(val):
                self.history["loss"].append(val)
                final_loss = val
            return loss

        try:
            self.optimizer.step(closure)
        except Exception as exc:
            print(f"  [L-BFGS] step() raised: {exc}")

        total_time = time.time() - t0
        iters_done = len(self.history["loss"])
        # DÜZELTİLDİ: `if best_loss` → `if best_loss is not None`
        # Önceki kod loss=0.0 için yanlış "no valid loss" yazıyordu
        best_loss  = min(self.history["loss"]) if self.history["loss"] else None
        l2_after   = val_fn(self.model) if val_fn else None

        if best_loss is not None:
            print(f"  L-BFGS complete: {total_time:.1f}s  |  "
                  f"iters={iters_done}  |  best_loss={best_loss:.4e}")
        else:
            print(f"  L-BFGS complete: {total_time:.1f}s  |  iters={iters_done}  |  no valid loss")
        if l2_after is not None:
            print(f"  L-BFGS final L2: {l2_after:.4e}")

        return {
            "phase":             "lbfgs",
            "steps":             iters_done,
            "requested_max_iter": self._max_iter,
            "history_size":      self._history_size,
            "tolerance_grad":    self._tolerance_grad,
            "tolerance_change":  self._tolerance_change,
            "final_loss":        best_loss,
            "train_time":        total_time,
            "history":           self.history,
        }


# ─────────────────────────────────────────────────────────────
# Bölüm 3 — PSO İnce Ayarı
# ─────────────────────────────────────────────────────────────

class PSOFinetuner:
    """
    Adam sonrası Particle Swarm Optimization ile global ince ayar.

    PSO L-BFGS'TEN BAĞIMSIZ ÇALIŞIR — adam_state'ten başlar.

    Hiper-parametreler (PSO standartları):
      w  = 0.7  : atalet katsayısı
      c1 = 1.5  : bilişsel katsayı
      c2 = 1.5  : sosyal katsayı
    """

    def __init__(self,
                 model:         nn.Module,
                 loss_fn:       Callable,
                 n_particles:   int   = 20,
                 w:             float = 0.7,
                 c1:            float = 1.5,
                 c2:            float = 1.5,
                 perturb_scale: float = 0.1):
        self.model         = model
        self.loss_fn       = loss_fn
        self.n_particles   = n_particles
        self.w             = w
        self.c1            = c1
        self.c2            = c2
        self.perturb_scale = perturb_scale
        self.history       = {"loss": [], "phase": "pso"}

        self.base_params = self._get_flat_params()
        self.n_dims      = len(self.base_params)

        # Adaptif perturb_scale: büyük ağlarda gürültüyü azalt
        if self.n_dims > 20_000:
            effective = perturb_scale * np.sqrt(1_000 / self.n_dims)
            print(f"  [PSO] Large network ({self.n_dims:,} dim): perturb_scale "
                  f"{perturb_scale:.4f} → {effective:.6f}")
            self.perturb_scale = effective

    def _get_flat_params(self) -> np.ndarray:
        return np.concatenate([
            p.data.cpu().numpy().flatten() for p in self.model.parameters()
        ])

    def _set_flat_params(self, flat: np.ndarray):
        offset = 0
        for p in self.model.parameters():
            size = p.numel()
            p.data = torch.tensor(
                flat[offset:offset + size].reshape(p.shape),
                dtype=torch.float32, device=DEVICE
            )
            offset += size

    def _eval_loss(self, batch: dict) -> float:
        self.model.eval()
        with torch.enable_grad():
            loss, _ = self.loss_fn(self.model, batch)
        return float(loss.item())

    def _model_copy_with(self, params: np.ndarray) -> nn.Module:
        m = deepcopy(self.model)
        offset = 0
        for p in m.parameters():
            size = p.numel()
            p.data = torch.tensor(
                params[offset:offset + size].reshape(p.shape),
                dtype=torch.float32, device=DEVICE
            )
            offset += size
        return m

    def finetune(self,
                 get_batch: Callable,
                 n_steps:   int = 30,
                 val_fn:    Optional[Callable] = None) -> Dict:
        print(f"\n{'─'*48}")
        print(f"  PSO FINE-TUNING  ({n_steps} steps, {self.n_particles} particles)")
        print(f"  Search dim: {self.n_dims}  |  perturb_scale: {self.perturb_scale}")
        print(f"{'─'*48}")

        t0 = time.time()

        positions  = (np.random.randn(self.n_particles, self.n_dims) *
                      self.perturb_scale + self.base_params)
        velocities = np.zeros((self.n_particles, self.n_dims))

        pbest_pos  = positions.copy()
        pbest_loss = np.full(self.n_particles, float("inf"))
        gbest_pos  = self.base_params.copy()
        gbest_loss = float("inf")

        batch = get_batch(0)

        for i in range(self.n_particles):
            self._set_flat_params(positions[i])
            lv = self._eval_loss(batch)
            pbest_loss[i] = lv
            if lv < gbest_loss:
                gbest_loss = lv
                gbest_pos  = positions[i].copy()

        print(f"  Initial best loss: {gbest_loss:.4e}")

        for step in range(1, n_steps + 1):
            r1 = np.random.rand(self.n_particles, self.n_dims)
            r2 = np.random.rand(self.n_particles, self.n_dims)

            velocities = (self.w  * velocities +
                          self.c1 * r1 * (pbest_pos - positions) +
                          self.c2 * r2 * (gbest_pos - positions))
            positions  = positions + velocities

            for i in range(self.n_particles):
                self._set_flat_params(positions[i])
                lv = self._eval_loss(batch)
                if lv < pbest_loss[i]:
                    pbest_loss[i] = lv
                    pbest_pos[i]  = positions[i].copy()
                if lv < gbest_loss:
                    gbest_loss = lv
                    gbest_pos  = positions[i].copy()

            self.history["loss"].append(gbest_loss)

            if step % 5 == 0:
                l2 = val_fn(self._model_copy_with(gbest_pos)) if val_fn else 0.0
                print(f"  Step {step:4d} | Best Loss: {gbest_loss:.4e} | L2: {l2:.4e}")

        self._set_flat_params(gbest_pos)

        total_time = time.time() - t0
        print(f"  PSO complete: {total_time:.1f}s  |  Final best loss: {gbest_loss:.4e}")
        return {
            "phase":      "pso",
            "steps":      n_steps,
            "particles":  self.n_particles,
            "final_loss": gbest_loss,
            "train_time": total_time,
            "history":    self.history,
        }


# ─────────────────────────────────────────────────────────────
# Bölüm 4 — Tam Eğitim Pipeline'ı
# ─────────────────────────────────────────────────────────────

def full_training_pipeline(model:                  nn.Module,
                            loss_fn:               Callable,
                            get_batch:             Callable,
                            val_fn:                Callable,
                            adam_epochs:           int   = 5000,
                            lbfgs_max_iter:        int   = 5000,
                            lbfgs_history_size:    int   = 20,
                            lbfgs_tolerance_grad:  float = 1e-7,
                            lbfgs_tolerance_change: float = 1e-9,
                            pso_steps:             int   = 50,
                            pso_particles:         int   = 20,
                            adam_lr:               float = 1e-3,
                            run_lbfgs:             bool  = False,
                            run_pso:               bool  = False) -> Dict:
    """
    Tam eğitim sırası:
      Adam → adam_state kaydedilir
      adam_state → L-BFGS  (bağımsız)
      adam_state → PSO      (bağımsız, L-BFGS sonrası değil)
    """
    results = {}

    # ── Faz 1: Adam
    adam = AdamTrainer(model, loss_fn, lr=adam_lr)
    results["adam"] = adam.train(get_batch, adam_epochs, val_fn=val_fn)

    l2_adam = val_fn(model)
    results["adam"]["final_l2"] = l2_adam
    print(f"\n  Adam final L2: {l2_adam:.4e}")

    adam_state = deepcopy(model.state_dict())

    # ── Faz 2: L-BFGS (Adam'dan bağımsız)
    # NOT: L-BFGS frozen (full-domain) batch kullanır — TemporalSkip pencereleri
    # Hessian tahmini tutarsız yapar; freeze ile tam domain'e geç.
    lbfgs_state = None
    if run_lbfgs:
        model.load_state_dict(adam_state)
        if hasattr(get_batch, "freeze"):
            get_batch.freeze()
        lbfgs = LBFGSFinetuner(
            model, loss_fn,
            max_iter         = lbfgs_max_iter,
            history_size     = lbfgs_history_size,
            tolerance_grad   = lbfgs_tolerance_grad,
            tolerance_change = lbfgs_tolerance_change,
        )
        results["lbfgs"] = lbfgs.finetune(get_batch, val_fn=val_fn)
        if hasattr(get_batch, "unfreeze"):
            get_batch.unfreeze()
        l2_lbfgs = val_fn(model)
        if not np.isfinite(l2_lbfgs):
            print(f"  [Warning] L-BFGS L2={l2_lbfgs}, restoring Adam weights")
            model.load_state_dict(adam_state)
            l2_lbfgs = l2_adam
        results["lbfgs"]["final_l2"] = l2_lbfgs
        lbfgs_state = deepcopy(model.state_dict())
        print(f"\n  L-BFGS final L2: {l2_lbfgs:.4e}")

    # ── Faz 3: PSO (Adam'dan bağımsız)
    pso_state = None
    if run_pso:
        model.load_state_dict(adam_state)
        if hasattr(get_batch, "freeze"):
            get_batch.freeze()
        pso = PSOFinetuner(model, loss_fn, n_particles=pso_particles)
        results["pso"] = pso.finetune(get_batch, pso_steps, val_fn=val_fn)
        if hasattr(get_batch, "unfreeze"):
            get_batch.unfreeze()
        l2_pso = val_fn(model)
        if not np.isfinite(l2_pso):
            print(f"  [Warning] PSO L2={l2_pso}, restoring Adam weights")
            model.load_state_dict(adam_state)
            l2_pso = l2_adam
        results["pso"]["final_l2"] = l2_pso
        pso_state = deepcopy(model.state_dict())
        print(f"\n  PSO final L2: {l2_pso:.4e}")

    # En iyi fazı seç
    candidates = {"adam": (l2_adam, adam_state)}
    if run_lbfgs and lbfgs_state is not None:
        l2_l = results["lbfgs"].get("final_l2", float("nan"))
        if np.isfinite(l2_l):
            candidates["lbfgs"] = (l2_l, lbfgs_state)
    if run_pso and pso_state is not None:
        l2_p = results["pso"].get("final_l2", float("nan"))
        if np.isfinite(l2_p):
            candidates["pso"] = (l2_p, pso_state)

    best_phase, (best_l2, best_state) = min(candidates.items(), key=lambda kv: kv[1][0])
    model.load_state_dict(best_state)

    # Özet tablosu
    print(f"\n{'='*52}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*52}")
    print(f"  {'Phase':<12} | {'L2 Error':>12} | {'Time (s)':>10} | {'Δ vs Adam':>12}")
    print(f"  {'─'*50}")
    print(f"  {'Adam':<12} | {l2_adam:>12.4e} | "
          f"{results['adam']['train_time']:>10.1f} | {'baseline':>12}")
    if run_lbfgs and "lbfgs" in results:
        l2_l  = results['lbfgs']['final_l2']
        delta = l2_adam - l2_l if np.isfinite(l2_l) else float('nan')
        l2_str = f"{l2_l:>12.4e}" if np.isfinite(l2_l) else "         NaN"
        d_str  = f"{delta:>+12.4e}" if np.isfinite(delta) else "         NaN"
        print(f"  {'Adam+LBFGS':<12} | {l2_str} | "
              f"{results['lbfgs']['train_time']:>10.1f} | {d_str}")
    if run_pso and "pso" in results:
        l2_p  = results['pso']['final_l2']
        delta = l2_adam - l2_p if np.isfinite(l2_p) else float('nan')
        l2_str = f"{l2_p:>12.4e}" if np.isfinite(l2_p) else "         NaN"
        d_str  = f"{delta:>+12.4e}" if np.isfinite(delta) else "         NaN"
        print(f"  {'Adam+PSO':<12} | {l2_str} | "
              f"{results['pso']['train_time']:>10.1f} | {d_str}")
    print(f"{'='*52}")
    print(f"  Best phase: {best_phase.upper()}")
    print(f"{'='*52}")

    results["best_phase"] = best_phase
    return results
