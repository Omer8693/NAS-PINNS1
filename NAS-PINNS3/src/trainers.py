"""
Training Module — Adam + L-BFGS + PSO
=======================================
Eğitim sırası (3 faz):
  Phase 1 : Adam            — bulk eğitim, tüm epoch'ların büyük kısmı
  Phase 2 : L-BFGS          — Adam ağırlıklarından başlar, Quasi-Newton ince ayarı
  Phase 3 : PSO             — Adam ağırlıklarından başlar (bağımsız), global arama

L-BFGS ve PSO BAĞIMSIZDIR:
  - İkisi de adam_state'ten başlar (PSO, L-BFGS sonrası değil)
  - Bu sayede hangisinin Adam'ı daha iyi iyileştirdiği test edilir
  - Karşılaştırma: Adam vs Adam+LBFGS vs Adam+PSO

PSO özelliği:
  - Ağırlıkları düzleştirilmiş numpy vektörü olarak temsil eder
  - Adam çözümü etrafında küçük pertürbasyonla parçacık bulutunu başlatır
  - GPU'da torch ile eval — büyük ağlarda pyswarms'tan daha verimli
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Callable, Optional
from copy import deepcopy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Bölüm 1 — Adam Eğitimi
# ─────────────────────────────────────────────────────────────

class AdamTrainer:
    """
    Adam optimizer ile PINN eğitimi.

    Özellikler:
      - StepLR öğrenme hızı düşüşü (her step_size epoch'ta gamma ile çarp)
      - Gradient clipping (max_norm=1.0) — patlayan gradient'ları önler
      - En iyi model kaydı (early stopping yok, ama en düşük kayıp saklanır)
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
        """Tek eğitim adımı — forward + backward + clip + step."""
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
        """
        n_epochs boyunca Adam eğitimi.

        get_batch(epoch) → her epoch yeni kollokasiyon batch'i döndürür.
        val_fn(model)    → L2 hatasını döndürür (baseline ile karşılaştırma için).
        """
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

            if epoch % print_every == 0:
                l2 = val_fn(self.model) if val_fn else 0.0
                self.history["l2"].append({"epoch": epoch, "l2": l2})

                # En iyi modeli sakla
                if details["L_total"] < best_loss:
                    best_loss  = details["L_total"]
                    best_state = deepcopy(self.model.state_dict())

                elapsed = time.time() - t0
                print(f"  Epoch {epoch:6d} | Loss: {details['L_total']:.4e} "
                      f"| L2: {l2:.4e} | LR: {self.scheduler.get_last_lr()[0]:.2e} "
                      f"| {elapsed:.0f}s")

        # En iyi ağırlıkları geri yükle
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
    Adam sonrası L-BFGS ile hassas ince ayar (fine-tuning).

    L-BFGS: sınırlı bellek Quasi-Newton yöntemi.
    Adam'ın düz bölgeye getirdiği çözümden Strong Wolfe ile optimize eder.

    Avantaj:
      - Birinci dereceden yöntemlere göre çok daha hızlı yakınsama
      - Loss yüzeyi yerel olarak kuadratik ise mükemmel performans

    Kısıt:
      - Büyük ağlarda bellek yoğun (history_size ile sınırlandırılır)
      - closure() pattern zorunlu — her adımda loss yeniden hesaplanır
    """

    def __init__(self,
                 model:        nn.Module,
                 loss_fn:      Callable,
                 lr:           float = 1.0,
                 max_iter:     int   = 100,
                 history_size: int   = 50):
        self.model   = model
        self.loss_fn = loss_fn
        self.history = {"loss": [], "phase": "lbfgs"}

        self.optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr             = lr,
            max_iter       = max_iter,
            history_size   = history_size,
            line_search_fn = "strong_wolfe",   # Armijo + Wolfe koşulları
        )

    def finetune(self,
                 get_batch: Callable,
                 n_steps:   int = 50,
                 val_fn:    Optional[Callable] = None) -> Dict:
        """
        n_steps L-BFGS adımı. Her adımda closure ile loss yeniden hesaplanır.
        Adam'dan başlar — full_training_pipeline bunu garantiler.
        """
        print(f"\n{'─'*48}")
        print(f"  L-BFGS FINE-TUNING  ({n_steps} steps)")
        print(f"{'─'*48}")

        t0 = time.time()

        for step in range(1, n_steps + 1):
            batch = get_batch(step)

            def closure():
                self.optimizer.zero_grad()
                loss, _ = self.loss_fn(self.model, batch)
                loss.backward()
                return loss

            loss_val = self.optimizer.step(closure)

            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.item()
            self.history["loss"].append(loss_val)

            if step % 10 == 0:
                l2 = val_fn(self.model) if val_fn else 0.0
                print(f"  Step {step:4d} | Loss: {loss_val:.4e} | L2: {l2:.4e}")

        total_time = time.time() - t0
        print(f"  L-BFGS complete: {total_time:.1f}s")
        return {
            "phase":      "lbfgs",
            "steps":      n_steps,
            "final_loss": self.history["loss"][-1] if self.history["loss"] else None,
            "train_time": total_time,
            "history":    self.history,
        }


# ─────────────────────────────────────────────────────────────
# Bölüm 3 — PSO İnce Ayarı
# ─────────────────────────────────────────────────────────────

class PSOFinetuner:
    """
    Adam sonrası Particle Swarm Optimization ile global ince ayar.

    PSO L-BFGS'TEN BAĞIMSIZ ÇALIŞIR — adam_state'ten başlar.

    Neden PSO?
      - L-BFGS yerel minimum'a takılabilir; PSO global arama yapar
      - Adam'ın bulduğu bölgenin etrafında parçacık bulutu oluşturur
      - Yerel minimum'dan kaçış için pertürbasyon ölçeği ayarlanabilir

    Torch tabanlı GPU implementasyonu:
      - Ağırlıklar düzleştirilmiş numpy vektörü olarak temsil edilir
      - Her parçacık eval'i: model'e yükle → forward → loss hesapla
      - no_grad ile hızlandırılmış değerlendirme

    Hiper-parametreler (PSO standartları):
      w  = 0.7   : atalet katsayısı (momentum)
      c1 = 1.5   : bilişsel katsayı (kendi en iyisine çekim)
      c2 = 1.5   : sosyal katsayı   (küresel en iyiye çekim)
    """

    def __init__(self,
                 model:         nn.Module,
                 loss_fn:       Callable,
                 n_particles:   int   = 20,
                 w:             float = 0.7,
                 c1:            float = 1.5,
                 c2:            float = 1.5,
                 perturb_scale: float = 0.01):   # pertürbasyon: Adam çözümünün %1'i
        self.model         = model
        self.loss_fn       = loss_fn
        self.n_particles   = n_particles
        self.w             = w
        self.c1            = c1
        self.c2            = c2
        self.perturb_scale = perturb_scale
        self.history       = {"loss": [], "phase": "pso"}

        # Adam ağırlıklarını düzleştirilmiş numpy vektörüne al
        self.base_params = self._get_flat_params()
        self.n_dims      = len(self.base_params)

    def _get_flat_params(self) -> np.ndarray:
        """Model ağırlıklarını tek boyutlu numpy vektörüne dönüştür."""
        return np.concatenate([
            p.data.cpu().numpy().flatten() for p in self.model.parameters()
        ])

    def _set_flat_params(self, flat: np.ndarray):
        """Düzleştirilmiş vektörü model parametrelerine yükle."""
        offset = 0
        for p in self.model.parameters():
            size = p.numel()
            p.data = torch.tensor(
                flat[offset:offset + size].reshape(p.shape),
                dtype=torch.float32, device=DEVICE
            )
            offset += size

    def _eval_loss(self, batch: dict) -> float:
        """Mevcut model ağırlıklarıyla batch kaybını hesapla.
        not_grad kullanılmaz — PDE rezidüeli autograd.grad gerektirir."""
        self.model.eval()
        with torch.enable_grad():
            loss, _ = self.loss_fn(self.model, batch)
        return float(loss.item())

    def _model_copy_with(self, params: np.ndarray) -> nn.Module:
        """Verilen ağırlıklarla modelin derin kopyasını oluştur (val_fn için)."""
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
        """
        PSO ile n_steps iterasyon.
        Parçacık bulutu Adam çözümü etrafında başlatılır.
        En iyi global pozisyon her iterasyonda güncellenir.
        """
        print(f"\n{'─'*48}")
        print(f"  PSO FINE-TUNING  ({n_steps} steps, {self.n_particles} particles)")
        print(f"  Search dim: {self.n_dims}  |  perturb_scale: {self.perturb_scale}")
        print(f"{'─'*48}")

        t0 = time.time()

        # Parçacık başlatma: Adam çözümü etrafında Gaussian pertürbasyon
        positions  = (np.random.randn(self.n_particles, self.n_dims) *
                      self.perturb_scale + self.base_params)
        velocities = np.zeros((self.n_particles, self.n_dims))

        pbest_pos  = positions.copy()
        pbest_loss = np.full(self.n_particles, float("inf"))
        gbest_pos  = self.base_params.copy()
        gbest_loss = float("inf")

        # Sabit batch: PSO değerlendirmesinde aynı kollokasiyon noktaları kullanılır
        batch = get_batch(0)

        # İlk değerlendirme — her parçacık için başlangıç kaybı
        for i in range(self.n_particles):
            self._set_flat_params(positions[i])
            lv = self._eval_loss(batch)
            pbest_loss[i] = lv
            if lv < gbest_loss:
                gbest_loss = lv
                gbest_pos  = positions[i].copy()

        print(f"  Initial best loss: {gbest_loss:.4e}")

        # PSO ana döngüsü
        for step in range(1, n_steps + 1):
            r1 = np.random.rand(self.n_particles, self.n_dims)
            r2 = np.random.rand(self.n_particles, self.n_dims)

            # Hız güncelleme: atalet + bilişsel + sosyal
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

        # En iyi global pozisyonu modele yükle
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

def full_training_pipeline(model:         nn.Module,
                            loss_fn:       Callable,
                            get_batch:     Callable,
                            val_fn:        Callable,
                            adam_epochs:   int   = 5000,
                            lbfgs_steps:   int   = 100,
                            pso_steps:     int   = 50,
                            pso_particles: int   = 20,
                            adam_lr:       float = 1e-3,
                            run_lbfgs:     bool  = True,
                            run_pso:       bool  = True) -> Dict:
    """
    Tam eğitim sırası ve sonuç özeti.

    Akış:
      Adam  →  adam_state kaydedilir
      adam_state → L-BFGS  (bağımsız)
      adam_state → PSO      (bağımsız, L-BFGS sonrası değil)

    Bu yapı sayesinde:
      - Üç sonuç da karşılaştırılabilir temel (Adam) üzerinden değerlendirilir
      - En iyi optimizer seçimi adil bir karşılaştırmayla yapılabilir
    """
    results = {}

    # ── Faz 1: Adam ──────────────────────────────────────────
    adam = AdamTrainer(model, loss_fn, lr=adam_lr)
    results["adam"] = adam.train(get_batch, adam_epochs, val_fn=val_fn)

    l2_adam = val_fn(model)
    results["adam"]["final_l2"] = l2_adam
    print(f"\n  Adam final L2: {l2_adam:.4e}")

    # Adam ağırlıklarını sakla — L-BFGS ve PSO buradan başlar
    adam_state = deepcopy(model.state_dict())

    # ── Faz 2: L-BFGS ────────────────────────────────────────
    if run_lbfgs:
        model.load_state_dict(adam_state)    # Adam noktasına geri dön
        lbfgs = LBFGSFinetuner(model, loss_fn)
        results["lbfgs"] = lbfgs.finetune(get_batch, lbfgs_steps, val_fn=val_fn)
        l2_lbfgs = val_fn(model)
        results["lbfgs"]["final_l2"] = l2_lbfgs
        print(f"\n  L-BFGS final L2: {l2_lbfgs:.4e}")
        lbfgs_state = deepcopy(model.state_dict())

    # ── Faz 3: PSO (Adam'dan bağımsız) ───────────────────────
    if run_pso:
        model.load_state_dict(adam_state)    # Adam noktasına geri dön (PSO bağımsız)
        pso = PSOFinetuner(model, loss_fn, n_particles=pso_particles)
        results["pso"] = pso.finetune(get_batch, pso_steps, val_fn=val_fn)
        l2_pso = val_fn(model)
        results["pso"]["final_l2"] = l2_pso
        print(f"\n  PSO final L2: {l2_pso:.4e}")

    # En iyi modeli tekrar yükle (PSO veya L-BFGS'ten hangisi daha iyiyse)
    if run_lbfgs and run_pso:
        if results["lbfgs"]["final_l2"] <= results["pso"]["final_l2"]:
            model.load_state_dict(lbfgs_state)
            best_phase = "lbfgs"
        else:
            # PSO ağırlıkları zaten model'de
            best_phase = "pso"
    elif run_lbfgs:
        model.load_state_dict(lbfgs_state)
        best_phase = "lbfgs"
    else:
        best_phase = "adam"

    # ── Özet Tablosu ─────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*52}")
    print(f"  {'Phase':<12} | {'L2 Error':>12} | {'Time (s)':>10} | {'Δ vs Adam':>12}")
    print(f"  {'─'*50}")
    print(f"  {'Adam':<12} | {l2_adam:>12.4e} | "
          f"{results['adam']['train_time']:>10.1f} | {'baseline':>12}")
    if run_lbfgs:
        delta = l2_adam - results['lbfgs']['final_l2']
        print(f"  {'Adam+LBFGS':<12} | {results['lbfgs']['final_l2']:>12.4e} | "
              f"{results['lbfgs']['train_time']:>10.1f} | {delta:>+12.4e}")
    if run_pso:
        delta = l2_adam - results['pso']['final_l2']
        print(f"  {'Adam+PSO':<12} | {results['pso']['final_l2']:>12.4e} | "
              f"{results['pso']['train_time']:>10.1f} | {delta:>+12.4e}")
    print(f"{'='*52}")
    print(f"  Best phase: {best_phase.upper()}")
    print(f"{'='*52}")

    results["best_phase"] = best_phase
    return results
