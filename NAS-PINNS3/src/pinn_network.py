"""
PINN Network Architecture
==========================
Kaynak: Wang & Zhong (2023) NAS-PINN paper'ı esas alınarak uyarlandı.

Orijinal NAS-PINN'den farklar:
  - DARTS mimari arama → NSGA-II / NSGA-III / Bayesian Optimization
  - Aktivasyon seçimi genişletildi: tanh, sin, swish, relu, gelu
  - Artık bağlantı (residual connection) opsiyonel olarak eklendi
  - Ağırlık başlatma: Xavier normal (PINN için önerilen)

Eğitim akışı:
  NAS araması (hızlı) → En iyi mimari seçilir → Adam (bulk) →
  L-BFGS (ince ayar) ve PSO (global ince ayar) — ikisi bağımsız, Adam'dan başlar
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

# Cihaz seçimi — physics_model.py ile tutarlı
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Bölüm 1 — Aktivasyon Fonksiyonları
# ─────────────────────────────────────────────────────────────
# PINN literatüründe sin aktivasyonu (Fourier özelliklerinden dolayı) sık kullanılır.
# swish (SiLU) türevlenebilirlik avantajı sunar.

class SinActivation(nn.Module):
    """Sin aktivasyon — yüksek frekanslı fiziksel fenomenler için."""
    def forward(self, x):
        return torch.sin(x)

ACTIVATIONS = {
    "tanh":  nn.Tanh(),
    "sin":   SinActivation(),
    "swish": nn.SiLU(),
    "relu":  nn.ReLU(),
    "gelu":  nn.GELU(),
}


# ─────────────────────────────────────────────────────────────
# Bölüm 2 — Temel PINN Ağı
# ─────────────────────────────────────────────────────────────

class PINNNet(nn.Module):
    """
    Tam bağlantılı PINN ağı.
    Mimari: [n_input → hidden_sizes[0] → ... → hidden_sizes[-1] → n_output]

    residual=True: giriş ile son gizli katman arasına skip connection eklenir.
    Bu, düzensiz (irregular) geometrilerde kayıp yüzeyini düzleştirir.

    NAS araması bu sınıfı farklı hidden_sizes konfigürasyonlarıyla test eder;
    en iyi konfigürasyon tam eğitim için seçilir.
    """

    def __init__(self,
                 n_input:      int,
                 n_output:     int,
                 hidden_sizes: List[int],
                 activation:   str  = "tanh",
                 residual:     bool = False):
        super().__init__()
        self.residual = residual
        self.act_name = activation

        # Aktivasyon fonksiyonu — her katman paylaşımlı (stateless)
        act = ACTIVATIONS.get(activation, nn.Tanh())

        # Gizli katmanlar
        layers = []
        in_dim = n_input
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act if isinstance(act, nn.Module) else nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_output))   # çıkış katmanı

        self.net = nn.Sequential(*layers)

        # Artık bağlantı projeksiyonu: n_input → hidden[-1] boyut uyumu için
        if residual:
            last_h = hidden_sizes[-1] if hidden_sizes else n_input
            self.skip = nn.Linear(n_input, last_h) if n_input != last_h else nn.Identity()
        else:
            self.skip = None

        self._init_weights()

    def _init_weights(self):
        """Xavier normal başlatma — PINN eğitiminde daha stabil yakınsama sağlar."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual and self.skip is not None:
            # Son Linear hariç tüm katmanlardan geç
            out = x
            layers_list = list(self.net)
            for layer in layers_list[:-1]:
                out = layer(out)
            # Skip bağlantısı: boyut uyumu varsa ekle
            skip_out = self.skip(x)
            if out.shape == skip_out.shape:
                out = out + skip_out
            return layers_list[-1](out)
        else:
            return self.net(x)

    def count_params(self) -> int:
        """Eğitilebilir parametre sayısını döndür (NAS değerlendirme kriteri)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_network_from_config(config: dict) -> PINNNet:
    """
    NAS aramasından dönen konfigürasyon sözlüğünden PINNNet oluştur.

    Beklenen config anahtarları:
      n_input    : giriş boyutu (örn. 3 → [t, x, y])
      n_output   : çıkış boyutu (örn. 1 → T, veya 3 → [T, ux, uy])
      n_layers   : gizli katman sayısı
      neurons    : her katman için nöron sayısı listesi
      activation : "tanh" | "sin" | "swish" | "relu" | "gelu"
      residual   : True | False
    """
    return PINNNet(
        n_input      = config.get("n_input",  3),
        n_output     = config.get("n_output", 1),
        hidden_sizes = config["neurons"][:config["n_layers"]],
        activation   = config.get("activation", "tanh"),
        residual     = config.get("residual", False)
    ).to(DEVICE)


# ─────────────────────────────────────────────────────────────
# Bölüm 3 — PINN Kayıp Fonksiyonu
# ─────────────────────────────────────────────────────────────

class PINNLoss(nn.Module):
    """
    Ağırlıklı PINN kaybı:
        L_total = ω_F·L_F + ω_B·L_B + ω_I·L_I [+ ω_D·L_D]

    Terimler:
      L_F : fizik rezidüeli kaybı     — PDE'nin ne kadar ihlal edildiği
      L_B : sınır koşulu kaybı        — quenching sınır koşulları
      L_I : başlangıç koşulu kaybı    — T(t=0) = T_solution = 540°C
      L_D : veri kaybı (opsiyonel)    — FEM veya deneysel ölçümler

    Sınır ve başlangıç ağırlıkları fizik ağırlığından yüksek tutulur —
    bu PINN eğitiminin standart pratiğidir.
    """

    def __init__(self,
                 w_physics:  float = 1.0,
                 w_boundary: float = 10.0,
                 w_initial:  float = 10.0,
                 w_data:     float = 5.0):
        super().__init__()
        self.w_f = w_physics
        self.w_b = w_boundary
        self.w_i = w_initial
        self.w_d = w_data

    def forward(self,
                pred_f:  torch.Tensor,             # fizik rezidüeli [N_f, 1]
                pred_b:  torch.Tensor,             # sınır tahmini [N_b, out]
                true_b:  torch.Tensor,             # sınır gerçeği
                pred_i:  torch.Tensor,             # başlangıç tahmini
                true_i:  torch.Tensor,             # başlangıç gerçeği
                pred_d:  Optional[torch.Tensor] = None,   # veri tahmini (opsiyonel)
                true_d:  Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:

        L_f = torch.mean(pred_f ** 2)
        L_b = torch.mean((pred_b - true_b) ** 2)
        L_i = torch.mean((pred_i - true_i) ** 2)

        loss = self.w_f * L_f + self.w_b * L_b + self.w_i * L_i

        details = {
            "L_physics":  float(L_f.item()),
            "L_boundary": float(L_b.item()),
            "L_initial":  float(L_i.item()),
        }

        # Gerçek FEM/CMM verisi varsa veri kaybı da eklenir
        if pred_d is not None and true_d is not None:
            L_d = torch.mean((pred_d - true_d) ** 2)
            loss = loss + self.w_d * L_d
            details["L_data"] = float(L_d.item())

        details["L_total"] = float(loss.item())
        return loss, details


# ─────────────────────────────────────────────────────────────
# Bölüm 4 — Kollokasiyon Noktası Üreticisi
# ─────────────────────────────────────────────────────────────

class CollocationSampler:
    """
    Mortensen subframe geometrisi için rasgele kollokasiyon noktaları üretir.

    Fiziksel alan:
      x ∈ [0, 1.3] m   — subframe uzunluğu
      y ∈ [0, 0.6] m   — subframe genişliği
      t ∈ [0,  30] s   — quenching süresi

    Üretilen nokta kümeleri:
      domain   — alan içi PDE noktaları
      boundary — quenching sınır koşulu noktaları (4 kenar)
      initial  — t=0 başlangıç koşulu noktaları
    """

    def __init__(self,
                 x_range:    Tuple[float, float] = (0.0, 1.3),
                 y_range:    Tuple[float, float] = (0.0, 0.6),
                 t_range:    Tuple[float, float] = (0.0, 30.0),
                 n_domain:   int = 2000,
                 n_boundary: int = 400,
                 n_initial:  int = 400,
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
        Alan içi kollokasiyon noktaları — Latin hypercube yerine uniform rasgele.
        t_window: belirli bir zaman dilimine kısıtlamak için [t_lo, t_hi]
        Dönüş: [N, 3] tensör, requires_grad=True (autograd için zorunlu)
        """
        t_lo, t_hi = t_window if t_window else self.t_range
        t = torch.rand(self.n_domain, 1, device=DEVICE) * (t_hi - t_lo) + t_lo
        x = torch.rand(self.n_domain, 1, device=DEVICE) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        y = torch.rand(self.n_domain, 1, device=DEVICE) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return torch.cat([t, x, y], dim=1).requires_grad_(True)

    def sample_boundary(self,
                        t_window: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Sınır noktaları — 4 kenardan eşit dağılımla örnekleme.
        Quenching HTC sınır koşulunun uygulandığı yüzeyler.
        """
        t_lo, t_hi = t_window if t_window else self.t_range
        n   = self.n_boundary // 4
        t   = torch.rand(n * 4, 1, device=DEVICE) * (t_hi - t_lo) + t_lo

        # 4 kenarın x, y koordinatları
        x0  = torch.zeros(n, 1, device=DEVICE)
        x1  = torch.full((n, 1), self.x_range[1], device=DEVICE)
        y0  = torch.zeros(n, 1, device=DEVICE)
        y1  = torch.full((n, 1), self.y_range[1], device=DEVICE)

        x_r = torch.rand(n * 2, 1, device=DEVICE) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        y_r = torch.rand(n * 2, 1, device=DEVICE) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

        x_side = torch.cat([x0,   x1,   x_r[:n], x_r[n:]], dim=0)
        y_side = torch.cat([y_r[:n], y_r[n:], y0, y1],     dim=0)

        return torch.cat([t, x_side, y_side], dim=1)

    def sample_initial(self) -> torch.Tensor:
        """
        Başlangıç koşulu noktaları — t=0, tüm alan üzerinde.
        T(t=0, x, y) = T_solution = 540°C başlangıç koşulunu uygular.
        """
        t = torch.zeros(self.n_initial, 1, device=DEVICE)
        x = torch.rand(self.n_initial, 1, device=DEVICE) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        y = torch.rand(self.n_initial, 1, device=DEVICE) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return torch.cat([t, x, y], dim=1)
