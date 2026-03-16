"""
PINN Network Architecture
==========================
Kaynak: Wang & Zhong (2023) NAS-PINN paper'ı esas alınarak uyarlandı.

Orijinal NAS-PINN'den farklar:
  - DARTS mimari arama → NSGA-II / NSGA-III / Bayesian Optimization
  - Aktivasyon seçimi genişletildi: tanh, sin, swish, relu, gelu
  - Artık bağlantı (residual connection) opsiyonel olarak eklendi
  - Ağırlık başlatma: Xavier normal (PINN için önerilen)

Not: CollocationSampler quenching problemine özgü olup
     problems/quenching.py dosyasına taşınmıştır.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from src.config import DEVICE


# ─────────────────────────────────────────────────────────────
# Bölüm 1 — Aktivasyon Fonksiyonları
# ─────────────────────────────────────────────────────────────

class SinActivation(nn.Module):
    """Sin aktivasyon — yüksek frekanslı fiziksel fenomenler için."""
    def forward(self, x):
        return torch.sin(x)


def _make_activation(name: str) -> nn.Module:
    """Her çağrıda yeni bir aktivasyon nesnesi döndür (paylaşımlı singleton riski yok)."""
    registry = {
        "tanh":  nn.Tanh,
        "sin":   SinActivation,
        "swish": nn.SiLU,
        "relu":  nn.ReLU,
        "gelu":  nn.GELU,
    }
    cls = registry.get(name, nn.Tanh)
    return cls()


# ─────────────────────────────────────────────────────────────
# Bölüm 2 — Temel PINN Ağı
# ─────────────────────────────────────────────────────────────

class PINNNet(nn.Module):
    """
    Tam bağlantılı PINN ağı.
    Mimari: [n_input → hidden_sizes[0] → ... → hidden_sizes[-1] → n_output]

    residual=True: giriş ile son gizli katman arasına skip connection eklenir.

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

        # Gizli katmanlar — her katman için ayrı aktivasyon nesnesi
        layers = []
        in_dim = n_input
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_make_activation(activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_output))

        self.net = nn.Sequential(*layers)

        # Artık bağlantı projeksiyonu: n_input → hidden[-1] boyut uyumu için
        if residual and hidden_sizes:
            last_h = hidden_sizes[-1]
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
            layers_list = list(self.net)
            out = x
            for layer in layers_list[:-1]:
                out = layer(out)
            skip_out = self.skip(x)
            if out.shape == skip_out.shape:
                out = out + skip_out
            else:
                # Boyut uyumsuzluğu: skip eklenmeden devam et (sessiz hata değil, log)
                print(f"  [WARN] Skip connection size mismatch: "
                      f"out={out.shape}, skip={skip_out.shape}")
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
      n_output   : çıkış boyutu (örn. 1 → T)
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
                pred_f:  torch.Tensor,
                pred_b:  torch.Tensor,
                true_b:  torch.Tensor,
                pred_i:  torch.Tensor,
                true_i:  torch.Tensor,
                pred_d:  Optional[torch.Tensor] = None,
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

        if pred_d is not None and true_d is not None:
            L_d = torch.mean((pred_d - true_d) ** 2)
            loss = loss + self.w_d * L_d
            details["L_data"] = float(L_d.item())

        details["L_total"] = float(loss.item())
        return loss, details
