"""
ts_model.py — Time-Stepping PINN Modeli
========================================
Level 1'den farkı: T(x,y,t_global) yerine T_next(x,y) tahmin eder.

Giriş boyutu: 4  →  [x, y, t_local_norm, T_prev_norm]
  - x, y          : uzamsal koordinat (normalize)
  - t_local_norm  : pencere içi zaman [0, 1]
  - T_prev_norm   : önceki adım sıcaklığı [0, 1]
Çıkış: T_next (°C)
"""

import torch
import torch.nn as nn
from typing import List, Dict


# Desteklenen aktivasyon fonksiyonları
_ACTIVATIONS: Dict[str, type] = {
    "tanh":  nn.Tanh,
    "relu":  nn.ReLU,
    "swish": nn.SiLU,
    "gelu":  nn.GELU,
}


class SinActivation(nn.Module):
    # Sinüs aktivasyon — quenching için tanh'dan ~2-4x daha iyi
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def build_mlp(n_input: int,
              n_output: int,
              neurons: List[int],
              activation: str = "tanh") -> nn.Sequential:
    # Tam bağlantılı MLP oluştur — aktivasyon tipi dışarıdan verilir
    act_cls = SinActivation if activation == "sin" else _ACTIVATIONS.get(activation, nn.Tanh)
    layers = []
    in_dim = n_input
    for out_dim in neurons:
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(act_cls())
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, n_output))
    return nn.Sequential(*layers)


class TimeStepperPINN(nn.Module):
    """
    Zaman-adım operatörü PINN.
    Her pencere [t_n, t_{n+k}] için ayrı çağrılır.
    """

    # Fiziksel normalizasyon sabitleri — baseline_paper [2026] proses parametreleri
    T_WATER = 20.0    # [°C] su sıcaklığı (alt sınır)
    T_INIT  = 540.0   # [°C] SHT çıkış (üst sınır)
    T_SPAN  = T_INIT - T_WATER   # 520°C

    def __init__(self, config: dict):
        super().__init__()
        # n_input sabit 4: x, y, t_local_norm, T_prev_norm
        self.net = build_mlp(
            n_input    = 4,
            n_output   = 1,
            neurons    = config["neurons"],
            activation = config.get("activation", "tanh"),
        )

    def _normalize_T(self, T: torch.Tensor) -> torch.Tensor:
        # Sıcaklığı [0, 1] aralığına normalize et
        return (T - self.T_WATER) / self.T_SPAN

    def _denormalize_T(self, T_norm: torch.Tensor) -> torch.Tensor:
        # [0, 1] aralığından gerçek sıcaklığa [°C] dön
        return T_norm * self.T_SPAN + self.T_WATER

    def forward(self,
                x:       torch.Tensor,
                y:       torch.Tensor,
                t_local: torch.Tensor,
                T_prev:  torch.Tensor,
                dt:      float) -> torch.Tensor:
        """
        İleri geçiş.
        Tüm giriş tensörleri shape (N, 1).
        T_prev: önceki adımın gerçek sıcaklığı [°C].
        dt: pencere genişliği [s] — normalizasyon için.
        Döndürür: T_next [°C], shape (N, 1).
        """
        # Tüm girişleri [0, 1] aralığına normalize et
        t_norm   = t_local / dt
        T_p_norm = self._normalize_T(T_prev)
        inp      = torch.cat([x, y, t_norm, T_p_norm], dim=1)
        out_norm = self.net(inp)
        return self._denormalize_T(out_norm)
