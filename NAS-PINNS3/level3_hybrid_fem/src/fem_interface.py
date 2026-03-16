"""
fem_interface.py — FEM Checkpoint Arayüzü
===========================================
Gerçek FEM çıktısı olmadığında analitik referansı FEM yerine kullanır.
İleride gerçek FEM (Abaqus / OpenFOAM) çıktısı bağlanabilir.

FEMCheckpoint: t anındaki sıcaklık alanını temsil eder.
  - save / load : numpy .npz formatında diske yaz/oku
  - from_analytical : analitik çözümden checkpoint üret (placeholder)
"""

import numpy as np
import torch
from pathlib import Path


class FEMCheckpoint:
    """
    Tek bir zaman adımının FEM (ya da analitik) sıcaklık alanı.
    Alanlar düz (flattened) tensör olarak saklanır — grid bağımsız.
    """

    def __init__(self,
                 t:       float,
                 T_field: torch.Tensor,
                 x_grid:  torch.Tensor,
                 y_grid:  torch.Tensor):
        # t: checkpoint zamanı [s]
        # T_field: (N,) sıcaklık değerleri [°C]
        # x_grid, y_grid: (N,) koordinatlar [m]
        self.t       = t
        self.T_field = T_field
        self.x_grid  = x_grid
        self.y_grid  = y_grid

    def save(self, path: str) -> None:
        # Checkpoint'i numpy .npz formatında kaydet
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 t       = np.array([self.t]),
                 T_field = self.T_field.cpu().numpy(),
                 x_grid  = self.x_grid.cpu().numpy(),
                 y_grid  = self.y_grid.cpu().numpy())

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")) -> "FEMCheckpoint":
        # Kaydedilmiş checkpoint'i yükle
        data = np.load(path)
        return cls(
            t       = float(data["t"][0]),
            T_field = torch.tensor(data["T_field"], dtype=torch.float32, device=device),
            x_grid  = torch.tensor(data["x_grid"],  dtype=torch.float32, device=device),
            y_grid  = torch.tensor(data["y_grid"],  dtype=torch.float32, device=device),
        )

    @classmethod
    def from_analytical(cls,
                        t:       float,
                        n_grid:  int = 50,
                        device:  torch.device = torch.device("cpu")) -> "FEMCheckpoint":
        """
        Gerçek FEM yokken analitik referansı checkpoint olarak kullan.
        2D Robin BC temel mod soğuma: α = 1.75e-3 /s (config.py ile aynı).
        T(t) = T_water + (T_init - T_water) · exp(-α·t)

        Not: Bu 0D soğuma modeli — uzaydan bağımsız.
        Gerçek 2D gradyan yoktur. Mekanik hesap için yeterli başlangıç.
        """
        alpha  = 1.75e-3   # [1/s] 2D Robin BC temel mod katsayısı
        T_w    = 20.0      # [°C] su sıcaklığı
        T_init = 540.0     # [°C] SHT çıkış sıcaklığı

        T_val = T_w + (T_init - T_w) * np.exp(-alpha * t)

        # Düzenli grid: X_MAX=1.3m, Y_MAX=0.6m
        x = torch.linspace(0.0, 1.3, n_grid, device=device)
        y = torch.linspace(0.0, 0.6, n_grid, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        T_field = torch.full(xx.shape, T_val, dtype=torch.float32, device=device)
        return cls(t, T_field.flatten(), xx.flatten(), yy.flatten())

    def interpolate_at(self,
                       x_query: torch.Tensor,
                       y_query: torch.Tensor) -> torch.Tensor:
        """
        Sorgu noktalarında (x_query, y_query) en yakın komşu interpolasyonu.
        Daha hassas yöntem için scipy.interpolate.griddata kullanılabilir.
        """
        # Basit en-yakın-komşu — yeterli hassasiyette
        dist = (self.x_grid.unsqueeze(0) - x_query.unsqueeze(1)) ** 2 \
             + (self.y_grid.unsqueeze(0) - y_query.unsqueeze(1)) ** 2
        idx  = torch.argmin(dist, dim=1)
        return self.T_field[idx]
