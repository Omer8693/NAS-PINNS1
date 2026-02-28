from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SearchLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, mask_levels: Sequence[int]) -> None:
        super().__init__()
        self.out_features = out_features
        self.mask_levels = list(mask_levels)

        self.op_skip = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        self.op_tanh = nn.Sequential(nn.Linear(in_features, out_features), nn.Tanh())
        self.op_sin = nn.Sequential(nn.Linear(in_features, out_features), SinActivation())

        # Operation relaxation (NAS-PINN style DARTS relaxation)
        self.alpha_ops = nn.Parameter(torch.randn(3) * 0.1)
        # Width/mask relaxation
        self.alpha_masks = nn.Parameter(torch.randn(len(self.mask_levels)) * 0.1)

    def _masked(self, x: torch.Tensor, level: int) -> torch.Tensor:
        k = min(int(level), x.shape[-1])
        mask = torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype)
        mask[:k] = 1.0
        return x * mask.unsqueeze(0)

    def mixed_ops(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.alpha_ops, dim=0)
        out_skip = self.op_skip(x)
        out_tanh = self.op_tanh(x)
        out_sin = self.op_sin(x)
        return weights[0] * out_skip + weights[1] * out_tanh + weights[2] * out_sin

    def forward(self, x: torch.Tensor, fixed_mask_idx: Optional[int] = None) -> torch.Tensor:
        mixed = self.mixed_ops(x)
        if fixed_mask_idx is None:
            w_mask = F.softmax(self.alpha_masks, dim=0)
            out = 0.0
            for idx, lvl in enumerate(self.mask_levels):
                out = out + w_mask[idx] * self._masked(mixed, lvl)
            return out
        idx = int(np.clip(fixed_mask_idx, 0, len(self.mask_levels) - 1))
        return self._masked(mixed, self.mask_levels[idx])


class SearchPINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: int,
        base_neurons: int,
        mask_levels: Sequence[int],
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_layers = int(hidden_layers)
        self.base_neurons = int(base_neurons)
        self.mask_levels = list(mask_levels)

        dims = [self.input_dim] + [self.base_neurons] * self.hidden_layers
        self.layers = nn.ModuleList(
            [SearchLayer(dims[i], dims[i + 1], self.mask_levels) for i in range(self.hidden_layers)]
        )
        self.output = nn.Linear(self.base_neurons, 1)

    def forward(self, x: torch.Tensor, mask_indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        h = x
        if mask_indices is None:
            for layer in self.layers:
                h = layer(h, fixed_mask_idx=None)
        else:
            if len(mask_indices) != len(self.layers):
                raise ValueError(f"mask_indices length {len(mask_indices)} != layer count {len(self.layers)}")
            for layer, m in zip(self.layers, mask_indices):
                h = layer(h, fixed_mask_idx=int(m))
        return self.output(h)

    def arch_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for layer in self.layers:
            params.extend([layer.alpha_ops, layer.alpha_masks])
        return params

    def mask_parameters(self) -> List[nn.Parameter]:
        return [layer.alpha_masks for layer in self.layers]

    def op_parameters(self) -> List[nn.Parameter]:
        return [layer.alpha_ops for layer in self.layers]

    def non_mask_parameters(self) -> List[nn.Parameter]:
        mask_ids = {id(p) for p in self.mask_parameters()}
        return [p for p in self.parameters() if id(p) not in mask_ids]

    def infer_best_masks(self) -> List[int]:
        masks = []
        for layer in self.layers:
            idx = int(torch.argmax(layer.alpha_masks).item())
            masks.append(idx)
        return masks


def clone_model_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model_state(model: nn.Module, state: dict) -> None:
    model.load_state_dict(state)


def flatten_model_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])


def set_model_from_flat_vector(model: nn.Module, flat: np.ndarray, device: torch.device) -> None:
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            n_param = param.numel()
            chunk = torch.from_numpy(flat[offset : offset + n_param]).view_as(param).to(device)
            param.copy_(chunk)
            offset += n_param


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def iter_model_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for p in model.parameters():
        yield p
