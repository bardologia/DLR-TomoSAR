from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from configuration.architectures import GammaNetConfig
from tools.loss.physical_loss    import PhysicalLoss
from ..blocks                    import build_activation


class TomoOperator:
    @staticmethod
    def forward(profiles: torch.Tensor, kz_map: torch.Tensor, x_axis: torch.Tensor, dx: float) -> torch.Tensor:
        tracks = [PhysicalLoss.synthesise_track(profiles, kz_map[:, track], x_axis, dx) for track in range(kz_map.shape[1])]

        return torch.stack(tracks, dim=1)

    @staticmethod
    def adjoint(measurements: torch.Tensor, kz_map: torch.Tensor, x_axis: torch.Tensor) -> torch.Tensor:
        n_tracks = kz_map.shape[1]

        accumulated = None
        for track in range(n_tracks):
            phase    = kz_map[:, track].unsqueeze(1) * x_axis.reshape(1, -1, 1, 1)
            steering = torch.polar(torch.ones_like(phase), phase)
            term     = (steering.conj() * measurements[:, track].unsqueeze(1)).real

            accumulated = term if accumulated is None else accumulated + term

        return accumulated


class ProfileProx(nn.Module):
    def __init__(self, hidden: int, kernel_size: int, activation: str):
        super().__init__()
        padding = kernel_size // 2

        self.refine = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=kernel_size, padding=padding),
            build_activation(activation),
            nn.Conv1d(hidden, 1, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, H, W = x.shape

        flat    = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, N)
        refined = flat + self.refine(flat)

        return refined.reshape(B, H, W, N).permute(0, 3, 1, 2)


class GammaNet(nn.Module):
    def __init__(self, config: GammaNetConfig | None = None):
        super().__init__()
        self.config = config if config is not None else GammaNetConfig()

        if self.config.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {self.config.n_iterations}")

        L = self.config.n_iterations

        self.raw_steps      = nn.Parameter(torch.full((L,), self._softplus_inverse(self.config.step_init)))
        self.raw_thresholds = nn.Parameter(torch.full((L,), self._softplus_inverse(self.config.threshold_init)))
        self.prox_blocks    = nn.ModuleList([ProfileProx(self.config.prox_hidden, self.config.prox_kernel_size, self.config.activation) for _ in range(L)])

    @staticmethod
    def _softplus_inverse(value: float) -> float:
        return math.log(math.expm1(value))

    def forward(self, measurements: torch.Tensor, kz_map: torch.Tensor, x_axis: torch.Tensor) -> torch.Tensor:
        if not measurements.is_complex():
            raise ValueError("GammaNet expects complex-valued coherence measurements of shape (B, T, H, W)")
        if measurements.shape != kz_map.shape:
            raise ValueError(f"measurements {tuple(measurements.shape)} and kz_map {tuple(kz_map.shape)} must share the (B, T, H, W) shape")

        dx        = float(x_axis[1] - x_axis[0])
        n_tracks  = kz_map.shape[1]
        lipschitz = n_tracks * x_axis.shape[0] * dx * dx

        steps      = functional.softplus(self.raw_steps)
        thresholds = functional.softplus(self.raw_thresholds)

        profile = functional.relu(TomoOperator.adjoint(measurements, kz_map, x_axis) / n_tracks)

        for iteration in range(self.config.n_iterations):
            residual = measurements - TomoOperator.forward(profile, kz_map, x_axis, dx)
            gradient = TomoOperator.adjoint(residual, kz_map, x_axis) / lipschitz

            profile = profile + steps[iteration] * gradient
            profile = self.prox_blocks[iteration](profile)
            profile = functional.relu(profile - thresholds[iteration])

        return profile
