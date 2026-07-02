from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


class CurveLoss:
    @staticmethod
    @lru_cache(maxsize=8)
    def gaussian_kernel(size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
        g      = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g      = g / g.sum()
        kernel = g[:, None] * g[None, :]
        return kernel[None, None, :, :].contiguous()

    @staticmethod
    def mse_diff(diff: torch.Tensor) -> torch.Tensor:
        return (diff * diff).mean()

    @staticmethod
    def l1_diff(diff: torch.Tensor) -> torch.Tensor:
        return diff.abs().mean()

    @staticmethod
    def huber_diff(diff: torch.Tensor, delta: float) -> torch.Tensor:
        abs_diff = diff.abs()
        val      = torch.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))
        return val.mean()

    @staticmethod
    def charbonnier_diff(diff: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt((diff * diff + eps * eps).clamp(min=eps * eps)).mean()

    @staticmethod
    def smooth_l1_diff(diff: torch.Tensor, beta: float) -> torch.Tensor:
        abs_diff = diff.abs()
        val      = torch.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
        return val.mean()

    @staticmethod
    def cosine(pred: torch.Tensor, target: torch.Tensor, axis: int) -> torch.Tensor:
        pred_norm   = torch.norm(pred,   dim=axis, keepdim=True)
        target_norm = torch.norm(target, dim=axis, keepdim=True)

        valid = (target_norm > 1e-3).squeeze(axis).float()

        p   = pred   / pred_norm.clamp(min=1e-3)
        t   = target / target_norm.clamp(min=1e-3)
        sim = (p * t).sum(dim=axis).clamp(-1.0, 1.0)
        n   = valid.sum().clamp(min=1.0)

        return ((1.0 - sim) * valid).sum() / n

    @staticmethod
    def spectral_coherence(pred: torch.Tensor, target: torch.Tensor, window: int) -> torch.Tensor:
        B, N, H, W = pred.shape
        p = pred.permute(  0, 2, 3, 1).reshape(B * H * W, 1, N)
        t = target.permute(0, 2, 3, 1).reshape(B * H * W, 1, N)

        pt = F.avg_pool1d(p * t, window, stride=1) * window
        p2 = F.avg_pool1d(p * p, window, stride=1) * window
        t2 = F.avg_pool1d(t * t, window, stride=1) * window

        floor = 1e-6
        valid = ((p2 > floor) & (t2 > floor)).float()
        coh   = (pt.abs() / (p2.clamp(min=floor) * t2.clamp(min=floor)).sqrt()).clamp(0.0, 1.0)
        n     = valid.sum().clamp(min=1.0)

        return ((1.0 - coh) * valid).sum() / n

    @staticmethod
    def ssim(
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int,
        sigma: float,
        data_range: float,
        k1: float,
        k2: float,
        axis: str,
    ) -> torch.Tensor:
        B, N, H, W = pred.shape
        dtype  = pred.dtype
        device = pred.device

        kernel  = CurveLoss.gaussian_kernel(window_size, sigma, dtype, device)
        padding = window_size // 2
        c1      = (k1 * data_range) ** 2
        c2      = (k2 * data_range) ** 2

        def conv(z: torch.Tensor) -> torch.Tensor:
            return F.conv2d(z, kernel, padding=padding)

        def ssim_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            xy_min = torch.minimum(x.amin(dim=(1, 2, 3), keepdim=True), y.amin(dim=(1, 2, 3), keepdim=True)).detach()
            xy_max = torch.maximum(x.amax(dim=(1, 2, 3), keepdim=True), y.amax(dim=(1, 2, 3), keepdim=True)).detach()
            rng    = (xy_max - xy_min).clamp(min=1e-6)
            x      = (x - xy_min) / rng
            y      = (y - xy_min) / rng

            mu_x  = conv(x);  mu_y = conv(y)
            mu_x2 = mu_x * mu_x;  mu_y2 = mu_y * mu_y;  mu_xy = mu_x * mu_y
            sx2   = torch.clamp(conv(x * x) - mu_x2, min=0.0)
            sy2   = torch.clamp(conv(y * y) - mu_y2, min=0.0)
            sxy   = conv(x * y) - mu_xy
            num   = (2.0 * mu_xy + c1) * (2.0 * sxy + c2)
            den   = (mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2)
            return (1.0 - num / den.clamp(min=1e-12)).mean()

        if axis == "elevation":
            xs = pred.permute(1, 0, 2, 3).reshape(-1, 1, H, W)
            ys = target.permute(1, 0, 2, 3).reshape(-1, 1, H, W)
        elif axis == "azimuth":
            xs = pred.permute(2, 0, 1, 3).reshape(-1, 1, N, W)
            ys = target.permute(2, 0, 1, 3).reshape(-1, 1, N, W)
        elif axis == "range":
            xs = pred.permute(3, 0, 1, 2).reshape(-1, 1, N, H)
            ys = target.permute(3, 0, 1, 2).reshape(-1, 1, N, H)
        else:
            raise ValueError(f"ssim_axis must be 'elevation', 'azimuth', or 'range', got '{axis}'")

        return ssim_slice(xs, ys)
