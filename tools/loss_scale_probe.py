from __future__ import annotations

import copy
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class LossScaleProbeConfig:
    enabled        : bool          = True
    n_batches      : int           = 10
    reference      : Optional[str] = None
    exit_after     : bool          = True
    enabled_losses : dict          = field(default_factory=dict)


class LossScaleProbe:

    _ALL_USE_FLAGS = [
        "use_mse_curve",
        "use_l1_curve",
        "use_huber_curve",
        "use_charbonnier_curve",
        "use_cosine_curve",
        "use_spectral_coherence",
        "use_ssim_curve",
        "use_param_l1",
        "use_param_huber",
        "use_smoothness_tv",
    ]

    _FLAG_TO_WEIGHT = {
        "use_mse_curve":          "weight_mse_curve",
        "use_l1_curve":           "weight_l1_curve",
        "use_huber_curve":        "weight_huber_curve",
        "use_charbonnier_curve":  "weight_charbonnier_curve",
        "use_cosine_curve":       "weight_cosine_curve",
        "use_spectral_coherence": "weight_spectral_coh",
        "use_ssim_curve":         "weight_ssim_curve",
        "use_param_l1":           "weight_param_l1",
        "use_param_huber":        "weight_param_huber",
        "use_smoothness_tv":      "weight_smoothness_tv",
    }

    def __init__(self, probe_cfg, loss_cfg, gaussian_cfg, norm_stats=None, logger=None):
        self.probe_cfg    = probe_cfg
        self.gaussian_cfg = gaussian_cfg
        self.norm_stats   = norm_stats
        self.logger       = logger
        self.loss_cfg     = copy.deepcopy(loss_cfg)

        for flag in self._ALL_USE_FLAGS:
            override = probe_cfg.enabled_losses.get(flag)
            setattr(self.loss_cfg, flag, override if override is not None else True)

        for flag, w_key in self._FLAG_TO_WEIGHT.items():
            if getattr(self.loss_cfg, flag, False):
                setattr(self.loss_cfg, w_key, 1.0)

    @staticmethod
    def _iqr_filter(values: list[float], k: float = 3.0) -> list[float]:
        if len(values) < 4:
            return values
        sorted_v = sorted(values)
        n        = len(sorted_v)
        q1       = sorted_v[n // 4]
        q3       = sorted_v[(3 * n) // 4]
        iqr      = q3 - q1
        lo, hi   = q1 - k * iqr, q3 + k * iqr
        filtered = [v for v in values if lo <= v <= hi]
        return filtered if len(filtered) >= 3 else values

    def run(self, train_loader, model, device, x_axis: torch.Tensor) -> dict[str, float]:
        if not self.probe_cfg.enabled:
            return {}

        from pipelines.training_pipeline.loss import Loss

        _null     = _NullLoggerTracker()
        criterion = Loss(
            x_axis       = x_axis,
            logger       = _null,
            tracker      = _null,
            gaussian_cfg = self.gaussian_cfg,
            loss_cfg     = self.loss_cfg,
            norm_stats   = self.norm_stats,
        )

        self.logger.section("[Loss Scale Probe]")
        self.logger.subsection(f"Averaging over {self.probe_cfg.n_batches} batches — all weights forced to 1.0")

        model.eval()
        accum: dict[str, list[float]] = defaultdict(list)

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= self.probe_cfg.n_batches:
                    break
                inputs, targets = batch[0].to(device), batch[1].to(device)
                result = criterion(model(inputs), targets)
                for name, val in result["components"].items():
                    if "/" not in name:
                        accum[name].append(float(val))

        if not accum:
            self.logger.warning("No loss components recorded — check model outputs.")
            return {}

        filtered = {k: self._iqr_filter(v) for k, v in accum.items()}
        means    = {k: sum(v) / len(v) for k, v in filtered.items()}
        n_total  = self.probe_cfg.n_batches

        ref     = self.probe_cfg.reference
        ref_val = means.get(ref) if ref is not None else None

        suggested = {
            name: ((ref_val / raw) if ref_val is not None else (1.0 / raw)) if raw > 0 else float("nan")
            for name, raw in means.items()
        }

        rows = [
            {
                "Term"             : name,
                "Raw value"        : f"{means[name]:.6f}",
                "Kept"             : f"{len(filtered[name])}/{len(accum[name])}",
                "Suggested weight" : f"{suggested[name]:.6f}" + ("  ← reference" if name == ref else ""),
            }
            for name in sorted(means)
        ]
        self.logger.metrics_table(rows, columns=["Term", "Raw value", "Kept", "Suggested weight"], title="Probe Results")

        formula = f"raw({ref}) / raw(i)" if ref_val is not None else "1 / raw(i)"
        self.logger.subsection(f"Formula : suggested_weight(i) = {formula}")
        self.logger.subsection("Update LossNormalizationConfig with these values, then tune weight_* (alpha) freely.")

        if ref is not None and ref not in means:
            self.logger.warning(f"Reference term '{ref}' not found — each term normalised to 1.0 independently.")

        model.train()

        if self.probe_cfg.exit_after:
            sys.exit(0)

        return suggested


class _NullLoggerTracker:
    def __getattr__(self, name):
        return lambda *a, **kw: None

    def section(self, *a, **kw):    pass
    def subsection(self, *a, **kw): pass
    def scalar(self, *a, **kw):     pass
    def add_scalar(self, *a, **kw): pass
