from __future__ import annotations

import copy
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ModelSummary:
    def __init__(self, logger, model: nn.Module):
        self.model        = model
        self.rows         = []
        self.total_params = 0
        self.logger       = logger

    def count_params(self, module: nn.Module):
        return sum(p.numel() for p in module.parameters())

    def count_own_params(self, module: nn.Module):
        return sum(p.numel() for p in module.parameters(recurse=False))

    def to_markdown(self, title="Model Summary") -> str:
        if not self.rows:
            return f"# {title}\n\nNo layers found."

        rows_fmt = [(name, typ, f"{params:,}") for name, typ, params in self.rows]

        col1 = max(len("Layer"), *(len(name) for name, _, _ in rows_fmt))
        col2 = max(len("Type"), *(len(typ) for _, typ, _ in rows_fmt))
        col3 = max(len("Parameters"), *(len(p) for _, _, p in rows_fmt))

        def line(a, b, c):
            return f"| {a:<{col1}} | {b:<{col2}} | {c:>{col3}} |"

        table = []
        table.append(line("Layer", "Type", "Parameters"))
        table.append(f"| {'-'*col1} | {'-'*col2} | {'-'*col3} |")
        for name, typ, params in rows_fmt:
            table.append(line(name, typ, params))

        total = f"{self.total_params:,}"

        md = []
        md.append(f"# {title}\n")
        md.extend(table)
        md.append(f"\n**Total Parameters:** `{total}`")
        return "\n".join(md)

    def run(self):
        self.logger.section("[Model Summary]")
        self.logger.info("Generating model architecture summary")

        self.total_params = sum(p.numel() for p in self.model.parameters())

        for name, module in self.model.named_modules():
            if name == "":
                continue
            own_params = self.count_own_params(module)
            self.rows.append((name, module.__class__.__name__, own_params))

    def save_markdown(self, path: str, title: str = "Model Summary"):
        md = self.to_markdown(title=title)
        Path(path).write_text(md, encoding="utf-8")
        self.logger.info(f"Model summary saved to {path}")


class ShapeLogger:
    def __init__(self, model, logger, include_types, docs_dir: str):
        self.model         = model
        self.include_types = include_types
        self.logger        = logger
        self.docs_dir      = Path(docs_dir)
        self.records       = []
        self.hooks         = []

    def _hook(self, name):
        def fn(module, inputs, output):
            x = inputs[0]
            in_shape  = tuple(x.shape) if hasattr(x, "shape") else str(type(x))
            if isinstance(output, tuple):
                if hasattr(output[0], "shape"):
                    out_shape = tuple(output[0].shape)
                else:
                    out_shape = f"tuple[{len(output)}]"
            else:
                out_shape = tuple(output.shape) if hasattr(output, "shape") else str(type(output))

            self.records.append((name, module.__class__.__name__, in_shape, out_shape))

        return fn

    def attach(self):
        self.logger.subsection("Hooks attached to layers for shape logging. \n")

        for name, module in self.model.named_modules():
            if name == "":
                continue

            if isinstance(module, self.include_types):
                self.hooks.append(module.register_forward_hook(self._hook(name)))

        return self

    def detach(self):
        if self.hooks == []:
            return

        self.logger.subsection("Hooks detached from layers. \n")

        for h in self.hooks:
            h.remove()

        self.hooks.clear()

    def clear(self):
        self.records.clear()

    def to_markdown(self, title: str = "Shape Log", sort_by_layer: bool = False) -> str:
        rows = list(self.records)
        if sort_by_layer:
            rows.sort(key=lambda r: r[0])

        def s(x):
            return str(x)

        def layer_cell(name: str) -> str:
            return f"`{name}`"

        col_names = ["Layer", "Type", "Input shape", "Output shape"]
        col_data = [
            [layer_cell(r[0]) for r in rows],
            [str(r[1]) for r in rows],
            [s(r[2]) for r in rows],
            [s(r[3]) for r in rows],
        ]

        widths = []
        for header, data in zip(col_names, col_data):
            widths.append(max([len(header)] + [len(v) for v in data]) if rows else len(header))

        def fmt_row(cells):
            return "| " + " | ".join(f"{c:<{w}}" for c, w in zip(cells, widths)) + " |"

        def fmt_sep():
            return "| " + " | ".join((":" + "-" * (w - 1)) if w > 1 else "-" for w in widths) + " |"

        lines = []
        lines.append(f"# {title}\n")
        lines.append(fmt_row(col_names))
        lines.append(fmt_sep())

        for (name, typ, ins, outs), layer_txt in zip(rows, col_data[0]):
            lines.append(fmt_row([layer_txt, str(typ), s(ins), s(outs)]))

        lines.append(f"\n**Records:** {len(rows)}")
        return "\n".join(lines)

    def save_markdown(self, filename="tensor_shape.md", title: str = "Shape Log", sort_by_layer: bool = False):
        md = self.to_markdown(title=title, sort_by_layer=sort_by_layer)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        path = self.docs_dir / filename
        path.write_text(md, encoding="utf-8")
        self.logger.subsection(f"Shape log saved to {path} \n")


class TrainingDocs:
    def __init__(self, model, model_cfg, logger, run_dir, enabled=True):
        self.model     = model
        self.model_cfg = model_cfg
        self.logger    = logger
        self.run_dir   = Path(run_dir)
        self.enabled   = enabled

    def emit_model_summary(self) -> None:
        if not self.enabled:
            return

        summary      = ModelSummary(self.logger, self.model)
        summary.run()
        summary_path = self.run_dir / "docs" / "model_summary.md"
        summary.save_markdown(str(summary_path))

    @torch.no_grad()
    def emit_shape_log(self, data_loader: DataLoader, device: torch.device) -> None:
        if not self.enabled:
            return

        include_types = getattr(self.model_cfg, "shape_logger_types", None)
        if include_types is None:
            return

        self.logger.section("[Shape Logger]")
        shape_logger = ShapeLogger(
            model         = self.model,
            logger        = self.logger,
            include_types = include_types,
            docs_dir      = self.run_dir / "docs",
        )
        shape_logger.attach()

        try:
            batch  = next(iter(data_loader))
            images = batch[0].to(device)
            self.model.eval()
            self.model(images)
        finally:
            shape_logger.detach()
            self.model.train()

        shape_logger.save_markdown(filename="shape_log.md", title="Tensor Shape Log")


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
        "use_total_power",
        "use_moments",
        "use_coherence_resyn",
        "use_covariance_match",
        "use_capon_cycle",
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
        "use_total_power":        "weight_total_power",
        "use_moments":            "weight_moments",
        "use_coherence_resyn":    "weight_coherence_resyn",
        "use_covariance_match":   "weight_covariance_match",
        "use_capon_cycle":        "weight_capon_cycle",
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
