from __future__ import annotations

import copy
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from rich.tree import Tree
from torch.utils.data import DataLoader

from tools import NullLogger, NullTracker
from tools.reporting.markdown import MarkdownDoc, MarkdownTable
from pipelines.backbone.training.loss import LOSS_TERMS


class LayerRecord:
    def __init__(self, name: str, module: nn.Module) -> None:
        self.name      = name
        self.module    = module
        self.type_name = module.__class__.__name__
        self.depth     = name.count(".") + 1
        self.trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        self.frozen    = sum(p.numel() for p in module.parameters(recurse=False) if not p.requires_grad)
        self.in_shape  = None
        self.out_shape = None
        self.visited   = False

    @property
    def own_params(self) -> int:
        return self.trainable + self.frozen


class ModelInspector:
    def __init__(self, model: nn.Module, logger, docs_dir, include_types=None) -> None:
        self.model         = model
        self.logger        = logger
        self.docs_dir      = Path(docs_dir)
        self.include_types = include_types
        self.records       = []
        self.hooks         = []

    def _shape_of(self, value):
        if hasattr(value, "shape"):
            return tuple(value.shape)
        if isinstance(value, (tuple, list)):
            head = value[0] if value else None
            return tuple(head.shape) if hasattr(head, "shape") else f"{type(value).__name__}[{len(value)}]"
        return type(value).__name__

    def _hook(self, record: LayerRecord):
        def fn(module, inputs, output):
            record.in_shape  = self._shape_of(inputs[0]) if inputs else None
            record.out_shape = self._shape_of(output)
            record.visited   = True
        return fn

    def _build(self) -> None:
        self.records = [LayerRecord(name, module) for name, module in self.model.named_modules() if name != ""]

    def _attach(self) -> None:
        self.hooks = [record.module.register_forward_hook(self._hook(record)) for record in self.records]

    def _detach(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def run(self, sample: torch.Tensor) -> "ModelInspector":
        self._build()
        self._attach()

        was_training = self.model.training
        self.model.eval()

        try:
            self.model(sample)
        finally:
            self._detach()
            if was_training:
                self.model.train()

        return self

    def totals(self) -> dict:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        size_mb   = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 ** 2

        return {
            "total"     : trainable + frozen,
            "trainable" : trainable,
            "frozen"    : frozen,
            "size_mb"   : size_mb,
        }

    def _selected(self) -> list[LayerRecord]:
        if self.include_types is None:
            return list(self.records)
        return [r for r in self.records if isinstance(r.module, self.include_types) or r.own_params > 0]

    def _shape_cell(self, shape) -> Optional[str]:
        return str(shape) if shape is not None else None

    def _node_label(self, record: LayerRecord) -> str:
        leaf  = record.name.rpartition(".")[2]
        label = f"[key]{leaf}[/key] [muted]{record.type_name}[/muted]"

        if record.visited:
            label += f"  {record.in_shape} [muted]->[/muted] {record.out_shape}"
        if record.own_params > 0:
            label += f"  [value]{record.own_params:,}[/value]"

        return label

    def render_console(self) -> None:
        totals = self.totals()
        tree   = Tree(f"[section]{self.model.__class__.__name__}[/section]")
        nodes  = {"": tree}

        for record in self.records:
            parent             = nodes.get(record.name.rpartition(".")[0], tree)
            nodes[record.name] = parent.add(self._node_label(record))

        self.logger.render(tree)
        self.logger.kv_table({
            "Total parameters"     : f"{totals['total']:,}",
            "Trainable parameters" : f"{totals['trainable']:,}",
            "Frozen parameters"    : f"{totals['frozen']:,}",
            "Parameter size"       : f"{totals['size_mb']:.2f} MB",
        }, title="Parameter Totals")

    def to_markdown(self, title: str = "Model Documentation") -> MarkdownDoc:
        totals = self.totals()
        rows   = self._selected()

        doc = MarkdownDoc(title)
        doc.bold_kv("Model",                self.model.__class__.__name__)
        doc.bold_kv("Total Parameters",     f"{totals['total']:,}")
        doc.bold_kv("Trainable Parameters", f"{totals['trainable']:,}")
        doc.bold_kv("Frozen Parameters",    f"{totals['frozen']:,}")
        doc.bold_kv("Parameter Size",       f"{totals['size_mb']:.2f} MB")
        doc.blank()

        columns = ["Layer", "Type", "Input shape", "Output shape", "Params", "Trainable", "Share %"]
        align   = ["left", "left", "left", "left", "right", "right", "right"]
        table   = MarkdownTable(columns, align=align)

        for r in rows:
            share = 100.0 * r.own_params / totals["total"] if totals["total"] else 0.0
            table.add_row(
                f"`{r.name}`",
                r.type_name,
                self._shape_cell(r.in_shape),
                self._shape_cell(r.out_shape),
                f"{r.own_params:,}",
                f"{r.trainable:,}",
                f"{share:.2f}",
            )

        doc.table(table)
        doc.bold_kv("Layers Documented", len(rows))
        return doc

    def save_markdown(self, filename: str = "model_doc.md", title: str = "Model Documentation") -> Path:
        path = self.to_markdown(title=title).save(self.docs_dir / filename)
        self.logger.subsection(f"Model documentation saved to {path}")
        return path


class TrainingDocs:
    def __init__(self, model, model_cfg, logger, run_dir, enabled=True):
        self.model     = model
        self.model_cfg = model_cfg
        self.logger    = logger
        self.run_dir   = Path(run_dir)
        self.enabled   = enabled

    @torch.no_grad()
    def emit(self, data_loader: DataLoader, device: torch.device) -> None:
        if not self.enabled:
            return

        self.logger.section("[Model Documentation]")

        inspector = ModelInspector(
            model         = self.model,
            logger        = self.logger,
            docs_dir      = self.run_dir / "docs",
            include_types = self.model_cfg.shape_logger_types,
        )

        batch  = next(iter(data_loader))
        sample = batch[0].to(device)

        inspector.run(sample)
        inspector.render_console()
        inspector.save_markdown()


@dataclass
class LossScaleProbeConfig:
    enabled        : bool          = True
    n_batches      : int           = 10
    reference      : Optional[str] = None
    exit_after     : bool          = True
    enabled_losses : dict          = field(default_factory=dict)


class LossScaleProbe:

    def __init__(self, probe_cfg, loss_cfg, gaussian_cfg, geometry_cfg, norm_stats=None, logger=None):
        self.probe_cfg    = probe_cfg
        self.gaussian_cfg = gaussian_cfg
        self.geometry_cfg = geometry_cfg
        self.norm_stats   = norm_stats
        self.logger       = logger
        self.loss_cfg     = copy.deepcopy(loss_cfg)

        for term in LOSS_TERMS:
            override = probe_cfg.enabled_losses.get(term.use_flag)
            setattr(self.loss_cfg, term.use_flag, override if override is not None else True)

        for term in LOSS_TERMS:
            if getattr(self.loss_cfg, term.use_flag, False):
                setattr(self.loss_cfg, term.weight_key, 1.0)

    def _build_criterion(self, x_axis: torch.Tensor):
        from pipelines.backbone.training.loss import Loss

        return Loss(
            x_axis       = x_axis,
            logger       = NullLogger(),
            tracker      = NullTracker(),
            gaussian_cfg = self.gaussian_cfg,
            loss_cfg     = self.loss_cfg,
            norm_stats   = self.norm_stats,
            geometry_cfg = self.geometry_cfg,
        )

    def _collect(self, criterion, train_loader, model, device) -> dict[str, list[float]]:
        model.eval()
        accum: dict[str, list[float]] = defaultdict(list)

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= self.probe_cfg.n_batches:
                    break

                inputs, targets = batch[0].to(device), batch[1].to(device)
                result          = criterion(model(inputs), targets)

                for name, val in result["components"].items():
                    if "/" not in name:
                        accum[name].append(float(val))

        model.train()

        return accum

    def _suggest(self, accum: dict[str, list[float]]) -> dict:
        filtered = {k: self._iqr_filter(v) for k, v in accum.items()}
        means    = {k: sum(v) / len(v) for k, v in filtered.items()}

        ref     = self.probe_cfg.reference
        ref_val = means.get(ref) if ref is not None else None

        suggested = {
            name: ((ref_val / raw) if ref_val is not None else (1.0 / raw)) if raw > 0 else float("nan")
            for name, raw in means.items()
        }

        return {
            "accum"     : accum,
            "filtered"  : filtered,
            "means"     : means,
            "ref"       : ref,
            "ref_val"   : ref_val,
            "suggested" : suggested,
        }

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

    def _emit_report(self, report: dict) -> None:
        accum     = report["accum"]
        filtered  = report["filtered"]
        means     = report["means"]
        ref       = report["ref"]
        ref_val   = report["ref_val"]
        suggested = report["suggested"]

        skipped = [name for name, weight in suggested.items() if not math.isfinite(weight)]

        for name in sorted(skipped):
            self.logger.warning(f"Probe term '{name}' has raw mean <= 0 (raw={means[name]:.6f}); no usable suggested weight — do not copy into LossNormalizationConfig.")

        rows = [
            {
                "Term"             : name,
                "Raw value"        : f"{means[name]:.6f}",
                "Kept"             : f"{len(filtered[name])}/{len(accum[name])}",
                "Suggested weight" : self._weight_cell(name, suggested[name], ref),
            }
            for name in sorted(means)
        ]
        self.logger.metrics_table(rows, columns=["Term", "Raw value", "Kept", "Suggested weight"], title="Probe Results")

        formula = f"raw({ref}) / raw(i)" if ref_val is not None else "1 / raw(i)"
        self.logger.subsection(f"Formula : suggested_weight(i) = {formula}")
        self.logger.subsection("Update LossNormalizationConfig with these values, then tune weight_* (alpha) freely.")

        if ref is not None and ref not in means:
            self.logger.warning(f"Reference term '{ref}' not found — each term normalised to 1.0 independently.")

    @staticmethod
    def _weight_cell(name: str, weight: float, ref) -> str:
        if not math.isfinite(weight):
            return "skipped: raw<=0"

        return f"{weight:.6f}" + ("  ← reference" if name == ref else "")

    def run(self, train_loader, model, device, x_axis: torch.Tensor) -> dict[str, float]:
        if not self.probe_cfg.enabled:
            return {}

        criterion = self._build_criterion(x_axis)

        self.logger.section("[Loss Scale Probe]")
        self.logger.subsection(f"Averaging over {self.probe_cfg.n_batches} batches — all weights forced to 1.0")

        accum = self._collect(criterion, train_loader, model, device)

        if not accum:
            self.logger.warning("No loss components recorded — check model outputs.")
            return {}

        report = self._suggest(accum)
        self._emit_report(report)

        if self.probe_cfg.exit_after:
            sys.exit(0)

        return report["suggested"]
