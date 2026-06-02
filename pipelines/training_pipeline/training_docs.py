from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tools.model_summary import ModelSummary
from tools.shape_logger  import ShapeLogger


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
