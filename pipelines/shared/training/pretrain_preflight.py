from __future__ import annotations

from pathlib import Path
from typing  import Callable, Optional

import torch

from tools.monitoring.logger             import Logger
from tools.training.pretraining          import PretrainContext, PretrainOrchestrator


class PretrainPreflight:
    def __init__(self, pretrain_config, training_config, build_context: Callable[[Logger, torch.device], PretrainContext], logdir: Path, label: Optional[str] = None) -> None:
        self.pretrain = pretrain_config
        self.training = training_config
        self.build    = build_context
        self.logdir   = Path(logdir)
        self.label    = label

    def _enabled(self) -> bool:
        return bool(self.pretrain.find_batch_size or self.pretrain.tune_loader or self.pretrain.run_overfit)

    def run(self) -> None:
        if not self._enabled():
            return

        result_dir = self.logdir / "pretrain"
        result_dir.mkdir(parents=True, exist_ok=True)

        logger = Logger(log_dir=str(result_dir / "logs"), name="pretrain", level="INFO")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        orchestrator = PretrainOrchestrator(
            pretrain_config = self.pretrain,
            training_config = self.training,
            build_context   = lambda: self.build(logger, device),
            logger          = logger,
            label           = self.label,
            result_dir      = result_dir,
        )

        try:
            orchestrator.run()
        finally:
            logger.close()
