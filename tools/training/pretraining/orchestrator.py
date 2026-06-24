from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib     import Path
from typing      import Callable, Optional

import torch

from tools.monitoring.logger              import Logger
from tools.training.pretraining.batch_finder import BatchSizeFinder
from tools.training.pretraining.loader_tuner import LoaderTuner
from tools.training.pretraining.overfit_gate import OverfitGate


@dataclass
class PretrainContext:
    dataset        : object
    model          : object
    to_model_input : Callable
    forward_loss   : Callable
    trial_step     : Callable[[int], float]
    run_overfit    : Callable[[], Optional[float]]
    device         : torch.device
    use_amp        : bool  = False
    context_gb     : float = 0.0
    on_oom         : Optional[Callable[[], None]] = None


class PretrainOrchestrator:
    def __init__(self, pretrain_config, training_config, build_context: Callable[[], PretrainContext], logger: Logger, label: Optional[str] = None, result_dir: Optional[Path] = None) -> None:
        self.pretrain   = pretrain_config
        self.training   = training_config
        self.build      = build_context
        self.logger     = logger
        self.label      = label
        self.result_dir = Path(result_dir) if result_dir is not None else None

    def _enabled(self) -> bool:
        return bool(self.pretrain.find_batch_size or self.pretrain.tune_loader or self.pretrain.run_overfit)

    def _release_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _find_batch_size(self, context: PretrainContext) -> None:
        self.logger.section("[Pretrain] Max batch-size finder")

        finder = BatchSizeFinder(
            trial_step = context.trial_step,
            budget_gb  = self.pretrain.vram_budget_gb,
            ceiling    = self.pretrain.max_batch,
            device     = context.device,
            logger     = self.logger,
            model_name = self.label,
            context_gb = context.context_gb,
            on_oom     = context.on_oom,
        )

        result = finder.run()

        if result["status"] != "PASS":
            raise SystemExit(f"max batch-size finder failed: {result['error']}")

        self.training.batch_size = int(result["batch_size"])
        self.logger.subsection(f"Resolved batch size: {self.training.batch_size} (peak {result['peak_gb']:.2f} GB, scale_lr_with_batch={self.training.scale_lr_with_batch})")

    def _tune_loader(self, context: PretrainContext) -> None:
        self.logger.section("[Pretrain] DataLoader tuner")

        tuner = LoaderTuner(
            dataset          = context.dataset,
            model            = context.model,
            to_model_input   = context.to_model_input,
            forward_loss     = context.forward_loss,
            device           = context.device,
            logger           = self.logger,
            use_amp          = context.use_amp,
            seed             = self.pretrain.seed,
            warmup_batches   = self.pretrain.warmup_batches,
            timed_batches    = self.pretrain.timed_batches,
            worker_counts    = self.pretrain.worker_counts,
            prefetch_factors = self.pretrain.prefetch_factors,
            data_wait_target = self.pretrain.data_wait_target,
        )

        choice = tuner.run(self.training.batch_size)

        if choice is None:
            return

        self.training.num_workers     = int(choice["num_workers"])
        self.training.prefetch_factor = int(choice["prefetch_factor"])
        self.logger.subsection(f"Resolved loader: workers={self.training.num_workers} prefetch={self.training.prefetch_factor} (pin_memory recommendation {choice['pin_memory']})")

    def _run_overfit(self, context: PretrainContext) -> None:
        self.logger.section("[Pretrain] Overfit gate")

        run_overfit = context.run_overfit
        del context
        self._release_cache()

        result_path = self.result_dir / "pretrain_overfit_result.json" if self.result_dir is not None else None

        gate = OverfitGate(
            run_overfit         = run_overfit,
            stop_threshold      = self.pretrain.overfit_stop_threshold,
            logger              = self.logger,
            label               = self.label,
            require_convergence = self.pretrain.overfit_require_convergence,
            abort_on_fail       = self.pretrain.overfit_abort_on_fail,
            result_path         = result_path,
        )

        gate.run()

    def run(self) -> None:
        if not self._enabled():
            return

        if self.pretrain.find_batch_size:
            self._find_batch_size(self.build())
            self._release_cache()

        if self.pretrain.tune_loader:
            self._tune_loader(self.build())
            self._release_cache()

        if self.pretrain.run_overfit:
            self._run_overfit(self.build())
            self._release_cache()
