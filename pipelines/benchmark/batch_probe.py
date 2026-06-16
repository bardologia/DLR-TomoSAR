from __future__ import annotations

import gc
import traceback

import torch
import torch.nn.functional as F

from configuration.benchmark import BenchmarkConfig
from configuration.sar.gaussian_config          import GaussianConfig
from models                                     import BACKBONE_CONFIG_REGISTRY, BACKBONE_IMAGE_SIZE_MODELS, get_backbone


class MaxBatchProbe:
    def __init__(self, config: BenchmarkConfig, model_name: str, overrides: dict) -> None:
        self.config     = config
        self.model_name = model_name
        self.overrides  = overrides

        self.budget_gb     = config.max_batch.vram_budget_gb
        self.ceiling       = config.max_batch.max_batch
        self.measure_steps = config.max_batch.measure_steps
        self.seed          = config.max_batch.seed

        self.image_size  = config.training.patch_size[0]
        self.in_channels = config.size_match.in_channels
        self.device      = torch.device("cuda")

        gaussian_cfg      = GaussianConfig.from_dataset(config.paths.dataset_path, n_gaussians=config.n_gaussians)
        self.out_channels = gaussian_cfg.params_per_gaussian * config.n_gaussians

    def _candidates(self) -> list[int]:
        sizes = []
        size  = 1

        while size < self.ceiling:
            sizes.append(size)
            size *= 2

        return sizes

    def _build_model(self):
        model_config = BACKBONE_CONFIG_REGISTRY[self.model_name]()

        for attribute, value in self.overrides.items():
            setattr(model_config, attribute, value)

        overrides = {"in_channels": self.in_channels, "out_channels": self.out_channels}
        if self.model_name in BACKBONE_IMAGE_SIZE_MODELS:
            overrides["image_size"] = self.image_size

        model, _ = get_backbone(self.model_name, config=model_config, **overrides)

        return model.to(self.device).train()

    def _trial(self, model, optimizer, batch_size: int) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        inputs  = torch.randn(batch_size, self.in_channels,  self.image_size, self.image_size, device=self.device)
        targets = torch.randn(batch_size, self.out_channels, self.image_size, self.image_size, device=self.device)

        for _ in range(self.measure_steps):
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss    = F.mse_loss(outputs, targets)

            loss.backward()
            optimizer.step()

        peak = torch.cuda.max_memory_allocated(self.device)

        del inputs, targets, outputs, loss

        return peak / (1024.0 ** 3)

    def run(self) -> dict:
        result = {
            "model"      : self.model_name,
            "status"     : None,
            "batch_size" : None,
            "peak_gb"    : None,
            "budget_gb"  : self.budget_gb,
            "ceiling"    : self.ceiling,
            "trials"     : [],
            "error"      : None,
        }

        try:
            torch.manual_seed(self.seed)

            model     = self._build_model()
            optimizer = torch.optim.Adam(model.parameters())
            best      = None

            for batch_size in self._candidates():
                try:
                    peak_gb = self._trial(model, optimizer, batch_size)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    result["trials"].append({"batch_size": batch_size, "peak_gb": None, "status": "OOM"})
                    break

                fits = peak_gb <= self.budget_gb
                result["trials"].append({"batch_size": batch_size, "peak_gb": peak_gb, "status": "FIT" if fits else "OVER"})

                if not fits:
                    break

                best = (batch_size, peak_gb)

            if best is None:
                result["status"] = "FAIL"
                result["error"]  = f"batch size 1 exceeds the {self.budget_gb:g} GB budget"
            else:
                result["batch_size"], result["peak_gb"] = best
                result["status"]                        = "PASS"

            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            result["status"] = "FAIL"
            result["error"]  = traceback.format_exc()

        return result
