from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.architectures import ResUNetConfig
from models.backbone.resunet import ResUNet
from tools.monitoring.logger import Logger

DEFAULT_RUN     = REPO_ROOT / "runs/resunet-conv-sorted_gt-K_5-hvn-none-param_l1_1_20260617_210314"
DEFAULT_TWIN    = REPO_ROOT / "test_data/data"
WINDOW          = 160
MASS_WINDOWS    = (8, 16, 24, 32, 40, 48, 56, 64, 96, 128)
PROBE_AZIMUTHS  = (150, 400, 780, 900)
PROBE_RANGES    = (100, 250, 400)


class TrainedModelLoader:
    def __init__(self, run_directory: Path):
        self.run_directory = run_directory

    def load(self) -> ResUNet:
        raw     = json.loads((self.run_directory / "meta/model_config.json").read_text())["config"]
        model   = ResUNet(ResUNetConfig(**raw))
        payload = torch.load(self.run_directory / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["params"], strict=True)
        return model.eval()


class RealInputAssembler:
    def __init__(self, run_directory: Path, twin_directory: Path):
        self.run_directory  = run_directory
        self.twin_directory = twin_directory

    def secondary_selection(self) -> list[int]:
        dataset_config = json.loads((self.run_directory / "meta/dataset_creation_config.json").read_text())
        twin_labels    = json.loads((self.twin_directory / "dataset.json").read_text())["pass_labels"]
        secondaries    = [label for label in twin_labels if label != twin_labels[0]]
        return [secondaries.index(label) for label in dataset_config["secondary_labels"]]

    def assemble(self) -> torch.Tensor:
        selection      = self.secondary_selection()
        primary        = np.load(self.twin_directory / "primary.npy")
        secondaries    = np.load(self.twin_directory / "secondaries.npy")[selection]
        interferograms = np.load(self.twin_directory / "interferograms.npy")[selection]
        stats          = json.loads((self.run_directory / "meta/normalization_stats.json").read_text())["input_stats"]["channels"]

        values   = [np.abs(primary)] + [np.abs(s) for s in secondaries] + [np.angle(i) for i in interferograms]
        channels = []
        for value, spec in zip(values, stats):
            x = np.log1p(value) if spec["apply_log1p"] else value
            channels.append((x - spec["loc"]) / spec["scale"])

        return torch.from_numpy(np.stack(channels)).float().unsqueeze(0)


class ReceptiveFieldMeasurement:
    def __init__(self, run_directory: Path = DEFAULT_RUN, twin_directory: Path = DEFAULT_TWIN):
        self.logger = Logger(log_dir="logs", name="receptive_field")
        self.model  = TrainedModelLoader(run_directory).load()
        self.inputs = RealInputAssembler(run_directory, twin_directory).assemble()

    def gradient_map(self) -> torch.Tensor:
        half        = WINDOW // 2
        probes      = [(az, rg) for az in PROBE_AZIMUTHS for rg in PROBE_RANGES]
        accumulated = torch.zeros(WINDOW, WINDOW)

        for az, rg in probes:
            x = self.inputs[:, :, az - half:az + half, rg - half:rg + half].clone().requires_grad_(True)
            y = self.model(x)
            y[0, :, half, half].abs().sum().backward()
            accumulated += x.grad.abs().sum(dim=(0, 1))

        return accumulated / len(probes)

    def report_sigma(self, grad: torch.Tensor) -> None:
        center = WINDOW // 2
        coords = torch.arange(WINDOW, dtype=torch.float32)
        rr, cc = torch.meshgrid(coords, coords, indexing="ij")
        weight = grad / grad.sum()
        var_az = (weight * (rr - center) ** 2).sum().item()
        var_rg = (weight * (cc - center) ** 2).sum().item()
        self.logger.info(f"trained ERF sigma: az {math.sqrt(var_az):.2f} px, rg {math.sqrt(var_rg):.2f} px")

    def report_mass(self, grad: torch.Tensor) -> None:
        center   = WINDOW // 2
        total    = grad.sum()
        previous = 0.0

        for w in MASS_WINDOWS:
            half = w // 2
            mass = (grad[center - half:center + half, center - half:center + half].sum() / total).item()
            self.logger.info(f"mass inside {w:3d}x{w:<3d}: {mass * 100:6.2f}%   marginal: {(mass - previous) * 100:5.2f}%")
            previous = mass

    def run(self) -> None:
        grad = self.gradient_map()
        self.report_sigma(grad)
        self.report_mass(grad)


if __name__ == "__main__":
    ReceptiveFieldMeasurement().run()
