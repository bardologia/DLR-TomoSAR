from __future__ import annotations

import json

import pytest
import torch

from configuration.training               import UnrolledEntryConfig
from models.unrolled                      import get_unrolled
from pipelines.unrolled.training.pipeline import UnrolledOverfitGate

from tests.backbone_training._helpers import identity_normalizer, x_axis_numpy

from tools.monitoring.logger import Logger


HW     = 8
TRACKS = 3


@pytest.fixture
def force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


class _StubDataset:
    def __init__(self, items):
        self.items     = items
        self.augmenter = object()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _dataset(n: int = 4) -> _StubDataset:
    gen   = torch.Generator().manual_seed(0)
    items = []

    for _ in range(n):
        img = torch.randn(2, HW, HW, generator=gen)
        gt  = torch.randn(6, HW, HW, generator=gen)
        kz  = torch.linspace(-0.15, 0.15, TRACKS).reshape(TRACKS, 1, 1).expand(TRACKS, HW, HW).contiguous()

        gt[0::3] = gt[0::3].abs() + 1.0
        gt[1::3] = 20.0 + 10.0 * gt[1::3].clamp(-1.5, 1.5)
        gt[2::3] = gt[2::3].abs() + 2.0

        items.append((img, gt, kz))

    return _StubDataset(items)


def _entry_config(**overrides) -> UnrolledEntryConfig:
    config = UnrolledEntryConfig()

    config.overfit_check.enabled         = True
    config.overfit_check.n_examples      = 2
    config.overfit_check.max_steps       = 4
    config.overfit_check.steps_per_epoch = 2
    config.overfit_check.pass_loss_ratio = 10.0

    for key, value in overrides.items():
        setattr(config.overfit_check, key, value)

    return config


def _run_gate(tmp_path, config: UnrolledEntryConfig, dataset: _StubDataset) -> None:
    logger    = Logger(log_dir=str(tmp_path / "logs"), name="overfit_gate", level="ERROR")
    model_cfg = get_unrolled("gamma_net", n_iterations=2, prox_hidden=4)[1]

    UnrolledOverfitGate(config, tmp_path, logger).run(model_cfg, x_axis_numpy(), 3, identity_normalizer(6), dataset)


def test_disabled_gate_is_a_no_op(tmp_path, force_cpu):
    _run_gate(tmp_path, _entry_config(enabled=False), _dataset())

    assert not (tmp_path / "meta" / "overfit_report.json").exists()
    assert not (tmp_path / "overfit_check").exists()


def test_passing_gate_writes_report_and_cleans_workdir(tmp_path, force_cpu):
    dataset = _dataset()

    _run_gate(tmp_path, _entry_config(), dataset)

    report = json.loads((tmp_path / "meta" / "overfit_report.json").read_text())

    assert report["passed"] is True
    assert len(report["epoch_losses"]) == 2
    assert report["sanitized_overrides"]["training.use_ema"] is False
    assert report["sanitized_overrides"]["training.warmup_enabled"] is False
    assert report["sanitized_overrides"]["measurement_noise_std"] == 0.0
    assert report["sanitized_overrides"]["model.steps_wd"] == 0.0
    assert report["sanitized_overrides"]["augmentation"] == "disabled"
    assert not (tmp_path / "overfit_check").exists()
    assert dataset.augmenter is not None


def test_failing_gate_aborts_and_cleans_workdir(tmp_path, force_cpu):
    config = _entry_config(pass_loss_ratio=0.0, stop_threshold=0.0)

    with pytest.raises(RuntimeError):
        _run_gate(tmp_path, config, _dataset())

    report = json.loads((tmp_path / "meta" / "overfit_report.json").read_text())

    assert report["passed"] is False
    assert not (tmp_path / "overfit_check").exists()
