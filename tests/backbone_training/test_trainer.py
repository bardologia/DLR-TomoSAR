from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configuration.training.general.loss import LossConfig
from pipelines.backbone.training.loss    import Loss
from pipelines.backbone.training.trainer import CurriculumController, Trainer

from tests.backbone_training._helpers import identity_normalizer, tiny_model, tiny_trainer_config, x_axis_numpy

from tools.monitoring.logger import Logger


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _loader(in_channels: int = 2, n_gaussians: int = 2, n: int = 6, hw: int = 16) -> DataLoader:
    gen  = torch.Generator().manual_seed(0)
    imgs = torch.randn(n, in_channels, hw, hw, generator=gen)
    tgt  = torch.randn(n, n_gaussians * 3, hw, hw, generator=gen)
    return DataLoader(TensorDataset(imgs, tgt), batch_size=2)


def _build_trainer(tmp_path, epochs: int = 1, n_gaussians: int = 2) -> Trainer:
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=n_gaussians)
    config           = tiny_trainer_config(n_gaussians=n_gaussians, epochs=epochs)
    logger           = Logger(log_dir=str(tmp_path / "logs"), name="trainer", level="ERROR")
    norm_stats       = identity_normalizer(n_gaussians * 3)

    return Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=norm_stats, emit_docs=False)


def test_trainer_builds_on_cpu_with_wired_components(tmp_path):
    trainer = _build_trainer(tmp_path)

    assert trainer.device.type == "cpu"
    assert isinstance(trainer.criterion, Loss)
    assert trainer.optimizer is not None
    assert trainer.lr_scheduler is not None
    assert trainer.early_stopping is not None
    assert trainer.warmup is not None
    assert isinstance(trainer.curriculum_controller, CurriculumController)


def test_trainer_optimizer_has_named_param_groups(tmp_path):
    trainer = _build_trainer(tmp_path)
    names   = {group.get("name") for group in trainer.optimizer.param_groups}

    assert "encoder"     in names
    assert "output_head" in names
    assert len(trainer.base_lrs) == len(trainer.optimizer.param_groups)


def test_single_train_step_updates_parameters(tmp_path):
    trainer = _build_trainer(tmp_path)
    loader  = _loader()

    before  = [p.detach().clone() for p in trainer.model.parameters()]
    avg     = trainer.train_epoch(loader, epoch=0)
    after   = list(trainer.model.parameters())

    assert isinstance(avg, float)
    assert torch.isfinite(torch.tensor(avg)).item()
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_compute_loss_returns_finite_total(tmp_path):
    trainer = _build_trainer(tmp_path)
    batch   = next(iter(_loader()))

    out     = trainer._compute_loss(batch)

    assert "total_loss" in out
    assert torch.isfinite(out["total_loss"]).item()


def test_one_epoch_fit_writes_checkpoint(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=1)
    loader  = _loader()

    train_losses, val_losses, best_val = trainer.train(loader, loader, loader)

    assert len(train_losses) == 1
    assert (tmp_path / "best_model.pt").exists()
    assert torch.isfinite(torch.tensor(best_val)).item()


def test_multi_epoch_fit_records_losses(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=3)
    loader  = _loader()

    train_losses, val_losses, _ = trainer.train(loader, loader, loader)

    assert len(train_losses) == 3
    assert all(torch.isfinite(torch.tensor(loss)).item() for loss in train_losses)
    assert len(val_losses) >= 1


def test_checkpoint_save_restore_round_trip(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=1)
    loader  = _loader()

    trainer.train(loader, loader, loader)

    saved = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=False)

    assert "params"        in saved
    assert "x_axis"        in saved
    assert "best_val_loss" in saved

    fresh_model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    fresh_model.load_state_dict(saved["params"])

    for name, param in fresh_model.state_dict().items():
        assert torch.equal(param, saved["params"][name])


def test_restore_best_loads_best_parameters(tmp_path):
    trainer = _build_trainer(tmp_path, epochs=2)
    loader  = _loader()

    trainer.train(loader, loader, loader)

    saved = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=False)

    for name, param in trainer.model.state_dict().items():
        assert torch.equal(param, saved["params"][name])


def test_early_stopping_reset_clears_state(tmp_path):
    trainer = _build_trainer(tmp_path)

    trainer.early_stopping(val_loss=1.0, epoch=0)
    trainer.early_stopping.reset()

    assert trainer.early_stopping.best_loss is None
    assert trainer.early_stopping.counter == 0
    assert trainer.early_stopping.triggered is False


def test_scheduler_reset_restores_base_lrs(tmp_path):
    trainer = _build_trainer(tmp_path)

    trainer.lr_scheduler.step(epoch=5)
    trainer.lr_scheduler.reset(epoch_offset=0)

    assert trainer.lr_scheduler.current_lrs == trainer.lr_scheduler.base_lrs


def test_curriculum_swap_disabled_is_noop(tmp_path):
    trainer = _build_trainer(tmp_path)

    swapped = trainer.curriculum_controller.maybe_swap(epoch=0)

    assert swapped is False
    assert trainer.criterion.loss_generation == 0


def test_curriculum_swap_replaces_loss_config(tmp_path):
    model, model_cfg = tiny_model(in_channels=2, n_gaussians=2)
    config           = tiny_trainer_config(n_gaussians=2, epochs=1)

    config.curriculum.enabled    = True
    config.curriculum.swap_epoch = 0
    config.curriculum.complete   = LossConfig(use_mse_curve=True, weight_mse_curve=1.0, param_match="none")

    logger     = Logger(log_dir=str(tmp_path / "logs"), name="curr", level="ERROR")
    norm_stats = identity_normalizer(6)
    trainer    = Trainer(model, model_cfg, x_axis_numpy(), config, tmp_path, logger, norm_stats=norm_stats, emit_docs=False)

    swapped = trainer.curriculum_controller.maybe_swap(epoch=0)

    assert swapped is True
    assert trainer.criterion.loss_generation == 1
    assert trainer.criterion.match_strategy  == "none"
