from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from tools.training.stopping import EarlyStopping, OverfitManager


def _es_config(patience=3, min_delta=0.0):
    return SimpleNamespace(early_stopping=SimpleNamespace(patience=patience, min_delta=min_delta))


def _overfit_config(enabled=True, max_steps=10, stop_threshold=1e-4, batch_size=2):
    return SimpleNamespace(
        overfit=SimpleNamespace(
            enabled        = enabled,
            max_steps      = max_steps,
            stop_threshold = stop_threshold,
            batch_size     = batch_size,
        )
    )


def test_early_stopping_first_call_sets_best(logger, tracker):
    es   = EarlyStopping(_es_config(patience=3), logger, tracker)
    stop = es(5.0, epoch=0)

    assert stop            is False
    assert es.best_loss    == 5.0
    assert es.best_epoch   == 0
    assert es.counter      == 0


def test_early_stopping_improvement_resets_counter(logger, tracker):
    es = EarlyStopping(_es_config(patience=3), logger, tracker)
    es(5.0, 0)
    es(6.0, 1)
    es(6.0, 2)
    assert es.counter == 2

    es(4.0, 3)
    assert es.counter    == 0
    assert es.best_loss  == 4.0
    assert es.best_epoch == 3


def test_early_stopping_triggers_after_patience(logger, tracker):
    es = EarlyStopping(_es_config(patience=3), logger, tracker)

    assert es(1.0, 0) is False
    assert es(2.0, 1) is False
    assert es(2.0, 2) is False
    assert es(2.0, 3) is True
    assert es.triggered is True


def test_early_stopping_does_not_trigger_below_patience(logger, tracker):
    es = EarlyStopping(_es_config(patience=5), logger, tracker)

    es(1.0, 0)
    for epoch in range(1, 5):
        assert es(2.0, epoch) is False
    assert es.counter == 4


def test_early_stopping_strict_less_than_for_improvement(logger, tracker):
    es = EarlyStopping(_es_config(patience=2), logger, tracker)

    es(1.0, 0)
    es(1.0, 1)
    assert es.counter == 1
    assert es.best_epoch == 0


def test_early_stopping_best_tracks_running_minimum(logger, tracker):
    es      = EarlyStopping(_es_config(patience=10), logger, tracker)
    losses  = [5.0, 4.0, 4.5, 3.0, 3.2, 2.0]

    for epoch, loss in enumerate(losses):
        es(loss, epoch)

    assert es.best_loss  == 2.0
    assert es.best_epoch == 5


def test_early_stopping_reset(logger, tracker):
    es = EarlyStopping(_es_config(patience=2), logger, tracker)
    es(1.0, 0)
    es(2.0, 1)
    es.reset()

    assert es.best_loss  is None
    assert es.counter    == 0
    assert es.best_epoch == -1
    assert es.triggered  is False


def test_overfit_disabled_passes_loaders_through(logger):
    mgr = OverfitManager(_overfit_config(enabled=False), logger)
    t, v, te = mgr.setup_loaders("train", "val", "test")

    assert (t, v, te) == ("train", "val", "test")
    assert mgr.check_stop(0.0) is False


def test_overfit_setup_loaders_replicates_single_batch(logger):
    mgr   = OverfitManager(_overfit_config(enabled=True, max_steps=6, batch_size=2), logger)
    batch = (torch.arange(8).reshape(4, 2),)
    loader = [batch, batch, batch]

    data_loader, val_loader, test_loader = mgr.setup_loaders(loader, loader, loader)

    assert len(data_loader)            == 3
    assert len(val_loader)             == 1
    assert len(test_loader)            == 1
    assert data_loader[0][0].shape[0]  == 2
    assert mgr._epoch_steps            == 3


def test_overfit_planned_epochs(logger):
    mgr   = OverfitManager(_overfit_config(enabled=True, max_steps=10, batch_size=1), logger)
    batch = (torch.zeros(4, 1),)
    loader = [batch, batch, batch, batch]

    mgr.setup_loaders(loader, loader, loader)
    assert mgr.planned_epochs() == math.ceil(10 / 4)


def test_overfit_check_stop_on_max_steps(logger):
    mgr   = OverfitManager(_overfit_config(enabled=True, max_steps=4, batch_size=1, stop_threshold=0.0), logger)
    batch = (torch.zeros(2, 1),)
    loader = [batch, batch]

    mgr.setup_loaders(loader, loader, loader)
    assert mgr.check_stop(1.0) is False
    assert mgr.check_stop(1.0) is True


def test_overfit_check_stop_on_threshold(logger):
    mgr   = OverfitManager(_overfit_config(enabled=True, max_steps=100, batch_size=1, stop_threshold=0.01), logger)
    batch = (torch.zeros(2, 1),)
    loader = [batch, batch]

    mgr.setup_loaders(loader, loader, loader)
    assert mgr.check_stop(0.001) is True


def test_early_stopping_min_delta_treats_small_gains_as_no_improvement(logger, tracker):
    es = EarlyStopping(_es_config(patience=2, min_delta=0.1), logger, tracker)

    assert es(1.0, 0) is False
    assert es(0.95, 1) is False
    assert es(0.92, 2) is True
    assert es.best_loss == pytest.approx(1.0)


def test_early_stopping_min_delta_accepts_large_gains(logger, tracker):
    es = EarlyStopping(_es_config(patience=2, min_delta=0.1), logger, tracker)

    es(1.0, 0)
    es(0.8, 1)

    assert es.best_loss  == pytest.approx(0.8)
    assert es.counter    == 0
    assert es.best_epoch == 1



def test_early_stopping_state_roundtrip(logger, tracker):
    es = EarlyStopping(_es_config(patience=3), logger, tracker)
    es(1.0, 0)
    es(1.2, 1)

    other = EarlyStopping(_es_config(patience=3), logger, tracker)
    other.load_state_dict(es.state_dict())

    assert other.best_loss  == pytest.approx(1.0)
    assert other.counter    == 1
    assert other.best_epoch == 0
    assert other.triggered is False
