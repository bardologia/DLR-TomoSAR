from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.backbone.training.diagnostics import ParamSampler, ReconstructionFigures

from tests.backbone_training._helpers import build_loss, valid_param_tensor


class FigureTracker:
    def __init__(self):
        self.figures    = []
        self.histograms = []

    def log_figure(self, tag, figure, step=None):
        self.figures.append((tag, figure, step))

    def log_histogram(self, tag, values, step=None):
        self.histograms.append((tag, values, step))


def _phys_params(batch=2, n_gaussians=2, hw=5, seed=0):
    return valid_param_tensor(batch, n_gaussians, hw, hw, seed=seed)


def test_sampler_inactive_until_begin():
    sampler = ParamSampler(params_per_gaussian=3, amp_zero_thr=1e-3)

    assert sampler.active is False

    sampler.begin()
    assert sampler.active is True

    sampler.end()
    assert sampler.active is False


def test_sampler_collects_active_values_per_param():
    sampler = ParamSampler(params_per_gaussian=3, amp_zero_thr=1e-3)
    sampler.begin()
    sampler.observe(_phys_params())

    hists = sampler.histograms()

    assert set(hists.keys()) == {"amp", "mu_m", "sigma_m"}
    assert all(isinstance(v, np.ndarray) for v in hists.values())
    assert all(v.size == 2 * 2 * 5 * 5 for v in hists.values())


def test_sampler_masks_inactive_slots():
    sampler = ParamSampler(params_per_gaussian=3, amp_zero_thr=1e-3)
    params  = _phys_params()

    params[:, 0:3] = 0.0

    sampler.begin()
    sampler.observe(params)

    hists = sampler.histograms()

    assert all(v.size == 2 * 1 * 5 * 5 for v in hists.values())


def test_sampler_empty_when_all_slots_inactive():
    sampler = ParamSampler(params_per_gaussian=3, amp_zero_thr=1e-3)

    sampler.begin()
    sampler.observe(torch.zeros(2, 6, 5, 5))

    assert sampler.histograms() == {}


def test_sampler_subsamples_to_batch_cap():
    sampler = ParamSampler(params_per_gaussian=3, amp_zero_thr=1e-3)
    sampler.begin()
    sampler.observe(_phys_params(batch=4, n_gaussians=2, hw=32))

    hists = sampler.histograms()

    assert all(v.size == ParamSampler.MAX_VALUES_PER_BATCH for v in hists.values())


def test_sampler_stops_at_total_cap():
    sampler = ParamSampler(params_per_gaussian=3, amp_zero_thr=1e-3)
    sampler.begin()

    n_batches = ParamSampler.MAX_VALUES_TOTAL // ParamSampler.MAX_VALUES_PER_BATCH + 2
    for i in range(n_batches):
        sampler.observe(_phys_params(batch=4, n_gaussians=2, hw=32, seed=i))

    hists = sampler.histograms()

    assert all(v.size <= ParamSampler.MAX_VALUES_TOTAL for v in hists.values())


def test_sampler_end_clears_store():
    sampler = ParamSampler(params_per_gaussian=3, amp_zero_thr=1e-3)
    sampler.begin()
    sampler.observe(_phys_params())
    sampler.end()

    assert sampler.histograms() == {}


def test_loss_feeds_active_sampler():
    sampler = ParamSampler(params_per_gaussian=3, amp_zero_thr=1e-3)
    loss    = build_loss(n_gaussians=2, sampler=sampler)
    pred    = valid_param_tensor(2, 2, 5, 5, seed=1)
    gt      = valid_param_tensor(2, 2, 5, 5, seed=2)

    loss(pred, gt)
    assert sampler.histograms() == {}

    sampler.begin()
    loss(pred, gt)
    assert set(sampler.histograms().keys()) == {"amp", "mu_m", "sigma_m"}


def _loader(batch=3, n_gaussians=2, hw=8):
    gen  = torch.Generator().manual_seed(0)
    imgs = torch.randn(batch, 2, hw, hw, generator=gen)
    tgt  = valid_param_tensor(batch, n_gaussians, hw, hw, seed=3)
    return [(imgs, tgt)]


def test_figures_capture_reference_once():
    figs = ReconstructionFigures(FigureTracker(), torch.device("cpu"))

    loader = _loader()
    figs.capture_reference(loader)
    first = figs._images

    figs.capture_reference(_loader(batch=1))

    assert figs._images is first
    assert len(figs._pixels) == ReconstructionFigures.NUM_PIXELS


def test_figures_log_noop_without_reference():
    tracker = FigureTracker()
    figs    = ReconstructionFigures(tracker, torch.device("cpu"))

    figs.log(model=None, criterion=None, epoch=0)

    assert tracker.figures == []


def test_figures_log_emits_one_figure_per_pixel():
    tracker = FigureTracker()
    figs    = ReconstructionFigures(tracker, torch.device("cpu"))
    figs.capture_reference(_loader())

    criterion = build_loss(n_gaussians=2)
    model     = lambda images: valid_param_tensor(images.shape[0], 2, images.shape[2], images.shape[3], seed=4)

    figs.log(model, criterion, epoch=5)

    assert len(tracker.figures) == ReconstructionFigures.NUM_PIXELS
    assert all(step == 5 for _, _, step in tracker.figures)
    assert all(tag.startswith("reconstruction/") and tag.endswith("/val") for tag, _, _ in tracker.figures)
