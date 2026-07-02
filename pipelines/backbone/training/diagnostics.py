from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


class ParamSampler:
    PARAM_NAMES          = ("amp", "mu_m", "sigma_m")
    MAX_VALUES_PER_BATCH = 2048
    MAX_VALUES_TOTAL     = 200_000

    def __init__(self, params_per_gaussian: int, amp_zero_thr: float):
        self.ppg    = params_per_gaussian
        self.thr    = amp_zero_thr
        self.active = False

        self._reset_store()

    def begin(self) -> None:
        self.active = True
        self._reset_store()

    @torch.no_grad()
    def observe(self, pred_params_phys: torch.Tensor) -> None:
        B, C, H, W = pred_params_phys.shape
        p          = pred_params_phys.reshape(B, C // self.ppg, self.ppg, H, W)
        active     = p[:, :, 0] > self.thr

        for i, name in enumerate(self.PARAM_NAMES):
            budget = self.MAX_VALUES_TOTAL - self._counts[name]

            if budget <= 0:
                continue

            values = self._subsample(p[:, :, i][active], min(self.MAX_VALUES_PER_BATCH, budget))

            if values.numel() == 0:
                continue

            self._values[name].append(values.detach().float().cpu())
            self._counts[name] += values.numel()

    @staticmethod
    def _subsample(values: torch.Tensor, cap: int) -> torch.Tensor:
        if values.numel() <= cap:
            return values

        idx = torch.linspace(0, values.numel() - 1, cap, device=values.device).long()
        return values[idx]

    def histograms(self) -> dict:
        return {name: torch.cat(chunks).numpy() for name, chunks in self._values.items() if chunks}

    def end(self) -> None:
        self.active = False
        self._reset_store()

    def _reset_store(self) -> None:
        self._values = {name: [] for name in self.PARAM_NAMES}
        self._counts = {name: 0  for name in self.PARAM_NAMES}


class ReconstructionFigures:
    NUM_PIXELS = 4

    def __init__(self, tracker, device):
        self.tracker = tracker
        self.device  = device

        self._images  = None
        self._targets = None
        self._pixels  = None

    def capture_reference(self, loader) -> None:
        if self._images is not None:
            return

        batch         = next(iter(loader))
        self._images  = batch[0].to(self.device)
        self._targets = batch[1].to(self.device)

        B, _, H, W   = self._targets.shape
        positions    = ((H // 4, W // 4), (H // 4, 3 * W // 4), (3 * H // 4, W // 4), (3 * H // 4, 3 * W // 4))
        self._pixels = tuple((i % B, h, w) for i, (h, w) in enumerate(positions[:self.NUM_PIXELS]))

    @torch.no_grad()
    def log(self, model, criterion, epoch: int) -> None:
        if self._images is None:
            return

        pred                    = model(self._images)
        pred_curves, exp_curves = criterion.curves(pred, self._targets)
        x                       = criterion.x_axis.detach().cpu().numpy()

        for b, h, w in self._pixels:
            pred_curve = pred_curves[b, :, h, w].float().cpu().numpy()
            gt_curve   = exp_curves[b, :, h, w].float().cpu().numpy()

            fig = self._figure(x, pred_curve, gt_curve)
            self.tracker.log_figure(f"reconstruction/sample{b}_h{h}_w{w}/val", fig, epoch)

    def _figure(self, x_axis, pred_curve, gt_curve):
        fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=120)

        ax.plot(x_axis, gt_curve,   color="black",   linestyle="--", linewidth=1.2, label="Ground truth")
        ax.plot(x_axis, pred_curve, color="#1f77b4", linestyle="-",  linewidth=1.4, label="Prediction")

        ax.set_xlabel("Elevation [m]")
        ax.set_ylabel("Amplitude")
        ax.legend(frameon=False)
        fig.tight_layout()

        return fig
