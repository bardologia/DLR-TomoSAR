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


class ExampleSelector:
    CANDIDATES_PER_BATCH = 256
    QUANTILES            = (0.1, 0.5, 0.9)
    OVERLAP_RATIO_RANGE  = (0.6, 1.5)
    SEPARATED_RATIO_MIN  = 1.5
    DISTANT_RATIO_MIN    = 3.0
    AMP_RATIO_MIN        = 2.0
    CATEGORIES           = ("single_gaussian", "two_overlapping", "two_separated", "two_distant")

    def __init__(self, params_per_gaussian: int, amp_zero_thr: float):
        self.ppg = params_per_gaussian
        self.thr = amp_zero_thr

    @torch.no_grad()
    def _batch_candidates(self, phys: torch.Tensor, offset: int) -> dict:
        B, C, H, W = phys.shape
        p          = phys.reshape(B, C // self.ppg, self.ppg, H, W)

        amp      = p[:, :, 0]
        mu       = p[:, :, 1]
        sigma    = p[:, :, 2]
        active   = amp > self.thr
        n_active = active.sum(dim=1)

        single     = n_active == 1
        single_amp = (amp * active).sum(dim=1)
        rows       = {"single_gaussian": self._rows(single, single_amp, offset)}

        if amp.shape[1] < 2:
            rows["two_overlapping"] = self._empty()
            rows["two_separated"]   = self._empty()
            rows["two_distant"]     = self._empty()
            return rows

        top2     = amp.topk(2, dim=1).indices
        mu_pair  = mu.gather(1, top2)
        sig_pair = sigma.gather(1, top2)
        amp_pair = amp.gather(1, top2)

        separation = (mu_pair[:, 0] - mu_pair[:, 1]).abs()
        sep_ratio  = separation / sig_pair.sum(dim=1).clamp_min(1e-6)
        amp_ratio  = amp_pair.amax(dim=1) / amp_pair.amin(dim=1).clamp_min(1e-6)

        pair      = n_active == 2
        overlap   = pair & (sep_ratio >= self.OVERLAP_RATIO_RANGE[0]) & (sep_ratio <= self.OVERLAP_RATIO_RANGE[1])
        separated = pair & (sep_ratio >= self.SEPARATED_RATIO_MIN) & (amp_ratio < self.AMP_RATIO_MIN)
        distant   = pair & (sep_ratio >= self.DISTANT_RATIO_MIN) & (amp_ratio >= self.AMP_RATIO_MIN)

        rows["two_overlapping"] = self._rows(overlap, sep_ratio, offset)
        rows["two_separated"]   = self._rows(separated, sep_ratio, offset)
        rows["two_distant"]     = self._rows(distant, amp_ratio, offset)

        return rows

    def _rows(self, mask: torch.Tensor, metric: torch.Tensor, offset: int) -> torch.Tensor:
        coords = mask.nonzero(as_tuple=False)

        if coords.shape[0] == 0:
            return self._empty()

        rows = torch.stack([coords[:, 0].float() + offset, coords[:, 1].float(), coords[:, 2].float(), metric[mask].float()], dim=1)
        return self._subsample(rows, self.CANDIDATES_PER_BATCH)

    @staticmethod
    def _subsample(rows: torch.Tensor, cap: int) -> torch.Tensor:
        if rows.shape[0] <= cap:
            return rows

        idx = torch.linspace(0, rows.shape[0] - 1, cap).long()
        return rows[idx]

    @staticmethod
    def _empty() -> torch.Tensor:
        return torch.zeros(0, 4)

    def _pick(self, pool: torch.Tensor) -> list:
        if pool.shape[0] == 0:
            return []

        order  = pool[:, 3].argsort()
        picked = []
        seen   = set()

        for q in self.QUANTILES:
            row   = order[round(q * (order.shape[0] - 1))]
            pixel = tuple(int(v) for v in pool[row, :3])

            if pixel not in seen:
                seen.add(pixel)
                picked.append(pixel)

        return picked

    def select(self, loader, denormalize) -> dict:
        pools  = {name: [] for name in self.CATEGORIES}
        offset = 0

        for batch in loader:
            phys = denormalize(batch[1].float())

            for name, rows in self._batch_candidates(phys, offset).items():
                pools[name].append(rows)

            offset += batch[0].shape[0]

        return {name: self._pick(torch.cat(chunks)) for name, chunks in pools.items()}


class ReconstructionFigures:
    HEADROOM = 1.3

    def __init__(self, tracker, device):
        self.tracker = tracker
        self.device  = device

        self._images   = None
        self._targets  = None
        self._pixels   = None
        self._ylims    = None
        self._disabled = False

    def _collect(self, loader, selected: dict):
        needed  = {g for rows in selected.values() for g, _, _ in rows}
        samples = {}
        offset  = 0

        for batch in loader:
            size = batch[0].shape[0]

            for g in needed:
                if offset <= g < offset + size:
                    samples[g] = (batch[0][g - offset], batch[1][g - offset])

            offset += size

        images  = []
        targets = []
        pixels  = []

        for name in ExampleSelector.CATEGORIES:
            for rank, (g, h, w) in enumerate(selected[name], start=1):
                image, target = samples[g]

                images.append(image)
                targets.append(target)
                pixels.append((len(pixels), h, w, f"{name}_{rank}"))

        return torch.stack(images), torch.stack(targets), pixels

    def _fixed_ylims(self, criterion) -> list:
        gt_phys   = criterion.norm_stats.denormalize_output(self._targets.float())
        gt_curves = criterion.reconstruct(gt_phys)

        return [(0.0, max(float(gt_curves[row, :, h, w].max()), 1e-6) * self.HEADROOM) for row, h, w, _ in self._pixels]

    def capture_reference(self, loader, criterion) -> None:
        if self._images is not None or self._disabled:
            return

        selector = ExampleSelector(criterion.gaussian_cfg.params_per_gaussian, criterion.loss_cfg.amp_zero_thr)
        selected = selector.select(loader, criterion.norm_stats.denormalize_output)

        for name in ExampleSelector.CATEGORIES:
            if len(selected[name]) < len(ExampleSelector.QUANTILES):
                criterion.logger.warning(f"Reconstruction figures: found {len(selected[name])}/{len(ExampleSelector.QUANTILES)} validation pixels for category '{name}'.")

        if not any(len(rows) for rows in selected.values()):
            criterion.logger.warning("Reconstruction figures disabled: no validation pixel matches any example category.")
            self._disabled = True
            return

        images, targets, pixels = self._collect(loader, selected)

        self._images  = images.to(self.device)
        self._targets = targets.to(self.device)
        self._pixels  = pixels
        self._ylims   = self._fixed_ylims(criterion)

    def _figure(self, x_axis, pred_curve, gt_curve, ylim):
        fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=120)

        ax.plot(x_axis, gt_curve,   color="black",   linestyle="--", linewidth=1.2, label="Ground truth")
        ax.plot(x_axis, pred_curve, color="#1f77b4", linestyle="-",  linewidth=1.4, label="Prediction")

        ax.set_xlabel("Elevation [m]")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(*ylim)
        ax.legend(frameon=False)
        fig.tight_layout()

        return fig

    @torch.no_grad()
    def log(self, model, criterion, epoch: int) -> None:
        if self._images is None:
            return

        pred                    = model(self._images)
        pred_curves, exp_curves = criterion.curves(pred, self._targets)
        x                       = criterion.x_axis.detach().cpu().numpy()

        for (row, h, w, tag), ylim in zip(self._pixels, self._ylims):
            pred_curve = pred_curves[row, :, h, w].float().cpu().numpy()
            gt_curve   = exp_curves[row, :, h, w].float().cpu().numpy()

            fig = self._figure(x, pred_curve, gt_curve, ylim)
            self.tracker.log_figure(f"reconstruction/{tag}/val", fig, epoch)
