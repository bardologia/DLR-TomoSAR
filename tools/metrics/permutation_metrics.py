from __future__ import annotations

import itertools
import torch
from tools.monitoring.logger import Logger


class PermutationMetrics:
    def __init__(self, config, logger: Logger | None = None) -> None:
        self.cfg     = config
        self.enabled = config.enabled

        if logger is not None:
            logger.kv_table({"Permutation metrics": "enabled" if self.enabled else "disabled"})

    @classmethod
    def silent(cls, cfg) -> PermutationMetrics:
        return cls(cfg, logger=None)

    @staticmethod
    def _mu_ordering_rate(pred_params: torch.Tensor, ppg: int, amp_threshold: float = 1e-3) -> float:
        B, C, H, W = pred_params.shape
        G = C // ppg
        p = pred_params.reshape(B, G, ppg, H, W)

        amp      = p[:, :, 0]
        mu       = p[:, :, 1]
        active   = amp > amp_threshold
        n_active = active.sum(dim=1)

        ordered       = mu[:, :-1] < mu[:, 1:]
        both_active   = active[:, :-1] & active[:, 1:]
        violations    = (~ordered) & both_active
        has_violation = violations.any(dim=1)

        multi_active = n_active >= 2
        denom        = multi_active.sum().item()
        if denom == 0:
            return float("nan")

        return ((~has_violation) & multi_active).sum().item() / denom

    @staticmethod
    def _assignment_cost_margin(pred_params: torch.Tensor, gt_params: torch.Tensor, ppg: int) -> dict[str, float]:
        B, C, H, W = pred_params.shape
        G          = C // ppg

        p    = pred_params.reshape(B, G, ppg, H, W)
        g    = gt_params.reshape(  B, G, ppg, H, W)
        p_mu = p[:, :, 1].permute(0, 2, 3, 1).reshape(B, H * W, G, 1)
        g_mu = g[:, :, 1].permute(0, 2, 3, 1).reshape(B, H * W, G, 1)

        cost_mat   = torch.nan_to_num(torch.cdist(p_mu, g_mu), nan=1e9, posinf=1e9)
        perms      = torch.tensor(list(itertools.permutations(range(G))), dtype=torch.long, device=pred_params.device)
        perm_mask  = torch.nn.functional.one_hot(perms, num_classes=G).to(cost_mat.dtype)
        perm_costs = torch.einsum("bsij,pij->bsp", cost_mat, perm_mask)

        sorted_costs, _ = perm_costs.sort(dim=-1)
        best   = sorted_costs[:, :, 0]
        second = sorted_costs[:, :, 1]

        resolved   = best > 1e-6

        margin     = (second - best).mean().item()
        rel_margin = ((second - best)[resolved] / best[resolved]).mean().item() if resolved.any() else 0.0
        ambiguous  = ((second - best) / (best + 1e-8) < 0.05).float().mean().item()

        return {
            "mean_margin":     margin,
            "mean_rel_margin": rel_margin,
            "ambiguous_frac":  ambiguous,
        }

    @staticmethod
    def _slot_activation_stats(pred_params: torch.Tensor, ppg: int, amp_threshold: float = 1e-3) -> dict[str, float]:
        B, C, H, W = pred_params.shape
        G      = C // ppg
        p      = pred_params.reshape(B, G, ppg, H, W)
        amp    = p[:, :, 0]
        active = (amp > amp_threshold).float()

        out: dict[str, float] = {}
        rates                 = []
        for i in range(G):
            rate                              = active[:, i].mean().item()
            out[f"active_rate/slot_{i}"]     = rate
            out[f"mean_amp/slot_{i}"]        = amp[:, i].mean().item()
            rates.append(rate)

        out["activation_rate_std"] = float(torch.tensor(rates).std().item())
        return out

    @staticmethod
    def _slot_mu_spread(pred_params: torch.Tensor, ppg: int) -> dict[str, float]:
        B, C, H, W = pred_params.shape
        G  = C // ppg
        p  = pred_params.reshape(B, G, ppg, H, W)
        mu = p[:, :, 1]

        out: dict[str, float] = {}
        means                 = []
        for i in range(G):
            m = mu[:, i].mean().item()
            s = mu[:, i].std().item()
            out[f"mu_mean/slot_{i}"]   = m
            out[f"mu_std/slot_{i}"]    = s
            means.append(m)

        out["mu_mean_spread"] = float(torch.tensor(means).std().item())
        return out

    @staticmethod
    def _placeholder_detection_stats(pred_params: torch.Tensor, gt_params: torch.Tensor, ppg: int, amp_threshold: float = 1e-3) -> dict[str, float]:
        B, C, H, W = pred_params.shape
        G = C // ppg
        p = pred_params.reshape(B, G, ppg, H, W)
        g = gt_params.reshape(  B, G, ppg, H, W)

        pred_ph = (p[:, :, 0] <= amp_threshold)
        gt_ph   = (g[:, :, 0] <= amp_threshold)

        out: dict[str, float] = {}

        for i in range(G):
            pp = pred_ph[:, i].float()
            gp = gt_ph[:, i].float()

            tp = (pp * gp).sum().item()
            fp = (pp * (1.0 - gp)).sum().item()
            fn = ((1.0 - pp) * gp).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2.0 * precision * recall / (precision + recall + 1e-8)

            out[f"placeholder/precision/slot_{i}"] = precision
            out[f"placeholder/recall/slot_{i}"]    = recall
            out[f"placeholder/f1/slot_{i}"]        = f1
            out[f"placeholder/gt_rate/slot_{i}"]   = gp.mean().item()
            out[f"placeholder/pred_rate/slot_{i}"] = pp.mean().item()

        pp_all = pred_ph.float()
        gp_all = gt_ph.float()

        tp_all = (pp_all * gp_all).sum().item()
        fp_all = (pp_all * (1.0 - gp_all)).sum().item()
        fn_all = ((1.0 - pp_all) * gp_all).sum().item()

        precision_all = tp_all / (tp_all + fp_all + 1e-8)
        recall_all    = tp_all / (tp_all + fn_all + 1e-8)
        f1_all        = 2.0 * precision_all * recall_all / (precision_all + recall_all + 1e-8)

        out["placeholder/precision"] = precision_all
        out["placeholder/recall"]    = recall_all
        out["placeholder/f1"]        = f1_all
        out["placeholder/gt_rate"]   = gp_all.mean().item()
        out["placeholder/pred_rate"] = pp_all.mean().item()

        return out

    @staticmethod
    def _active_count_stats(
        pred_params: torch.Tensor,
        gt_params: torch.Tensor,
        ppg: int,
        amp_threshold: float = 1e-3,
    ) -> dict[str, float]:
        B, C, H, W = pred_params.shape
        G = C // ppg
        p = pred_params.reshape(B, G, ppg, H, W)
        g = gt_params.reshape(  B, G, ppg, H, W)

        pred_n = (p[:, :, 0] > amp_threshold).sum(dim=1).float()
        gt_n   = (g[:, :, 0] > amp_threshold).sum(dim=1).float()

        out: dict[str, float] = {}
        out["count/mae"] = (pred_n - gt_n).abs().mean().item()
        out["count/bias"] = (pred_n - gt_n).mean().item()

        for k_gt in range(G + 1):
            mask = (gt_n == k_gt)
            if mask.sum() == 0:
                continue
            for k_pred in range(G + 1):
                frac = ((pred_n == k_pred) & mask).float().sum().item() / mask.float().sum().item()
                if frac > 0:
                    out[f"count/confusion/gt{k_gt}_pred{k_pred}"] = frac

        return out

    @staticmethod
    def _permutation_consensus(
        pred_params: torch.Tensor,
        gt_params: torch.Tensor,
        ppg: int,
    ) -> dict[str, float]:
        B, C, H, W = pred_params.shape
        G          = C // ppg

        p    = pred_params.reshape(B, G, ppg, H, W)
        g    = gt_params.reshape(  B, G, ppg, H, W)
        p_mu = p[:, :, 1].permute(0, 2, 3, 1).reshape(B, H * W, G, 1)
        g_mu = g[:, :, 1].permute(0, 2, 3, 1).reshape(B, H * W, G, 1)

        cost_mat   = torch.nan_to_num(torch.cdist(p_mu, g_mu), nan=1e9, posinf=1e9)
        perms      = torch.tensor(list(itertools.permutations(range(G))), dtype=torch.long, device=pred_params.device)
        perm_mask  = torch.nn.functional.one_hot(perms, num_classes=G).to(cost_mat.dtype)
        perm_costs = torch.einsum("bsij,pij->bsp", cost_mat, perm_mask)

        best_perm_idx = perm_costs.argmin(dim=-1)

        out: dict[str, float] = {}

        consensus_per_sample = []
        for b in range(B):
            counts = torch.bincount(best_perm_idx[b], minlength=len(perms)).float()
            consensus_per_sample.append((counts.max() / counts.sum()).item())
        out["consensus/mean"] = float(torch.tensor(consensus_per_sample).mean().item())
        out["consensus/min"]  = float(torch.tensor(consensus_per_sample).min().item())

        all_idx  = best_perm_idx.reshape(-1)
        counts_g = torch.bincount(all_idx, minlength=len(perms)).float()
        out["consensus/global_dominant_frac"] = (counts_g.max() / counts_g.sum()).item()

        return out

    @staticmethod
    def _amplitude_calibration(
        pred_params: torch.Tensor,
        gt_params: torch.Tensor,
        ppg: int,
        amp_threshold: float = 1e-3,
    ) -> dict[str, float]:
        B, C, H, W = pred_params.shape
        G = C // ppg
        p = pred_params.reshape(B, G, ppg, H, W)
        g = gt_params.reshape(  B, G, ppg, H, W)

        pred_amp  = p[:, :, 0]
        gt_active = g[:, :, 0] > amp_threshold

        out: dict[str, float] = {}
        overall_active_amp   = []
        overall_inactive_amp = []

        for i in range(G):
            act_mask = gt_active[:, i]
            ina_mask = ~act_mask

            if act_mask.any():
                m = pred_amp[:, i][act_mask].mean().item()
                out[f"amp_cal/active_gt/slot_{i}"]   = m
                overall_active_amp.append(m)

            if ina_mask.any():
                m = pred_amp[:, i][ina_mask].mean().item()
                out[f"amp_cal/inactive_gt/slot_{i}"] = m
                overall_inactive_amp.append(m)

        if overall_active_amp and overall_inactive_amp:
            mean_act = float(torch.tensor(overall_active_amp).mean().item())
            mean_ina = float(torch.tensor(overall_inactive_amp).mean().item())
            out["amp_cal/active_gt"]    = mean_act
            out["amp_cal/inactive_gt"]  = mean_ina
            out["amp_cal/gap"]          = mean_act - mean_ina

        return out

    @staticmethod
    def _sigma_degeneration(
        pred_params: torch.Tensor,
        gt_params: torch.Tensor,
        ppg: int,
        amp_threshold: float = 1e-3,
    ) -> dict[str, float]:
        if ppg < 3:
            return {}

        B, C, H, W = pred_params.shape
        G = C // ppg
        p = pred_params.reshape(B, G, ppg, H, W)
        g = gt_params.reshape(  B, G, ppg, H, W)

        pred_sigma = p[:, :, 2]
        gt_active  = g[:, :, 0] > amp_threshold

        out: dict[str, float] = {}
        for i in range(G):
            act_mask = gt_active[:, i]
            ina_mask = ~act_mask

            if act_mask.any():
                s = pred_sigma[:, i][act_mask]
                out[f"sigma/active_gt/mean/slot_{i}"] = s.mean().item()
                out[f"sigma/active_gt/std/slot_{i}"]  = s.std().item()

            if ina_mask.any():
                s = pred_sigma[:, i][ina_mask]
                out[f"sigma/inactive_gt/mean/slot_{i}"] = s.mean().item()
                out[f"sigma/inactive_gt/std/slot_{i}"]  = s.std().item()

        act_s = pred_sigma[gt_active]
        ina_s = pred_sigma[~gt_active]
        if act_s.numel() > 0:
            out["sigma/active_gt/mean"]   = act_s.mean().item()
        if ina_s.numel() > 0:
            out["sigma/inactive_gt/mean"] = ina_s.mean().item()

        return out

    @torch.no_grad()
    def compute(self, pred_params: torch.Tensor, gt_params: torch.Tensor, ppg: int) -> dict[str, float]:
        if not self.enabled:
            return {}
        
        cfg                              = self.cfg
        metrics: dict[str, float]        = {}
        metrics["perm/mu_ordering_rate"] = self._mu_ordering_rate(pred_params, ppg, cfg.amp_threshold)
       
        metrics.update({f"perm/{k}": v for k, v in self._assignment_cost_margin(pred_params, gt_params, ppg).items()})
        metrics.update({f"perm/{k}": v for k, v in self._slot_activation_stats(pred_params, ppg, cfg.amp_threshold).items()})
        metrics.update({f"perm/{k}": v for k, v in self._slot_mu_spread(pred_params, ppg).items()})
        metrics.update({f"perm/{k}": v for k, v in self._placeholder_detection_stats(pred_params, gt_params, ppg, cfg.amp_threshold).items()})
        metrics.update({f"perm/{k}": v for k, v in self._active_count_stats(pred_params, gt_params, ppg, cfg.amp_threshold).items()})
        metrics.update({f"perm/{k}": v for k, v in self._permutation_consensus(pred_params, gt_params, ppg).items()})
        metrics.update({f"perm/{k}": v for k, v in self._amplitude_calibration(pred_params, gt_params, ppg, cfg.amp_threshold).items()})
        metrics.update({f"perm/{k}": v for k, v in self._sigma_degeneration(pred_params, gt_params, ppg, cfg.amp_threshold).items()})
       
        return metrics
