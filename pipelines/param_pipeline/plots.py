from __future__ import annotations

import gc
from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm     as cm
import matplotlib.pyplot as plt
import numpy             as np
from scipy.stats         import gaussian_kde, pearsonr, spearmanr

from pipelines.shared.plotting      import PlotBase
from pipelines.shared.preprocessing import ProfilePreprocessor
from tools.gaussians                import GaussianMixture
from tools.logger                   import Logger


class SpatialMapPlotter(PlotBase):
    def __init__(self, n_gaussians : int, logger : Logger, fig_dpi : int = 150, save_dpi : int = 300) -> None:
        self.n_gaussians = n_gaussians
        self.logger      = logger
        self.fig_dpi     = fig_dpi
        self.save_dpi    = save_dpi

    def _plot_discrete_k_map(self, k_map : np.ndarray, title : str, cbar_label : str, out_path : Path) -> Path:
        Az, R   = k_map.shape
        levels  = list(range(self.n_gaussians + 1))
        n_total = k_map.size

        text_lines   = ["share of all pixels:"] + [f"$K={k}$: {(k_map == k).sum() / n_total * 100:.1f}%" for k in levels if (k_map == k).sum() > 0]
        text_overlay = "\n".join(text_lines)

        return self._imshow_figure(
            k_map,
            x_label        = "range [px]",
            y_label        = "azimuth [px]",
            title          = title,
            cmap           = None,
            extent         = [0, R, Az, 0],
            colorbar_label = cbar_label,
            figsize        = (8, 6),
            discrete       = True,
            levels         = levels,
            text_overlay   = text_overlay,
            path           = out_path,
        )

    def _plot_spatial_maps(
        self,
        maps_dict  : Dict[str, np.ndarray],
        keys       : List[str],
        map_titles : List[str],
        group_name : str,
        col_label  : str,
        out_dir    : Path,
        cmap       : str = "plasma",
    ) -> Dict[str, Path]:
        valid_pairs = [(k, t) for k, t in zip(keys, map_titles) if k in maps_dict]
        if not valid_pairs:
            self.logger.warning(f"Spatial map group '{group_name}' skipped: no data")
            return {}

        cmap_obj = self._cmap_with_bad(cmap)
        saved    : Dict[str, Path] = {}

        for key, title in valid_pairs:
            data  = maps_dict[key].astype(np.float32)
            Az, R = data.shape

            if not np.isfinite(data).any():
                self.logger.warning(f"Spatial map '{key}' in group '{group_name}' skipped: field is entirely masked, no active pixels")
                continue

            vmin, vmax = self._shared_clim(data)

            saved[key] = self._imshow_figure(
                data,
                x_label        = "range [px]",
                y_label        = "azimuth [px]",
                title          = title,
                cmap           = cmap_obj,
                vmin           = vmin,
                vmax           = vmax,
                extent         = [0, R, Az, 0],
                colorbar_label = col_label,
                figsize        = (8, 6),
                path           = out_dir / f"{key}.png",
            )

        return saved

    def _plot_r2_spatial_map(self, r2_map : np.ndarray, out_path : Path) -> Path:
        Az, R    = r2_map.shape
        cmap_obj = self._cmap_with_bad("RdYlGn")
        vmin     = float(np.nanpercentile(r2_map, 1.0))
        vmax     = 1.0

        return self._imshow_figure(
            r2_map,
            x_label        = "range [px]",
            y_label        = "azimuth [px]",
            title          = rf"Per-pixel $R^2$ of Gaussian fit  (colour floor at $p_1={vmin:.2f}$)",
            cmap           = cmap_obj,
            vmin           = vmin,
            vmax           = vmax,
            extent         = [0, R, Az, 0],
            colorbar_label = r"$R^2$",
            figsize        = (8, 6),
            path           = out_path,
        )

    def _plot_snr_map(self, snr_db_map : np.ndarray, out_path : Path) -> Path:
        Az, R    = snr_db_map.shape
        cmap_obj = self._cmap_with_bad("viridis")
        vmin     = float(np.nanpercentile(snr_db_map, 1.0))
        vmax     = float(np.nanpercentile(snr_db_map, 99.0))

        return self._imshow_figure(
            snr_db_map,
            x_label        = "range [px]",
            y_label        = "azimuth [px]",
            title          = "Per-pixel peak-to-floor profile contrast  (uncalibrated proxy, not calibrated SNR)",
            cmap           = cmap_obj,
            vmin           = vmin,
            vmax           = vmax,
            extent         = [0, R, Az, 0],
            colorbar_label = "peak-to-floor contrast [dB]",
            figsize        = (8, 6),
            path           = out_path,
        )


class DistributionPlotter(PlotBase):
    def __init__(self, n_gaussians : int, logger : Logger, amp_threshold : float, fig_dpi : int = 150, save_dpi : int = 300) -> None:
        self.n_gaussians   = n_gaussians
        self.logger        = logger
        self.amp_threshold = amp_threshold
        self.fig_dpi       = fig_dpi
        self.save_dpi      = save_dpi

    def _plot_r2_distribution(self, r2_map : np.ndarray, summary : dict, out_dir : Path) -> Dict[str, Path]:
        flat  = r2_map.reshape(-1).astype(np.float64)
        valid = flat[np.isfinite(flat)]

        qs       = [10, 25, 50, 75, 90]
        pvals    = np.percentile(valid, qs) if valid.size > 0 else [float("nan")] * 5
        pcolors  = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
        stats_tx = rf"mean={summary['r2_mean']:.4f},  median={summary['r2_median']:.4f},  neg\_frac={summary['r2_neg_frac']:.3f}"
        saved    : Dict[str, Path] = {}

        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        lo      = float(np.percentile(valid, 0.5))
        hi      = float(np.percentile(valid, 99.5))
        ax.hist(valid, bins=120, range=(lo, hi), density=True, color="#1f77b4", alpha=0.60, edgecolor="none", label="PDF")

        if valid.size > 2:
            kde = gaussian_kde(valid, bw_method="scott")
            xs  = np.linspace(lo, hi, 600)
            ax.plot(xs, kde(xs), color="black", lw=1.4, label="KDE")

        for q, pv, pc in zip(qs, pvals, pcolors):
            ax.axvline(pv, color=pc, lw=1.1, ls="--", label=f"$p_{{{q}}}={pv:.3f}$")

        ax.set_xlabel(r"coefficient of determination $R^2$")
        ax.set_ylabel(r"probability density")
        ax.set_title(r"$R^2$ distribution across all pixels" + "\n" + stats_tx)
        ax.legend(fontsize=8, framealpha=0.90, ncol=2)
        ax.grid(True, which="major", lw=0.3, alpha=0.40)
        fig.tight_layout()

        saved["r2_pdf"] = self._save(fig, out_dir / "r2_pdf.png")

        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        cdf_x   = np.sort(valid)
        cdf_y   = np.arange(1, valid.size + 1) / valid.size
        ax.plot(cdf_x, cdf_y, color="#1f77b4", lw=1.4)

        for q, pv, pc in zip(qs, pvals, pcolors):
            ax.axvline(pv, color=pc, lw=1.0, ls="--")
            ax.axhline(q / 100.0, color=pc, lw=0.6, ls=":")
            ax.text(pv + 0.005, q / 100.0 + 0.012, f"$p_{{{q}}}$", fontsize=7, color=pc)

        ax.set_xlabel(r"$R^2$")
        ax.set_ylabel(r"cumulative fraction")
        ax.set_title(r"$R^2$ empirical CDF")
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="major", lw=0.3, alpha=0.40)
        fig.tight_layout()

        saved["r2_cdf"] = self._save(fig, out_dir / "r2_cdf.png")

        return saved

    def _plot_parameter_distributions(self, parameters_array : np.ndarray, out_dir : Path) -> Dict[str, Path]:
        specs = [
            ("amp",   r"amplitude $A_k$",             "amp"),
            ("mu",    r"centroid height $\mu_k$ [m]", "mu"),
            ("sigma", r"spread $\sigma_k$ [m]",       "sigma"),
        ]
        saved: Dict[str, Path] = {}

        for tag, ylabel, file_tag in specs:
            fig, ax = plt.subplots(figsize=(max(6, 2.2 * self.n_gaussians), 5))

            data_by_slot : List[np.ndarray] = []
            labels       : List[str]        = []

            for k in range(self.n_gaussians):
                amp_flat = parameters_array[3 * k].reshape(-1)
                active   = amp_flat >= self.amp_threshold
                if tag == "amp":
                    ch_idx = 3 * k
                elif tag == "mu":
                    ch_idx = 3 * k + 1
                else:
                    ch_idx = 3 * k + 2
                vals = parameters_array[ch_idx].reshape(-1)
                vals = vals[active & np.isfinite(vals)]
                data_by_slot.append(vals)
                labels.append(f"$g_{{{k + 1}}}$")

            non_empty = [d for d in data_by_slot if d.size > 0]
            if not non_empty:
                plt.close(fig)
                continue

            palette   = [cm.tab10(i) for i in range(self.n_gaussians)]
            positions = list(range(1, self.n_gaussians + 1))

            self._violin_with_iqr(ax, data_by_slot, palette)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_xlabel("Gaussian slot")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Distribution of {ylabel}  (active pixels only, IQR box + violin)")
            ax.grid(True, axis="y", which="major", lw=0.3, alpha=0.40)
            fig.tight_layout()

            path                  = out_dir / f"{file_tag}_distribution.png"
            saved[f"{tag}_dist"]  = self._save(fig, path)

        return saved

    def _plot_param_joint_distributions(self, parameters_array : np.ndarray, out_dir : Path) -> Dict[str, Path]:
        amp_pool : List[np.ndarray] = []
        mu_pool  : List[np.ndarray] = []
        sig_pool : List[np.ndarray] = []

        for k in range(self.n_gaussians):
            amp_flat = parameters_array[3 * k].reshape(-1)
            active   = amp_flat >= self.amp_threshold
            amp_pool.append(amp_flat[active])
            mu_pool .append(parameters_array[3 * k + 1].reshape(-1)[active])
            sig_pool.append(parameters_array[3 * k + 2].reshape(-1)[active])

        amps = np.concatenate(amp_pool) if amp_pool else np.empty(0, dtype=np.float32)

        if amps.size == 0:
            self.logger.warning("Joint parameter distribution plots skipped: no active components")
            return {}

        mus      = np.concatenate(mu_pool)
        sigs     = np.concatenate(sig_pool)
        log_amps = np.log10(np.maximum(amps, 1e-12))

        pairs = [
            ("joint_mu_sigma",  mus,  sigs,     r"$\mu$ [m]",    r"$\sigma$ [m]"),
            ("joint_mu_amp",    mus,  log_amps, r"$\mu$ [m]",    r"$\log_{10} A$"),
            ("joint_sigma_amp", sigs, log_amps, r"$\sigma$ [m]", r"$\log_{10} A$"),
        ]

        saved : Dict[str, Path] = {}

        for name, x, y, x_label, y_label in pairs:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            xs, ys  = self._paired_subsample([x, y], 400_000)
            hb      = ax.hexbin(xs, ys, gridsize=70, bins="log", cmap="viridis", mincnt=1)
            fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02).set_label("component count")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{y_label} vs {x_label}  (active components, all slots pooled)")
            fig.tight_layout()

            saved[name] = self._save(fig, out_dir / f"{name}.png")

        return saved

    def _plot_k_ambiguity_distribution(self, metrics_dict : dict, per_k_summary : dict, out_dir : Path) -> Dict[str, Path]:
        rel    = metrics_dict["k_relative_margin_map"].reshape(-1)
        best_k = metrics_dict["best_k_map"].reshape(-1)
        ok     = np.isfinite(rel)

        if int(ok.sum()) == 0:
            self.logger.warning("K-selection ambiguity plots skipped: no data")
            return {}

        log_rel   = np.log10(np.maximum(rel[ok], 1e-9))
        bk_valid  = best_k[ok]
        threshold = float(per_k_summary["ambiguity_threshold"])
        saved     : Dict[str, Path] = {}

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        sample  = self._subsample(log_rel, 2_000_000)
        ax.hist(sample, bins=120, density=True, color="#1f77b4", alpha=0.65, edgecolor="none")
        ax.axvline(np.log10(threshold),       color="#d62728", lw=1.2, ls="--", label=f"ambiguity threshold ({threshold:.2f})")
        ax.axvline(float(np.median(log_rel)), color="black",   lw=1.2, ls=":",  label="median")
        ax.set_xlabel(r"$\log_{10}$ relative selection margin  $(\mathcal{L}_{2\mathrm{nd}} - \mathcal{L}_{K^*}) / \mathcal{L}_{K^*}$")
        ax.set_ylabel("probability density")
        ax.set_title("K-selection ambiguity distribution")
        ax.legend(framealpha=0.90)
        ax.grid(True, which="major", lw=0.3, alpha=0.40)

        amb_frac = per_k_summary["k_ambiguous_fraction"]
        if np.isfinite(amb_frac):
            ax.text(0.02, 0.98, f"ambiguous fraction: {amb_frac:.3f}", transform=ax.transAxes, fontsize=9, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.88))

        fig.tight_layout()
        saved["k_ambiguity_distribution"] = self._save(fig, out_dir / "k_ambiguity_distribution.png")

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        k_vals  = list(range(1, self.n_gaussians + 1))
        palette = [cm.tab10((k - 1) % 10) for k in k_vals]
        data    = [self._subsample(log_rel[bk_valid == k], 200_000, seed=k) for k in k_vals]
        self._violin_with_iqr(ax, data, palette)
        ax.axhline(np.log10(threshold), color="#d62728", lw=1.0, ls="--")
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"$K={k}$" for k in k_vals])
        ax.set_xlabel(r"selected model order $K^*$")
        ax.set_ylabel(r"$\log_{10}$ relative selection margin")
        ax.set_title("Ambiguity conditioned on selected model order")
        ax.grid(True, axis="y", which="major", lw=0.3, alpha=0.40)
        fig.tight_layout()

        saved["k_ambiguity_by_k"] = self._save(fig, out_dir / "k_ambiguity_by_k.png")

        return saved


class MetricsBarPlotter(PlotBase):
    def __init__(self, n_gaussians : int, logger : Logger, fig_dpi : int = 150, save_dpi : int = 300) -> None:
        self.n_gaussians = n_gaussians
        self.logger      = logger
        self.fig_dpi     = fig_dpi
        self.save_dpi    = save_dpi

    def _plot_global_metrics_summary(self, summary : dict, out_dir : Path) -> Dict[str, Path]:
        qs      = [10, 25, 50, 75, 90]
        pvals   = [summary[f"r2_p{q}"] for q in qs]
        pcolors = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
        saved   : Dict[str, Path] = {}

        fig, ax = plt.subplots(figsize=(6.4, 5))
        bars    = ax.bar(range(len(qs)), pvals, color=pcolors, alpha=0.80, edgecolor="white", lw=0.5)
        ax.set_xticks(range(len(qs)))
        ax.set_xticklabels([f"$p_{{{q}}}$" for q in qs])
        ax.set_ylabel(r"$R^2$")
        ax.set_ylim(bottom=min(0.0, min(v for v in pvals if np.isfinite(v))) - 0.05)
        ax.set_title(r"$R^2$ percentiles across all pixels")
        r2_mean = summary["r2_mean"]

        if np.isfinite(r2_mean):
            ax.axhline(r2_mean, color="black", lw=1.0, ls="--", label=f"mean $= {r2_mean:.4f}$")
            ax.legend(framealpha=0.90)

        ax.grid(True, axis="y", which="major", lw=0.3, alpha=0.40)

        for bar, val in zip(bars, pvals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        saved["r2_percentiles"] = self._save(fig, out_dir / "r2_percentiles.png")

        fig, ax = plt.subplots(figsize=(6.4, 5))
        n_K     = self.n_gaussians
        k_vals  = list(range(1, n_K + 1))
        fracs   = [summary[f"frac_{k}_fitted"] for k in k_vals]
        cols    = [cm.tab10((k - 1) % 10) for k in k_vals]
        bars    = ax.bar(k_vals, fracs, color=cols, alpha=0.80, edgecolor="white", lw=0.5)

        frac_unfitted = summary["frac_0_active"]
        n_fitted      = summary["n_fitted"]

        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"$K={k}$" for k in k_vals])
        ax.set_ylabel("fraction of fitted pixels")
        ax.set_ylim(0, 1.08)
        ax.set_title("Active-Gaussian count distribution over fitted pixels")
        ax.grid(True, axis="y", which="major", lw=0.3, alpha=0.40)

        for bar, val in zip(bars, fracs):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        if np.isfinite(frac_unfitted):
            note = f"unfitted ($K=0$) share of all pixels $= {frac_unfitted:.3f}$"
            if np.isfinite(n_fitted):
                note = f"{note}\nfitted pixels $= {int(n_fitted)}$"
            ax.text(0.98, 0.98, note, transform=ax.transAxes, fontsize=8, va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.88))

        fig.tight_layout()
        saved["active_count_distribution"] = self._save(fig, out_dir / "active_count_distribution.png")

        return saved

    def _plot_mse_penalty_per_k(self, mse_per_k : np.ndarray, per_k_summary : dict, out_dir : Path) -> Dict[str, Path]:
        k_vals  = list(range(1, self.n_gaussians + 1))
        palette = [cm.tab10((k - 1) % 10) for k in k_vals]
        saved   : Dict[str, Path] = {}

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        data    = [np.log10(np.maximum(self._subsample(mse_per_k[k - 1].reshape(-1), 200_000, seed=k), 1e-12)) for k in k_vals]
        self._violin_with_iqr(ax, data, palette)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"$K={k}$" for k in k_vals])
        ax.set_xlabel(r"model order $K$")
        ax.set_ylabel(r"$\log_{10}\,\mathrm{MSE}$")
        ax.set_title("Residual MSE distribution per model order")
        ax.grid(True, axis="y", which="major", lw=0.3, alpha=0.40)
        fig.tight_layout()

        saved["mse_per_k_distribution"] = self._save(fig, out_dir / "mse_per_k_distribution.png")

        fig, ax   = plt.subplots(figsize=(6.4, 4.8))
        mse_means = [per_k_summary[f"k{k}_mse_mean"]       for k in k_vals]
        pty_means = [per_k_summary[f"k{k}_penalty_mean"]   for k in k_vals]
        tot_means = [per_k_summary[f"k{k}_penalised_mean"] for k in k_vals]
        ax.bar(k_vals, mse_means,                   color="#1f77b4", alpha=0.78, edgecolor="white", lw=0.5, label="mean MSE (peak-normalised profile)")
        ax.bar(k_vals, pty_means, bottom=mse_means, color="#d62728", alpha=0.78, edgecolor="white", lw=0.5, label=r"mean penalty $\lambda_K K \bar{A}_{\mathrm{norm}}$")
        ax.plot(k_vals, tot_means, color="black", lw=1.3, ls="--", marker="o", ms=4, label="mean penalised score")
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"$K={k}$" for k in k_vals])
        ax.set_xlabel(r"model order $K$")
        ax.set_ylabel("score (peak-normalised profile units)")
        ax.set_title("Penalised score decomposition per model order")
        ax.legend(framealpha=0.90)
        ax.grid(True, axis="y", which="major", lw=0.3, alpha=0.40)
        fig.tight_layout()

        saved["penalised_score_decomposition"] = self._save(fig, out_dir / "penalised_score_decomposition.png")

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        wins    = [per_k_summary[f"k{k}_win_fraction"] for k in k_vals]
        bars    = ax.bar(k_vals, wins, color=palette, alpha=0.80, edgecolor="white", lw=0.5)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f"$K={k}$" for k in k_vals])
        ax.set_xlabel(r"model order $K$")
        ax.set_ylabel("fraction of active pixels")
        ax.set_ylim(0, 1.08)
        ax.set_title("Selected model order distribution")
        ax.grid(True, axis="y", which="major", lw=0.3, alpha=0.40)

        for bar, val in zip(bars, wins):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        saved["selected_k_distribution"] = self._save(fig, out_dir / "selected_k_distribution.png")

        return saved

    def _plot_snr_vs_fit_quality(
        self,
        snr_db_map     : np.ndarray,
        r2_map         : np.ndarray,
        best_k_map     : Optional[np.ndarray],
        rel_margin_map : Optional[np.ndarray],
        snr_summary    : dict,
        out_dir        : Path,
    ) -> Dict[str, Path]:
        saved : Dict[str, Path] = {}

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        s, r    = self._paired_subsample([snr_db_map, np.maximum(r2_map, -1.0)], 400_000)
        hb      = ax.hexbin(s, r, gridsize=70, bins="log", cmap="magma", mincnt=1)
        fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02).set_label("pixel count")

        centers, medians = self._binned_median(s, r)
        ax.plot(centers, medians, color="cyan", lw=1.6, label=r"binned median $R^2$")
        ax.set_xlabel("peak-to-floor contrast [dB]")
        ax.set_ylabel(r"$R^2$  (floored at $-1$)")
        ax.set_title("Fit quality vs peak-to-floor contrast  (uncalibrated proxy, not calibrated SNR)")
        ax.legend(loc="lower right", framealpha=0.90)

        pearson  = float(pearsonr(s, r)[0])  if s.size > 1 else float("nan")
        spearman = float(spearmanr(s, r)[0]) if s.size > 1 else float("nan")
        ax.text(0.02, 0.98, f"Pearson $r$ = {pearson:.3f}\nSpearman $\\rho$ = {spearman:.3f}\n(on plotted, $-1$-floored $R^2$)", transform=ax.transAxes, fontsize=9, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.88))

        fig.tight_layout()
        saved["snr_vs_r2"] = self._save(fig, out_dir / "snr_vs_r2.png")

        if best_k_map is not None:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            k_vals  = list(range(1, self.n_gaussians + 1))
            palette = [cm.tab10((k - 1) % 10) for k in k_vals]
            snr_f   = snr_db_map.reshape(-1)
            bk_f    = best_k_map.reshape(-1)
            data    = [self._subsample(snr_f[bk_f == k], 200_000, seed=k) for k in k_vals]
            self._violin_with_iqr(ax, data, palette)
            ax.set_xticks(k_vals)
            ax.set_xticklabels([f"$K={k}$" for k in k_vals])
            ax.set_xlabel(r"selected model order $K^*$")
            ax.set_ylabel("peak-to-floor contrast [dB]")
            ax.set_title("Peak-to-floor contrast conditioned on selected model order  (uncalibrated proxy, not calibrated SNR)")
            ax.grid(True, axis="y", which="major", lw=0.3, alpha=0.40)
            fig.tight_layout()

            saved["snr_by_selected_k"] = self._save(fig, out_dir / "snr_by_selected_k.png")

        if rel_margin_map is not None:
            fig, ax  = plt.subplots(figsize=(6.4, 4.8))
            s_m, rel = self._paired_subsample([snr_db_map, rel_margin_map], 400_000)
            log_rel  = np.log10(np.maximum(rel, 1e-9))
            hb       = ax.hexbin(s_m, log_rel, gridsize=70, bins="log", cmap="magma", mincnt=1)
            fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02).set_label("pixel count")

            centers, medians = self._binned_median(s_m, log_rel)
            ax.plot(centers, medians, color="cyan", lw=1.6, label="binned median margin")
            ax.set_xlabel("peak-to-floor contrast [dB]")
            ax.set_ylabel(r"$\log_{10}$ relative selection margin")
            ax.set_title("K-selection ambiguity vs peak-to-floor contrast  (uncalibrated proxy, not calibrated SNR)")
            ax.legend(loc="lower right", framealpha=0.90)
            fig.tight_layout()

            saved["snr_vs_k_ambiguity"] = self._save(fig, out_dir / "snr_vs_k_ambiguity.png")

        return saved


class ExampleFitPlotter(PlotBase):
    def __init__(
        self,
        n_gaussians      : int,
        logger           : Logger,
        threshold_factor : float,
        truncation_index : int,
        n_fits_per_k     : int,
        amp_threshold    : float,
        fig_dpi          : int = 150,
        save_dpi         : int = 300,
    ) -> None:
        self.n_gaussians      = n_gaussians
        self.logger           = logger
        self.threshold_factor = threshold_factor
        self.truncation_index = truncation_index
        self.n_fits_per_k     = n_fits_per_k
        self.amp_threshold    = amp_threshold
        self.fig_dpi          = fig_dpi
        self.save_dpi         = save_dpi

    def _select_pixels_by_k(self, best_k_map : np.ndarray, r2_map : np.ndarray, seed : int = 42) -> Dict[int, np.ndarray]:
        rng    = np.random.default_rng(seed)
        flat_k = best_k_map.reshape(-1)
        flat_r = r2_map.reshape(-1)
        H, W   = best_k_map.shape
        finite = np.isfinite(flat_r)

        groups : Dict[int, np.ndarray] = {}

        for K in range(1, self.n_gaussians + 1):
            idx = np.where(finite & (flat_k == K))[0]

            if idx.size == 0:
                groups[K] = np.empty((0, 2), dtype=np.int32)
                continue

            chosen    = rng.choice(idx, size=min(self.n_fits_per_k, idx.size), replace=False)
            groups[K] = np.stack([(chosen // W).astype(np.int32), (chosen % W).astype(np.int32)], axis=1)

        return groups

    def _reconstruct_pixel(self, params : np.ndarray, height_axis : np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        return GaussianMixture.evaluate_pixel(params, height_axis, self.n_gaussians)

    def _extract_pixel_profiles(self, tomogram_path : Path, all_pixels : np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        tomogram_mmap = np.load(str(tomogram_path), mmap_mode="r")

        pixel_profiles : Dict[Tuple[int, int], np.ndarray] = {}
        for az, rg in all_pixels.tolist():
            raw                      = np.abs(np.array(tomogram_mmap[:, az, rg])).astype(np.float32)
            pixel_profiles[(az, rg)] = ProfilePreprocessor.apply(raw, self.threshold_factor, self.truncation_index)

        del tomogram_mmap
        gc.collect()

        return pixel_profiles

    def _plot_example_fits(
        self,
        parameters_array : np.ndarray,
        pixel_profiles   : Dict[Tuple[int, int], np.ndarray],
        height_axis      : np.ndarray,
        pixels_by_k      : Dict[int, np.ndarray],
        r2_map           : np.ndarray,
        out_dir          : Path,
    ) -> Dict[str, Path]:
        comp_colors = [cm.tab10(i) for i in range(self.n_gaussians)]
        saved       : Dict[str, Path] = {}

        for K, pixels in pixels_by_k.items():
            if pixels.shape[0] == 0:
                continue

            k_color = cm.tab10((K - 1) % 10)
            k_label = rf"$K^*={K}$  ({K} Gaussian{'s' if K > 1 else ''})"
            k_dir   = out_dir / f"k{K}"
            k_dir.mkdir(parents=True, exist_ok=True)

            for az, rg in pixels.tolist():
                profile = pixel_profiles.get((az, rg))
                if profile is None:
                    continue

                profile      = profile.astype(np.float64)
                params       = parameters_array[:, az, rg].astype(np.float64)
                total, comps = self._reconstruct_pixel(params, height_axis.astype(np.float64))
                residual     = profile - total
                r2_val       = float(r2_map[az, rg]) if np.isfinite(r2_map[az, rg]) else float("nan")

                fig, ax = plt.subplots(figsize=(5.6, 4.4))
                ax.plot(height_axis, profile, color="black",   lw=1.5, label="data", zorder=4)
                ax.plot(height_axis, total,   color=k_color,   lw=1.4, ls="--", label="fit", zorder=5)

                for k, comp in enumerate(comps):
                    if float(params[3 * k]) >= self.amp_threshold:
                        ax.fill_between(height_axis, comp, alpha=0.20, color=comp_colors[k], zorder=2)
                        ax.plot(height_axis, comp, color=comp_colors[k], lw=0.9, alpha=0.85, label=f"$g_{{{k + 1}}}$")

                ax.set_title(f"Example fit — {k_label}\naz={az},  rg={rg},  $R^2={r2_val:.3f}$", fontsize=10)
                ax.set_xlabel(r"height $h$ [m]")
                ax.set_ylabel(r"backscatter intensity")
                ax.grid(True, which="major", lw=0.25, alpha=0.40)
                ax.legend(fontsize=8, framealpha=0.90, ncol=2)
                fig.tight_layout()

                saved[f"k{K}_az{az}_rg{rg}_fit"] = self._save(fig, k_dir / f"az{az}_rg{rg}_fit.png")

                fig, ax = plt.subplots(figsize=(5.6, 2.8))
                ax.plot(height_axis, residual, color="0.35", lw=0.9, zorder=3)
                ax.axhline(0.0, color="black", lw=0.7)
                ax.fill_between(height_axis, residual, 0.0, where=residual >= 0, color="#1f77b4", alpha=0.25, zorder=2)
                ax.fill_between(height_axis, residual, 0.0, where=residual < 0,  color="#d62728", alpha=0.25, zorder=2)

                ax.set_title(f"Fit residual — az={az},  rg={rg},  $R^2={r2_val:.3f}$", fontsize=10)
                ax.set_xlabel(r"height $h$ [m]")
                ax.set_ylabel(r"$\varepsilon = \mathrm{data} - \mathrm{fit}$")
                ax.grid(True, which="major", lw=0.25, alpha=0.40)
                fig.tight_layout()

                saved[f"k{K}_az{az}_rg{rg}_residual"] = self._save(fig, k_dir / f"az{az}_rg{rg}_residual.png")

        return saved

    def run(
        self,
        parameters_array : np.ndarray,
        best_k_map       : np.ndarray,
        r2_map           : np.ndarray,
        height_axis      : np.ndarray,
        tomogram_path    : Path,
        out_dir          : Path,
    ) -> Dict[str, Path]:
        self.logger.subsection("Loading tomogram for example fit plots (memory-mapped)")

        pixels_by_k = self._select_pixels_by_k(best_k_map, r2_map)
        non_empty   = [px for px in pixels_by_k.values() if px.shape[0] > 0]
        all_pixels  = np.concatenate(non_empty, axis=0) if non_empty else np.empty((0, 2), dtype=np.int32)

        self.logger.subsection(f"Extracting {all_pixels.shape[0]} pixel profiles for example fits")
        pixel_profiles = self._extract_pixel_profiles(tomogram_path, all_pixels)

        self.logger.subsection(f"Plotting example fits  ({self.n_fits_per_k} pixels × up to {self.n_gaussians} K groups)")
        return self._plot_example_fits(parameters_array, pixel_profiles, height_axis, pixels_by_k, r2_map, out_dir)


class FittingResultPlotter(PlotBase):
    def __init__(
        self,
        output_directory : Path,
        n_gaussians      : int,
        logger           : Logger,
        threshold_factor : float,
        truncation_index : int,
        fig_dpi          : int   = 150,
        save_dpi         : int   = 300,
        n_fits_per_k     : int   = 5,
        amp_threshold    : float = 1e-3,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.n_gaussians      = n_gaussians
        self.logger           = logger
        self.threshold_factor = threshold_factor
        self.truncation_index = truncation_index
        self.fig_dpi          = fig_dpi
        self.save_dpi         = save_dpi
        self.n_fits_per_k     = n_fits_per_k
        self.amp_threshold    = amp_threshold
        self._images_dir      = self.output_directory / "images"

        self.spatial_plotter      = SpatialMapPlotter(n_gaussians=n_gaussians, logger=logger, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.distribution_plotter = DistributionPlotter(n_gaussians=n_gaussians, logger=logger, amp_threshold=amp_threshold, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.metrics_bar_plotter  = MetricsBarPlotter(n_gaussians=n_gaussians, logger=logger, fig_dpi=fig_dpi, save_dpi=save_dpi)
        self.example_fit_plotter  = ExampleFitPlotter(
            n_gaussians      = n_gaussians,
            logger           = logger,
            threshold_factor = threshold_factor,
            truncation_index = truncation_index,
            n_fits_per_k     = n_fits_per_k,
            amp_threshold    = amp_threshold,
            fig_dpi          = fig_dpi,
            save_dpi         = save_dpi,
        )

    def _setup_output_dirs(self) -> Dict[str, Path]:
        dirs = {
            "colormaps"    : self._images_dir / "colormaps",
            "example_fits" : self._images_dir / "example_fits",
            "distributions": self._images_dir / "distributions",
            "metrics"      : self._images_dir / "metrics",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def _run_spatial_maps(self, metrics_dict : dict, r2_map : np.ndarray, activity_map : np.ndarray, dirs : Dict[str, Path]) -> Dict[str, Path]:
        saved : Dict[str, Path] = {}

        self.logger.subsection("Plotting active-Gaussian count colormap")
        saved["n_gaussians_map"] = self.spatial_plotter._plot_discrete_k_map(
            activity_map,
            rf"Number of active Gaussians per pixel  ($A_k \geq {self.amp_threshold:g}$)",
            r"active Gaussians $K$",
            dirs["colormaps"] / "n_gaussians_map.png",
        )

        if "best_k_map" in metrics_dict:
            self.logger.subsection("Plotting selected model-order colormap")
            saved["best_k_map"] = self.spatial_plotter._plot_discrete_k_map(
                metrics_dict["best_k_map"],
                r"Selected model order $K^*$ per pixel  (penalised-MSE minimiser)",
                r"selected $K^*$",
                dirs["colormaps"] / "best_k_map.png",
            )

        self.logger.subsection("Plotting R² spatial map")
        saved["r2_spatial_map"] = self.spatial_plotter._plot_r2_spatial_map(r2_map, dirs["colormaps"] / "r2_map.png")

        self.logger.subsection("Plotting amplitude spatial maps")
        amp_keys   = [f"amp_{k}"   for k in range(self.n_gaussians)]
        amp_titles = [f"$A_{{{k + 1}}}$  amplitude  (inactive pixels masked)" for k in range(self.n_gaussians)]
        saved.update(self.spatial_plotter._plot_spatial_maps(
            metrics_dict, amp_keys, amp_titles,
            "Gaussian amplitude maps",
            r"$A_k$",
            dirs["colormaps"] / "amplitude_maps",
            cmap="plasma",
        ))

        self.logger.subsection("Plotting height-centroid (μ) spatial maps")
        mu_keys   = [f"mu_{k}"    for k in range(self.n_gaussians)]
        mu_titles = [rf"$\mu_{{{k + 1}}}$  centroid [m]  (inactive pixels masked)" for k in range(self.n_gaussians)]
        saved.update(self.spatial_plotter._plot_spatial_maps(
            metrics_dict, mu_keys, mu_titles,
            r"Gaussian centroid height maps",
            r"$\mu_k$ [m]",
            dirs["colormaps"] / "mu_maps",
            cmap="RdYlGn",
        ))

        self.logger.subsection("Plotting sigma spatial maps")
        sig_keys   = [f"sigma_{k}" for k in range(self.n_gaussians)]
        sig_titles = [rf"$\sigma_{{{k + 1}}}$  spread [m]  (inactive pixels masked)" for k in range(self.n_gaussians)]
        saved.update(self.spatial_plotter._plot_spatial_maps(
            metrics_dict, sig_keys, sig_titles,
            r"Gaussian spread maps",
            r"$\sigma_k$ [m]",
            dirs["colormaps"] / "sigma_maps",
            cmap="viridis",
        ))

        if self.n_gaussians >= 2:
            self.logger.subsection("Plotting μ-separation maps")
            sep_keys   = [f"mu_sep_{k}_{k + 1}" for k in range(self.n_gaussians - 1)]
            sep_titles = [rf"$|\mu_{{{k + 2}}} - \mu_{{{k + 1}}}|$  [m]  (both active)" for k in range(self.n_gaussians - 1)]
            saved.update(self.spatial_plotter._plot_spatial_maps(
                metrics_dict, sep_keys, sep_titles,
                r"Adjacent centroid separation maps",
                "separation [m]",
                dirs["colormaps"] / "mu_separation_maps",
                cmap="magma",
            ))

        return saved

    def _run_distributions(self, parameters_array : np.ndarray, r2_map : np.ndarray, summary : dict, dirs : Dict[str, Path]) -> Dict[str, Path]:
        saved : Dict[str, Path] = {}

        self.logger.subsection("Plotting R² distribution and CDF")
        saved.update(self.distribution_plotter._plot_r2_distribution(r2_map, summary, dirs["distributions"]))

        self.logger.subsection("Plotting parameter distributions")
        saved.update(self.distribution_plotter._plot_parameter_distributions(parameters_array, dirs["distributions"]))

        self.logger.subsection("Plotting joint parameter distributions")
        saved.update(self.distribution_plotter._plot_param_joint_distributions(parameters_array, dirs["distributions"]))

        return saved

    def _run_metrics_and_snr(self, metrics_dict : dict, r2_map : np.ndarray, summary : dict, dirs : Dict[str, Path]) -> Dict[str, Path]:
        saved : Dict[str, Path] = {}

        self.logger.subsection("Plotting global metrics summary")
        saved.update(self.metrics_bar_plotter._plot_global_metrics_summary(summary, dirs["metrics"]))

        snr_db_map    = metrics_dict["snr_db_map"]
        per_k_summary = metrics_dict["per_k_summary"]

        if snr_db_map is not None:
            self.logger.subsection("Plotting SNR spatial map")
            saved["snr_map"] = self.spatial_plotter._plot_snr_map(snr_db_map, dirs["colormaps"] / "snr_map.png")

        if "mse_per_k" in metrics_dict:
            self.logger.subsection("Plotting per-K MSE and penalty decomposition")
            saved.update(self.metrics_bar_plotter._plot_mse_penalty_per_k(metrics_dict["mse_per_k"], per_k_summary, dirs["metrics"]))

        if "k_relative_margin_map" in metrics_dict:
            self.logger.subsection("Plotting K-selection ambiguity maps and distribution")
            margin_keys   = ["k_margin_prev_map", "k_margin_next_map", "k_relative_margin_map"]
            margin_titles = [r"margin to $K^*-1$  (small = ambiguous choice)", r"margin to $K^*+1$  (small = ambiguous choice)", "relative margin to runner-up  (small = ambiguous choice)"]

            saved.update(self.spatial_plotter._plot_spatial_maps(
                metrics_dict, margin_keys, margin_titles,
                r"K-selection margin maps",
                "penalised-score margin",
                dirs["colormaps"] / "k_ambiguity_maps",
                cmap="cividis",
            ))
            saved.update(self.distribution_plotter._plot_k_ambiguity_distribution(metrics_dict, per_k_summary, dirs["distributions"]))

        if snr_db_map is not None:
            self.logger.subsection("Plotting SNR against fit quality and K-selection ambiguity")
            best_k_map     = metrics_dict["best_k_map"]            if "best_k_map"            in metrics_dict else None
            rel_margin_map = metrics_dict["k_relative_margin_map"] if "k_relative_margin_map" in metrics_dict else None
            saved.update(self.metrics_bar_plotter._plot_snr_vs_fit_quality(
                snr_db_map, r2_map,
                best_k_map,
                rel_margin_map,
                metrics_dict["snr_summary"],
                dirs["metrics"],
            ))

        return saved

    def _run_example_fits(self, parameters_array : np.ndarray, metrics_dict : dict, r2_map : np.ndarray, height_axis : np.ndarray, tomogram_path : Path, dirs : Dict[str, Path]) -> Dict[str, Path]:
        if "best_k_map" not in metrics_dict:
            self.logger.warning("Example fit plots skipped: best_k_map unavailable")
            return {}

        fits = self.example_fit_plotter.run(parameters_array, metrics_dict["best_k_map"], r2_map, height_axis, tomogram_path, dirs["example_fits"])

        saved : Dict[str, Path] = {}
        for key, path in fits.items():
            saved[f"example_fit_{key}"] = path

        return saved

    def run(self, parameters_array : np.ndarray, metrics_dict : dict, metadata : dict, tomogram_path : Path) -> Dict[str, Path]:
        self.logger.section("[Fitting Result Plots]")
        self._apply_style()
        dirs = self._setup_output_dirs()

        r2_map       = metrics_dict["r2_map"]
        activity_map = metrics_dict["activity_map"]
        height_axis  = metrics_dict["height_axis"]
        summary      = metrics_dict["global_summary"]

        saved : Dict[str, Path] = {}

        saved.update(self._run_spatial_maps(metrics_dict, r2_map, activity_map, dirs))
        saved.update(self._run_distributions(parameters_array, r2_map, summary, dirs))
        saved.update(self._run_metrics_and_snr(metrics_dict, r2_map, summary, dirs))
        saved.update(self._run_example_fits(parameters_array, metrics_dict, r2_map, height_axis, tomogram_path, dirs))

        self.logger.subsection(f"Saved {len(saved)} figures → {self._images_dir}")
        return saved
