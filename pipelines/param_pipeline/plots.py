from __future__ import annotations

import gc
from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm     as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy             as np
from scipy.stats         import gaussian_kde

from tools.logger import Logger


_SCIENTIFIC_RC: dict = {
    "font.family"         : "serif",
    "font.serif"          : ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset"    : "dejavuserif",
    "font.size"           : 11,
    "axes.titlesize"      : 12,
    "axes.labelsize"      : 11,
    "xtick.labelsize"     : 9,
    "ytick.labelsize"     : 9,
    "legend.fontsize"     : 9,
    "axes.linewidth"      : 0.8,
    "xtick.direction"     : "in",
    "ytick.direction"     : "in",
    "xtick.top"           : True,
    "ytick.right"         : True,
    "xtick.minor.visible" : True,
    "ytick.minor.visible" : True,
    "image.interpolation" : "nearest",
    "savefig.bbox"        : "tight",
    "pdf.fonttype"        : 42,
    "ps.fonttype"         : 42,
}

_TIER_COLOR  = {"low": "#d62728", "mid": "#ff7f0e", "high": "#2ca02c"}
_TIER_LABEL  = {"low": r"Low $R^2$  ($\leq p_{25}$)", "mid": r"Mid $R^2$  ($p_{40}$–$p_{60}$)", "high": r"High $R^2$ ($\geq p_{75}$)"}


class FittingResultPlotter:
    def __init__(
        self,
        output_directory : Path,
        n_gaussians      : int,
        logger           : Logger,
        fig_dpi          : int   = 150,
        save_dpi         : int   = 300,
        n_fits_per_tier  : int   = 5,
        amp_threshold    : float = 1e-3,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.n_gaussians      = n_gaussians
        self.logger           = logger
        self.fig_dpi          = fig_dpi
        self.save_dpi         = save_dpi
        self.n_fits_per_tier  = n_fits_per_tier
        self.amp_threshold    = amp_threshold
        self._images_dir      = self.output_directory / "images"

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

    def _apply_style(self) -> None:
        plt.rcParams.update(_SCIENTIFIC_RC)
        plt.rcParams["figure.dpi"]  = self.fig_dpi
        plt.rcParams["savefig.dpi"] = self.save_dpi

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _shared_clim(*arrays: np.ndarray, q_lo: float = 1.0, q_hi: float = 99.0) -> Tuple[float, float]:
        flat = np.concatenate([a.reshape(-1) for a in arrays])
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return (0.0, 1.0)
        return float(np.percentile(flat, q_lo)), float(np.percentile(flat, q_hi))

    @staticmethod
    def _cmap_with_bad(name: str, bad_color: str = "0.88") -> mcolors.Colormap:
        cmap = plt.cm.get_cmap(name).copy()
        cmap.set_bad(color=bad_color)
        return cmap

    def _select_pixels_by_r2_tiers(self, r2_map : np.ndarray, seed : int = 42) -> Dict[str, np.ndarray]:
        rng   = np.random.default_rng(seed)
        flat  = r2_map.reshape(-1)
        H, W  = r2_map.shape
        valid = np.where(np.isfinite(flat))[0]
        r2_v  = flat[valid]

        p25, p40, p60, p75 = np.percentile(r2_v, [25, 40, 60, 75])

        def _sample(mask: np.ndarray, n: int) -> np.ndarray:
            idx = valid[mask]
            if idx.size == 0:
                return np.empty((0, 2), dtype=np.int32)
            chosen = rng.choice(idx, size=min(n, idx.size), replace=False)
            return np.stack([(chosen // W).astype(np.int32), (chosen % W).astype(np.int32)], axis=1)

        return {
            "low"  : _sample(r2_v <= p25,                          self.n_fits_per_tier),
            "mid"  : _sample((r2_v >= p40) & (r2_v <= p60),        self.n_fits_per_tier),
            "high" : _sample(r2_v >= p75,                          self.n_fits_per_tier),
        }

    def _reconstruct_pixel(self, params : np.ndarray, height_axis : np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        components : List[np.ndarray] = []
        total      = np.zeros_like(height_axis, dtype=np.float64)
        for k in range(self.n_gaussians):
            a   = float(params[3 * k    ])
            mu  = float(params[3 * k + 1])
            sig = float(params[3 * k + 2])
            c   = a * np.exp(-((height_axis - mu) ** 2) / (2.0 * sig ** 2 + 1e-12))
            components.append(c)
            total += c
        
        return total, components

    def _plot_n_gaussians_map(self, activity_map : np.ndarray, out_path : Path) -> Path:
        Az, R    = activity_map.shape
        n_K      = self.n_gaussians
        levels   = list(range(n_K + 2))
        palette  = plt.cm.get_cmap("tab10", len(levels))
        cmap     = mcolors.ListedColormap([palette(i) for i in range(len(levels))])
        bounds   = [l - 0.5 for l in levels] + [levels[-1] + 0.5]
        norm     = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax  = plt.subplots(figsize=(8, 6))
        im       = ax.imshow(activity_map, cmap=cmap, norm=norm, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(r"Number of active Gaussians per pixel  ($A_k \geq 10^{-3}$)")

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=levels, boundaries=bounds)
        cb.set_label(r"active Gaussians $K$")
        cb.ax.set_yticklabels([str(v) for v in levels])

        n_total    = activity_map.size
        text_lines = [f"$K={k}$: {(activity_map == k).sum() / n_total * 100:.1f}\\%" for k in levels if (activity_map == k).sum() > 0]
        ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes, fontsize=8, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.88))

        fig.tight_layout()
        return self._save(fig, out_path)

    def _plot_spatial_maps(
        self,
        maps_dict  : Dict[str, np.ndarray],
        keys       : List[str],
        row_titles : List[str],
        fig_title  : str,
        col_label  : str,
        out_path   : Path,
        cmap       : str = "plasma",
    ) -> Path:
        valid_pairs = [(k, t) for k, t in zip(keys, row_titles) if k in maps_dict]
        if not valid_pairs:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.set_axis_off()
            ax.set_title(fig_title + "\n(no data)")
            return self._save(fig, out_path)

        n       = len(valid_pairs)
        arr0    = maps_dict[valid_pairs[0][0]]
        Az, R   = arr0.shape
        cmap_obj = self._cmap_with_bad(cmap)

        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)

        for col, (key, title) in enumerate(valid_pairs):
            ax   = axes[0, col]
            data = maps_dict[key].astype(np.float32)
            vmin, vmax = self._shared_clim(data)
            im   = ax.imshow(data, cmap=cmap_obj, vmin=vmin, vmax=vmax, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("range [px]")
            if col == 0:
                ax.set_ylabel("azimuth [px]")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(col_label)

        fig.suptitle(fig_title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return self._save(fig, out_path)

    def _plot_example_fits(
        self,
        parameters_array : np.ndarray,                                              
        pixel_profiles   : Dict[Tuple[int, int], np.ndarray],                      
        height_axis      : np.ndarray,                                              
        pixels_by_tier   : Dict[str, np.ndarray],                                  
        r2_map           : np.ndarray,                                             
        out_dir          : Path,
    ) -> Dict[str, Path]:
        comp_colors = [cm.tab10(i) for i in range(self.n_gaussians)]
        saved       : Dict[str, Path] = {}

        for tier, pixels in pixels_by_tier.items():
            if pixels.shape[0] == 0:
                continue

            n_px  = pixels.shape[0]
            fig   = plt.figure(figsize=(3.8 * n_px, 5.8), constrained_layout=True)
            gs    = fig.add_gridspec(2, n_px, height_ratios=[3, 1])

            for col, (az, rg) in enumerate(pixels.tolist()):
                profile  = pixel_profiles.get((az, rg))
                if profile is None:
                    continue
                profile  = profile.astype(np.float64)
                params   = parameters_array[:, az, rg].astype(np.float64)
                total, comps = self._reconstruct_pixel(params, height_axis.astype(np.float64))
                residual = profile - total
                r2_val   = float(r2_map[az, rg]) if np.isfinite(r2_map[az, rg]) else float("nan")

                ax_top = fig.add_subplot(gs[0, col])
                ax_bot = fig.add_subplot(gs[1, col], sharex=ax_top)

                ax_top.plot(height_axis, profile, color="black",               lw=1.5, label="data",  zorder=4)
                ax_top.plot(height_axis, total,   color=_TIER_COLOR[tier],     lw=1.4, ls="--", label="fit", zorder=5)

                for k, comp in enumerate(comps):
                    if float(params[3 * k]) >= self.amp_threshold:
                        ax_top.fill_between(height_axis, comp, alpha=0.20, color=comp_colors[k], zorder=2)
                        ax_top.plot(height_axis, comp, color=comp_colors[k], lw=0.9, alpha=0.85, label=f"$g_{{{k + 1}}}$" if col == 0 else "_nolegend_")

                ax_top.set_title(f"az={az},  rg={rg}\n$R^2={r2_val:.3f}$", fontsize=9)
                ax_top.tick_params(labelbottom=False)
                ax_top.grid(True, which="major", lw=0.25, alpha=0.40)
                if col == 0:
                    ax_top.set_ylabel(r"backscatter intensity")

                ax_bot.plot(height_axis, residual, color="0.35", lw=0.9, zorder=3)
                ax_bot.axhline(0.0, color="black", lw=0.7)
                ax_bot.fill_between(height_axis, residual, 0.0, where=residual >= 0, color="#1f77b4", alpha=0.25, zorder=2)
                ax_bot.fill_between(height_axis, residual, 0.0, where=residual < 0,  color="#d62728", alpha=0.25, zorder=2)
                ax_bot.set_xlabel(r"height $h$ [m]", fontsize=9)
                ax_bot.grid(True, which="major", lw=0.25, alpha=0.40)
                if col == 0:
                    ax_bot.set_ylabel(r"$\varepsilon = \mathrm{data} - \mathrm{fit}$", fontsize=8)

            handles, leg_labels = fig.axes[0].get_legend_handles_labels()
            fig.legend(handles, leg_labels, loc="upper right", fontsize=8, framealpha=0.90, ncol=2 + self.n_gaussians)
            fig.suptitle(f"Example fits — {_TIER_LABEL[tier]}", fontsize=13, y=1.02)

            path        = out_dir / f"tier_{tier}.png"
            saved[tier] = self._save(fig, path)

        return saved

    def _plot_r2_distribution(self, r2_map : np.ndarray, summary : dict, out_path : Path) -> Path:
        flat  = r2_map.reshape(-1).astype(np.float64)
        valid = flat[np.isfinite(flat)]

        qs      = [10, 25, 50, 75, 90]
        pvals   = np.percentile(valid, qs) if valid.size > 0 else [float("nan")] * 5
        pcolors = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax  = axes[0]
        lo  = float(np.percentile(valid, 0.5))
        hi  = float(np.percentile(valid, 99.5))
        ax.hist(valid, bins=120, range=(lo, hi), density=True, color="#1f77b4", alpha=0.60, edgecolor="none", label="PDF")
        
        if valid.size > 2:
            kde = gaussian_kde(valid, bw_method="scott")
            xs  = np.linspace(lo, hi, 600)
            ax.plot(xs, kde(xs), color="black", lw=1.4, label="KDE")
        
        for q, pv, pc in zip(qs, pvals, pcolors):
            ax.axvline(pv, color=pc, lw=1.1, ls="--", label=f"$p_{{{q}}}={pv:.3f}$")
       
        ax.set_xlabel(r"coefficient of determination $R^2$")
        ax.set_ylabel(r"probability density")
        ax.set_title(r"$R^2$ distribution across all pixels")
        ax.legend(fontsize=8, framealpha=0.90, ncol=2)
        ax.grid(True, which="major", lw=0.3, alpha=0.40)

        ax2   = axes[1]
        cdf_x = np.sort(valid)
        cdf_y = np.arange(1, valid.size + 1) / valid.size
        ax2.plot(cdf_x, cdf_y, color="#1f77b4", lw=1.4)
       
        for q, pv, pc in zip(qs, pvals, pcolors):
            ax2.axvline(pv, color=pc, lw=1.0, ls="--")
            ax2.axhline(q / 100.0, color=pc, lw=0.6, ls=":")
            ax2.text(pv + 0.005, q / 100.0 + 0.012, f"$p_{{{q}}}$", fontsize=7, color=pc)
       
        ax2.set_xlabel(r"$R^2$")
        ax2.set_ylabel(r"cumulative fraction")
        ax2.set_title(r"$R^2$ empirical CDF")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, which="major", lw=0.3, alpha=0.40)

        fig.suptitle(
            rf"$R^2$ statistics — "
            rf"mean={summary['r2_mean']:.4f},  "
            rf"median={summary['r2_median']:.4f},  "
            rf"neg\_frac={summary['r2_neg_frac']:.3f}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
       
        return self._save(fig, out_path)

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

            plot_data  = [d if d.size > 0 else np.array([0.0]) for d in data_by_slot]
            parts      = ax.violinplot(plot_data, positions=positions, showmedians=True, showextrema=False, widths=0.7)
            
            for body, color in zip(parts["bodies"], palette):
                body.set_facecolor(color)
                body.set_alpha(0.60)
            
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.8)

            for k, (slot_data, color) in enumerate(zip(data_by_slot, palette)):
                if slot_data.size < 4:
                    continue
                q25, q50, q75 = np.percentile(slot_data, [25, 50, 75])
                ax.vlines(k + 1, q25, q75, color=color, lw=3.0, zorder=3)
                ax.scatter(k + 1, q50, color="white", s=22, zorder=5, edgecolors=color, linewidths=1.2)

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

    def _plot_r2_spatial_map(self, r2_map : np.ndarray, out_path : Path) -> Path:
        Az, R    = r2_map.shape
        cmap_obj = self._cmap_with_bad("RdYlGn")
        vmin     = max(-1.0, float(np.nanpercentile(r2_map, 1.0)))
        vmax     = 1.0

        fig, ax = plt.subplots(figsize=(8, 6))
        im      = ax.imshow(r2_map, cmap=cmap_obj, vmin=vmin, vmax=vmax, extent=[0, R, Az, 0], aspect="auto", interpolation="nearest")
        ax.set_xlabel("range [px]")
        ax.set_ylabel("azimuth [px]")
        ax.set_title(r"Per-pixel $R^2$ of Gaussian fit")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(r"$R^2$")
        fig.tight_layout()
        
        return self._save(fig, out_path)

    def _plot_global_metrics_summary(self, summary : dict, out_path : Path) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        qs      = [10, 25, 50, 75, 90]
        pvals   = [summary.get(f"r2_p{q}", float("nan")) for q in qs]
        pcolors = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

        ax   = axes[0]
        bars = ax.bar(range(len(qs)), pvals, color=pcolors, alpha=0.80, edgecolor="white", lw=0.5)
        ax.set_xticks(range(len(qs)))
        ax.set_xticklabels([f"$p_{{{q}}}$" for q in qs])
        ax.set_ylabel(r"$R^2$")
        ax.set_ylim(bottom=min(0.0, min(v for v in pvals if np.isfinite(v))) - 0.05)
        ax.set_title(r"$R^2$ percentiles across all pixels")
        r2_mean = summary.get("r2_mean", float("nan"))
       
        if np.isfinite(r2_mean):
            ax.axhline(r2_mean, color="black", lw=1.0, ls="--", label=f"mean $= {r2_mean:.4f}$")
            ax.legend(framealpha=0.90)
       
        ax.grid(True, axis="y", which="major", lw=0.3, alpha=0.40)
       
        for bar, val in zip(bars, pvals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax2    = axes[1]
        n_K    = self.n_gaussians
        k_vals = list(range(n_K + 1))
        fracs  = [summary.get(f"frac_{k}_active", float("nan")) for k in k_vals]
        cols2  = [cm.tab10(k % 10) for k in k_vals]
        bars2  = ax2.bar(k_vals, fracs, color=cols2, alpha=0.80, edgecolor="white", lw=0.5)
       
        ax2.set_xticks(k_vals)
        ax2.set_xticklabels([f"$K={k}$" for k in k_vals])
        ax2.set_ylabel("fraction of pixels")
        ax2.set_ylim(0, 1.08)
        ax2.set_title("Active-Gaussian count distribution")
        ax2.grid(True, axis="y", which="major", lw=0.3, alpha=0.40)
       
        for bar, val in zip(bars2, fracs):
            if np.isfinite(val):
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle("Global fitting quality summary", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        
        return self._save(fig, out_path)

    def run(self, parameters_array : np.ndarray, metrics_dict : dict, metadata : dict, tomogram_path : Path) -> Dict[str, Path]:
        self.logger.section("[Fitting Result Plots]")
        self._apply_style()
        dirs = self._setup_output_dirs()


        r2_map       = metrics_dict["r2_map"]                                       
        activity_map = metrics_dict["activity_map"]                                 
        height_axis  = metrics_dict["height_axis"]                                  
        summary      = metrics_dict["global_summary"]

        saved: Dict[str, Path] = {}

        self.logger.subsection("Plotting active-Gaussian count colormap")
        saved["n_gaussians_map"] = self._plot_n_gaussians_map(activity_map, dirs["colormaps"] / "n_gaussians_map.png")

        self.logger.subsection("Plotting R\u00b2 spatial map")
        saved["r2_spatial_map"] = self._plot_r2_spatial_map(r2_map, dirs["colormaps"] / "r2_map.png")

        self.logger.subsection("Plotting amplitude spatial maps")
        amp_keys   = [f"amp_{k}"   for k in range(self.n_gaussians)]
        amp_titles = [f"$A_{{{k + 1}}}$  amplitude" for k in range(self.n_gaussians)]
        saved["amplitude_maps"] = self._plot_spatial_maps(
            metrics_dict, amp_keys, amp_titles,
            "Gaussian amplitude maps  (inactive pixels masked)",
            r"$A_k$",
            dirs["colormaps"] / "amplitude_maps.png",
            cmap="plasma",
        )

        self.logger.subsection("Plotting height-centroid (\u03bc) spatial maps")
        mu_keys   = [f"mu_{k}"    for k in range(self.n_gaussians)]
        mu_titles = [rf"$\mu_{{{k + 1}}}$  centroid [m]" for k in range(self.n_gaussians)]
        saved["mu_maps"] = self._plot_spatial_maps(
            metrics_dict, mu_keys, mu_titles,
            r"Gaussian centroid height $\mu_k$ maps  (inactive pixels masked)",
            r"$\mu_k$ [m]",
            dirs["colormaps"] / "mu_maps.png",
            cmap="RdYlGn",
        )

        self.logger.subsection("Plotting sigma spatial maps")
        sig_keys   = [f"sigma_{k}" for k in range(self.n_gaussians)]
        sig_titles = [rf"$\sigma_{{{k + 1}}}$  spread [m]" for k in range(self.n_gaussians)]
        
        saved["sigma_maps"] = self._plot_spatial_maps(
            metrics_dict, sig_keys, sig_titles,
            r"Gaussian spread $\sigma_k$ maps  (inactive pixels masked)",
            r"$\sigma_k$ [m]",
            dirs["colormaps"] / "sigma_maps.png",
            cmap="viridis",
        )

        if self.n_gaussians >= 2:
            self.logger.subsection("Plotting \u03bc-separation maps")
            sep_keys   = [f"mu_sep_{k}_{k + 1}" for k in range(self.n_gaussians - 1)]
            sep_titles = [rf"$|\mu_{{{k + 2}}} - \mu_{{{k + 1}}}|$  [m]" for k in range(self.n_gaussians - 1)]
            
            saved["mu_separation_maps"] = self._plot_spatial_maps(
                metrics_dict, sep_keys, sep_titles,
                r"Adjacent centroid separation $|\mu_{k+1} - \mu_k|$  (both active)",
                "separation [m]",
                dirs["colormaps"] / "mu_separation_maps.png",
                cmap="magma",
            )

        self.logger.subsection("Plotting R\u00b2 distribution and CDF")
        saved["r2_distribution"] = self._plot_r2_distribution(r2_map, summary, dirs["distributions"] / "r2_distribution.png")

        self.logger.subsection("Plotting parameter distributions")
        for key, path in self._plot_parameter_distributions(parameters_array, dirs["distributions"]).items():
            saved[key] = path

        self.logger.subsection("Plotting global metrics summary")
        saved["global_summary"] = self._plot_global_metrics_summary(summary, dirs["metrics"] / "global_summary.png")

        self.logger.subsection("Loading tomogram for example fit plots (memory-mapped)")
        tomogram_mmap = np.load(str(tomogram_path), mmap_mode="r")                 # (n_elev, Az, R)

        pixels_by_tier = self._select_pixels_by_r2_tiers(r2_map)
        all_pixels     = np.concatenate([px for px in pixels_by_tier.values() if px.shape[0] > 0], axis=0)

        self.logger.subsection(f"Extracting {all_pixels.shape[0]} pixel profiles for example fits")
        pixel_profiles: Dict[Tuple[int, int], np.ndarray] = {}
        for az, rg in all_pixels.tolist():
            pixel_profiles[(az, rg)] = np.array(tomogram_mmap[:, az, rg], dtype=np.float32)

        del tomogram_mmap
        gc.collect()

        self.logger.subsection(f"Plotting example fits  ({self.n_fits_per_tier} pixels × 3 tiers)")
        for tier, path in self._plot_example_fits(parameters_array, pixel_profiles, height_axis, pixels_by_tier, r2_map, dirs["example_fits"]).items():
            saved[f"example_fits_{tier}"] = path

        self.logger.subsection(f"Saved {len(saved)} figures \u2192 {self._images_dir}")
        return saved
