from __future__ import annotations

from pathlib import Path
from typing  import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class Plotter:

    PALETTE = ("#1f3b73", "#c1272d", "#2a9d8f", "#e9c46a", "#6a4c93", "#264653")

    RC_PARAMS = {
        "font.family"       : "serif",
        "font.serif"        : ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
        "mathtext.fontset"  : "cm",
        "font.size"         : 10,
        "axes.titlesize"    : 11,
        "axes.labelsize"    : 10,
        "legend.fontsize"   : 9,
        "xtick.labelsize"   : 9,
        "ytick.labelsize"   : 9,
        "axes.linewidth"    : 0.8,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.grid"         : True,
        "grid.linestyle"    : ":",
        "grid.linewidth"    : 0.6,
        "grid.alpha"        : 0.5,
        "lines.linewidth"   : 1.4,
        "lines.markersize"  : 3.0,
        "figure.dpi"        : 120,
        "savefig.dpi"       : 300,
        "savefig.bbox"      : "tight",
        "pdf.fonttype"      : 42,
        "ps.fonttype"       : 42,
    }

    def __init__(self, images_dir: Path, embed_dir: Path, recon_dir: Path) -> None:
        self.images_dir = Path(images_dir)
        self.embed_dir  = Path(embed_dir)
        self.recon_dir  = Path(recon_dir)
        for d in (self.images_dir, self.embed_dir, self.recon_dir):
            d.mkdir(parents=True, exist_ok=True)
        mpl.rcParams.update(self.RC_PARAMS)

    def _save(self, fig, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix(".png"))
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)
        return out_path.with_suffix(".png")

    def plot_loss_component(self,
                            component  : str,
                            epochs     : Sequence[int],
                            train_vals : Sequence[float],
                            val_vals   : Sequence[float] | None) -> Path:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(epochs, train_vals, label="train", color=self.PALETTE[0])
        if val_vals is not None and len(val_vals):
            ax.plot(epochs, val_vals, label="val", color=self.PALETTE[1], linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(component.replace("_", " "))
        ax.set_title(component.replace("_", " ").title())
        ax.legend(frameon=False)
        fig.tight_layout()
        return self._save(fig, self.images_dir / f"loss_{component}")

    def plot_loss_overview(self,
                           components : Sequence[str],
                           epochs     : Sequence[int],
                           series     : dict[str, dict[str, Sequence[float] | None]]) -> Path:
        n     = len(components)
        cols  = min(3, n)
        rows  = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.6), squeeze=False)
        for i, comp in enumerate(components):
            ax = axes[i // cols][i % cols]
            tr = series[comp].get("train")
            va = series[comp].get("val")
            if tr is not None and len(tr):
                ax.plot(epochs, tr, label="train", color=self.PALETTE[0])
            if va is not None and len(va):
                ax.plot(epochs, va, label="val", color=self.PALETTE[1], linestyle="--")
            ax.set_title(comp.replace("_", " ").title())
            ax.set_xlabel("Epoch")
            if i == 0:
                ax.legend(frameon=False)
        for j in range(len(components), rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle("Training Loss Overview")
        fig.tight_layout()
        return self._save(fig, self.images_dir / "loss_overview")

    def plot_reconstruction_gallery(self,
                                    split_name  : str,
                                    tag         : str,
                                    profiles    : np.ndarray,
                                    recons      : np.ndarray,
                                    errors      : np.ndarray,
                                    indices     : np.ndarray) -> Path:
        k    = profiles.shape[0]
        cols = 4
        rows = int(np.ceil(k / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.0), squeeze=False)
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            if i < k:
                x = np.arange(profiles.shape[1])
                ax.plot(x, profiles[i], color=self.PALETTE[0], label="target")
                ax.plot(x, recons[i],   color=self.PALETTE[1], linestyle="--", label="recon")
                ax.set_title(f"idx={int(indices[i])}  mse={errors[i]:.2e}", fontsize=8)
                ax.tick_params(labelsize=7)
                if i == 0:
                    ax.legend(frameon=False, fontsize=7)
            else:
                ax.axis("off")
        fig.suptitle(f"{split_name} — {tag}")
        fig.tight_layout()
        return self._save(fig, self.recon_dir / f"gallery_{split_name}_{tag}")

    def plot_embedding_scatter(self,
                               split_name : str,
                               points_2d  : np.ndarray,
                               tag        : str,
                               title      : str,
                               color_by   : np.ndarray | None = None) -> Path:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        if color_by is not None:
            sc = ax.scatter(points_2d[:, 0], points_2d[:, 1], c=color_by, s=4, alpha=0.6, cmap="viridis")
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("recon MSE")
        else:
            ax.scatter(points_2d[:, 0], points_2d[:, 1], s=4, alpha=0.6, color=self.PALETTE[0])
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.set_title(f"{split_name} — {title}")
        fig.tight_layout()
        return self._save(fig, self.embed_dir / f"{split_name}_{tag}")

    def plot_embedding_spectrum(self,
                                split_name : str,
                                eigenvalues: np.ndarray) -> Path:
        eig    = np.sort(eigenvalues)[::-1]
        eig    = np.clip(eig, 1e-12, None)
        ratio  = eig / eig.sum()
        cumrat = np.cumsum(ratio)

        fig, ax1 = plt.subplots(figsize=(6.0, 3.6))
        idx = np.arange(1, len(eig) + 1)
        ax1.bar(idx, ratio, color=self.PALETTE[0], alpha=0.7, label="explained ratio")
        ax1.set_xlabel("Latent component")
        ax1.set_ylabel("Explained variance ratio")
        ax2 = ax1.twinx()
        ax2.plot(idx, cumrat, color=self.PALETTE[1], marker="o", label="cumulative")
        ax2.set_ylabel("Cumulative explained variance")
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(False)
        ax1.set_title(f"{split_name} — Latent spectrum")
        fig.tight_layout()
        return self._save(fig, self.embed_dir / f"{split_name}_spectrum")

    def plot_error_histogram(self,
                             split_name : str,
                             errors     : np.ndarray) -> Path:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.hist(errors, bins=60, color=self.PALETTE[0], alpha=0.85, edgecolor="white")
        ax.set_xlabel("Reconstruction MSE (per profile)")
        ax.set_ylabel("Count")
        ax.set_title(f"{split_name} — Reconstruction MSE distribution")
        ax.set_yscale("log")
        fig.tight_layout()
        return self._save(fig, self.recon_dir / f"error_hist_{split_name}")
