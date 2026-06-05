from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy             as np


class PlotBase:
    SCIENTIFIC_RC: dict = {
        "font.family"         : "serif",
        "font.serif"          : ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset"    : "dejavuserif",
        "font.size"           : 11,
        "axes.titlesize"      : 12,
        "axes.labelsize"      : 11,
        "xtick.labelsize"     : 10,
        "ytick.labelsize"     : 10,
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

    fig_dpi  : int = 150
    save_dpi : int = 150

    def _apply_style(self) -> None:
        plt.rcParams.update(self.SCIENTIFIC_RC)
        plt.rcParams["figure.dpi"]  = self.fig_dpi
        plt.rcParams["savefig.dpi"] = self.save_dpi

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return path

    @staticmethod
    def _shared_clim(*arrays: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> Tuple[float, float]:
        flat = np.concatenate([a.reshape(-1) for a in arrays])
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return (0.0, 1.0)
        return float(np.percentile(flat, q_low)), float(np.percentile(flat, q_high))

    @staticmethod
    def _cmap_with_bad(name: str, bad_color: str = "0.88") -> mcolors.Colormap:
        cmap = plt.cm.get_cmap(name).copy()
        cmap.set_bad(color=bad_color)
        return cmap

    @staticmethod
    def _normalize_01(arr: np.ndarray) -> np.ndarray:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi - lo < 1e-12:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    @staticmethod
    def _triple_panel(
        fig,
        axes,
        panels    : List[Tuple[np.ndarray, str, str, float, float]],
        x_label   : str,
        int_label : str,
        extent    : list,
        origin    : str,
    ) -> None:

        for ax_i, (data, label, cm_used, vlo, vhi) in zip(axes, panels):
            im = ax_i.imshow(data, cmap=cm_used, vmin=vlo, vmax=vhi, extent=extent, aspect="auto", origin=origin)
            ax_i.set_title(label)
            ax_i.set_xlabel(x_label)
            lbl_cb = int_label if cm_used == panels[0][2] else "|error|"
            fig.colorbar(im, ax=ax_i, fraction=0.045, pad=0.02).set_label(lbl_cb)
