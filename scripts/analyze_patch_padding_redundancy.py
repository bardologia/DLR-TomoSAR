from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy             as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.monitoring.logger import Logger
from tools.reporting.plotting import PlotBase

SPLIT_AZIMUTH = 13000
SPLIT_RANGE   = 4000

PATCH_AZ_MIN  = 1000
PATCH_AZ_MAX  = 13000
PATCH_RG_MIN  = 500
PATCH_RG_MAX  = 4000

BOXCAR_WINDOW = 20
SWEEP_STEPS   = 600

ACCENT = "#B03052"
GOOD   = "#1E6E46"
INK    = "#282828"

OUTPUT_DIR = Path.home() / "tomosar_results" / "patch_padding_redundancy"


class PatchPaddingModel:
    def __init__(self, w: int = BOXCAR_WINDOW):
        self.w = w

    def padded_area(self, patch_az, patch_rg):
        return (patch_az + 2 * self.w) * (patch_rg + 2 * self.w)

    def fresh_core(self, patch_az, patch_rg):
        core_az = np.clip(patch_az - 2 * self.w, 0, None)
        core_rg = np.clip(patch_rg - 2 * self.w, 0, None)
        return core_az * core_rg

    def padding_pixels(self, patch_az, patch_rg):
        return self.padded_area(patch_az, patch_rg) - patch_az * patch_rg

    def two_w_pixels(self, patch_az, patch_rg):
        return patch_az * patch_rg - self.fresh_core(patch_az, patch_rg)

    def patch_count(self, split_az, split_rg, patch_az, patch_rg):
        return np.ceil(split_az / patch_az) * np.ceil(split_rg / patch_rg)

    def evaluate(self, split_az, split_rg, patch_az, patch_rg) -> dict:
        padded    = self.padded_area(patch_az, patch_rg)
        core      = self.fresh_core(patch_az, patch_rg)
        padding   = self.padding_pixels(patch_az, patch_rg)
        two_w     = self.two_w_pixels(patch_az, patch_rg)
        redundant = padding + two_w

        return dict(
            padding_thickness = self.w,
            padding_pixels    = padding,
            two_w_pixels      = two_w,
            fresh_pixels      = core,
            padded_area       = padded,
            patch_count       = self.patch_count(split_az, split_rg, patch_az, patch_rg),
            pct_redundant     = 100.0 * redundant / padded,
            pct_two_w         = 100.0 * two_w / padded,
            pct_fresh         = 100.0 * core / padded,
        )


class PatchRedundancySweep(PlotBase):
    save_dpi = 300

    def __init__(self, model: PatchPaddingModel, output_dir: Path = OUTPUT_DIR):
        self.model      = model
        self.output_dir = output_dir
        self.logger     = Logger(log_dir="logs", name="patch_padding_redundancy")

    def patch_grid(self):
        fraction = np.linspace(0.0, 1.0, SWEEP_STEPS)
        patch_az = np.round(PATCH_AZ_MIN + fraction * (PATCH_AZ_MAX - PATCH_AZ_MIN)).astype(int)
        patch_rg = np.round(PATCH_RG_MIN + fraction * (PATCH_RG_MAX - PATCH_RG_MIN)).astype(int)
        return patch_az, patch_rg

    def evaluate_sweep(self, patch_az, patch_rg):
        return self.model.evaluate(SPLIT_AZIMUTH, SPLIT_RANGE, patch_az, patch_rg)

    def log_summary(self, patch_az, patch_rg, results):
        self.logger.section("Patch padding / redundancy sweep")
        self.logger.info(f"split = {SPLIT_AZIMUTH} az x {SPLIT_RANGE} rg  |  boxcar w = {self.model.w}")
        self.logger.info(f"padding thickness (constant) = {self.model.w} px on every side")

        marks = np.linspace(0, len(patch_az) - 1, 7).astype(int)
        for i in marks:
            self.logger.info(
                f"patch {patch_az[i]:5d} x {patch_rg[i]:4d}  |  "
                f"pad={results['padding_pixels'][i]:10.0f}  "
                f"red%={results['pct_redundant'][i]:5.2f}  "
                f"2w%={results['pct_two_w'][i]:5.2f}  "
                f"fresh%={results['pct_fresh'][i]:5.2f}  "
                f"patches={results['patch_count'][i]:4.0f}"
            )

    def plot_pct_redundant(self, patch_az, results):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(patch_az, results["pct_redundant"], color=ACCENT, lw=2.0)
        ax.set_xlabel("patch size — azimuth (px), range scales $500\\!\\to\\!4000$")
        ax.set_ylabel("redundant pixels (%)")
        ax.set_title("Redundant share falls as the patch grows")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return self._save(fig, self.output_dir / "pct_redundant.png")

    def plot_pct_fresh(self, patch_az, results):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(patch_az, results["pct_fresh"], color=GOOD, lw=2.0)
        ax.set_xlabel("patch size — azimuth (px), range scales $500\\!\\to\\!4000$")
        ax.set_ylabel("fresh pixels outside $2w$, not redundant (%)")
        ax.set_title("Fresh share rises as the patch grows")
        fig.tight_layout()
        return self._save(fig, self.output_dir / "pct_fresh.png")

    def plot_padding_pixels(self, patch_az, results):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(patch_az, results["padding_pixels"] / 1e6, color=INK, lw=2.0)
        ax.set_xlabel("patch size — azimuth (px), range scales $500\\!\\to\\!4000$")
        ax.set_ylabel("padding pixels (millions), thickness fixed at $w$")
        ax.set_title("Padding grows in count but stays a $w$-thick band")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return self._save(fig, self.output_dir / "padding_pixels.png")

    def plot_two_w_pixels(self, patch_az, results):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(patch_az, results["two_w_pixels"] / 1e6, color=ACCENT, lw=2.0)
        ax.set_xlabel("patch size — azimuth (px), range scales $500\\!\\to\\!4000$")
        ax.set_ylabel("real pixels inside the $2w$ band (millions)")
        ax.set_title("Contaminated $2w$ pixels, excluding the padding")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return self._save(fig, self.output_dir / "two_w_pixels.png")

    def run(self) -> None:
        self._apply_style()

        patch_az, patch_rg = self.patch_grid()
        results            = self.evaluate_sweep(patch_az, patch_rg)

        self.log_summary(patch_az, patch_rg, results)

        saved = [
            self.plot_pct_redundant(patch_az, results),
            self.plot_pct_fresh(patch_az, results),
            self.plot_padding_pixels(patch_az, results),
            self.plot_two_w_pixels(patch_az, results),
        ]

        for path in saved:
            self.logger.ok(f"wrote {path}")


if __name__ == "__main__":
    PatchRedundancySweep(PatchPaddingModel()).run()
