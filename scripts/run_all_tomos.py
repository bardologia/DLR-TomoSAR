"""Batch runner for tomo_gif + refactored profile/compare pipelines."""

import sys
from pathlib import Path
from itertools import product
from time import time
from dataclasses import dataclass

# ── make sure sibling modules are importable ───────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tomo_gif import make_tomo_gif
from tomo_pixel_profiles_ref import (
    ProfileConfig,
    ProfileParallelConfig,
    ProfileFitter,
    ProfilePlotter,
    SliceReconstructor,
    SweepGifGenerator,
    ParameterStorage,
    TomogramLoader,
)
from tomo_compare_ngauss_ref import (
    TomoConfig,
    GaussianFitter,
    TomoPlotter,
    SarReconstructor,
    ParallelConfig,
)

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these to taste
# ══════════════════════════════════════════════════════════════════════
TOMO_DIR   = Path("/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/tomograms")
OUTPUT_DIR = Path("/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR")

METHODS = ["Capon", "MSF"]
WINDOWS = ["w10_20", "w15_30", "w20_40"]

# tomo_pixel_profiles settings
RN_IDX            = 500
N_GAUSSIANS_PROF  = 3
HEIGHT_AXIS_RANGE = (-20, 80)
SAVE_FITTED       = True
PROFILE_N_WORKERS = 64  # set to your server CPU budget; None = auto (cpu_count-1)

# tomo_compare_ngauss settings
N_GAUSS_RANGE      = range(1, 6)
N_EXAMPLE_PROFILES = 6
SAVE_COMPARE       = False
SAR_N_WORKERS      = 64  # set to your server CPU budget; None = auto (cpu_count-1)
SAR_GIF_FPS        = 10  # 0 to disable height-sweep GIF
SAR_GIF_N_FRAMES   = 150

# tomo_gif settings
GIF_SLICE_STEP = 1
GIF_FPS        = 50
GIF_DPI        = 100
GIF_CMAP       = "jet"
GIF_KEEP       = False
GIF_N_WORKERS  = 84   # set to your server CPU budget; None = auto (cpu_count-1)

# Which pipelines to run (toggle individually)
RUN_GIF          = True
RUN_PROFILES     = True
RUN_COMPARE      = True
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RunContext:
    tomo_path: Path

    @property
    def tomo_name(self) -> str:
        return self.tomo_path.stem


def build_tomo_path(method: str, window: str) -> Path:
    return TOMO_DIR / f"tomo_17sartom0102_Lhv_{method}_{window}.hd5"


def run_gif_pipeline(tomo_path: Path):
    print(f"\n{'─'*60}")
    print(f"  GIF  →  {tomo_path.name}")
    print(f"{'─'*60}")
    make_tomo_gif(
        tomo_file_path=str(tomo_path),
        output_dir=str(OUTPUT_DIR),
        slice_step=GIF_SLICE_STEP,
        fps=GIF_FPS,
        dpi=GIF_DPI,
        cmap=GIF_CMAP,
        keep_frames=GIF_KEEP,
        n_workers=GIF_N_WORKERS,
    )


def _build_profile_config(ctx: RunContext) -> ProfileConfig:
    return ProfileConfig(
        tomo_file=str(ctx.tomo_path),
        output_dir=OUTPUT_DIR,
        height_axis_range=HEIGHT_AXIS_RANGE,
        n_gaussians=N_GAUSSIANS_PROF,
        range_index=RN_IDX,
        n_display_profiles=N_EXAMPLE_PROFILES,
        save_results=SAVE_FITTED,
        parallel=ProfileParallelConfig(enabled=True, n_workers=PROFILE_N_WORKERS, method="fork"),
    )


def run_profiles_pipeline(ctx: RunContext):
    print(f"\n{'─'*60}")
    print(f"  PIXEL PROFILES  →  {ctx.tomo_path.name}")
    print(f"{'─'*60}")
    config = _build_profile_config(ctx)

    fitter = ProfileFitter(config)
    plotter = ProfilePlotter(config)
    reconstructor = SliceReconstructor()
    gif_generator = SweepGifGenerator(config)

    slide_abs, height_axis = TomogramLoader.load_from_config(config)
    fit_result = fitter.fit_all_profiles(slide_abs, height_axis)
    reconstructor.reconstruct(slide_abs, height_axis, fit_result)

    if SAVE_FITTED:
        ParameterStorage.save(config, fit_result, height_axis)

    plotter.plot_slice_with_profiles(slide_abs, height_axis, fit_result)
    plotter.plot_parameter_distributions(fit_result)
    plotter.plot_parameter_maps(fit_result)
    gif_generator.generate(slide_abs, height_axis, fit_result)

    import matplotlib.pyplot as plt
    plt.close("all")


def _build_compare_config(ctx: RunContext) -> TomoConfig:
    return TomoConfig(
        tomo_file=str(ctx.tomo_path),
        output_dir=OUTPUT_DIR,
        height_axis_range=HEIGHT_AXIS_RANGE,
        n_gauss_range=list(N_GAUSS_RANGE),
        range_index=RN_IDX,
        n_example_profiles=N_EXAMPLE_PROFILES,
        save_params=SAVE_COMPARE,
        sar_gif_fps=SAR_GIF_FPS,
        sar_gif_n_frames=SAR_GIF_N_FRAMES,
        parallel=ParallelConfig(enabled=True, n_workers=SAR_N_WORKERS, method="fork"),
    )


def run_compare_pipeline(ctx: RunContext):
    print(f"\n{'─'*60}")
    print(f"  COMPARE N_GAUSS  →  {ctx.tomo_path.name}")
    print(f"{'─'*60}")
    config = _build_compare_config(ctx)

    fitter = GaussianFitter(config)
    plotter = TomoPlotter(config)
    sar_reconstructor = SarReconstructor(config)

    slide_abs, height_axis = TomogramLoader.load_slice(config.tomo_file, config.range_index, config.height_axis_range)
    results = fitter.fit_all_orders(slide_abs, height_axis)
    example_pixels = fitter.pick_example_pixels(slide_abs, results)

    plotter.plot_slice_comparison(slide_abs, height_axis, results)
    plotter.plot_peak_heights(results)
    plotter.plot_example_fits(slide_abs, height_axis, results, example_pixels)
    plotter.plot_fit_quality(slide_abs, height_axis, results)
    plotter.plot_residual_metrics(results)
    sar_reconstructor.evaluate_sar_reconstruction()

    import matplotlib.pyplot as plt
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    combos = list(product(METHODS, WINDOWS))
    total  = len(combos)
    t_global = time()

    for i, (method, window) in enumerate(combos, 1):
        tomo_path = build_tomo_path(method, window)
        if not tomo_path.exists():
            print(f"\n  File not found, skipping: {tomo_path}")
            continue

        ctx = RunContext(tomo_path=tomo_path)

        print(f"\n{'═'*60}")
        print(f"  [{i}/{total}]  {method} — {window}")
        print(f"{'═'*60}")
        t0 = time()

        if RUN_GIF:
            run_gif_pipeline(tomo_path)

        if RUN_PROFILES:
            run_profiles_pipeline(ctx)

        if RUN_COMPARE:
            run_compare_pipeline(ctx)

        print(f"\n  ✔  {method} — {window} done in {time()-t0:.1f} s")

    print(f"\n{'═'*60}")
    print(f"  All done!  Total time: {time()-t_global:.1f} s")
    print(f"{'═'*60}")
