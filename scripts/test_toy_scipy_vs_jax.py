"""
test_toy_scipy_vs_jax.py
========================
Runs both the CPU (scipy) and GPU (JAX) parameter extraction backends on the
toy dataset and reports per-parameter mean absolute difference.

Run:
    cd /ste/rnd/User/vice_vi
    conda run -n stetools python DLR-TomoSAR/scripts/test_toy_scipy_vs_jax.py
"""
import sys, time
sys.path.insert(0, "DLR-TomoSAR")

import numpy as np
from pathlib import Path

from configuration.param_extraction_config import FitSettings, FitMode
from pipelines.param_extraction_pipeline.fitting import ParameterExtractor
from tools.logger import Logger

TOY_PATH     = Path("Dataset/toy")
TOMO_NAME    = "tomofull_1000a1050a500a550_dtmf_Xtomo_id2X.npy"
HEIGHT_RANGE = (-20.0, 80.0)
N_GAUSSIANS  = 3


def main() -> None:
    fit_settings = FitSettings(
        number_of_gaussians = N_GAUSSIANS,
        max_fit_iterations  = 5000,
        fit_config          = FitMode.Adaptive(),
    )
    logger    = Logger(name="toy_cmp")
    tomo_path = TOY_PATH / "data" / TOMO_NAME
    tomo      = np.load(str(tomo_path), mmap_mode="r", allow_pickle=False)

    print(f"Tomogram : {tomo.shape}  dtype={tomo.dtype}")
    print(f"Height   : {HEIGHT_RANGE}  N_gaussians={N_GAUSSIANS}  n_params={3*N_GAUSSIANS}")
    print()

    print("Running scipy CPU …")
    t0 = time.perf_counter()
    cpu_extractor = ParameterExtractor(
        parameter_extraction = fit_settings,
        parameter_workers    = 8,
        logger               = logger,
        use_gpu              = False,
    )
    cpu_params = cpu_extractor.run(tomo_path, HEIGHT_RANGE)
    cpu_time   = time.perf_counter() - t0
    print(f"  done in {cpu_time:.1f}s   shape={cpu_params.shape}\n")

    print("Running JAX GPU …")
    t0 = time.perf_counter()
    gpu_extractor = ParameterExtractor(
        parameter_extraction = fit_settings,
        parameter_workers    = 1,
        logger               = logger,
        use_gpu              = True,
        gpu_batch_size       = 256,
        adam_steps           = 2000,
    )
    gpu_params = gpu_extractor.run(tomo_path, HEIGHT_RANGE)
    gpu_time   = time.perf_counter() - t0
    print(f"  done in {gpu_time:.1f}s   shape={gpu_params.shape}\n")

    param_names = []
    for g in range(1, N_GAUSSIANS + 1):
        param_names += [f"amp{g}", f"mu{g}", f"sig{g}"]

    assert cpu_params.shape == gpu_params.shape, "Shape mismatch!"

    diff = np.abs(cpu_params.astype(np.float64) - gpu_params.astype(np.float64))

    print(f"{'param':<8}  {'mean|Δ|':>10}  {'median|Δ|':>10}  {'max|Δ|':>10}  {'rel%':>8}")
    print("-" * 56)
    for i, name in enumerate(param_names):
        d   = diff[i]
        ref = np.abs(cpu_params[i]).mean()
        rel = 100.0 * d.mean() / ref if ref > 1e-12 else float("nan")
        print(f"{name:<8}  {d.mean():>10.4f}  {np.median(d):>10.4f}  {d.max():>10.4f}  {rel:>7.2f}%")

    print()
    overall_rel = 100.0 * diff.mean() / (np.abs(cpu_params).mean() + 1e-12)
    print(f"Overall mean |Δ| : {diff.mean():.4f}   ({overall_rel:.2f}% of scipy mean magnitude)")
    print(f"Speedup          : {cpu_time / gpu_time:.1f}×")


if __name__ == "__main__":
    main()
