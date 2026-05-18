import sys; sys.path.insert(0, 'DLR-TomoSAR')
import numpy as np
from pathlib import Path
from configuration.param_extraction_config import FitSettings, FitMode
from pipelines.param_extraction_pipeline.fitting import ParameterExtractor
from pipelines.param_extraction_pipeline.gpu_fitting import GPUParameterExtractor, AdamKernel, JaxGaussianModel
from tools.logger import Logger
import jax.numpy as jnp

if __name__ == '__main__':
    TOMO = Path('Dataset/toy/data/tomofull_1000a1050a500a550_dtmf_Xtomo_id2X.npy')
    HR   = (-20.0, 80.0)
    fs   = FitSettings(number_of_gaussians=3, max_fit_iterations=5000, fit_config=FitMode.Adaptive())
    lg   = Logger(name='dbg')
    tomo = np.load(str(TOMO), mmap_mode='r')
    H, Az, R = tomo.shape

    gpu  = ParameterExtractor(fs, 1, lg, use_gpu=True, gpu_batch_size=256, adam_steps=2000)
    gp   = gpu.run(TOMO, HR)
    ext  = gpu._gpu_extractor

    height = np.linspace(HR[0], HR[1], H, dtype=np.float32)
    fit_cfg = fs.fit_config
    threshold = float(getattr(fit_cfg, 'threshold_factor', 0.0))
    trunc     = int(getattr(fit_cfg, 'truncation_index', H))

    abs_b = np.abs(tomo).astype(np.float32)
    if threshold > 0:
        abs_b = np.where(abs_b > abs_b.max(axis=0, keepdims=True)*threshold, abs_b, 0.0)
    if trunc < H:
        abs_b[trunc:] = 0.0

    profiles = abs_b.transpose(2,1,0).reshape(R*Az, H).copy()
    active   = profiles.max(axis=1) > 1e-7

    fitted_flat = gp.transpose(2,1,0).reshape(R*Az, 9)

    sig_cols = fitted_flat[active, 2::3]
    print(f'Sig stats: min={sig_cols.min():.6f}  median={np.median(sig_cols):.4f}  % near zero (< 1e-4): {(sig_cols < 1e-4).mean()*100:.1f}%')
    print(f'Sig < 1e-4 count: {(sig_cols < 1e-4).sum()} out of {sig_cols.size}')

    r2 = ext._compute_r2_batch(fitted_flat[active], profiles[active], height)
    finite = np.isfinite(r2)
    print(f'R² finite={finite.sum()}/{len(r2)}  mean={r2[finite].mean() if finite.any() else float("nan"):.4f}')
    print(f'R² < 0: {(r2[finite] < 0).sum()}  R² ∈ [0,0.5]: {((r2[finite]>=0)&(r2[finite]<0.5)).sum()}  R² > 0.9: {(r2[finite]>=0.9).sum()}')
