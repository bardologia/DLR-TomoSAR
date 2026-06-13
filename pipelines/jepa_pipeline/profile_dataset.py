from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from tools.gaussians import GaussianMixture
from tools.logger    import Logger


class ProfileDataset(Dataset):
    def __init__(
        self,
        param_arrays    : list[np.ndarray],
        x_axis          : np.ndarray,
        normalizer,
        n_gaussians     : int,
        amp_zero_thr    : float = 1e-3,
        pixel_subsample : float = 1.0,
        keep_empty_frac : float = 0.05,
        seed            : int   = 0,
        logger          : Logger | None = None,
    ) -> None:
        self.x_axis      = np.asarray(x_axis, dtype=np.float32)
        self.n_gaussians = int(n_gaussians)
        self.ppg         = 3

        amps_list, mus_list, sigs_list, norm_list = [], [], [], []
        for params in param_arrays:
            params = np.asarray(params, dtype=np.float32)
            C      = params.shape[0]
            flat   = params.reshape(C, -1)

            amps = flat[0::3].T
            mus  = flat[1::3].T
            sigs = flat[2::3].T

            params_norm = normalizer.normalize_output(params) if normalizer is not None else params
            params_norm = params_norm.reshape(C, -1).T

            amps_list.append(amps)
            mus_list.append(mus)
            sigs_list.append(sigs)
            norm_list.append(params_norm)

        self.amps        = np.concatenate(amps_list, axis=0)
        self.mus         = np.concatenate(mus_list,  axis=0)
        self.sigs        = np.concatenate(sigs_list, axis=0)
        self.params_norm = np.concatenate(norm_list, axis=0)

        active = (self.amps > amp_zero_thr).any(axis=1)
        rng    = np.random.default_rng(seed)

        active_idx = np.flatnonzero(active)
        empty_idx  = np.flatnonzero(~active)

        if 0.0 < pixel_subsample < 1.0:
            keep       = max(1, int(round(len(active_idx) * pixel_subsample)))
            active_idx = rng.choice(active_idx, size=keep, replace=False)

        if keep_empty_frac > 0.0 and len(empty_idx) > 0:
            keep      = max(1, int(round(len(empty_idx) * keep_empty_frac)))
            empty_idx = rng.choice(empty_idx, size=min(keep, len(empty_idx)), replace=False)
        else:
            empty_idx = np.empty(0, dtype=np.int64)

        self.index = np.concatenate([active_idx, empty_idx]).astype(np.int64)
        rng.shuffle(self.index)

        if logger is not None:
            logger.section("[ProfileDataset]")
            logger.kv_table({
                "Total pixels"  : int(self.amps.shape[0]),
                "Active pixels" : int(active.sum()),
                "Kept active"   : int(len(active_idx)),
                "Kept empty"    : int(len(empty_idx)),
                "Profile length": int(self.x_axis.shape[0]),
                "Gaussians"     : self.n_gaussians,
            })

    @classmethod
    def from_patch_dataset(cls, patch_dataset, x_axis, normalizer, n_gaussians, **kwargs) -> "ProfileDataset":
        parts        = getattr(patch_dataset, "parts", None)
        datasets     = parts if parts is not None else [patch_dataset]
        param_arrays = [ds.gt_parameters for ds in datasets]
        return cls(param_arrays=param_arrays, x_axis=x_axis, normalizer=normalizer, n_gaussians=n_gaussians, **kwargs)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int):
        idx   = self.index[i]
        curve = GaussianMixture.evaluate_batch(
            self.x_axis,
            self.amps[idx:idx + 1],
            self.mus[idx:idx + 1],
            self.sigs[idx:idx + 1],
        )[0]
        return curve.astype(np.float32), self.params_norm[idx].astype(np.float32)
