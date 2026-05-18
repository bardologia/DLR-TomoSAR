from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pipelines.dataset_creation_pipeline.load import TomoPatchDataset
from tools.logger                             import Logger

from .config        import AugmentationConfig, ContrastiveView, DataConfig
from .normalization import ProfileNormalizer


class Augmenter:

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config
        self._rng   = np.random.default_rng(config.seed)

    def __call__(self, profile: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        out = profile.clone()
        n   = out.shape[-1]

        if cfg.scale_range is not None:
            scale = float(self._rng.uniform(cfg.scale_range[0], cfg.scale_range[1]))
            out   = out * scale

        if cfg.jitter_std > 0:
            noise = torch.from_numpy(self._rng.normal(0.0, cfg.jitter_std, size=n).astype(np.float32))
            out   = out + noise

        if cfg.shift_max > 0:
            shift = int(self._rng.integers(-cfg.shift_max, cfg.shift_max + 1))
            if shift != 0:
                out = torch.roll(out, shifts=shift, dims=-1)

        if cfg.mask_prob > 0 and self._rng.random() < cfg.mask_prob:
            width = int(self._rng.integers(1, cfg.mask_max_width + 1))
            start = int(self._rng.integers(0, max(1, n - width)))
            out[..., start:start + width] = 0.0

        return out


class ProfileDataset(Dataset):

    def __init__(
        self,
        tomo_dataset : TomoPatchDataset,
        data_config  : DataConfig,
        split_name   : str,
        logger       : Logger,
    ) -> None:
        self.tomo_dataset = tomo_dataset
        self.data_config  = data_config
        self.split_name   = split_name
        self.logger       = logger

        self.patch_h, self.patch_w = tomo_dataset.grid.grid.patch_size
        self.pixels_per_patch      = self.patch_h * self.patch_w
        self.profile_length        = int(tomo_dataset.target_channels)

        if data_config.profile_length != self.profile_length:
            self.logger.warning(
                f"[ProfileDataset:{split_name}] data_config.profile_length="
                f"{data_config.profile_length} overridden to {self.profile_length} from tomogram."
            )
            data_config.profile_length = self.profile_length

        self.normalizer = ProfileNormalizer(data_config)
        self.augmenter  = Augmenter(data_config.augmentation)

        self._cache_patch_idx : int | None          = None
        self._cache_target    : torch.Tensor | None = None

        full_size = len(self.tomo_dataset) * self.pixels_per_patch
        max_p     = data_config.max_profiles
        if max_p is not None and max_p < full_size:
            rng           = np.random.default_rng(data_config.sampling_seed)
            self._indices : np.ndarray | None = rng.choice(full_size, size=max_p, replace=False)
            self._indices.sort()
            effective = max_p
        else:
            self._indices = None
            effective     = full_size

        self.logger.subsection(
            f"[ProfileDataset:{split_name}] patches={len(tomo_dataset)}  "
            f"pixels/patch={self.pixels_per_patch}  N={self.profile_length}  "
            f"full_profiles={full_size:,}  used_profiles={effective:,}"
            + (f"  (subsampled from {full_size:,})" if self._indices is not None else "")
        )

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self.tomo_dataset) * self.pixels_per_patch

    def _load_patch(self, patch_idx: int) -> torch.Tensor:
        if self._cache_patch_idx == patch_idx and self._cache_target is not None:
            return self._cache_target
        item                  = self.tomo_dataset[patch_idx]
        target                = item[1]
        self._cache_patch_idx = patch_idx
        self._cache_target    = target
        return target

    def _pixel_coords(self, pixel_idx: int) -> Tuple[int, int]:
        return divmod(pixel_idx, self.patch_w)

    def _neighbor_coords(self, row: int, col: int) -> Tuple[int, int]:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        np.random.shuffle(deltas)
        for dr, dc in deltas:
            r, c = row + dr, col + dc
            if 0 <= r < self.patch_h and 0 <= c < self.patch_w:
                return r, c
        return row, col

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if self._indices is not None:
            idx = int(self._indices[idx])
        patch_idx, pixel_idx = divmod(idx, self.pixels_per_patch)
        target_patch         = self._load_patch(patch_idx)
        row, col             = self._pixel_coords(pixel_idx)

        profile_a, scale_a = self.normalizer.apply(target_patch[:, row, col])
        view = self.data_config.contrastive_view

        if view == ContrastiveView.augmentation:
            profile_b = self.augmenter(profile_a)
        elif view == ContrastiveView.neighbor:
            r2, c2       = self._neighbor_coords(row, col)
            profile_b, _ = self.normalizer.apply(target_patch[:, r2, c2])
        elif view == ContrastiveView.both:
            r2, c2       = self._neighbor_coords(row, col)
            profile_b, _ = self.normalizer.apply(target_patch[:, r2, c2])
            profile_b    = self.augmenter(profile_b)
        else:
            raise ValueError(f"Unknown contrastive view: {view}")

        return profile_a, profile_b, scale_a


class LoaderBuilder:

    def __init__(
        self,
        batch_size         : int,
        num_workers        : int,
        logger             : Logger,
        shuffle_train      : bool       = True,
        pin_memory         : bool       = True,
        persistent_workers : bool       = False,
        prefetch_factor    : int | None = None,
    ) -> None:
        self.batch_size         = batch_size
        self.num_workers        = num_workers
        self.logger             = logger
        self.shuffle_train      = shuffle_train
        self.pin_memory         = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor    = prefetch_factor

    def build(
        self,
        train_dataset : ProfileDataset,
        val_dataset   : ProfileDataset,
        test_dataset  : ProfileDataset,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        common = {
            "batch_size"         : self.batch_size,
            "num_workers"        : self.num_workers,
            "pin_memory"         : self.pin_memory,
            "multiprocessing_context" : "forkserver" if self.num_workers > 0 else None,
            "persistent_workers" : self.persistent_workers,
            "prefetch_factor"    : self.prefetch_factor,
        }
        train_loader = DataLoader(train_dataset, shuffle=self.shuffle_train, drop_last=True,  **common)
        val_loader   = DataLoader(val_dataset,   shuffle=False,              drop_last=False, **common)
        test_loader  = DataLoader(test_dataset,  shuffle=False,              drop_last=False, **common)

        self.logger.section("[Profile Loaders Built]")
        self.logger.subsection(f"Train batches : {len(train_loader)}")
        self.logger.subsection(f"Val   batches : {len(val_loader)}")
        self.logger.subsection(f"Test  batches : {len(test_loader)}")
        return train_loader, val_loader, test_loader
