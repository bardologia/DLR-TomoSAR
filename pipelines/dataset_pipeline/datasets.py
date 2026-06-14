from __future__ import annotations

import bisect
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset

from configuration.data.dataset_config              import AugmentationConfig, InputConfig, OutputConfig
from pipelines.dataset_pipeline.normalization  import Normalizer
from pipelines.dataset_pipeline.spatial        import Patcher
from tools.monitoring.logger                              import Logger
from tools.reproducibility                     import Reproducibility


class SpatialAugmenter:
    def __init__(self, config: AugmentationConfig, logger, seed: int = 0):
        self.config = config
        self.logger = logger
        self.seed   = int(seed)
        self._rng   = np.random.default_rng(self.seed)

        self.logger.section("[Data Augmentation]")
        self.logger.kv_table(
            {
                "Flip Horizontal" : self.config.p_flip_h,
                "Flip Vertical"   : self.config.p_flip_v,
                "Rotate 90°"      : self.config.p_rot90,
                "Noise"           : f"std={self.config.noise_std} (normalized units) p={self.config.p_noise}",
            },
            title="Augmentation Config",
        )

    def __call__(self, input_tensor: np.ndarray, gt_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        flip_h  = self._rng.random() < self.config.p_flip_h
        flip_v  = self._rng.random() < self.config.p_flip_v
        rotate  = self.config.p_rot90 > 0.0 and self._rng.random() < self.config.p_rot90
        k       = int(self._rng.integers(1, 4)) if rotate else 0

        sl_h    = slice(None, None, -1) if flip_h else slice(None)
        sl_v    = slice(None, None, -1) if flip_v else slice(None)
        sl      = (Ellipsis, sl_v, sl_h)

        input_view = input_tensor[sl]
        gt_view    = gt_params[sl]

        if k:
            input_view = np.rot90(input_view, k=k, axes=(-2, -1))
            gt_view    = np.rot90(gt_view, k=k, axes=(-2, -1))

        input_tensor = np.ascontiguousarray(input_view)
        gt_params    = np.ascontiguousarray(gt_view)

        return input_tensor, gt_params

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def add_noise(self, normalized_input: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self.config.p_noise:
            return normalized_input

        noise = self._rng.normal(0.0, self.config.noise_std, normalized_input.shape).astype(normalized_input.dtype)

        return normalized_input + noise


class PatchDataset(Dataset):
    def __init__(
        self,
        inputs           : np.ndarray,
        gt_parameters    : np.ndarray,
        grid             : Patcher,
        input_config     : InputConfig,
        output_config    : OutputConfig,
        split_name       : str,
        n_secondaries    : int,
        n_interferograms : int,
        normalizer       : Optional[Normalizer]       = None,
        x_axis           : Optional[np.ndarray]       = None,
        n_gaussians      : int                        = 1,
        augmenter        : Optional[SpatialAugmenter] = None,
        dem              : Optional[np.ndarray]       = None,
    ) -> None:

        self.inputs         = inputs
        self.gt_parameters  = gt_parameters
        self.grid           = grid
        self.input_config   = input_config
        self.output_config  = output_config
        self.split_name     = split_name
        self.normalizer     = normalizer
        self.x_axis         = x_axis
        self.n_gaussians    = n_gaussians
        self.augmenter      = augmenter
        self.dem            = dem
        self.input_layers   = int(inputs.shape[0])

        self.n_secondaries    = n_secondaries
        self.n_interferograms = n_interferograms

        expected_layers = 1 + n_secondaries + n_interferograms
        if expected_layers != self.input_layers:
            raise ValueError(f"Input stack has {self.input_layers} layers but the dataset artifacts describe {expected_layers} (1 primary + {n_secondaries} secondaries + {n_interferograms} interferograms)")

        if input_config.use_secondaries and n_secondaries == 0:
            raise ValueError("Input config requests secondaries but the dataset stack contains none. Rebuild the dataset with secondaries or set use_secondaries=False.")

        if input_config.use_interferograms and n_interferograms == 0:
            raise ValueError("Input config requests interferograms but the dataset stack contains none. Rebuild the dataset with interferograms or set use_interferograms=False.")

        self.input_channels = input_config.total_channels(n_secondaries, n_interferograms)

        self.output_channel_indices = output_config.selected_indices(n_gaussians = n_gaussians)
        self.gt_channels            = len(self.output_channel_indices)

    def __len__(self) -> int:
        return self.grid.grid.number_of_patches

    def _build_input_tensor(self, complex_patch: np.ndarray, dem_patch: Optional[np.ndarray] = None) -> np.ndarray:
        primary_data        = complex_patch[                       : 1                                              ]
        secondaries_data    = complex_patch[1                      : 1 + self.n_secondaries                         ]
        interferograms_data = complex_patch[1 + self.n_secondaries : 1 + self.n_secondaries + self.n_interferograms ]

        p_h, p_w     = complex_patch.shape[-2], complex_patch.shape[-1]
        input_tensor = np.empty((self.input_channels, p_h, p_w), dtype=np.float32)

        offset = 0

        if self.input_config.use_primary:
            n = self.input_config.primary_channels_per_pass
            self.input_config.primary_representation.convert_into(input_tensor[offset:offset + n], primary_data)
            offset += n

        if self.input_config.use_secondaries:
            n = self.n_secondaries * self.input_config.secondaries_channels_per_pass
            self.input_config.secondaries_representation.convert_into(input_tensor[offset:offset + n], secondaries_data)
            offset += n

        if self.input_config.use_interferograms:
            n = self.n_interferograms * self.input_config.interferograms_channels_per_pass
            self.input_config.interferograms_representation.convert_into(input_tensor[offset:offset + n], interferograms_data)
            offset += n

        if self.input_config.use_dem and dem_patch is not None:
            input_tensor[offset] = dem_patch
            offset += 1

        return input_tensor

    def _build_output_tensor(self, gt_patch: np.ndarray) -> np.ndarray:
        output_tensor = gt_patch[self.output_channel_indices, ...]

        return np.ascontiguousarray(output_tensor, dtype=np.float32)

    def _normalize_input_tensor(self, input_tensor: np.ndarray) -> np.ndarray:
        if self.normalizer is None:
            return input_tensor

        return self.normalizer.normalize_input(input_tensor)

    def _normalize_gt_params(self, gt_params: np.ndarray) -> np.ndarray:
        if self.normalizer is None or self.normalizer.stats.output_stats is None:
            return gt_params

        return self.normalizer.normalize_output(gt_params)

    def __getitem__(self, idx: int):
        complex_patch = self.grid.extract(self.inputs, idx)
        dem_patch     = self.grid.extract(self.dem, idx) if (self.input_config.use_dem and self.dem is not None) else None
        input_tensor  = self._build_input_tensor(complex_patch, dem_patch)

        gt_patch  = self.grid.extract(self.gt_parameters, idx)
        gt_params = self._build_output_tensor(gt_patch)

        if self.augmenter is not None and self.split_name == "train":
            input_tensor, gt_params = self.augmenter(input_tensor, gt_params)

        input_tensor = self._normalize_input_tensor(input_tensor)
        gt_params    = self._normalize_gt_params(gt_params)

        if self.augmenter is not None and self.split_name == "train":
            input_tensor = self.augmenter.add_noise(input_tensor)

        return input_tensor, gt_params


class MultiRegionDataset(Dataset):
    def __init__(self, parts: list[PatchDataset]) -> None:
        if not parts:
            raise ValueError("MultiRegionDataset requires at least one part")

        self.parts   = parts
        self.offsets = []

        total = 0
        for part in parts:
            self.offsets.append(total)
            total += len(part)
        self.total = total

        first = parts[0]

        self.input_config           = first.input_config
        self.output_config          = first.output_config
        self.split_name             = first.split_name
        self.x_axis                 = first.x_axis
        self.n_gaussians            = first.n_gaussians
        self.input_layers           = first.input_layers
        self.n_secondaries          = first.n_secondaries
        self.n_interferograms       = first.n_interferograms
        self.input_channels         = first.input_channels
        self.output_channel_indices = first.output_channel_indices
        self.gt_channels            = first.gt_channels

    @property
    def normalizer(self):
        return self.parts[0].normalizer

    @normalizer.setter
    def normalizer(self, normalizer) -> None:
        for part in self.parts:
            part.normalizer = normalizer

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.total
        if idx < 0 or idx >= self.total:
            raise IndexError(f"Index {idx} out of range for {self.total} patches")

        part_index = bisect.bisect_right(self.offsets, idx) - 1

        return self.parts[part_index][idx - self.offsets[part_index]]


class Loader:
    @staticmethod
    def build(
        train_dataset : PatchDataset,
        val_dataset   : PatchDataset,
        test_dataset  : PatchDataset,
        batch_size    : int,
        num_workers   : int,
        logger        : Logger,
        pin_memory    : bool = True,
        shuffle_train : bool = True,
        seed          : int  = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        logger.section("[Loaders]")
        logger.kv_table({
            "Batch size":    batch_size,
            "Num workers":   num_workers,
            "Pin memory":    pin_memory,
            "Shuffle train": shuffle_train,
            "Seed":          seed,
        })

        worker_init = Reproducibility.worker_init(seed) if num_workers > 0 else None

        _base = dict(
            batch_size         = batch_size,
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            persistent_workers = num_workers > 0,
            prefetch_factor    = 8 if num_workers > 0 else None,
            worker_init_fn     = worker_init,
        )

        train_loader = DataLoader(train_dataset, shuffle = shuffle_train, drop_last = True,  generator = Reproducibility.generator(seed), **_base)
        val_loader   = DataLoader(val_dataset,   shuffle = False,         drop_last = False, **_base)
        test_loader  = DataLoader(test_dataset,  shuffle = False,         drop_last = False, **_base)

        logger.kv_table({
            "Train batches": len(train_loader),
            "Val batches":   len(val_loader),
            "Test batches":  len(test_loader),
        })

        return train_loader, val_loader, test_loader
