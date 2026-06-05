from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data                        import Dataset
from configuration.dataset_config            import InputConfig, OutputConfig
from pipelines.dataset_pipeline.normalizer   import Normalizer
from pipelines.dataset_pipeline.patch        import Patcher
from pipelines.dataset_pipeline.augmentation import SpatialAugmenter


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
        self.n_slaves         = n_secondaries

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

        return input_tensor, gt_params
