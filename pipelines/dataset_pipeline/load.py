from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from torch.utils.data                        import DataLoader, Dataset
from configuration.dataset_config            import InputConfig, OutputConfig
from pipelines.dataset_pipeline.normalize    import Stats, Normalizer
from pipelines.dataset_pipeline.patch        import Patcher
from tools.logger                            import Logger
from pipelines.dataset_pipeline.augmentation import SpatialAugmenter


class PatchDataset(Dataset):
    def __init__(
        self,
        inputs        : np.ndarray,
        gt_parameters : np.ndarray,
        grid          : Patcher,
        input_config  : InputConfig,
        output_config : OutputConfig,
        split_name    : str,
        norm_stats    : Optional[Stats]      = None,
        x_axis        : Optional[np.ndarray] = None,
        n_gaussians   : int                  = 1,
        augmenter     : Optional[SpatialAugmenter] = None,
    ) -> None:

        self.inputs         = inputs
        self.gt_parameters  = gt_parameters
        self.grid           = grid
        self.input_config   = input_config
        self.output_config  = output_config
        self.split_name     = split_name
        self.norm_stats     = Normalizer(norm_stats) if norm_stats is not None else None
        self.x_axis         = x_axis
        self.n_gaussians    = n_gaussians
        self.augmenter      = augmenter
        self.input_layers   = int(inputs.shape[0])

        n_rest              = self.input_layers - 1  # subtract primary
        self.n_secondaries  = n_rest // 2 if (input_config.use_secondaries and input_config.use_interferograms) else (n_rest if (input_config.use_secondaries or input_config.use_interferograms) else 0)
        self.n_slaves       = self.n_secondaries
        self.input_channels = input_config.total_channels(self.n_secondaries)

        self.output_channel_indices = output_config.selected_indices(n_gaussians = n_gaussians)
        self.gt_channels            = len(self.output_channel_indices)

    def __len__(self) -> int:
        return self.grid.grid.number_of_patches

    def _build_input_tensor(self, complex_patch: np.ndarray) -> np.ndarray:
        complex_data = complex_patch[None, ...]
      
        primary_data        = complex_data[:, :1]
        secondaries_data    = complex_data[:, 1:1 + self.n_secondaries]
        interferograms_data = complex_data[:, 1 + self.n_secondaries:]

        parts: list[np.ndarray] = []
        
        if self.input_config.use_primary:
            parts.append(self.input_config.primary_representation.convert(primary_data))
       
        if self.input_config.use_secondaries:
            parts.append(self.input_config.secondaries_representation.convert(secondaries_data))

        if self.input_config.use_interferograms:
            parts.append(self.input_config.interferograms_representation.convert(interferograms_data))

        input_tensor = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        input_tensor = input_tensor[0]  

        return np.ascontiguousarray(input_tensor, dtype=np.float32)

    def _build_output_tensor(self, gt_patch: np.ndarray) -> np.ndarray:
        output_tensor = gt_patch[self.output_channel_indices, ...]
       
        return np.ascontiguousarray(output_tensor, dtype=np.float32)
        
    def _normalize_input_tensor(self, input_tensor: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return input_tensor
        
        return self.norm_stats.normalize_input(input_tensor)

    def _normalize_gt_params(self, gt_params: np.ndarray) -> np.ndarray:
        if self.norm_stats is None or self.norm_stats.stats.output_stats is None:
            return gt_params
        
        return self.norm_stats.normalize_output(gt_params)

    def __getitem__(self, idx: int):
        complex_patch = self.grid.extract(self.inputs, idx)
        input_tensor  = self._build_input_tensor(complex_patch)

        gt_patch  = self.grid.extract(self.gt_parameters, idx)
        gt_params = self._build_output_tensor(gt_patch)

        if self.augmenter is not None and self.split_name == "train":
            input_tensor, gt_params = self.augmenter(input_tensor, gt_params)

        input_tensor = self._normalize_input_tensor(input_tensor)
        gt_params    = self._normalize_gt_params(gt_params)

        return input_tensor, gt_params


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
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        logger.section("[Loaders]")
        logger.kv_table({
            "Batch size":    batch_size,
            "Num workers":   num_workers,
            "Pin memory":    pin_memory,
            "Shuffle train": shuffle_train,
        })

        _base = dict(
            batch_size         = batch_size,
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            persistent_workers = num_workers > 0,
            prefetch_factor    = 8 if num_workers > 0 else None,
        )

        train_loader = DataLoader(train_dataset, shuffle = shuffle_train, drop_last  = True, **_base,)
        val_loader   = DataLoader(val_dataset,   shuffle = False,         drop_last  = False, **_base,)
        test_loader  = DataLoader(test_dataset,  shuffle = False,         drop_last  = False, **_base,)
         
        logger.kv_table({
            "Train batches": len(train_loader),
            "Val batches":   len(val_loader),
            "Test batches":  len(test_loader),
        })

        return train_loader, val_loader, test_loader
