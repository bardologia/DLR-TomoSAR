from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset

from configuration.dataset_config                     import InputConfig, OutputConfig
from pipelines.dataset_creation_pipeline.normalize    import Stats, Normalizer
from pipelines.dataset_creation_pipeline.patch        import Patcher
from tools.logger                                     import Logger


class PatchDataset(Dataset):
    def __init__(
        self,
        inputs        : np.ndarray,
        gt_parameters : np.ndarray,
        grid          : Patcher,
        input_config  : InputConfig,
        output_config : OutputConfig,
        split_name    : str,
        logger        : Logger,
        norm_stats    : Optional[Stats]      = None,
        x_axis        : Optional[np.ndarray] = None,
        n_gaussians   : int                  = 1,
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
        self.passes         = int(inputs.shape[0])
        self.n_slaves       = self.passes - 1
        self.input_channels = input_config.total_channels(self.n_slaves)

        self.output_channel_indices = output_config.selected_indices(n_gaussians = int(gt_parameters.shape[0]) // 3)
        self.gt_channels            = len(self.output_channel_indices)

        logger.section(f"[Dataset:{split_name}]")
        logger.subsection(f"Patches  : {len(self):>6}")
        logger.subsection(f"Passes   : {self.passes}")
        logger.subsection(f"Input ch : {self.input_channels}")
        logger.subsection(f"GT ch    : {self.gt_channels}")

        logger.section("Pipeline Flux")
        logger.subsection("input patch -> input tensor + gt params -> (norm) input tensor + (norm) gt params \n")

    def __len__(self) -> int:
        return self.grid.grid.number_of_patches

    def _build_input_tensor(self, complex_patch: np.ndarray) -> np.ndarray:
        complex_data = complex_patch[None, ...]
      
        master_data = complex_data[:, :1]
        slave_data  = complex_data[:, 1:] 

        parts: list[np.ndarray] = []
        
        if self.input_config.use_master:
            parts.append(self.input_config.master_representation.convert(master_data))
       
        if self.input_config.use_slaves:
            parts.append(self.input_config.slaves_representation.convert(slave_data))

        if self.input_config.use_interferograms:
            interferograms = slave_data * np.conj(master_data)
            parts.append(self.input_config.interferograms_representation.convert(interferograms))

        input_tensor = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        
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
        logger.subsection(f"Batch size   : {batch_size}")
        logger.subsection(f"Num workers  : {num_workers}")
        logger.subsection(f"Pin memory   : {pin_memory}")
        logger.subsection(f"Shuffle train : {shuffle_train}")

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
         
        logger.subsection(f"Train batches : {len(train_loader)}")
        logger.subsection(f"Val   batches : {len(val_loader)}")
        logger.subsection(f"Test  batches : {len(test_loader)} \n")

        return train_loader, val_loader, test_loader
