from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset

from configuration.dataset_config                     import InputConfig
from pipelines.dataset_creation_pipeline.normalize    import Stats, Normalizer
from pipelines.dataset_creation_pipeline.patch        import Patcher
from tools.logger                                     import Logger


class PatchDataset(Dataset):
    def __init__(
        self,
        inputs           : np.ndarray,
        gt_parameters    : np.ndarray,
        grid             : Patcher,
        input_config     : InputConfig,
        split_name       : str,
        logger           : Logger,
        norm_stats       : Optional[Stats] = None,
        x_axis           : Optional[np.ndarray]         = None,
        n_gaussians      : int                          = 1,
    ) -> None:

        self.inputs           = inputs
        self.gt_parameters    = gt_parameters
        self.grid             = grid
        self.input_config     = input_config
        self.split_name       = split_name
        self.norm_stats       = Normalizer(norm_stats) if norm_stats is not None else None
        self.x_axis           = x_axis
        self.n_gaussians      = n_gaussians
        self.passes           = int(inputs.shape[0])
        self.n_slaves         = self.passes - 1
        self.input_channels   = input_config.total_channels(self.n_slaves)
        self.gt_channels      = int(gt_parameters.shape[0])

        logger.subsection(
            f"[Dataset:{split_name}] patches={len(self):>6}  passes={self.passes}  "
            f"in_ch={self.input_channels}  gt_ch={self.gt_channels}  "
            f"normalized={self.norm_stats is not None}"
        )

    def __len__(self) -> int:
        return self.grid.grid.number_of_patches

    def __getitem__(self, idx: int):
        complex_patch = self.grid.extract(self.inputs, idx)
        converted     = self.input_config.build_tensor(complex_patch[None, ...])[0]
        input_tensor  = np.ascontiguousarray(converted, dtype=np.float32)

        gt_params = np.ascontiguousarray(self.grid.extract(self.gt_parameters, idx), dtype=np.float32)

        if self.norm_stats is not None:
            input_tensor = self.norm_stats.normalize_input(input_tensor)

        if self.norm_stats is not None and self.norm_stats.stats.output_stats is not None:
            gt_params = self.norm_stats.normalize_output(gt_params)

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


        _base = dict(
            batch_size              = batch_size,
            num_workers             = num_workers,
            pin_memory              = pin_memory,
            persistent_workers      = num_workers > 0,
            prefetch_factor         = 8 if num_workers > 0 else None,
        )

        train_loader = DataLoader(train_dataset, shuffle = shuffle_train, drop_last  = True, **_base,)
        val_loader   = DataLoader(val_dataset,   shuffle = False,         drop_last  = False, **_base,)
        test_loader  = DataLoader(test_dataset,  shuffle = False,         drop_last  = False, **_base,)
         

        logger.section("[Loaders Built]")
        logger.subsection(f"Workers       : {num_workers}  (spawn)")
        logger.subsection(f"Train batches : {len(train_loader)}")
        logger.subsection(f"Val   batches : {len(val_loader)}")
        logger.subsection(f"Test  batches : {len(test_loader)}")

        return train_loader, val_loader, test_loader
