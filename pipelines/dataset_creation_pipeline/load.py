from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from configuration.dataset_config  import InputConfig, TargetMode
from pipelines.dataset_creation_pipeline.normalize    import NormalizationStats, Normalizer
from pipelines.dataset_creation_pipeline.patch        import Patcher
from tools.logger                                     import Logger


class TomoPatchDataset(Dataset):
    def __init__(
        self,
        inputs           : np.ndarray,
        targets          : np.ndarray,
        gt_parameters    : np.ndarray,
        grid             : Patcher,
        input_config     : InputConfig,
        split_name       : str,
        logger           : Logger,
        norm_stats       : Optional[NormalizationStats] = None,
        target_mode      : TargetMode                   = TargetMode.RAW,
        x_axis           : Optional[np.ndarray]         = None,
        n_gaussians      : int                          = 1,
    ) -> None:
        
        self.inputs           = inputs
        self.targets          = targets
        self.gt_parameters    = gt_parameters
        self.grid             = grid
        self.input_config     = input_config
        self.split_name       = split_name
        self.logger           = logger
        self.norm_stats       = Normalizer(norm_stats) if norm_stats is not None else None
        self.target_mode      = target_mode
        self.x_axis           = x_axis
        self.n_gaussians      = n_gaussians        
        self.passes           = int(inputs.shape[0])
        self.n_slaves         = self.passes - 1
        self.input_channels   = input_config.total_channels(self.n_slaves)
        self.target_channels  = int(targets.shape[0])
        self.gt_channels      = int(gt_parameters.shape[0])

        self.logger.subsection(
            f"[Dataset:{split_name}] patches={len(self):>6}  passes={self.passes}  "
            f"in_ch={self.input_channels}  tgt_ch={self.target_channels}  gt_ch={self.gt_channels}  "
            f"target_mode={self.target_mode.value}  normalized={self.norm_stats is not None}"
        )

    def __len__(self) -> int:
        return self.grid.grid.number_of_patches

    def __getitem__(self, idx: int):
        complex_patch = self.grid.extract(self.inputs,  idx)
        target_patch  = self.grid.extract(self.targets, idx)
        converted     = self.input_config.build_tensor(complex_patch[None, ...])[0]

        input_tensor  = torch.from_numpy(np.ascontiguousarray(converted)).float()
        gt_t          = torch.from_numpy(np.ascontiguousarray(self.grid.extract(self.gt_parameters, idx))).float()

        if self.target_mode == TargetMode.GAUSSIAN_FIT:
            x      = torch.from_numpy(self.x_axis).float()
            _, H, W = gt_t.shape
            out_t  = torch.zeros(x.shape[0], H, W, dtype=torch.float32)
            xv     = x.view(-1, 1, 1)
            
            for k in range(self.n_gaussians):
                a   = gt_t[3 * k    ].unsqueeze(0)
                mu  = gt_t[3 * k + 1].unsqueeze(0)
                sig = gt_t[3 * k + 2].unsqueeze(0)
                out_t = out_t + a * torch.exp(-((xv - mu) ** 2) / (2.0 * sig * sig + 1e-8))
            
            target_tensor = out_t
        else:
            target_tensor = torch.from_numpy(np.ascontiguousarray(np.abs(target_patch))).float()

        if self.norm_stats is not None:
            input_tensor = self.norm_stats.normalize_input(input_tensor)
        
        if self.norm_stats is not None and self.norm_stats.stats.output_stats is not None:
            gt_t = self.norm_stats.normalize_output(gt_t)
        
        return input_tensor, target_tensor, gt_t


class LoaderBuilder:
    @staticmethod
    def build(
        train_dataset : TomoPatchDataset,
        val_dataset   : TomoPatchDataset,
        test_dataset  : TomoPatchDataset,
        batch_size    : int,
        num_workers   : int,
        logger        : Logger,
        shuffle_train : bool = True,
        pin_memory    : bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        common = {
            "batch_size"  : batch_size,
            "num_workers" : num_workers,
            "pin_memory"  : pin_memory,
            "drop_last"   : False,
        }

        train_loader = DataLoader(train_dataset, shuffle=shuffle_train, **common)
        val_loader   = DataLoader(val_dataset,   shuffle=False,         **common)
        test_loader  = DataLoader(test_dataset,  shuffle=False,         **common)

        logger.section("[Loaders Built]")
        logger.subsection(f"Train batches : {len(train_loader)}")
        logger.subsection(f"Val   batches : {len(val_loader)}")
        logger.subsection(f"Test  batches : {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
