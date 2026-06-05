from __future__ import annotations

import bisect

from torch.utils.data import Dataset

from pipelines.dataset_pipeline.dataset import PatchDataset


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
        self.n_slaves               = first.n_slaves
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
