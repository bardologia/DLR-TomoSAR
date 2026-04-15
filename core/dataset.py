from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from STEtools.ste_io import rrat

from core.config import ChannelNorm, SplitConfig

logger = logging.getLogger(__name__)

# Pre-computed channel counts (avoids rebuilding a dict on every access).
_CHANNELS_PER_PASS: dict[str, int] = {
    "real_imag":     2,
    "mag_real_imag": 3,
    "mag_angle":     2,
    "mag_ri_angle":  4,
    "angle_only":    1,
    "mag_only":      1,
}


class Representation(Enum):
    REAL_IMAG     = "real_imag"
    MAG_REAL_IMAG = "mag_real_imag"
    MAG_ANGLE     = "mag_angle"
    MAG_RI_ANGLE  = "mag_ri_angle"
    ANGLE_ONLY    = "angle_only"
    MAG_ONLY      = "mag_only"

    def convert(self, data: np.ndarray) -> np.ndarray:
        n_samples, n_passes, h, w = data.shape
        cpp = _CHANNELS_PER_PASS[self.value]

        # --- vectorised intermediate quantities (computed once) -----------
        magnitude     = np.abs(data)                              # (N,P,H,W)
        log_magnitude = np.log1p(magnitude)                       # log(1+|z|)
        safe_mag      = np.where(magnitude > 0, magnitude, 1.0)
        normalised_re = data.real / safe_mag
        normalised_im = data.imag / safe_mag
        phase         = np.angle(data)

        # Select channels for each mode (all arrays are (N, P, H, W)).
        Re, Im = data.real, data.imag
        channels = {
            Representation.REAL_IMAG:     [Re, Im],
            Representation.MAG_REAL_IMAG: [log_magnitude, normalised_re, normalised_im],
            Representation.MAG_ANGLE:     [log_magnitude, phase],
            Representation.MAG_RI_ANGLE:  [log_magnitude, normalised_re, normalised_im, phase],
            Representation.ANGLE_ONLY:    [phase],
            Representation.MAG_ONLY:      [log_magnitude],
        }[self]

        # Interleave: for each pass p, place its channels contiguously.
        out = np.empty((n_samples, n_passes * cpp, h, w), dtype=np.float32)
        for c, arr in enumerate(channels):
            out[:, c::cpp] = arr  # arr[:, p] → out[:, p*cpp + c] for all p

        return np.nan_to_num(out, nan=0.0)


class BaseDataset(Dataset):
    inputs:    torch.Tensor
    targets:   torch.Tensor
    transform: Callable | None

    @staticmethod
    def _flatten_patch_grid(arr: np.ndarray) -> np.ndarray:
        ny, nx, *rest = arr.shape
        return arr.reshape(ny * nx, *rest)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.inputs[idx], self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class SLCPatchDataset(BaseDataset):
    def __init__(
        self,
        filepath:      str | Path,
        target_passes: Sequence[int],
        input_repr:    Representation = Representation.MAG_REAL_IMAG,
        target_repr:   Representation = Representation.MAG_ANGLE,
        pass_subset:   list | slice | None = None,
        transform:     Callable | None = None,
    ) -> None:
        
        raw = rrat(str(filepath))  # (Ny, Nx, P, H, W)
        if pass_subset is not None:
            raw = raw[:, :, pass_subset, ...]
        logger.info("SLCPatchDataset  raw=%s  file=%s", raw.shape, filepath)

        patches    = self._flatten_patch_grid(raw)
        all_idx    = set(range(patches.shape[1]))
        target_set = set(target_passes)
        input_idx  = sorted(all_idx - target_set)
        target_idx = sorted(target_set)

        self.inputs    = torch.from_numpy(input_repr.convert(patches[:, input_idx]))
        self.targets   = torch.from_numpy(target_repr.convert(patches[:, target_idx]))
        self.transform = transform


class ParameterEstimationDataset(BaseDataset):
    def __init__(
        self,
        input_path:   str | Path,
        target_path:  str | Path,
        input_repr:   Representation = Representation.MAG_ANGLE,
        target_norms: Sequence[ChannelNorm] | None = None,
        transform:    Callable | None = None,
    ) -> None:
        
        flat_inputs  = self._flatten_patch_grid(np.load(input_path))
        flat_targets = self._flatten_patch_grid(np.load(target_path))
        
        logger.info("ParameterEstimationDataset  in=%s  tgt=%s",flat_inputs.shape, flat_targets.shape)

        self.inputs = torch.from_numpy(input_repr.convert(flat_inputs))

        targets = flat_targets.astype(np.float32)
        if target_norms:
            ChannelNorm.apply(targets, target_norms)

        self.targets   = torch.nan_to_num(torch.from_numpy(targets))
        self.transform = transform


class FeatureMapDataset(BaseDataset):
    def __init__(
        self,
        directory:        str | Path,
        input_files:      Sequence[str],
        target_files:     Sequence[str],
        input_transforms: dict[str, Callable] | None = None,
        target_norms:     Sequence[ChannelNorm] | None = None,
        transform:        Callable | None = None,
    ) -> None:
        directory = Path(directory)
        input_transforms = input_transforms or {}

        inputs  = self._stack_files(directory, input_files, input_transforms)
        targets = self._stack_files(directory, target_files)

        if target_norms:
            ChannelNorm.apply(targets, target_norms)

        self.inputs    = torch.from_numpy(inputs)
        self.targets   = torch.from_numpy(targets)
        self.transform = transform
        logger.info("FeatureMapDataset  in=%s  tgt=%s", self.inputs.shape, self.targets.shape)

    @staticmethod
    def _stack_files(
        directory:  Path,
        filenames:  Sequence[str],
        transforms: dict[str, Callable] | None = None,
    ) -> np.ndarray:
        transforms = transforms or {}
        arrays: list[np.ndarray] = []
        for fname in filenames:
            arr = np.load(directory / fname).astype(np.float32)
            if fname in transforms:
                arr = transforms[fname](arr)
            if arr.ndim == 3:                       # (N, H, W) → (N, 1, H, W)
                arr = arr[:, np.newaxis]
            arrays.append(arr)
        return np.concatenate(arrays, axis=1)


class TensorPairDataset(BaseDataset):
    def __init__(
        self,
        inputs:    torch.Tensor | np.ndarray,
        targets:   torch.Tensor | np.ndarray,
        transform: Callable | None = None,
    ) -> None:
        
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs.astype(np.float32))
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets.astype(np.float32))
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"Sample-count mismatch: inputs={inputs.shape[0]}, targets={targets.shape[0]}")

        self.inputs    = inputs
        self.targets   = targets
        self.transform = transform

    @classmethod
    def from_numpy_files(
        cls,
        input_path:  str | Path,
        target_path: str | Path,
        transform:   Callable | None = None,
    ) -> TensorPairDataset:
        """Convenience constructor that loads two ``.npy`` files."""
        return cls(np.load(input_path), np.load(target_path), transform)


class DataPipeline:
    def __init__(
        self,
        dataset:     Dataset,
        split:       SplitConfig = SplitConfig(),
        batch_size:  int  = 32,
        num_workers: int  = 0,
        pin_memory:  bool = True,
        seed:        int  = 42,
    ) -> None:
        self.dataset     = dataset
        self.split       = split
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory
        self.seed        = seed

    def build(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Perform the split and return ``(train, val, test)`` loaders."""
        sizes     = self.split.to_sizes(len(self.dataset))
        generator = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds, test_ds = random_split(self.dataset, sizes, generator=generator)

        loaders = (
            self._make_loader(train_ds, shuffle=True),
            self._make_loader(val_ds,   shuffle=False),
            self._make_loader(test_ds,  shuffle=False),
        )
        logger.info(
            "DataPipeline  train=%d  val=%d  test=%d  batch=%d",
            len(train_ds), len(val_ds), len(test_ds), self.batch_size,
        )
        return loaders

    @classmethod
    def from_slc_patches(
        cls,
        filepath:      str | Path,
        target_passes: Sequence[int],
        input_repr:    Representation = Representation.MAG_REAL_IMAG,
        target_repr:   Representation = Representation.MAG_ANGLE,
        pass_subset:   list | slice | None = None,
        transform:     Callable | None = None,
        **pipeline_kw,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        
        ds = SLCPatchDataset(
            filepath, target_passes, input_repr, target_repr,
            pass_subset=pass_subset, transform=transform,
        )
        return cls(dataset=ds, **pipeline_kw).build()

    @classmethod
    def from_parameter_estimation(
        cls,
        input_path:   str | Path,
        target_path:  str | Path,
        input_repr:   Representation = Representation.MAG_ANGLE,
        target_norms: Sequence[ChannelNorm] | None = None,
        transform:    Callable | None = None,
        **pipeline_kw,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        
        ds = ParameterEstimationDataset(
            input_path, target_path, input_repr,
            target_norms=target_norms, transform=transform,
        )
        
        return cls(dataset=ds, **pipeline_kw).build()

    @classmethod
    def from_feature_maps(
        cls,
        directory:        str | Path,
        input_files:      Sequence[str],
        target_files:     Sequence[str],
        input_transforms: dict[str, Callable] | None = None,
        target_norms:     Sequence[ChannelNorm] | None = None,
        transform:        Callable | None = None,
        **pipeline_kw,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        
        ds = FeatureMapDataset(
            directory, input_files, target_files,
            input_transforms=input_transforms,
            target_norms=target_norms, transform=transform,
        )
        
        return cls(dataset=ds, **pipeline_kw).build()

    @classmethod
    def from_numpy_files(
        cls,
        input_path:  str | Path,
        target_path: str | Path,
        transform:   Callable | None = None,
        **pipeline_kw,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        
        ds = TensorPairDataset.from_numpy_files(input_path, target_path, transform=transform)
        
        return cls(dataset=ds, **pipeline_kw).build()


    def _make_loader(self, subset: Dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset     = subset,
            batch_size  = self.batch_size,
            shuffle     = shuffle,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
        )
