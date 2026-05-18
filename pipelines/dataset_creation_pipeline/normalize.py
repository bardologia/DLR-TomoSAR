from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib     import Path
from typing      import ClassVar, Optional

import numpy as np

from configuration.dataset_config import ChannelStats, InputConfig, InputNormalizationMode, OutputNormalizationMode, Representation
from tools.logger                 import Logger


@dataclass
class Stats:
    input_stats  : Optional[ChannelStats]    = None
    output_stats : Optional[ChannelStats]    = None
    input_mode   : InputNormalizationMode    = InputNormalizationMode.PER_CHANNEL
    output_mode  : OutputNormalizationMode   = OutputNormalizationMode.DISABLED

    _REPRESENTATION_SLOT_KINDS: ClassVar[dict[Representation, list[str]]] = {
        Representation.REAL_IMAG     : ["raw_re_im",  "raw_re_im"],
        Representation.MAG_REAL_IMAG : ["log_mag",    "norm_re_im", "norm_re_im"],
        Representation.MAG_ANGLE     : ["log_mag",    "phase"],
        Representation.MAG_RI_ANGLE  : ["log_mag",    "norm_re_im", "norm_re_im", "phase"],
        Representation.ANGLE_ONLY    : ["phase"],
        Representation.MAG_ONLY      : ["log_mag"],
    }

    @staticmethod
    def _build_input_channel_to_group(input_config: InputConfig, n_slaves: int) -> list[str]:
        slot_kinds_map = Stats._REPRESENTATION_SLOT_KINDS
        keys: list[str] = []

        def _append_block(rep: Representation, n_passes: int, source_kind: str) -> None:
            slot_kinds = slot_kinds_map[rep]
            cpp        = len(slot_kinds)
            for i in range(n_passes * cpp):
                slot = i % cpp
                keys.append(f"{source_kind}/{slot_kinds[slot]}")

        if input_config.use_master:
            _append_block(input_config.master_representation, n_passes=1, source_kind="pass")

        if n_slaves > 0:
            if input_config.use_slaves:
                _append_block(input_config.slaves_representation, n_passes=n_slaves, source_kind="pass")
            if input_config.use_interferograms:
                _append_block(input_config.interferograms_representation, n_passes=n_slaves, source_kind="ifg")

        return keys

    @staticmethod
    def _build_output_channel_to_group(n_channels: int, params_per_gaussian: int) -> list[str]:
        role_names = ["a", "mu", "sig"] + [f"p{i}" for i in range(3, params_per_gaussian)]
        return [role_names[i % params_per_gaussian] for i in range(n_channels)]

    @staticmethod
    def _per_channel_stats(
        per_ch_count : np.ndarray,
        per_ch_mean  : np.ndarray,
        per_ch_m2    : np.ndarray,
    ) -> ChannelStats:
        
        std = np.sqrt(per_ch_m2 / np.maximum(per_ch_count - 1, 1))
        std = np.where(std < 1e-8, 1.0, std)
        return ChannelStats(mean=per_ch_mean.tolist(), std=std.tolist())

    @staticmethod
    def _collapse_to_groups(per_ch_count : np.ndarray, per_ch_mean : np.ndarray, per_ch_m2 : np.ndarray, group_keys : list[str]) -> ChannelStats:
        n         = len(group_keys)
        grp_count : dict[str, float] = {}
        grp_mean  : dict[str, float] = {}
        grp_m2    : dict[str, float] = {}

        for i in range(n):
            key            = group_keys[i]
            c_i, m_i, s_i = float(per_ch_count[i]), float(per_ch_mean[i]), float(per_ch_m2[i])
            if key not in grp_count:
                grp_count[key] = c_i
                grp_mean[key]  = m_i
                grp_m2[key]    = s_i
                continue
            
            c_a, m_a, s_a = grp_count[key], grp_mean[key], grp_m2[key]
            c_b, m_b, s_b = c_i,            m_i,            s_i
            total = c_a + c_b
            
            if total <= 0:
                continue
            
            delta          = m_b - m_a
            grp_mean[key]  = m_a + delta * c_b / total
            grp_m2[key]    = s_a + s_b + delta * delta * c_a * c_b / total
            grp_count[key] = total

        means = np.empty(n, dtype=np.float64)
        stds  = np.empty(n, dtype=np.float64)
        for i, key in enumerate(group_keys):
            c        = grp_count[key]
            means[i] = grp_mean[key]
            var      = grp_m2[key] / max(c - 1.0, 1.0)
            stds[i]  = float(np.sqrt(var))
        
        stds = np.where(stds < 1e-8, 1.0, stds)
        
        return ChannelStats(mean=means.tolist(), std=stds.tolist())

    @staticmethod
    def _log_grouping(logger: Logger, label: str, group_keys: list[str]) -> None:
        seen: dict[str, list[int]] = {}
        
        for i, k in enumerate(group_keys):
            seen.setdefault(k, []).append(i)
        
        logger.subsection(f"{label} grouping ({len(seen)} groups):")
        for k, idxs in seen.items():
            preview = ",".join(str(i) for i in idxs[:8]) + ("..." if len(idxs) > 8 else "")
            logger.subsection(f"  {k:<24s} -> {len(idxs)} ch  [{preview}]")

    @staticmethod
    def compute_from_dataset(
        dataset,
        logger              : Logger,
        input_config        : InputConfig,
        n_slaves            : int,
        params_per_gaussian : int                     = 3,
        input_mode          : InputNormalizationMode  = InputNormalizationMode.PER_CHANNEL,
        output_mode         : OutputNormalizationMode = OutputNormalizationMode.DISABLED,
        max_samples         : int                     = 0,
        num_workers         : int                     = 4,
        batch_size          : int                     = 512,
    ) -> "Stats":
        from torch.utils.data import DataLoader, Subset

        logger.section("[Normalization Statistics Computation]")
        logger.subsection(f"Input  mode : {input_mode.value}")
        logger.subsection(f"Output mode : {output_mode.value}")

        n_total = len(dataset)
        n_use   = min(n_total, max_samples) if max_samples > 0 else n_total
        
        if n_use < n_total:
            rng     = np.random.default_rng(42)
            indices = rng.choice(n_total, size=n_use, replace=False)
            indices.sort()
            subset  = Subset(dataset, indices.tolist())
        else:
            subset  = dataset
        
        logger.subsection(f"Samples used : {n_use:,} / {n_total:,}")

        sample      = dataset[0]
        in_first    = sample[0] if isinstance(sample, (tuple, list)) else sample
        in_channels = int(in_first.shape[0])
        has_gt      = isinstance(sample, (tuple, list)) and len(sample) >= 2
        gt_channels = int(sample[1].shape[0]) if has_gt else 0

        logger.subsection(f"Input channels : {in_channels}")
        if has_gt:
            logger.subsection(f"Output channels: {gt_channels}")
        elif output_mode is not OutputNormalizationMode.DISABLED:
            logger.warning("Output normalisation requested but dataset has no gt_parameters; disabling.")
            output_mode = OutputNormalizationMode.DISABLED

        in_count = np.zeros(in_channels, dtype=np.int64)
        in_mean  = np.zeros(in_channels, dtype=np.float64)
        in_m2    = np.zeros(in_channels, dtype=np.float64)

        do_output = has_gt and (output_mode is not OutputNormalizationMode.DISABLED)
        out_count = np.zeros(gt_channels, dtype=np.int64)   if do_output else None
        out_mean  = np.zeros(gt_channels, dtype=np.float64) if do_output else None
        out_m2    = np.zeros(gt_channels, dtype=np.float64) if do_output else None

        def _accumulate_batch(arr: np.ndarray, count, mean, m2) -> None:
            arr = np.asarray(arr, dtype=np.float64)
            
            if arr.ndim == 1:                   
                arr = arr[np.newaxis, :, np.newaxis]
            elif arr.ndim == 2:                  
                arr = arr[:, :, np.newaxis]
            
            n_ch  = arr.shape[1]
            flat  = arr.reshape(arr.shape[0], n_ch, -1) 
            flat  = flat.transpose(1, 0, 2).reshape(n_ch, -1) 

            batch_n    = flat.shape[1]
            batch_mean = flat.mean(axis=1)                       
            batch_m2   = ((flat - batch_mean[:, None]) ** 2).sum(axis=1)  #

            total    = count + batch_n
            delta    = batch_mean - mean
            mean[:]  = mean + delta * batch_n / total
            m2[:]    = m2 + batch_m2 + delta ** 2 * count * batch_n / total
            count[:] = total

        loader = DataLoader(
            subset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = False,
            multiprocessing_context = "spawn" if num_workers > 0 else None,
            drop_last   = False,
        )

        for batch in loader:
            inp = batch[0] if isinstance(batch, (tuple, list)) else batch
            _accumulate_batch(np.asarray(inp), in_count, in_mean, in_m2)

            if do_output:
                gt = np.asarray(batch[1])
                _accumulate_batch(gt, out_count, out_mean, out_m2)

        in_groups = Stats._build_input_channel_to_group(input_config, n_slaves)
        in_names  = in_groups if len(in_groups) == in_channels else [f"ch{i}" for i in range(in_channels)]

        if input_mode is InputNormalizationMode.PER_CHANNEL:
            input_stats       = Stats._per_channel_stats(in_count, in_mean, in_m2)
            input_stats.names = in_names
        else:
            input_stats       = Stats._collapse_to_groups(in_count, in_mean, in_m2, in_groups)
            input_stats.names = in_names
            Stats._log_grouping(logger, "Input", in_groups)

        logger.subsection("Input stats per channel:")
        for c in range(in_channels):
            logger.subsection(f"  ch{c:>3d}  mean={input_stats.mean[c]:+.6f}  std={input_stats.std[c]:.6f}")

        output_stats: Optional[ChannelStats] = None
        if do_output:
            out_groups = Stats._build_output_channel_to_group(gt_channels, params_per_gaussian)
            if output_mode is OutputNormalizationMode.PER_CHANNEL:
                output_stats       = Stats._per_channel_stats(out_count, out_mean, out_m2)
                output_stats.names = out_groups
            else:
                output_stats       = Stats._collapse_to_groups(out_count, out_mean, out_m2, out_groups)
                output_stats.names = out_groups
                Stats._log_grouping(logger, "Output", out_groups)

            logger.subsection("Output stats per channel:")
            for c in range(gt_channels):
                logger.subsection(f"  ch{c:>3d}  mean={output_stats.mean[c]:+.6f}  std={output_stats.std[c]:.6f}")

        return Stats(
            input_stats  = input_stats,
            output_stats = output_stats,
            input_mode   = input_mode,
            output_mode  = output_mode,
        )

    def save(self, directory: Path, logger: Logger) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / "normalization_stats.json"

        payload = {
            "input_mode"   : self.input_mode.value,
            "output_mode"  : self.output_mode.value,
            "input_stats"  : self.input_stats.as_dict()  if self.input_stats  else None,
            "output_stats" : self.output_stats.as_dict() if self.output_stats else None,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        logger.subsection(f"-> Normalization stats saved: {out_path}")
        return out_path

    @classmethod
    def load(cls, directory: Path, logger: Logger) -> "Stats":
        path = Path(directory) / "normalization_stats.json"
        if not path.exists():
            logger.warning(f"No normalization stats found at {path}. Running without normalization.")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        input_stats  = ChannelStats.from_dict(payload["input_stats"])  
        output_stats = ChannelStats.from_dict(payload["output_stats"]) 
        input_mode   = InputNormalizationMode(payload["input_mode"])
        output_mode  = OutputNormalizationMode(payload["output_mode"])

        logger.subsection(f"<- Normalization stats loaded: {path}")
        if input_stats:
            logger.subsection(f"   Input  channels : {input_stats.n_channels}  (mode={input_mode.value})")
        if output_stats:
            logger.subsection(f"   Output channels : {output_stats.n_channels}  (mode={output_mode.value})")

        return cls(
            input_stats  = input_stats,
            output_stats = output_stats,
            input_mode   = input_mode,
            output_mode  = output_mode,
        )

    def __repr__(self) -> str:
        in_ch  = self.input_stats.n_channels  if self.input_stats  else 0
        out_ch = self.output_stats.n_channels if self.output_stats else 0
        return (f"Stats(input_ch={in_ch}, output_ch={out_ch}, input_mode={self.input_mode.value}, output_mode={self.output_mode.value})")


class Normalizer:
    def __init__(self, stats: Stats) -> None:
        self.stats = stats

    @staticmethod
    def _broadcast_shape(ndim: int) -> tuple:
        if ndim == 4:
            return (1, -1, 1, 1)
        elif ndim == 3:
            return (-1, 1, 1)
        return (-1,)

    def _mean_std_numpy(self, channel_stats, ndim: int):
        shape = self._broadcast_shape(ndim)
        mean  = np.asarray(channel_stats.mean, dtype=np.float32).reshape(shape)
        std   = np.asarray(channel_stats.std,  dtype=np.float32).reshape(shape)
        return mean, std

    def normalize_input(self, tensor: np.ndarray) -> np.ndarray:
        if self.stats.input_stats is None:
            return tensor
        mean, std = self._mean_std_numpy(self.stats.input_stats, tensor.ndim)
        return (tensor - mean) / std

    def normalize_output(self, tensor: np.ndarray) -> np.ndarray:
        if self.stats.output_stats is None:
            return tensor
        mean, std = self._mean_std_numpy(self.stats.output_stats, tensor.ndim)
        return (tensor - mean) / std

    def denormalize_output(self, tensor):
        if self.stats.output_stats is None:
            return tensor
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                shape = self._broadcast_shape(tensor.ndim)
                mean  = torch.tensor(self.stats.output_stats.mean, dtype=torch.float32, device=tensor.device).reshape(shape)
                std   = torch.tensor(self.stats.output_stats.std,  dtype=torch.float32, device=tensor.device).reshape(shape)
                return tensor * std + mean
        except ImportError:
            pass
        mean, std = self._mean_std_numpy(self.stats.output_stats, tensor.ndim)
        return tensor * std + mean

    def __repr__(self) -> str:
        return f"Normalizer(stats={self.stats!r})"
