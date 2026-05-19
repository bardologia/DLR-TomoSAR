from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional

import numpy as np
import torch

from configuration.dataset_config import ChannelStats, InputConfig, InputNormalizationMode, OutputConfig, OutputNormalizationMode, Representation
from tools.logger                 import Logger
from torch.utils.data             import DataLoader
from torch.utils.data             import Subset


@dataclass
class Stats:
    input_stats  : Optional[ChannelStats]    = None
    output_stats : Optional[ChannelStats]    = None
    input_mode   : InputNormalizationMode    = InputNormalizationMode.PER_CHANNEL
    output_mode  : OutputNormalizationMode   = OutputNormalizationMode.DISABLED

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out_path  = directory / "normalization_stats.json"

        payload = {
            "input_mode"   : self.input_mode.value,
            "output_mode"  : self.output_mode.value,
            "input_stats"  : self.input_stats.as_dict()  if self.input_stats  else None,
            "output_stats" : self.output_stats.as_dict() if self.output_stats else None,
        }
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        return out_path

    @classmethod
    def load(cls, directory: Path, logger: Logger) -> "Stats":
        path = Path(directory) / "normalization_stats.json"
        if not path.exists():
            logger.warning(f"No normalization stats found at {path}. Running without normalization.")
            
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        input_stats  = ChannelStats.from_dict(payload["input_stats"])  
        output_stats = ChannelStats.from_dict(payload["output_stats"]) 
        input_mode   = InputNormalizationMode(payload["input_mode"])
        output_mode  = OutputNormalizationMode(payload["output_mode"])

        logger.section(f"[Normalization stats loaded]")
        logger.subsection(f"Stats path      : {path}")
        logger.subsection(f"Input  mode     : {input_mode.value}")
        logger.subsection(f"Output mode     : {output_mode.value}")
        logger.subsection(f"Input  channels : {input_stats.n_channels}  (mode={input_mode.value})")
        logger.subsection(f"Output channels : {output_stats.n_channels}  (mode={output_mode.value})")

        return cls(
            input_stats  = input_stats,
            output_stats = output_stats,
            input_mode   = input_mode,
            output_mode  = output_mode,
        )


class StatsComputer:
    @staticmethod
    def _build_input_to_group(input_config : InputConfig, n_slaves : int) -> list[str]:
        map: dict[Representation, list[str]] = {
            Representation.REAL_IMAG     : ["raw_re_im",  "raw_re_im"],
            Representation.MAG_REAL_IMAG : ["log_mag",    "norm_re_im", "norm_re_im"],
            Representation.MAG_ANGLE     : ["log_mag",    "phase"],
            Representation.MAG_RI_ANGLE  : ["log_mag",    "norm_re_im", "norm_re_im", "phase"],
            Representation.ANGLE_ONLY    : ["phase"],
            Representation.MAG_ONLY      : ["log_mag"],
        }

        keys: list[str] = []

        def _append_block(rep: Representation, n_passes: int, source_kind: str) -> None:
            slot_kinds = map[rep]
            cpp        = len(slot_kinds)
            for i in range(n_passes * cpp):
                slot = i % cpp
                keys.append(f"{source_kind}/{slot_kinds[slot]}")

        if input_config.use_master:
            _append_block(input_config.master_representation, n_passes=1, source_kind="pass")

        if input_config.use_slaves:
            _append_block(input_config.slaves_representation, n_passes=n_slaves, source_kind="pass")

        if input_config.use_interferograms:
            _append_block(input_config.interferograms_representation, n_passes=n_slaves, source_kind="ifg")

        return keys

    @staticmethod
    def _build_output_to_group(n_channels : int, role_names : list[str]) -> list[str]:
        ppg = len(role_names)
        return [role_names[i % ppg] for i in range(n_channels)]

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
            key           = group_keys[i]
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
    def _log_grouping(logger : Logger, label : str, group_keys : list[str]) -> None:
        if len(group_keys) == 0:
            logger.subsection(f"{label} grouping (0 groups, 0 channels)")
            return

        seen: dict[str, list[int]] = {}
        for i, k in enumerate(group_keys):
            seen.setdefault(k, []).append(i)

        logger.subsection(f"{label} grouping ({len(seen)} groups, {len(group_keys)} channels):")

        def _compact_ranges(indices: list[int], max_items: int = 6) -> str:
            ranges: list[tuple[int, int]] = []
            start = indices[0]
            prev  = indices[0]

            for idx in indices[1:]:
                if idx == prev + 1:
                    prev = idx
                    continue
                ranges.append((start, prev))
                start = idx
                prev  = idx

            ranges.append((start, prev))

            parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in ranges[:max_items]]
            if len(ranges) > max_items:
                parts.append("...")

            return ", ".join(parts)

        for k, idxs in sorted(seen.items(), key=lambda kv: (kv[1][0], kv[0])):
            preview = _compact_ranges(sorted(idxs))
            logger.subsection(f"  {k:<24s} -> {len(idxs):>3d} ch  [{preview}]")

    @staticmethod
    def _get_dataset_subset(dataset, max_samples : int) -> tuple:
        n_total = len(dataset)
        n_use   = min(n_total, max_samples) if max_samples > 0 else n_total

        if n_use < n_total:
            rng     = np.random.default_rng(42)
            indices = rng.choice(n_total, size=n_use, replace=False)
            indices.sort()
            subset  = Subset(dataset, indices.tolist())
        else:
            subset  = dataset

        return subset, n_use, n_total

    @staticmethod
    def _initialize_accumulators(in_channels : int, gt_channels : int, do_output : bool) -> tuple:
        in_count = np.zeros(in_channels, dtype=np.int64)
        in_mean  = np.zeros(in_channels, dtype=np.float64)
        in_m2    = np.zeros(in_channels, dtype=np.float64)

        out_count = np.zeros(gt_channels, dtype=np.int64)   if do_output else None
        out_mean  = np.zeros(gt_channels, dtype=np.float64) if do_output else None
        out_m2    = np.zeros(gt_channels, dtype=np.float64) if do_output else None

        return in_count, in_mean, in_m2, out_count, out_mean, out_m2

    @staticmethod
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
        batch_m2   = ((flat - batch_mean[:, None]) ** 2).sum(axis=1)

        total    = count + batch_n
        delta    = batch_mean - mean
        mean[:]  = mean + delta * batch_n / total
        m2[:]    = m2 + batch_m2 + delta ** 2 * count * batch_n / total
        count[:] = total

    @staticmethod
    def _process_batches(
        subset,
        in_count, in_mean, in_m2,
        out_count, out_mean, out_m2,
        do_input    : bool,
        do_output   : bool,
        num_workers : int,
        batch_size  : int,
    ) -> None:

        loader = DataLoader(
            subset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = False,
            drop_last   = False,
        )

        for batch in loader:
            if do_input:
                inp = batch[0] if isinstance(batch, (tuple, list)) else batch
                StatsComputer._accumulate_batch(np.asarray(inp), in_count, in_mean, in_m2)

            if do_output:
                gt = np.asarray(batch[1])
                StatsComputer._accumulate_batch(gt, out_count, out_mean, out_m2)

    @staticmethod
    def _compute_input_stats(
        logger       : Logger,
        input_config : InputConfig,
        n_slaves     : int,
        input_mode   : InputNormalizationMode,
        in_channels  : int,
        in_count,
        in_mean,
        in_m2,
    ) -> ChannelStats:
        in_groups = StatsComputer._build_input_to_group(input_config, n_slaves)

        if input_mode is InputNormalizationMode.PER_CHANNEL:
            input_stats       = StatsComputer._per_channel_stats(in_count, in_mean, in_m2)
            input_stats.names = in_groups
        else:
            input_stats       = StatsComputer._collapse_to_groups(in_count, in_mean, in_m2, in_groups)
            input_stats.names = in_groups

        if input_mode is not InputNormalizationMode.PER_CHANNEL:
            StatsComputer._log_grouping(logger, "Input", in_groups)

        logger.section("[Input stats per channel]")
        for c in range(in_channels):
            logger.subsection(f" Channel {c:>3d}  mean={input_stats.mean[c]:+.6f},  std={input_stats.std[c]:.6f}")
        logger.write("\n")

        return input_stats

    @staticmethod
    def _compute_output_stats(
        logger        : Logger,
        output_config : OutputConfig,
        output_mode   : OutputNormalizationMode,
        gt_channels   : int,
        out_count,
        out_mean,
        out_m2,
    ) -> ChannelStats:
        out_groups = StatsComputer._build_output_to_group(gt_channels, output_config.role_names)

        if output_mode is OutputNormalizationMode.PER_CHANNEL:
            output_stats       = StatsComputer._per_channel_stats(out_count, out_mean, out_m2)
            output_stats.names = out_groups
        else:
            output_stats       = StatsComputer._collapse_to_groups(out_count, out_mean, out_m2, out_groups)
            output_stats.names = out_groups

        if output_mode is not OutputNormalizationMode.PER_CHANNEL:
            StatsComputer._log_grouping(logger, "Output", out_groups)

        logger.section("[Output stats per channel]")
        for c in range(gt_channels):
            logger.subsection(f" Channel {c:>3d}  mean={output_stats.mean[c]:+.6f},  std={output_stats.std[c]:.6f}")
        logger.write("\n")

        return output_stats

    @staticmethod
    def compute_from_dataset(
        dataset,
        logger              : Logger,
        input_config        : InputConfig,
        output_config       : OutputConfig,
        n_slaves            : int,
        input_mode          : InputNormalizationMode  = InputNormalizationMode.PER_CHANNEL,
        output_mode         : OutputNormalizationMode = OutputNormalizationMode.DISABLED,
        max_samples         : int                     = 0,
        num_workers         : int                     = 4,
        batch_size          : int                     = 512,
    ) -> Stats:

        logger.section("[Normalization Statistics Computation]")
        logger.subsection(f"Input  mode : {input_mode.value}")
        logger.subsection(f"Output mode : {output_mode.value}")

        subset, n_use, n_total = StatsComputer._get_dataset_subset(
            dataset     = dataset,
            max_samples = max_samples,
        )

        logger.subsection(f"Samples used : {n_use:,} / {n_total:,}")

        sample      = dataset[0]
        in_first    = sample[0] if isinstance(sample, (tuple, list)) else sample
        in_channels = int(in_first.shape[0])
        gt_channels = int(sample[1].shape[0])

        logger.subsection(f"Input channels : {in_channels}")
        logger.subsection(f"Output channels: {gt_channels} \n")

        do_input  = input_mode is not InputNormalizationMode.DISABLED
        do_output = output_mode is not OutputNormalizationMode.DISABLED

        in_count, in_mean, in_m2, out_count, out_mean, out_m2 = StatsComputer._initialize_accumulators(
            in_channels = in_channels if do_input else 0,
            gt_channels = gt_channels,
            do_output   = do_output,
        )

        StatsComputer._process_batches(
            subset      = subset,
            in_count    = in_count if do_input else None,
            in_mean     = in_mean  if do_input else None,
            in_m2       = in_m2    if do_input else None,
            out_count   = out_count,
            out_mean    = out_mean,
            out_m2      = out_m2,
            do_input    = do_input,
            do_output   = do_output,
            num_workers = num_workers,
            batch_size  = batch_size,
        )

        input_stats: Optional[ChannelStats] = None
        if do_input:
            input_stats = StatsComputer._compute_input_stats(
                logger       = logger,
                input_config = input_config,
                n_slaves     = n_slaves,
                input_mode   = input_mode,
                in_channels  = in_channels,
                in_count     = in_count,
                in_mean      = in_mean,
                in_m2        = in_m2,
            )

        output_stats: Optional[ChannelStats] = None
        if do_output:
            output_stats = StatsComputer._compute_output_stats(
                logger        = logger,
                output_config = output_config,
                output_mode   = output_mode,
                gt_channels   = gt_channels,
                out_count     = out_count,
                out_mean      = out_mean,
                out_m2        = out_m2,
            )

        return Stats(
            input_stats  = input_stats,
            output_stats = output_stats,
            input_mode   = input_mode,
            output_mode  = output_mode,
        )


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
        if isinstance(tensor, torch.Tensor):
            shape = self._broadcast_shape(tensor.ndim)
            mean  = torch.tensor(self.stats.input_stats.mean, dtype=torch.float32, device=tensor.device).reshape(shape)
            std   = torch.tensor(self.stats.input_stats.std,  dtype=torch.float32, device=tensor.device).reshape(shape)
        else:
            mean, std = self._mean_std_numpy(self.stats.input_stats, tensor.ndim)
        
        return (tensor - mean) / std

    def normalize_output(self, tensor: np.ndarray) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            shape = self._broadcast_shape(tensor.ndim)
            mean  = torch.tensor(self.stats.output_stats.mean, dtype=torch.float32, device=tensor.device).reshape(shape)
            std   = torch.tensor(self.stats.output_stats.std,  dtype=torch.float32, device=tensor.device).reshape(shape)
        else:
            mean, std = self._mean_std_numpy(self.stats.output_stats, tensor.ndim)
        
        return (tensor - mean) / std

    def denormalize_input(self, tensor):
        if isinstance(tensor, torch.Tensor):
            shape = self._broadcast_shape(tensor.ndim)
            mean  = torch.tensor(self.stats.input_stats.mean, dtype=torch.float32, device=tensor.device).reshape(shape)
            std   = torch.tensor(self.stats.input_stats.std,  dtype=torch.float32, device=tensor.device).reshape(shape)
        else:
            mean, std = self._mean_std_numpy(self.stats.input_stats, tensor.ndim)
        
        return tensor * std + mean

    def denormalize_output(self, tensor):
        if isinstance(tensor, torch.Tensor):
            shape = self._broadcast_shape(tensor.ndim)
            mean  = torch.tensor(self.stats.output_stats.mean, dtype=torch.float32, device=tensor.device).reshape(shape)
            std   = torch.tensor(self.stats.output_stats.std,  dtype=torch.float32, device=tensor.device).reshape(shape)
        else:
            mean, std = self._mean_std_numpy(self.stats.output_stats, tensor.ndim)
            
        return tensor * std + mean
