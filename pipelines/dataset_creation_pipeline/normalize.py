from __future__ import annotations

import json
from dataclasses import dataclass
from enum        import Enum
from pathlib     import Path
from typing      import ClassVar, Optional

import numpy as np
import torch

from configuration.dataset_config import InputConfig, Representation
from tools.logger                 import Logger


class InputNormalizationMode(Enum):
    PER_CHANNEL = "per_channel"
    GROUPED     = "grouped"


class OutputNormalizationMode(Enum):
    DISABLED    = "disabled"
    PER_CHANNEL = "per_channel"
    GROUPED     = "grouped"


@dataclass
class ChannelStats:
    mean  : list[float]
    std   : list[float]
    names : Optional[list[str]] = None

    @property
    def n_channels(self) -> int:
        return len(self.mean)

    def as_dict(self) -> dict:
        entries = []
        for i, (m, s) in enumerate(zip(self.mean, self.std)):
            entry: dict = {"name": self.names[i] if self.names else f"ch{i}", "mean": m, "std": s}
            entries.append(entry)
        return {"channels": entries}

    @classmethod
    def from_dict(cls, payload: dict) -> "ChannelStats":
        if "channels" in payload:
            entries = payload["channels"]
            return cls(
                mean  = [e["mean"]  for e in entries],
                std   = [e["std"]   for e in entries],
                names = [e["name"]  for e in entries],
            )
        return cls(mean=list(payload["mean"]), std=list(payload["std"]))


@dataclass
class NormalizationStats:
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
        slot_kinds_map = NormalizationStats._REPRESENTATION_SLOT_KINDS
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
    def _collapse_to_groups(
        per_ch_count : np.ndarray,
        per_ch_mean  : np.ndarray,
        per_ch_m2    : np.ndarray,
        group_keys   : list[str],
    ) -> ChannelStats:
        
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
    ) -> "NormalizationStats":
        logger.section("[Normalization Statistics Computation]")
        logger.subsection(f"Input  mode : {input_mode.value}")
        logger.subsection(f"Output mode : {output_mode.value}")

        n_total = len(dataset)
        n_use   = min(n_total, max_samples) if max_samples > 0 else n_total
        indices = np.arange(n_total)
        if n_use < n_total:
            rng     = np.random.default_rng(42)
            indices = rng.choice(n_total, size=n_use, replace=False)
            indices.sort()
        logger.subsection(f"Samples used : {n_use:,} / {n_total:,}")

        sample      = dataset[0]
        in_first    = sample[0] if isinstance(sample, (tuple, list)) else sample
        in_channels = int(in_first.shape[0])
        has_gt      = isinstance(sample, (tuple, list)) and len(sample) >= 3
        gt_channels = int(sample[2].shape[0]) if has_gt else 0
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

        def _accumulate(arr: np.ndarray, count, mean, m2, n_ch: int) -> None:
            for c in range(n_ch):
                channel_data = arr[c].ravel().astype(np.float64)
                batch_n      = channel_data.size
                if batch_n == 0:
                    continue
                batch_mean = channel_data.mean()
                batch_m2   = ((channel_data - batch_mean) ** 2).sum()

                total    = count[c] + batch_n
                delta    = batch_mean - mean[c]
                mean[c]  = mean[c] + delta * batch_n / total
                m2[c]    = m2[c] + batch_m2 + delta ** 2 * count[c] * batch_n / total
                count[c] = total

        for idx in indices:
            sample = dataset[int(idx)]
            inp    = sample[0] if isinstance(sample, (tuple, list)) else sample
            if isinstance(inp, torch.Tensor):
                inp = inp.numpy()
            _accumulate(inp, in_count, in_mean, in_m2, in_channels)

            if do_output:
                gt = sample[2]
                if isinstance(gt, torch.Tensor):
                    gt = gt.numpy()
                _accumulate(gt, out_count, out_mean, out_m2, gt_channels)

        in_groups = NormalizationStats._build_input_channel_to_group(input_config, n_slaves)
        in_names  = in_groups if len(in_groups) == in_channels else [f"ch{i}" for i in range(in_channels)]

        if input_mode is InputNormalizationMode.PER_CHANNEL:
            input_stats       = NormalizationStats._per_channel_stats(in_count, in_mean, in_m2)
            input_stats.names = in_names
        else:
            if len(in_groups) != in_channels:
                logger.warning(
                    f"Group key count ({len(in_groups)}) != input channels ({in_channels}); "
                    "falling back to PER_CHANNEL stats for input."
                )
                input_stats       = NormalizationStats._per_channel_stats(in_count, in_mean, in_m2)
                input_stats.names = in_names
            else:
                input_stats       = NormalizationStats._collapse_to_groups(in_count, in_mean, in_m2, in_groups)
                input_stats.names = in_names
                NormalizationStats._log_grouping(logger, "Input", in_groups)

        logger.subsection("Input stats per channel:")
        for c in range(in_channels):
            logger.subsection(f"  ch{c:>3d}  mean={input_stats.mean[c]:+.6f}  std={input_stats.std[c]:.6f}")

        output_stats: Optional[ChannelStats] = None
        if do_output:
            out_groups = NormalizationStats._build_output_channel_to_group(gt_channels, params_per_gaussian)
            if output_mode is OutputNormalizationMode.PER_CHANNEL:
                output_stats       = NormalizationStats._per_channel_stats(out_count, out_mean, out_m2)
                output_stats.names = out_groups
            else:
                output_stats       = NormalizationStats._collapse_to_groups(out_count, out_mean, out_m2, out_groups)
                output_stats.names = out_groups
                NormalizationStats._log_grouping(logger, "Output", out_groups)

            logger.subsection("Output stats per channel:")
            for c in range(gt_channels):
                logger.subsection(f"  ch{c:>3d}  mean={output_stats.mean[c]:+.6f}  std={output_stats.std[c]:.6f}")

        return NormalizationStats(
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
    def load(cls, directory: Path, logger: Logger) -> "NormalizationStats":
        path = Path(directory) / "normalization_stats.json"
        if not path.exists():
            logger.warning(f"No normalization stats found at {path}. Running without normalization.")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        input_stats  = ChannelStats.from_dict(payload["input_stats"])  if payload.get("input_stats")  else None
        output_stats = ChannelStats.from_dict(payload["output_stats"]) if payload.get("output_stats") else None
        input_mode   = InputNormalizationMode(payload.get("input_mode",  InputNormalizationMode.PER_CHANNEL.value))
        output_mode  = OutputNormalizationMode(payload.get("output_mode", OutputNormalizationMode.DISABLED.value))

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
        return (f"NormalizationStats(input_ch={in_ch}, output_ch={out_ch}, input_mode={self.input_mode.value}, output_mode={self.output_mode.value})")


class Normalizer:
    def __init__(self, stats: NormalizationStats) -> None:
        self.stats = stats

    def _input_mean_std(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)
        mean  = torch.tensor(self.stats.input_stats.mean, dtype=tensor.dtype, device=tensor.device).view(*shape)
        std   = torch.tensor(self.stats.input_stats.std,  dtype=tensor.dtype, device=tensor.device).view(*shape)
        return mean, std

    def normalize_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.stats.input_stats is None:
            return tensor
        mean, std = self._input_mean_std(tensor)
        return (tensor - mean) / std

    def denormalize_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.stats.input_stats is None:
            return tensor
        mean, std = self._input_mean_std(tensor)
        return tensor * std + mean

    def _output_mean_std(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim == 4:
            shape = (1, -1, 1, 1)
        elif tensor.ndim == 3:
            shape = (-1, 1, 1)
        else:
            shape = (-1,)
        mean = torch.tensor(self.stats.output_stats.mean, dtype=tensor.dtype, device=tensor.device).view(*shape)
        std  = torch.tensor(self.stats.output_stats.std,  dtype=tensor.dtype, device=tensor.device).view(*shape)
        return mean, std

    def normalize_output(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.stats.output_stats is None:
            return tensor
        mean, std = self._output_mean_std(tensor)
        return (tensor - mean) / std

    def denormalize_output(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.stats.output_stats is None:
            return tensor
        mean, std = self._output_mean_std(tensor)
        return tensor * std + mean

    def __repr__(self) -> str:
        return f"Normalizer(stats={self.stats!r})"
