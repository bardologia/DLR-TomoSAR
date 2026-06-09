from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from configuration.norm_config    import ChannelStats, ChannelStrategy, NormMethod
from configuration.dataset_config import InputConfig, OutputConfig
from pipelines.shared.io          import FileIO
from tools.logger                 import Logger
from tools.ranges                 import RangeFormatter


@dataclass
class Stats:
    input_stats  : Optional[ChannelStats]  = None
    output_stats : Optional[ChannelStats]  = None

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        out_path  = directory / "normalization_stats.json"

        payload = {
            "input_stats"  : self.input_stats.as_dict()  if self.input_stats  else None,
            "output_stats" : self.output_stats.as_dict() if self.output_stats else None,
        }

        return FileIO.save_json(payload, out_path, indent=4)

    @classmethod
    def load(cls, directory: Path, logger: Logger) -> "Stats":
        path = Path(directory) / "normalization_stats.json"
        if not path.exists():
            raise FileNotFoundError(f"Normalization stats not found at '{path}'.")

        payload = FileIO.load_json(path)

        input_stats  = ChannelStats.from_dict(payload["input_stats"])
        output_stats = ChannelStats.from_dict(payload["output_stats"])

        logger.section("[Normalization stats loaded]")
        logger.kv_table({
            "Stats path":      path,
            "Input channels":  input_stats.n_channels,
            "Output channels": output_stats.n_channels,
        })

        return cls(input_stats = input_stats, output_stats = output_stats)

    @classmethod
    def merge(cls, input_only: "Stats", output_only: "Stats") -> "Stats":
        return cls(
            input_stats  = input_only.input_stats,
            output_stats = output_only.output_stats,
        )


class StatsComputer:
    @staticmethod
    def _input_to_group(input_config : InputConfig, n_secondaries : int, n_interferograms : int) -> list[str]:
        return input_config.channel_group_keys(n_secondaries, n_interferograms)

    @staticmethod
    def _log_grouping(logger : Logger, label : str, group_keys : list[str]) -> None:
        if len(group_keys) == 0:
            logger.subsection(f"{label} grouping (0 groups, 0 channels)")
            return

        seen: dict[str, list[int]] = {}
        for i, k in enumerate(group_keys):
            seen.setdefault(k, []).append(i)

        logger.subsection(f"{label} grouping ({len(seen)} groups, {len(group_keys)} channels):")

        rows = {
            k: f"{len(idxs):d} ch  [{RangeFormatter.compact(sorted(idxs))}]"
            for k, idxs in sorted(seen.items(), key=lambda kv: (kv[1][0], kv[0]))
        }
        logger.kv_table(rows)

    @staticmethod
    def _get_subset(dataset, max_samples : int) -> tuple:
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
    def _collect(
        subset,
        group_keys         : list[str],
        num_workers        : int,
        batch_size         : int,
        max_vals_per_group : int = 1_000_000,
    ) -> dict[str, np.ndarray]:

        unique_groups  = list(dict.fromkeys(group_keys))
        group_channels = {g: [i for i, k in enumerate(group_keys) if k == g] for g in unique_groups}
        needs_data     = {g for g in unique_groups if ChannelStrategy.from_slot(g).norm_method is not NormMethod.FIXED_DIV_PI}

        collected: dict[str, list[np.ndarray]] = {g: [] for g in needs_data}
        if not collected:
            return {}

        n_batches_est        = max(len(subset) // max(batch_size, 1), 1)
        vals_per_ch_batch    = {g: max(64, max_vals_per_group // (n_batches_est * max(len(channels), 1))) for g, channels in group_channels.items()}

        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        rng    = np.random.default_rng(42)

        for batch in loader:
            inp = batch[0] if isinstance(batch, (tuple, list)) else batch
            arr = np.asarray(inp, dtype=np.float32)

            if arr.ndim == 3:
                arr = arr[np.newaxis]

            for g, channels in group_channels.items():
                if g not in needs_data:
                    continue

                budget = vals_per_ch_batch[g]

                for ch in channels:
                    flat = arr[:, ch].ravel()

                    if len(flat) > budget:
                        idx  = rng.choice(len(flat), budget, replace=False)
                        flat = flat[idx]

                    collected[g].append(flat)

        return {g: np.concatenate(v) for g, v in collected.items() if v}

    @staticmethod
    def _fit_input(logger : Logger, group_keys : list[str], collected : dict[str, np.ndarray]) -> ChannelStats:
        unique_groups    = list(dict.fromkeys(group_keys))
        group_strategies = {g: ChannelStrategy.from_slot(g) for g in unique_groups}

        group_mean_std: dict[str, tuple[float, float]] = {g: group_strategies[g].fit(collected.get(g, np.array([]))) for g in unique_groups}

        n          = len(group_keys)
        locs       = [group_mean_std[group_keys[i]][0] for i in range(n)]
        scales     = [group_mean_std[group_keys[i]][1] for i in range(n)]
        strategies = [group_strategies[group_keys[i]]  for i in range(n)]

        logger.section("[Input stats per channel]")
        rows = []
        for c in range(n):
            strat = strategies[c]
            rows.append({
                "Ch":       str(c),
                "Slot":     group_keys[c],
                "Method":   strat.norm_method.value,
                "log1p":    str(strat.apply_log1p),
                "loc":      f"{locs[c]:+.6f}",
                "scale":    f"{scales[c]:.6f}",
            })
        logger.metrics_table(rows, ["Ch", "Slot", "Method", "log1p", "loc", "scale"])

        return ChannelStats(
            loc        = locs,
            scale      = scales,
            names      = group_keys,
            strategies = strategies,
        )

    @staticmethod
    def _fit_output(
        logger        : Optional[Logger],
        role_pools    : dict[str, np.ndarray],
        output_config : "OutputConfig",
        n_gaussians   : int,
    ) -> ChannelStats:

        role_fit   : dict[str, tuple[float, float]]      = {key: output_config.strategy_for(key).fit(pool) for key, pool in role_pools.items()}
        role_strat : dict[str, ChannelStrategy] = {key: output_config.strategy_for(key) for key in role_pools}

        selected       = output_config.selected_indices(n_gaussians)
        _local_to_role = {0: "out/amp", 1: "out/mu", 2: "out/sigma"}

        locs:       list[float]                    = []
        scales:     list[float]                    = []
        names:      list[str]                      = []
        strategies: list[ChannelStrategy] = []

        for out_ch, full_ch in enumerate(selected):
            g        = full_ch // 3
            role_key = _local_to_role[full_ch % 3]
            m, s     = role_fit[role_key]
            locs.append(m)
            scales.append(s)
            names.append(f"G{g+1}_{role_key.split('/')[1]}")
            strategies.append(role_strat[role_key])

        if logger is not None:
            logger.section("[Output stats from params]")
            rows = []
            for key, (m, s) in role_fit.items():
                strat = role_strat[key]
                rows.append({
                    "Channel":  key,
                    "loc":      f"{m:.5f}",
                    "scale":    f"{s:.5f}",
                    "Method":    strat.norm_method.value,
                    "log1p":    str(strat.apply_log1p),
                })
            logger.metrics_table(rows, ["Channel", "loc", "scale", "Method", "log1p"])

        return ChannelStats(
            loc        = locs,
            scale      = scales,
            names      = names,
            strategies = strategies,
        )

    @staticmethod
    def compute_input_stats(
        dataset,
        logger           : Logger,
        input_config     : InputConfig,
        n_secondaries    : int,
        n_interferograms : int,
        max_samples      : int = 0,
        num_workers      : int = 4,
        batch_size       : int = 512,
    ) -> Stats:

        logger.section("[Input Normalization Statistics]")
        subset, n_use, n_total = StatsComputer._get_subset(dataset, max_samples)
        sample      = dataset[0]
        in_first    = sample[0] if isinstance(sample, (tuple, list)) else sample
        in_channels = int(in_first.shape[0])

        logger.kv_table({
            "Strategy":       "auto-selected per slot-kind (grouped across passes/ifgs)",
            "Samples":        f"{n_use:,} / {n_total:,}",
            "Input channels": in_channels,
        })

        group_keys = StatsComputer._input_to_group(input_config, n_secondaries, n_interferograms)
        assert len(group_keys) == in_channels, (f"Group key count ({len(group_keys)}) != tensor channels ({in_channels})")

        logger.section("[Input grouping by slot-kind]")
        StatsComputer._log_grouping(logger, "Input", group_keys)

        collected = StatsComputer._collect(
            subset      = subset,
            group_keys  = group_keys,
            num_workers = num_workers,
            batch_size  = batch_size,
        )

        input_stats = StatsComputer._fit_input(
            logger     = logger,
            group_keys = group_keys,
            collected  = collected,
        )

        return Stats(
            input_stats  = input_stats,
            output_stats = None,
        )

    @staticmethod
    def _train_gt_parameters(dataset) -> list[np.ndarray]:
        if hasattr(dataset, "parts"):
            return [part.gt_parameters for part in dataset.parts]

        return [dataset.gt_parameters]

    @staticmethod
    def compute_output_stats(
        dataset,
        n_gaussians   : int,
        output_config : "OutputConfig",
        amp_threshold : float = 1e-2,
        logger        : Optional[Logger] = None,
    ) -> Stats:
        regions = StatsComputer._train_gt_parameters(dataset)

        amp_pool_vals:  list[np.ndarray] = []
        mu_pool_vals:   list[np.ndarray] = []
        sig_pool_vals:  list[np.ndarray] = []

        for params in regions:
            for g in range(n_gaussians):
                a_flat   = params[g * 3 + 0].ravel().astype(np.float64)
                mu_flat  = params[g * 3 + 1].ravel().astype(np.float64)
                sig_flat = params[g * 3 + 2].ravel().astype(np.float64)
                active   = a_flat > amp_threshold

                amp_pool_vals.append(a_flat[active])
                mu_pool_vals.append(mu_flat[active])
                sig_pool_vals.append(sig_flat[active])

        role_pools = {
            "out/amp"   : np.concatenate(amp_pool_vals),
            "out/mu"    : np.concatenate(mu_pool_vals),
            "out/sigma" : np.concatenate(sig_pool_vals),
        }

        output_stats = StatsComputer._fit_output(
            logger        = logger,
            role_pools    = role_pools,
            output_config = output_config,
            n_gaussians   = n_gaussians,
        )

        return Stats(
            input_stats  = None,
            output_stats = output_stats,
        )

    @staticmethod
    def _detach_augmenters(dataset) -> list:
        parts    = dataset.parts if hasattr(dataset, "parts") else [dataset]
        detached = [(part, part.augmenter) for part in parts]

        for part in parts:
            part.augmenter = None

        return detached

    @staticmethod
    def _restore_augmenters(detached: list) -> None:
        for part, augmenter in detached:
            part.augmenter = augmenter

    @staticmethod
    def compute(
        dataset,
        logger           : Logger,
        input_config     : InputConfig,
        output_config    : "OutputConfig",
        n_secondaries    : int,
        n_interferograms : int,
        n_gaussians      : int,
        max_samples      : int = 0,
        num_workers      : int = 4,
        batch_size       : int = 512,
    ) -> Stats:

        detached = StatsComputer._detach_augmenters(dataset)

        try:
            input_only = StatsComputer.compute_input_stats(
                dataset          = dataset,
                logger           = logger,
                input_config     = input_config,
                n_secondaries    = n_secondaries,
                n_interferograms = n_interferograms,
                num_workers      = num_workers,
                max_samples      = max_samples,
                batch_size       = batch_size,
            )
        finally:
            StatsComputer._restore_augmenters(detached)

        output_only = StatsComputer.compute_output_stats(
            dataset       = dataset,
            n_gaussians   = n_gaussians,
            output_config = output_config,
            logger        = logger,
        )

        return Stats.merge(input_only, output_only)


class Normalizer:
    def __init__(self, stats: Stats) -> None:
        self.stats = stats

        self._vectors: dict[int, dict] = {}

    def _channel_vectors(self, stats: ChannelStats) -> dict:
        key = id(stats)
        if key in self._vectors:
            return self._vectors[key]

        loc       = np.asarray(stats.loc,   dtype=np.float32)
        scale     = np.asarray(stats.scale, dtype=np.float32)
        log1p     = np.asarray([strat.apply_log1p for strat in stats.strategies], dtype=bool)
        inv_scale = (1.0 / scale).astype(np.float32)

        vectors = {
            "loc"       : loc,
            "scale"     : scale,
            "inv_scale" : inv_scale,
            "log1p"     : log1p,
        }
        self._vectors[key] = vectors

        return vectors

    def _apply_normalization(self, tensor, stats: ChannelStats, inverse: bool):
        is_torch = isinstance(tensor, torch.Tensor)
        vectors  = self._channel_vectors(stats)

        shape    = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)

        if is_torch:
            device    = tensor.device
            loc       = torch.as_tensor(vectors["loc"],       device=device).reshape(shape)
            scale     = torch.as_tensor(vectors["scale"],     device=device).reshape(shape)
            inv_scale = torch.as_tensor(vectors["inv_scale"], device=device).reshape(shape)
            log1p     = torch.as_tensor(vectors["log1p"],     device=device).reshape(shape)

            if not inverse:
                x   = torch.where(log1p, torch.log1p(torch.clamp(tensor, min=0.0)), tensor)
                out = (x - loc) * inv_scale
            else:
                x   = tensor * scale + loc
                out = torch.where(log1p, torch.expm1(x), x)

            return out

        loc       = vectors["loc"].reshape(shape)
        scale     = vectors["scale"].reshape(shape)
        inv_scale = vectors["inv_scale"].reshape(shape)
        log1p     = vectors["log1p"].reshape(shape)

        if not inverse:
            x   = np.where(log1p, np.log1p(np.maximum(tensor, 0.0)), tensor)
            out = (x - loc) * inv_scale
        else:
            x   = tensor * scale + loc
            out = np.where(log1p, np.expm1(x), x)

        return np.ascontiguousarray(out, dtype=np.float32)

    def normalize_input(self, tensor: np.ndarray) -> np.ndarray:
        return self._apply_normalization(tensor, self.stats.input_stats, inverse=False)

    def normalize_output(self, tensor) -> np.ndarray:
        return self._apply_normalization(tensor, self.stats.output_stats, inverse=False)

    def denormalize_input(self, tensor):
        return self._apply_normalization(tensor, self.stats.input_stats, inverse=True)

    def denormalize_output(self, tensor):
        return self._apply_normalization(tensor, self.stats.output_stats, inverse=True)
