from __future__ import annotations

from pathlib import Path
from typing  import Optional

import numpy as np

from configuration.norm_config        import ChannelStats, ChannelStrategy, NormMethod
from configuration.dataset_config     import InputConfig, OutputConfig
from pipelines.dataset_pipeline.stats import Stats
from tools.logger                     import Logger
from torch.utils.data                 import DataLoader
from torch.utils.data                 import Subset


class StatsComputer:
    @staticmethod
    def _input_to_group(input_config : InputConfig, n_slaves : int) -> list[str]:
        return input_config.channel_group_keys(n_slaves)

    @staticmethod
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
            k: f"{len(idxs):d} ch  [{StatsComputer._compact_ranges(sorted(idxs))}]"
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

        n_batches_est     = max(len(subset) // max(batch_size, 1), 1)
        first_group_chs   = len(next(iter(group_channels.values())))
        vals_per_ch_batch = max(64, max_vals_per_group // (n_batches_est * max(first_group_chs, 1)))

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
                for ch in channels:
                    flat = arr[:, ch].ravel()

                    if len(flat) > vals_per_ch_batch:
                        idx  = rng.choice(len(flat), vals_per_ch_batch, replace=False)
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
            rows = [
                {
                    "Channel":  key,
                    "loc":      f"{m:.5f}",
                    "scale":    f"{s:.5f}",
                    "Method":    strat.norm_method.value,
                    "log1p":    str(strat.apply_log1p),
                }
                for key, (m, s) in role_fit.items()
                for strat in [role_strat[key]]
            ]
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
        logger       : Logger,
        input_config : InputConfig,
        n_slaves     : int,
        max_samples  : int = 0,
        num_workers  : int = 4,
        batch_size   : int = 512,
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

        group_keys = StatsComputer._input_to_group(input_config, n_slaves)
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
    def compute_output_stats(
        params_path   : Path,
        n_gaussians   : int,
        output_config : "OutputConfig",
        amp_threshold : float = 1e-2,
        logger        : Optional[Logger] = None,
    ) -> Stats:
        params = np.load(params_path, mmap_mode="r")

        amp_pool_vals:  list[np.ndarray] = []
        mu_pool_vals:   list[np.ndarray] = []
        sig_pool_vals:  list[np.ndarray] = []

        for g in range(n_gaussians):
            a_flat   = params[g * 3 + 0].ravel().astype(np.float64)
            mu_flat  = params[g * 3 + 1].ravel().astype(np.float64)
            sig_flat = params[g * 3 + 2].ravel().astype(np.float64)
            active   = a_flat > amp_threshold

            amp_pool_vals.append(a_flat)
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
    def compute(
        dataset,
        params_path   : Path,
        logger        : Logger,
        input_config  : InputConfig,
        output_config : "OutputConfig",
        n_slaves      : int,
        n_gaussians   : int,
        max_samples   : int = 0,
        num_workers   : int = 4,
        batch_size    : int = 512,
    ) -> Stats:

        input_only = StatsComputer.compute_input_stats(
            dataset      = dataset,
            logger       = logger,
            input_config = input_config,
            n_slaves     = n_slaves,
            num_workers  = num_workers,
            max_samples  = max_samples,
            batch_size   = batch_size,
        )

        output_only = StatsComputer.compute_output_stats(
            params_path   = params_path,
            n_gaussians   = n_gaussians,
            output_config = output_config,
            logger        = logger,
        )

        return Stats.merge(input_only, output_only)
