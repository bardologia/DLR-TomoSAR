from __future__ import annotations

from typing import Optional

import numpy as np

from configuration.dataset import InputConfig, OutputConfig
from configuration.normalization import ChannelStats, ChannelStrategy, NormalizationConfig, NormMethod
from tools.monitoring.logger           import Logger
from tools.reporting.ranges            import RangeFormatter

from pipelines.backbone.dataset.stats import Stats


class StatsComputer:
    @staticmethod
    def _parts(dataset) -> list:
        return list(dataset.parts) if hasattr(dataset, "parts") else [dataset]

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
    def _sample_flat(flat: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
        if flat.size <= budget:
            return np.asarray(flat)

        return flat[rng.choice(flat.size, size=budget, replace=False)]

    @staticmethod
    def _collect(
        parts              : list,
        input_config       : InputConfig,
        group_keys         : list[str],
        strategies         : dict[str, ChannelStrategy],
        max_vals_per_group : int = 1_000_000,
    ) -> dict[str, np.ndarray]:

        unique_groups  = list(dict.fromkeys(group_keys))
        group_channels = {g: [i for i, k in enumerate(group_keys) if k == g] for g in unique_groups}
        needs_data     = {g for g in unique_groups if strategies[g].norm_method is not NormMethod.FIXED_DIV_PI}

        collected: dict[str, list[np.ndarray]] = {g: [] for g in needs_data}
        if not collected:
            return {}

        rng    = np.random.default_rng(42)
        budget = {g: max(64, max_vals_per_group // (len(channels) * len(parts))) for g, channels in group_channels.items()}

        for part in parts:
            inputs        = part.inputs
            n_secondaries = part.n_secondaries

            sections = [
                (input_config.use_primary,        inputs[0:1],             input_config.primary_representation,        "pass"),
                (input_config.use_secondaries,    inputs[1:1 + n_secondaries], input_config.secondaries_representation,    "pass"),
                (input_config.use_interferograms, inputs[1 + n_secondaries:],  input_config.interferograms_representation, "ifg"),
            ]

            for enabled, layers, representation, prefix in sections:
                if not enabled:
                    continue

                keys   = [f"{prefix}/{kind}" for kind in representation.slot_kinds]
                needed = [key for key in keys if key in needs_data]
                if not needed:
                    continue

                layer_budget = max(budget[key] for key in needed)

                for layer in layers:
                    sample   = StatsComputer._sample_flat(np.asarray(layer).ravel(), layer_budget, rng)
                    channels = representation.channel_values(sample)

                    for key, values in zip(keys, channels):
                        if key in needs_data:
                            collected[key].append(np.asarray(values[:budget[key]], dtype=np.float64))

            if input_config.use_dem and "dem/elevation" in needs_data:
                dem_flat = np.asarray(part.dem, dtype=np.float64).ravel()
                collected["dem/elevation"].append(StatsComputer._sample_flat(dem_flat, budget["dem/elevation"], rng))

        return {g: np.concatenate(v) for g, v in collected.items() if v}

    @staticmethod
    def _fit_input(logger : Logger, group_keys : list[str], strategies : dict[str, ChannelStrategy], collected : dict[str, np.ndarray]) -> ChannelStats:
        unique_groups    = list(dict.fromkeys(group_keys))
        group_strategies = strategies

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
    def compute_input_stats(
        dataset,
        logger             : Logger,
        input_config       : InputConfig,
        n_secondaries      : int,
        n_interferograms   : int,
        normalization      : NormalizationConfig = None,
        max_vals_per_group : int = 1_000_000,
    ) -> Stats:

        normalization = normalization if normalization is not None else NormalizationConfig()
        parts         = StatsComputer._parts(dataset)

        group_keys  = input_config.channel_group_keys(n_secondaries, n_interferograms)
        in_channels = int(parts[0].input_channels)
        if len(group_keys) != in_channels:
            raise ValueError(f"Group key count ({len(group_keys)}) does not match the dataset input channels ({in_channels}).")

        logger.section("[Input Normalization Statistics]")
        logger.kv_table({
            "Strategy":       f"{normalization.input_strategy} (per slot-kind, grouped across passes/ifgs)",
            "Source":         f"{len(parts)} region array(s), up to {max_vals_per_group:,} values per group",
            "Input channels": in_channels,
        })

        strategies = {g: normalization.strategy("input", g) for g in dict.fromkeys(group_keys)}

        logger.section("[Input grouping by slot-kind]")
        StatsComputer._log_grouping(logger, "Input", group_keys)

        collected = StatsComputer._collect(
            parts              = parts,
            input_config       = input_config,
            group_keys         = group_keys,
            strategies         = strategies,
            max_vals_per_group = max_vals_per_group,
        )

        input_stats = StatsComputer._fit_input(
            logger     = logger,
            group_keys = group_keys,
            strategies = strategies,
            collected  = collected,
        )

        return Stats(
            input_stats  = input_stats,
            output_stats = None,
        )

    @staticmethod
    def _fit_output(
        logger        : Optional[Logger],
        role_pools    : dict[str, np.ndarray],
        output_config : "OutputConfig",
        n_gaussians   : int,
    ) -> ChannelStats:

        role_fit   : dict[str, tuple[float, float]] = {key: output_config.strategy_for(key).fit(pool) for key, pool in role_pools.items()}
        role_strat : dict[str, ChannelStrategy]     = {key: output_config.strategy_for(key) for key in role_pools}

        selected       = output_config.selected_indices(n_gaussians)
        _local_to_role = {0: "out/amp", 1: "out/mu", 2: "out/sigma"}

        locs       : list[float]           = []
        scales     : list[float]           = []
        names      : list[str]             = []
        strategies : list[ChannelStrategy] = []

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
    def compute_output_stats(
        dataset,
        n_gaussians   : int,
        output_config : "OutputConfig",
        amp_threshold : float = 1e-3,
        logger        : Optional[Logger] = None,
    ) -> Stats:
        regions = [part.gt_parameters for part in StatsComputer._parts(dataset)]

        amp_pool_vals : list[np.ndarray] = []
        mu_pool_vals  : list[np.ndarray] = []
        sig_pool_vals : list[np.ndarray] = []

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
    def compute(
        dataset,
        logger           : Logger,
        input_config     : InputConfig,
        output_config    : "OutputConfig",
        n_secondaries    : int,
        n_interferograms : int,
        n_gaussians      : int,
        normalization    : NormalizationConfig = None,
    ) -> Stats:

        normalization = normalization if normalization is not None else NormalizationConfig()

        input_only = StatsComputer.compute_input_stats(
            dataset          = dataset,
            logger           = logger,
            input_config     = input_config,
            n_secondaries    = n_secondaries,
            n_interferograms = n_interferograms,
            normalization    = normalization,
        )

        output_only = StatsComputer.compute_output_stats(
            dataset       = dataset,
            n_gaussians   = n_gaussians,
            output_config = output_config,
            logger        = logger,
        )

        return Stats.merge(input_only, output_only, normalization.clamp())
