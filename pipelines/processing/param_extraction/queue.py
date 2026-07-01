from __future__ import annotations

from pathlib import Path

from configuration.param_extraction import ExtractionConfig, FitMode, FitSettings


class DatasetQueueResolver:
    def __init__(self, base_path: Path, dataset_filter: list) -> None:
        self.base_path      = base_path
        self.dataset_filter = dataset_filter

    def resolve(self) -> list[Path]:
        if not isinstance(self.dataset_filter, (list, tuple)):
            raise TypeError(f"dataset_filter must be a list of dataset names, got {type(self.dataset_filter).__name__}: {self.dataset_filter!r}")

        if not self.base_path.is_dir():
            raise NotADirectoryError(f"dataset_base_path does not exist: {self.base_path}")

        dataset_dirs = sorted(
            [d for d in self.base_path.iterdir() if d.is_dir()]
            if not self.dataset_filter
            else [self.base_path / str(name) for name in self.dataset_filter]
        )

        invalid = [d for d in dataset_dirs if not (d / "data").is_dir()]
        if invalid:
            names = ", ".join(d.name for d in invalid)
            raise NotADirectoryError(f"Queue entries without a data/ directory under {self.base_path}: {names}")

        return dataset_dirs


class ExtractionPlanResolver:
    def __init__(self, entry_config, dataset_dirs: list[Path]) -> None:
        self.entry_config = entry_config
        self.dataset_dirs = dataset_dirs

    def resolve(self) -> list[ExtractionConfig]:
        self._validate()

        plans = []
        for processed_data_path in self.dataset_dirs:
            for k_max in self.entry_config.fit_k_values:
                for lambda_k in self.entry_config.fit_lambda_values:
                    for mode in self.entry_config.fit_modes:
                        plans.append(self._build_plan(processed_data_path, k_max, lambda_k, mode))

        return plans

    def _validate(self) -> None:
        for name in ("fit_k_values", "fit_lambda_values", "fit_modes"):
            value = getattr(self.entry_config, name)
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{name} must be a non-empty list, got {value!r}")

        permutations = len(self.dataset_dirs) * len(self.entry_config.fit_k_values) * len(self.entry_config.fit_lambda_values) * len(self.entry_config.fit_modes)
        if self.entry_config.output_suffix and permutations > 1:
            raise ValueError(f"output_suffix is a fixed name but the sweep expands to {permutations} permutations that would all collide on it; leave output_suffix unset so each permutation gets its auto-encoded name")

    def _build_plan(self, processed_data_path: Path, k_max, lambda_k, mode: str) -> ExtractionConfig:
        fit_amplitude, fit_mean = FitMode.free_flags(mode)

        fit_config = FitMode.SigmaOnly(
            k_max              = int(k_max),
            lambda_k           = float(lambda_k),
            sigma_init_divisor = self.entry_config.fit_sigma_init_divisor,
            fit_amplitude      = fit_amplitude,
            fit_mean           = fit_mean,
        )

        return ExtractionConfig(
            processed_data_path = processed_data_path,
            pyrat_directory     = self.entry_config.pyrat_directory,

            output_prefix = self.entry_config.output_prefix,
            output_suffix = self.entry_config.output_suffix,

            height_range = self.entry_config.height_range,

            fit_settings = FitSettings(fit_config=fit_config),

            range_batch_size  = self.entry_config.range_batch_size,
            parameter_workers = self.entry_config.parameter_workers,
        )
