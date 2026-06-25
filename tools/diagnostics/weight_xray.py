from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional

import numpy as np
import torch

from configuration.diagnostics import WeightXrayConfig, WeightXrayEntryConfig, WeightXrayThresholds
from tools.runtime.run_selector                    import RunSelector


SEVERITY_RANK = {"ok": 0, "info": 1, "warning": 2, "critical": 3}


@dataclass
class Issue:
    severity : str
    code     : str
    message  : str


@dataclass
class LayerReport:
    name  : str
    role  : str
    shape : tuple
    dtype : str
    count : int

    n_nonfinite : int
    n_nan       : int
    n_inf       : int

    mean     : float
    std      : float
    minimum  : float
    maximum  : float
    max_abs  : float
    mean_abs : float
    l2_norm  : float

    frac_zero : float
    frac_dead : float
    cv        : float
    kurtosis  : float

    fan_in            : Optional[int]   = None
    out_units         : Optional[int]   = None
    dead_output_units : Optional[int]   = None
    dead_output_frac  : Optional[float] = None
    dead_input_units  : Optional[int]   = None
    spectral_norm     : Optional[float] = None
    stable_rank       : Optional[float] = None
    effective_rank    : Optional[float] = None
    rank_ratio        : Optional[float] = None
    condition_number  : Optional[float] = None
    init_ratio        : Optional[float] = None
    duplicate_units   : Optional[int]   = None

    issues : list = field(default_factory=list)

    @property
    def severity(self) -> str:
        worst = max((SEVERITY_RANK[issue.severity] for issue in self.issues), default=0)
        return {0: "ok", 1: "info", 2: "warning", 3: "critical"}[worst]


class StateDictResolver:
    def __init__(self, config: WeightXrayConfig) -> None:
        self.config = config

    def _load_object(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint file at {path}; point --checkpoint-path at a .pt file or a run directory containing {self.config.checkpoint_filename}")

        return torch.load(str(path), map_location="cpu", weights_only=False)

    def _is_tensor_mapping(self, candidate) -> bool:
        if not isinstance(candidate, dict) or not candidate:
            return False

        tensor_values = sum(1 for value in candidate.values() if torch.is_tensor(value))
        return tensor_values >= max(1, int(0.5 * len(candidate)))

    def _select_mapping(self, obj) -> dict:
        if self._is_tensor_mapping(obj):
            return obj

        if isinstance(obj, dict):
            for key in self.config.state_dict_keys:
                if key in obj and self._is_tensor_mapping(obj[key]):
                    return obj[key]

            for value in obj.values():
                if self._is_tensor_mapping(value):
                    return value

        raise ValueError(f"Could not locate a parameter tensor mapping in the checkpoint; searched top level and keys {self.config.state_dict_keys}. Found type {type(obj).__name__}")

    def resolve(self) -> dict:
        path = self.config.resolved_checkpoint_path
        obj  = self._load_object(path)
        return self._select_mapping(obj)


class WeightAnalyzer:
    def __init__(self, config: WeightXrayConfig) -> None:
        self.config = config

    def _classify_role(self, name: str, ndim: int) -> str:
        lowered = name.lower()

        if lowered.endswith("num_batches_tracked"):
            return "counter"
        if lowered.endswith("running_mean"):
            return "running_mean"
        if lowered.endswith("running_var"):
            return "running_var"
        if lowered.endswith("bias"):
            return "bias"
        if lowered.endswith("weight"):
            return "weight" if ndim >= 2 else "norm_scale"
        return "other"

    def _basic_stats(self, values: np.ndarray) -> dict:
        finite     = np.isfinite(values)
        n_nan      = int(np.isnan(values).sum())
        n_inf      = int(np.isinf(values).sum())
        clean      = values[finite].astype(np.float64)

        if clean.size == 0:
            return {"n_nan": n_nan, "n_inf": n_inf, "mean": 0.0, "std": 0.0, "minimum": 0.0, "maximum": 0.0, "max_abs": 0.0, "mean_abs": 0.0, "l2_norm": 0.0, "frac_zero": 0.0, "frac_dead": 1.0, "cv": 0.0, "kurtosis": 0.0}

        abs_clean = np.abs(clean)
        mean      = float(clean.mean())
        std       = float(clean.std())
        mean_abs  = float(abs_clean.mean())

        centred   = clean - mean
        variance  = float((centred ** 2).mean())
        kurtosis  = float((centred ** 4).mean() / (variance ** 2)) - 3.0 if variance > 0 else 0.0

        return {
            "n_nan"     : n_nan,
            "n_inf"     : n_inf,
            "mean"      : mean,
            "std"       : std,
            "minimum"   : float(clean.min()),
            "maximum"   : float(clean.max()),
            "max_abs"   : float(abs_clean.max()),
            "mean_abs"  : mean_abs,
            "l2_norm"   : float(np.sqrt((clean ** 2).sum())),
            "frac_zero" : float((clean == 0.0).mean()),
            "frac_dead" : float((abs_clean < self.config.thresholds.dead_abs_threshold).mean()),
            "cv"        : float(std / mean_abs) if mean_abs > 0 else 0.0,
            "kurtosis"  : kurtosis,
        }

    def _matrix_stats(self, values: np.ndarray) -> dict:
        matrix    = values.reshape(values.shape[0], -1).astype(np.float64)
        matrix    = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        out_units = matrix.shape[0]
        fan_in    = matrix.shape[1]

        row_norms = np.sqrt((matrix ** 2).sum(axis=1))
        col_norms = np.sqrt((matrix ** 2).sum(axis=0))

        max_row = float(row_norms.max()) if row_norms.size else 0.0
        max_col = float(col_norms.max()) if col_norms.size else 0.0

        dead_row_cut = self.config.thresholds.dead_unit_norm_frac * max_row
        dead_col_cut = self.config.thresholds.dead_unit_norm_frac * max_col

        dead_output = int((row_norms <= dead_row_cut).sum()) if max_row > 0 else out_units
        dead_input  = int((col_norms <= dead_col_cut).sum()) if max_col > 0 else fan_in

        stats = {
            "fan_in"            : fan_in,
            "out_units"         : out_units,
            "dead_output_units" : dead_output,
            "dead_output_frac"  : float(dead_output / out_units) if out_units else 0.0,
            "dead_input_units"  : dead_input,
            "init_ratio"        : float(matrix.std() * np.sqrt(fan_in)) if fan_in > 0 else 0.0,
        }

        stats.update(self._spectral_stats(matrix, out_units, fan_in))
        stats["duplicate_units"] = self._duplicate_units(matrix, row_norms)
        return stats

    def _spectral_stats(self, matrix: np.ndarray, out_units: int, fan_in: int) -> dict:
        if min(out_units, fan_in) > self.config.svd_max_dim:
            return {"spectral_norm": None, "stable_rank": None, "effective_rank": None, "rank_ratio": None, "condition_number": None}

        singular = np.linalg.svd(matrix, compute_uv=False)
        singular = singular[singular > 0]
        if singular.size == 0:
            return {"spectral_norm": 0.0, "stable_rank": 0.0, "effective_rank": 0.0, "rank_ratio": 0.0, "condition_number": 0.0}

        spectral     = float(singular[0])
        frobenius_sq = float((singular ** 2).sum())
        probabilities = singular / singular.sum()
        entropy       = float(-(probabilities * np.log(probabilities)).sum())

        return {
            "spectral_norm"    : spectral,
            "stable_rank"      : frobenius_sq / (spectral ** 2),
            "effective_rank"   : float(np.exp(entropy)),
            "rank_ratio"       : float(np.exp(entropy) / min(out_units, fan_in)),
            "condition_number" : float(spectral / singular[-1]),
        }

    def _duplicate_units(self, matrix: np.ndarray, row_norms: np.ndarray) -> Optional[int]:
        out_units = matrix.shape[0]
        if out_units < 2 or out_units > self.config.duplicate_max_units:
            return None

        safe       = np.where(row_norms > 0, row_norms, 1.0)
        normalized = matrix / safe[:, None]
        gram       = np.abs(normalized @ normalized.T)
        np.fill_diagonal(gram, 0.0)

        alive    = row_norms > 0
        pair_mask = np.triu(gram >= self.config.thresholds.duplicate_cosine, k=1)
        pair_mask = pair_mask & alive[:, None] & alive[None, :]
        return int(pair_mask.sum())

    def analyze_tensor(self, name: str, tensor: torch.Tensor) -> Optional[LayerReport]:
        role = self._classify_role(name, tensor.dim())
        if role == "counter" or tensor.dim() == 0 or not torch.is_floating_point(tensor):
            return None

        values = tensor.detach().to(torch.float32).cpu().numpy()
        basics = self._basic_stats(values)

        report = LayerReport(
            name        = name,
            role        = role,
            shape       = tuple(tensor.shape),
            dtype       = str(tensor.dtype).replace("torch.", ""),
            count       = int(values.size),
            n_nonfinite = basics["n_nan"] + basics["n_inf"],
            n_nan       = basics["n_nan"],
            n_inf       = basics["n_inf"],
            mean        = basics["mean"],
            std         = basics["std"],
            minimum     = basics["minimum"],
            maximum     = basics["maximum"],
            max_abs     = basics["max_abs"],
            mean_abs    = basics["mean_abs"],
            l2_norm     = basics["l2_norm"],
            frac_zero   = basics["frac_zero"],
            frac_dead   = basics["frac_dead"],
            cv          = basics["cv"],
            kurtosis    = basics["kurtosis"],
        )

        if role == "weight" and tensor.dim() >= 2:
            matrix_stats = self._matrix_stats(values)
            for key, value in matrix_stats.items():
                setattr(report, key, value)

        return report

    def analyze(self, state_dict: dict) -> list[LayerReport]:
        reports = []
        for name, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
                continue

            report = self.analyze_tensor(name, tensor)
            if report is not None:
                reports.append(report)

        return reports


class IssueDetector:
    def __init__(self, thresholds: WeightXrayThresholds) -> None:
        self.thresholds = thresholds

    def _check_finite(self, report: LayerReport) -> list[Issue]:
        if report.n_nonfinite == 0:
            return []
        return [Issue("critical", "nonfinite", f"{report.n_nonfinite} non-finite values ({report.n_nan} NaN, {report.n_inf} Inf)")]

    def _check_dead(self, report: LayerReport) -> list[Issue]:
        if report.frac_dead >= self.thresholds.dead_fraction_critical:
            return [Issue("critical", "dead_tensor", f"{report.frac_dead:.1%} of values are effectively zero")]
        if report.frac_dead >= self.thresholds.dead_fraction_warn:
            return [Issue("warning", "high_sparsity", f"{report.frac_dead:.1%} of values are effectively zero")]
        return []

    def _check_uniform(self, report: LayerReport) -> list[Issue]:
        if report.std <= self.thresholds.constant_std_threshold:
            return [Issue("critical", "constant", f"tensor is constant (std={report.std:.2e})")]
        if report.role in ("weight", "norm_scale") and report.count >= 8 and report.cv <= self.thresholds.uniform_cv_threshold:
            return [Issue("warning", "uniform", f"near-uniform weights (cv={report.cv:.3f})")]
        return []

    def _check_explode(self, report: LayerReport) -> list[Issue]:
        issues = []
        if report.max_abs >= self.thresholds.explode_abs_threshold:
            issues.append(Issue("warning", "large_magnitude", f"max |w| = {report.max_abs:.2e}"))
        if report.spectral_norm is not None and report.spectral_norm >= self.thresholds.spectral_norm_warn:
            issues.append(Issue("warning", "large_spectral_norm", f"spectral norm = {report.spectral_norm:.2e}"))
        return issues

    def _check_rank(self, report: LayerReport) -> list[Issue]:
        if report.rank_ratio is None:
            return []
        if report.rank_ratio <= self.thresholds.rank_ratio_critical:
            return [Issue("critical", "rank_collapse", f"effective rank ratio = {report.rank_ratio:.2f}")]
        if report.rank_ratio <= self.thresholds.rank_ratio_warn:
            return [Issue("warning", "rank_deficient", f"effective rank ratio = {report.rank_ratio:.2f}")]
        return []

    def _check_units(self, report: LayerReport) -> list[Issue]:
        if report.dead_output_frac is None or report.dead_output_units == 0:
            return []
        if report.dead_output_frac >= self.thresholds.dead_unit_fraction_warn:
            return [Issue("warning", "dead_neurons", f"{report.dead_output_units}/{report.out_units} output units have ~zero norm")]
        return [Issue("info", "dead_neurons", f"{report.dead_output_units}/{report.out_units} output units have ~zero norm")]

    def _check_duplicates(self, report: LayerReport) -> list[Issue]:
        if not report.duplicate_units or report.out_units is None:
            return []
        fraction = report.duplicate_units / report.out_units
        severity = "warning" if fraction >= self.thresholds.duplicate_fraction_warn else "info"
        return [Issue(severity, "duplicate_units", f"{report.duplicate_units} collinear output-unit pairs (|cos| >= {self.thresholds.duplicate_cosine})")]

    def _check_init(self, report: LayerReport) -> list[Issue]:
        if report.role != "weight" or report.init_ratio is None or report.init_ratio == 0.0:
            return []
        if report.init_ratio < self.thresholds.init_ratio_low:
            return [Issue("info", "scale_low", f"std*sqrt(fan_in) = {report.init_ratio:.2f} (unusually small)")]
        if report.init_ratio > self.thresholds.init_ratio_high:
            return [Issue("info", "scale_high", f"std*sqrt(fan_in) = {report.init_ratio:.2f} (unusually large)")]
        return []

    def _check_norm_layer(self, report: LayerReport, values: np.ndarray) -> list[Issue]:
        if report.role == "norm_scale":
            fraction = float((np.abs(values) < self.thresholds.norm_scale_dead_value).mean())
            if fraction >= self.thresholds.norm_scale_dead_frac:
                return [Issue("warning", "scale_collapse", f"{fraction:.1%} of normalisation scales are ~zero")]
        if report.role == "running_var":
            fraction = float((values < self.thresholds.running_var_dead_value).mean())
            if fraction >= self.thresholds.running_var_dead_frac:
                return [Issue("warning", "dead_channel_var", f"{fraction:.1%} of running variances are ~zero")]
        return []

    def _check_bias(self, report: LayerReport) -> list[Issue]:
        if report.role == "bias" and report.max_abs >= self.thresholds.bias_abs_threshold:
            return [Issue("warning", "large_bias", f"max |bias| = {report.max_abs:.2e}")]
        return []

    def detect(self, report: LayerReport, values: np.ndarray) -> None:
        report.issues.extend(self._check_finite(report))
        report.issues.extend(self._check_dead(report))
        report.issues.extend(self._check_uniform(report))
        report.issues.extend(self._check_explode(report))
        report.issues.extend(self._check_rank(report))
        report.issues.extend(self._check_units(report))
        report.issues.extend(self._check_duplicates(report))
        report.issues.extend(self._check_init(report))
        report.issues.extend(self._check_norm_layer(report, values))
        report.issues.extend(self._check_bias(report))

    def run(self, reports: list[LayerReport], state_dict: dict) -> list[LayerReport]:
        for report in reports:
            values = state_dict[report.name].detach().to(torch.float32).cpu().numpy().reshape(-1)
            self.detect(report, values)
        return reports


class XraySummarizer:
    def build(self, reports: list[LayerReport], checkpoint_path: Path) -> dict:
        issues       = [issue for report in reports for issue in report.issues]
        flagged      = [report for report in reports if report.severity != "ok"]
        total_params = sum(report.count for report in reports)

        severity_counts = {level: 0 for level in ("critical", "warning", "info", "ok")}
        for report in reports:
            severity_counts[report.severity] += 1

        code_counts: dict = {}
        for issue in issues:
            code_counts[issue.code] = code_counts.get(issue.code, 0) + 1

        return {
            "checkpoint"      : str(checkpoint_path),
            "tensors"         : len(reports),
            "parameters"      : total_params,
            "flagged_tensors" : len(flagged),
            "issues"          : len(issues),
            "severity_counts" : severity_counts,
            "issue_codes"     : dict(sorted(code_counts.items(), key=lambda item: item[1], reverse=True)),
            "verdict"         : self._verdict(severity_counts),
        }

    def _verdict(self, severity_counts: dict) -> str:
        if severity_counts["critical"] > 0:
            return "critical issues detected"
        if severity_counts["warning"] > 0:
            return "warnings detected"
        if severity_counts["info"] > 0:
            return "minor observations only"
        return "clean"


class WeightXray:
    def __init__(self, config: WeightXrayConfig, logger) -> None:
        self.config = config
        self.logger = logger

    def _resolve(self) -> dict:
        self.logger.subsection(f"Loading checkpoint: {self.config.resolved_checkpoint_path}")
        state_dict = StateDictResolver(self.config).resolve()
        self.logger.ok(f"Resolved {len(state_dict)} tensors")
        return state_dict

    def _analyze(self, state_dict: dict) -> list[LayerReport]:
        reports = WeightAnalyzer(self.config).analyze(state_dict)
        self.logger.ok(f"Analysed {len(reports)} parameter tensors")
        return reports

    def _detect(self, reports: list[LayerReport], state_dict: dict) -> list[LayerReport]:
        return IssueDetector(self.config.thresholds).run(reports, state_dict)

    def _summarize(self, reports: list[LayerReport]) -> dict:
        return XraySummarizer().build(reports, self.config.resolved_checkpoint_path)

    def _plot(self, reports: list[LayerReport], state_dict: dict) -> list[Path]:
        if not self.config.make_plots:
            return []

        from tools.diagnostics.weight_xray_plots import WeightXrayPlots

        return WeightXrayPlots(self.config).render(reports, state_dict)

    def _report(self, reports: list[LayerReport], summary: dict, plot_paths: list[Path]) -> dict:
        from tools.diagnostics.weight_xray_report import WeightXrayReport

        return WeightXrayReport(self.config).write(self.logger, reports, summary, plot_paths)

    def run(self) -> dict:
        self.logger.section("Model weight x-ray")

        state_dict = self._resolve()
        reports    = self._analyze(state_dict)
        reports    = self._detect(reports, state_dict)
        summary    = self._summarize(reports)
        plot_paths = self._plot(reports, state_dict)
        outputs    = self._report(reports, summary, plot_paths)

        return {"summary": summary, "outputs": outputs}


class WeightXrayBatch:
    def __init__(self, entry_config: WeightXrayEntryConfig, logger) -> None:
        self.entry_config = entry_config
        self.logger       = logger

    def _select_runs(self) -> list[Path]:
        selector = RunSelector(self.entry_config.runs_dir, self.entry_config.checkpoint_filename, self.logger)

        if self.entry_config.run_filter:
            return selector.filter(self.entry_config.run_filter)
        if sys.stdin.isatty():
            return selector.select()
        return selector.all()

    def _xray_run(self, run_dir: Path) -> dict:
        config = self.entry_config.to_config(run_dir)
        result = WeightXray(config, self.logger).run()

        self.logger.ok(f"{run_dir.name}: {result['summary']['verdict']} -> {config.report_directory}")
        return result

    def run(self) -> list[dict]:
        run_dirs = self._select_runs()
        return [self._xray_run(run_dir) for run_dir in run_dirs]
