import re
import sys
from pathlib import Path

_vault = next(p for p in Path(__file__).resolve().parents if (p / "code" / "tools" / "logger.py").exists())
sys.path.insert(0, str(_vault / "code"))

from tools.logger import Logger


class ReportParser:

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir

    def parse_metrics(self) -> dict:
        text   = (self.results_dir / "metrics_comparison.md").read_text()
        values = {}

        for block in text.split("\n## "):
            lines = [line for line in block.splitlines() if line.startswith("|")]
            if len(lines) < 3:
                continue

            header = [cell.strip() for cell in lines[0].strip("|").split("|")]
            keys   = [re.sub(r"[`↓↑ ]", "", cell) for cell in header]

            for line in lines[2:]:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                run   = cells[0].strip("`").split("/")[-1]
                for key, cell in zip(keys[1:], cells[1:]):
                    parsed = self.parse_cell(cell)
                    if parsed is not None:
                        values.setdefault(run, {})[key] = parsed

        return values

    def parse_val_loss(self, values: dict) -> None:
        text = (self.results_dir / "overview.md").read_text()

        for line in text.splitlines():
            if not line.startswith("| `"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 5:
                continue
            run    = cells[0].strip("`").split("/")[-1]
            parsed = self.parse_cell(cells[4])
            if parsed is not None and run in values:
                values[run]["best_val_loss"] = parsed

    def parse_cell(self, cell: str) -> tuple | None:
        match = re.match(r"\*{0,2}(-?[\d.]+(?:e[+-]?\d+)?)\*{0,2}\s*±\s*(-?[\d.]+(?:e[+-]?\d+)?)", cell.replace("**", ""))
        if match is None:
            return None
        return float(match.group(1)), float(match.group(2))

    def load(self) -> dict:
        values = self.parse_metrics()
        self.parse_val_loss(values)
        return values


class CellFormatter:

    def mean_text(self, value: float, decimals: int) -> str:
        text = f"{value:.{decimals}f}"
        if text.startswith("-"):
            text = f"${text}$"
        return text

    def std_text(self, value: float, decimals: int) -> str:
        text = f"{value:.{decimals}f}"
        if text.startswith("0."):
            text = text[1:]
        return text

    def cell(self, mean: float, std: float, decimals: int, bold: bool) -> str:
        mean_text = self.mean_text(mean, decimals)
        if bold:
            mean_text = f"\\textbf{{{mean_text}}}"
        return f"\\seedpm{{{mean_text}}}{{{self.std_text(std, decimals)}}}"


class SpreadGate:

    THRESHOLD = 0.05

    def spread(self, best: float, worst: float) -> float:
        base = min(abs(best), abs(worst))

        if best == worst:
            return 0.0
        if base == 0.0:
            return float("inf")

        return abs(best - worst) / base

    def clears(self, best: float, worst: float) -> bool:
        return self.spread(best, worst) > self.THRESHOLD


class TableEmitter:

    def __init__(self, values: dict, runs: list, formatter: CellFormatter, gate: SpreadGate) -> None:
        self.values    = values
        self.runs      = runs
        self.formatter = formatter
        self.gate      = gate

    def best_flags(self, means: list, decimals: int, direction: str, target: float | None) -> list:
        displayed = [round(mean, decimals) for mean in means]

        if target is not None:
            gaps     = [abs(value - round(target, decimals)) for value in displayed]
            ranked   = sorted(zip(gaps, displayed))
            flags    = [gap == ranked[0][0] for gap in gaps]
            extremes = (ranked[0][1], ranked[-1][1])
        else:
            best     = min(displayed) if direction == "down" else max(displayed)
            worst    = max(displayed) if direction == "down" else min(displayed)
            flags    = [value == best for value in displayed]
            extremes = (best, worst)

        if not self.gate.clears(*extremes):
            return [False] * len(flags)

        return flags

    def data_row(self, label: str, key: str, direction: str, decimals: int, gt_key: str | None, gt_decimals: int | None) -> str:
        pairs = [self.values[run][key] for run in self.runs]
        means = [pair[0] for pair in pairs]

        target = None
        prefix = ""
        if gt_key is not None:
            target = self.values[self.runs[0]][gt_key][0]
            prefix = f"{self.formatter.mean_text(target, gt_decimals if gt_decimals is not None else decimals)} & "

        flags = self.best_flags(means, decimals, direction, target)
        cells = [self.formatter.cell(mean, std, decimals, flag) for (mean, std), flag in zip(pairs, flags)]
        return f"    {label} & {prefix}{' & '.join(cells)} \\\\"

    def raw_row(self, label: str, key: str, decimals: int) -> str:
        pairs = [self.values[run][key] for run in self.runs]
        cells = [self.formatter.cell(mean, std, decimals, False) for mean, std in pairs]
        return f"    {label} & --- & {' & '.join(cells)} \\\\"

    def emit(self, rows: list, gt_span: int | None) -> str:
        lines = []
        for row in rows:
            if row[0] == "group":
                span = gt_span if gt_span is not None else len(self.runs) + 1
                lines.append(f"    \\multicolumn{{{span}}}{{@{{}}l}}{{{row[1]}}} \\\\")
            elif row[0] == "space":
                lines.append("    \\addlinespace" + (f"[{row[1]}]" if len(row) > 1 else ""))
            elif row[0] == "raw":
                lines.append(self.raw_row(row[1], row[2], row[3]))
            else:
                label, key, direction, decimals = row[:4]
                gt_key      = row[4] if len(row) > 4 else None
                gt_decimals = row[5] if len(row) > 5 else None
                lines.append(self.data_row(label, key, direction, decimals, gt_key, gt_decimals))
        return "\n".join(lines)


class DualRound2Tables:

    K5_RUNS    = [f"dual_resunet-set_pred-hungarian-K_5-hv-A-param_l1_1_di-{arm}" for arm in ["full-full", "full-ifg", "full-pass", "ifg-full", "ifg-pass", "pass-full", "pass-ifg"]]
    RATIO_RUNS = [f"dual_resunet-set_pred-hungarian-K_2-hv-A-param_l1_1_dr-{arm}" for arm in ["50-50", "60-40", "70-30", "80-20", "90-10"]]
    RATIO_K5_RUNS = [f"dual_resunet-set_pred-hungarian-K_5-hv-A-param_l1_1_dr-{arm}" for arm in ["50-50", "60-40", "70-30", "80-20", "90-10"]]
    SINGLE_RUN    = "unet_skip-set_pred-hungarian-K_2-hv-A__param_l1"

    def __init__(self, repo_root: Path) -> None:
        self.repo_root  = repo_root
        self.output_dir = repo_root / "presentations" / "full_project_story" / "table_fragments"
        self.formatter  = CellFormatter()
        self.gate       = SpreadGate()
        self.logger     = Logger(log_dir=str(self.output_dir / "logs"), name="gen_dual_round2_tables")

    def k5_headline_rows(self) -> list:
        return [
            ("group", r"\emph{training (validation)}"),
            (r"val loss $\downarrow$", "best_val_loss", "down", 3),
            ("space",),
            ("group", r"\emph{reconstruction (curve)}"),
            (r"curve $R^2$ $\uparrow$", "overall_r2_gt", "up", 3),
            (r"curve MAE $\downarrow$", "curve_mae_gt", "down", 3),
            (r"PSNR [dB] $\uparrow$", "psnr_db_gt", "up", 1),
            (r"SSIM elev $\uparrow$", "ssim_gt_elev_mean", "up", 3),
            (r"SSIM range $\uparrow$", "ssim_gt_range_mean", "up", 3),
            (r"SSIM azim $\uparrow$", "ssim_gt_azimuth_mean", "up", 3),
            (r"profile cos med\,$\uparrow$", "pixel_cosine_gt_median", "up", 3),
            ("space",),
            ("group", r"\emph{detection (matched)}"),
            (r"precision $\uparrow$", "matched_precision", "up", 3),
            (r"recall $\uparrow$", "matched_recall", "up", 3),
            (r"matched F1 $\uparrow$", "matched_f1", "up", 3),
            (r"count exact $\uparrow$", "count_exact_frac", "up", 3),
            (r"\quad under $\downarrow$", "count_under_frac", "down", 3),
            (r"\quad over $\downarrow$", "count_over_frac", "down", 3),
            (r"peak err med $\downarrow$", "pixel_peak_err_units_median_gt", "down", 2),
            (r"peak err p95 $\downarrow$", "pixel_peak_err_units_p95_gt", "down", 1),
        ]

    def k5_detection_rows(self) -> list:
        return [
            (r"precision $\uparrow$", "matched_precision", "up", 3),
            *[(rf"\quad $k{{=}}{k}$", f"matched_precision_gt{k}", "up", 3) for k in range(1, 6)],
            ("space",),
            (r"recall $\uparrow$", "matched_recall", "up", 3),
            *[(rf"\quad $k{{=}}{k}$", f"matched_recall_gt{k}", "up", 3) for k in range(1, 6)],
            ("space",),
            ("group", r"\emph{count acc $\mid$ pred $k$ $\uparrow$}"),
            *[(rf"\quad pred $k{{=}}{k}$", f"count_acc_pred{k}", "up", 3) for k in range(1, 6)],
        ]

    def k5_component_rows(self) -> list:
        return [
            (r"$\mu$ MAE $\downarrow$", "matched_mu_mae", "down", 2),
            *[(rf"\quad $k{{=}}{k}$", f"matched_mu_mae_gt{k}", "down", 2) for k in range(1, 6)],
            ("space",),
            (r"$\sigma$ MAE $\downarrow$", "matched_sig_mae", "down", 2),
            *[(rf"\quad $k{{=}}{k}$", f"matched_sig_mae_gt{k}", "down", 2) for k in range(1, 6)],
            ("space",),
            (r"$a$ MAE $\downarrow$", "matched_amp_mae", "down", 3),
            *[(rf"\quad $k{{=}}{k}$", f"matched_amp_mae_gt{k}", "down", 3) for k in range(1, 6)],
        ]

    def k5_stats_rows(self) -> list:
        return [
            ("group", r"\textcolor{soft}{\emph{predicted active fraction --- share of pixels where the slot fires}}"),
            *[(rf"\; slot {s}", f"slot_{s}_active_pred_frac", "up", 3, f"slot_{s}_active_gt_frac", 3) for s in range(5)],
            ("space", "1.5pt"),
            ("group", r"\textcolor{soft}{\emph{$a$ --- predicted mean over GT-active pixels}}"),
            *[(rf"\; slot {s}", f"slot_{s}_amp_active_pred_mean", "up", 2, f"slot_{s}_amp_active_gt_mean", 2) for s in range(5)],
            ("space", "1.5pt"),
            ("group", r"\textcolor{soft}{\emph{$a$ --- predicted mean over GT-inactive pixels (leakage)}}"),
            *[(rf"\; slot {s}", f"slot_{s}_amp_inactive_pred_mean", "down", 2, f"slot_{s}_amp_inactive_gt_mean", 2) for s in range(1, 5)],
            ("space", "1.5pt"),
            ("group", r"\textcolor{soft}{\emph{distribution width --- std over GT-active pixels}}"),
            (r"\; $a$ --- slot 0", "slot_0_amp_active_pred_std", "up", 2, "slot_0_amp_active_gt_std", 2),
            (r"\; $\mu$ --- slot 0", "slot_0_mu_active_pred_std", "up", 2, "slot_0_mu_active_gt_std", 2),
            (r"\; $\sigma$ --- slot 0", "slot_0_sig_active_pred_std", "up", 2, "slot_0_sig_active_gt_std", 2),
            (r"\; $a$ --- slot 1", "slot_1_amp_active_pred_std", "up", 2, "slot_1_amp_active_gt_std", 2),
            (r"\; $\mu$ --- slot 1", "slot_1_mu_active_pred_std", "up", 2, "slot_1_mu_active_gt_std", 2),
            (r"\; $\sigma$ --- slot 1", "slot_1_sig_active_pred_std", "up", 2, "slot_1_sig_active_gt_std", 2),
        ]

    def k5_frames(self) -> dict:
        emitter = TableEmitter(ReportParser(self.repo_root / "results" / "K5" / "dual_input_k5").load(), self.K5_RUNS, self.formatter, self.gate)

        return {
            "kdA": emitter.emit(self.k5_headline_rows(), None),
            "kdB": emitter.emit(self.k5_detection_rows(), None),
            "kdC": emitter.emit(self.k5_component_rows(), None),
            "kdD": emitter.emit(self.k5_stats_rows(), len(self.K5_RUNS) + 2),
        }

    def ratio_k5_frames(self) -> dict:
        emitter = TableEmitter(ReportParser(self.repo_root / "results" / "K5" / "dual_ratio_k5").load(), self.RATIO_K5_RUNS, self.formatter, self.gate)

        return {
            "krA": emitter.emit(self.k5_headline_rows(), None),
            "krB": emitter.emit(self.k5_detection_rows(), None),
            "krC": emitter.emit(self.k5_component_rows(), None),
            "krD": emitter.emit(self.k5_stats_rows(), len(self.RATIO_K5_RUNS) + 2),
        }

    def ratio_frames(self) -> dict:
        values                  = ReportParser(self.repo_root / "results" / "K2" / "dual_k2_ratio").load()
        values[self.SINGLE_RUN] = ReportParser(self.repo_root / "results" / "K2" / "Benchmark").load()[self.SINGLE_RUN]
        emitter                 = TableEmitter(values, self.RATIO_RUNS + [self.SINGLE_RUN], self.formatter, self.gate)

        headline = emitter.emit([
            ("group", r"\emph{training (validation)}"),
            (r"val loss $\downarrow$", "best_val_loss", "down", 3),
            ("space",),
            ("group", r"\emph{reconstruction (curve)}"),
            (r"curve $R^2$ $\uparrow$", "overall_r2_gt", "up", 3),
            (r"curve MAE $\downarrow$", "curve_mae_gt", "down", 4),
            (r"PSNR [dB] $\uparrow$", "psnr_db_gt", "up", 1),
            (r"SSIM elev $\uparrow$", "ssim_gt_elev_mean", "up", 4),
            (r"SSIM range $\uparrow$", "ssim_gt_range_mean", "up", 4),
            (r"SSIM azim $\uparrow$", "ssim_gt_azimuth_mean", "up", 4),
            (r"profile cos med\,$\uparrow$", "pixel_cosine_gt_median", "up", 3),
            ("space",),
            ("group", r"\emph{detection (matched)}"),
            (r"precision $\uparrow$", "matched_precision", "up", 3),
            (r"recall $\uparrow$", "matched_recall", "up", 3),
            (r"matched F1 $\uparrow$", "matched_f1", "up", 3),
            (r"count exact $\uparrow$", "count_exact_frac", "up", 3),
            (r"\quad under $\downarrow$", "count_under_frac", "down", 3),
            (r"\quad over $\downarrow$", "count_over_frac", "down", 3),
            (r"peak err med $\downarrow$", "pixel_peak_err_units_median_gt", "down", 2),
            (r"peak err p95 $\downarrow$", "pixel_peak_err_units_p95_gt", "down", 1),
        ], None)

        by_count = emitter.emit([
            (r"precision $\uparrow$", "matched_precision", "up", 3),
            (r"\quad $k{=}1$", "matched_precision_gt1", "up", 3),
            (r"\quad $k{=}2$", "matched_precision_gt2", "up", 3),
            ("space",),
            (r"recall $\uparrow$", "matched_recall", "up", 3),
            (r"\quad $k{=}1$", "matched_recall_gt1", "up", 3),
            (r"\quad $k{=}2$", "matched_recall_gt2", "up", 3),
            ("space",),
            ("group", r"\emph{count acc $\mid$ pred $k$}"),
            (r"\quad pred $k{=}1$ $\uparrow$", "count_acc_pred1", "up", 3),
            (r"\quad pred $k{=}2$ $\uparrow$", "count_acc_pred2", "up", 3),
            ("space",),
            (r"$\mu$ MAE $\downarrow$", "matched_mu_mae", "down", 2),
            (r"\quad $k{=}1$", "matched_mu_mae_gt1", "down", 2),
            (r"\quad $k{=}2$", "matched_mu_mae_gt2", "down", 2),
            ("space",),
            (r"$\sigma$ MAE $\downarrow$", "matched_sig_mae", "down", 3),
            (r"\quad $k{=}1$", "matched_sig_mae_gt1", "down", 3),
            (r"\quad $k{=}2$", "matched_sig_mae_gt2", "down", 3),
            ("space",),
            (r"$a$ MAE $\downarrow$", "matched_amp_mae", "down", 3),
            (r"\quad $k{=}1$", "matched_amp_mae_gt1", "down", 3),
            (r"\quad $k{=}2$", "matched_amp_mae_gt2", "down", 3),
        ], None)

        stats = emitter.emit([
            ("group", r"\textcolor{soft}{\emph{slot 0 --- strongest scatterer; GT-active on every pixel}}"),
            (r"\; active frac", "slot_0_active_pred_frac", "up", 3, "slot_0_active_gt_frac", 3),
            (r"\; $a$ --- active", "slot_0_amp_active_pred_mean", "up", 2, "slot_0_amp_active_gt_mean", 2),
            (r"\; $\mu$ --- active", "slot_0_mu_active_pred_mean", "up", 2, "slot_0_mu_active_gt_mean", 2),
            (r"\; $\sigma$ --- active", "slot_0_sig_active_pred_mean", "up", 2, "slot_0_sig_active_gt_mean", 2),
            ("space", "1.5pt"),
            ("group", r"\textcolor{soft}{\emph{slot 1 --- second scatterer; GT-active on 26\% of pixels}}"),
            (r"\; active frac", "slot_1_active_pred_frac", "up", 3, "slot_1_active_gt_frac", 3),
            (r"\; $a$ --- active", "slot_1_amp_active_pred_mean", "up", 2, "slot_1_amp_active_gt_mean", 2),
            (r"\; $a$ --- inact.", "slot_1_amp_inactive_pred_mean", "down", 2, "slot_1_amp_inactive_gt_mean", 2),
            (r"\; $\mu$ --- active", "slot_1_mu_active_pred_mean", "up", 2, "slot_1_mu_active_gt_mean", 2),
            ("raw", r"\; $\mu$ --- inact.", "slot_1_mu_inactive_pred_mean", 2),
            (r"\; $\sigma$ --- active", "slot_1_sig_active_pred_mean", "up", 2, "slot_1_sig_active_gt_mean", 2),
            ("raw", r"\; $\sigma$ --- inact.", "slot_1_sig_inactive_pred_mean", 2),
            ("space", "1.5pt"),
            ("group", r"\textcolor{soft}{\emph{distribution width --- std over GT-active pixels}}"),
            (r"\; $a$ --- slot 0", "slot_0_amp_active_pred_std", "up", 2, "slot_0_amp_active_gt_std", 2),
            (r"\; $\mu$ --- slot 0", "slot_0_mu_active_pred_std", "up", 2, "slot_0_mu_active_gt_std", 2),
            (r"\; $\sigma$ --- slot 0", "slot_0_sig_active_pred_std", "up", 2, "slot_0_sig_active_gt_std", 2),
            (r"\; $a$ --- slot 1", "slot_1_amp_active_pred_std", "up", 2, "slot_1_amp_active_gt_std", 2),
            (r"\; $\mu$ --- slot 1", "slot_1_mu_active_pred_std", "up", 2, "slot_1_mu_active_gt_std", 2),
            (r"\; $\sigma$ --- slot 1", "slot_1_sig_active_pred_std", "up", 2, "slot_1_sig_active_gt_std", 2),
        ], len(self.RATIO_RUNS) + 2)

        return {"drA": headline, "drB": by_count, "drC": stats}

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        fragments = {}
        fragments.update(self.k5_frames())
        fragments.update(self.ratio_frames())
        fragments.update(self.ratio_k5_frames())

        for tag, body in fragments.items():
            path = self.output_dir / f"{tag}.tex"
            path.write_text(body + "\n")
            self.logger.info(f"wrote {path}")


if __name__ == "__main__":
    DualRound2Tables(_vault / "code" / "DLR-TomoSAR").run()
