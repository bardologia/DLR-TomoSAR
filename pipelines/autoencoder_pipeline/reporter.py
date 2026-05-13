from __future__ import annotations

from dataclasses import asdict
from datetime    import datetime
from pathlib     import Path
from typing      import Any

from .config import AutoencoderConfig


class Reporter:

    def __init__(self,
                 ae_config     : AutoencoderConfig,
                 run_directory : Path) -> None:
        self.ae_config     = ae_config
        self.run_directory = Path(run_directory)
        self.report_path   = Path(ae_config.io.report_path or self.run_directory / "report.md")

    def write(self,
              history          : list[dict[str, float]],
              best_epoch       : int,
              best_val_total   : float,
              inference_results: dict[str, dict[str, Any]],
              checkpoint_paths : dict[str, str]) -> Path:
        lines: list[str] = []
        lines += self._section_header()
        lines += self._section_configuration()
        lines += self._section_training(history, best_epoch, best_val_total)
        for split_name, summary in inference_results.items():
            lines += self._section_inference(split_name, summary)
        lines += self._section_checkpoints(checkpoint_paths)
        lines += self._section_artifacts(inference_results)

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        return self.report_path

    def _section_header(self) -> list[str]:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [
            f"# Autoencoder Run Report",
            "",
            f"- **Run directory** : `{self.run_directory}`",
            f"- **Generated**     : {ts}",
            f"- **Profile length**: {self.ae_config.profile_length}",
            f"- **Latent dim**    : {self.ae_config.latent_dim}",
            f"- **Encoder**       : {self.ae_config.encoder.backbone.value}",
            f"- **Decoder**       : {self.ae_config.decoder.backbone.value}",
            "",
        ]

    def _section_configuration(self) -> list[str]:
        cfg   = self.ae_config
        lines = ["## Configuration", "", "### Model", "",
                 "| Field | Value |", "|---|---|",
                 f"| latent_dim | {cfg.latent_dim} |",
                 f"| profile_length | {cfg.profile_length} |",
                 f"| encoder.backbone | {cfg.encoder.backbone.value} |",
                 f"| encoder.channels | {cfg.encoder.channels} |",
                 f"| encoder.proj_dim | {cfg.encoder.proj_dim} |",
                 f"| encoder.use_projection_head | {cfg.encoder.use_projection_head} |",
                 f"| decoder.backbone | {cfg.decoder.backbone.value} |",
                 f"| decoder.channels | {cfg.decoder.channels} |",
                 ""]
        lines += ["### Loss weights", "",
                  "| Term | Weight |", "|---|---|",
                  f"| reconstruction | {cfg.loss.reconstruction_weight} |",
                  f"| variance | {cfg.loss.variance_weight} |",
                  f"| covariance | {cfg.loss.covariance_weight} |",
                  f"| contrastive | {cfg.loss.contrastive_weight} |",
                  ""]
        lines += ["### Trainer", "",
                  "| Field | Value |", "|---|---|",
                  f"| epochs | {cfg.trainer.epochs} |",
                  f"| optimizer | {cfg.trainer.optimizer} |",
                  f"| learning_rate | {cfg.trainer.learning_rate} |",
                  f"| weight_decay | {cfg.trainer.weight_decay} |",
                  f"| scheduler | {cfg.trainer.scheduler} |",
                  f"| grad_clip | {cfg.trainer.grad_clip} |",
                  f"| use_amp | {cfg.trainer.use_amp} |",
                  ""]
        lines += ["### Data", "",
                  "| Field | Value |", "|---|---|",
                  f"| normalize | {cfg.data.normalize} |",
                  f"| log_compress | {cfg.data.log_compress} |",
                  f"| contrastive_view | {cfg.data.contrastive_view.value} |",
                  f"| max_profiles | {cfg.data.max_profiles} |",
                  ""]
        return lines

    def _section_training(self,
                          history        : list[dict[str, float]],
                          best_epoch     : int,
                          best_val_total : float) -> list[str]:
        if not history:
            return ["## Training", "", "_No epochs were logged._", ""]

        last  = history[-1]
        first = history[0]
        epochs_run = int(last.get("epoch", len(history)))
        components = ["total", "reconstruction", "variance", "covariance", "contrastive"]

        lines = ["## Training", "",
                 f"- Epochs run: **{epochs_run}**",
                 f"- Best epoch: **{best_epoch}**",
                 f"- Best val/total: **{best_val_total:.6e}**",
                 "",
                 "### Final-epoch losses", "",
                 "| Component | Train | Val |", "|---|---|---|"]
        for c in components:
            tr = last.get(f"train/{c}")
            va = last.get(f"val/{c}")
            tr_s = f"{tr:.6e}" if tr is not None else "—"
            va_s = f"{va:.6e}" if va is not None else "—"
            lines.append(f"| {c} | {tr_s} | {va_s} |")
        lines.append("")

        lines += ["### Initial-epoch losses (for reference)", "",
                  "| Component | Train | Val |", "|---|---|---|"]
        for c in components:
            tr = first.get(f"train/{c}")
            va = first.get(f"val/{c}")
            tr_s = f"{tr:.6e}" if tr is not None else "—"
            va_s = f"{va:.6e}" if va is not None else "—"
            lines.append(f"| {c} | {tr_s} | {va_s} |")
        lines.append("")
        lines.append("![Loss overview](images/loss_overview.png)")
        lines.append("")
        return lines

    def _section_inference(self, split_name: str, summary: dict[str, Any]) -> list[str]:
        recon = summary.get("reconstruction_stats", {})
        embed = summary.get("embedding_stats", {})
        lines = [f"## Inference — `{split_name}` split", "",
                 f"- Profiles evaluated: **{summary.get('num_profiles', 0):,}**",
                 f"- Profile length: **{summary.get('profile_length', '—')}**",
                 f"- Latent dim: **{summary.get('latent_dim', '—')}**",
                 ""]

        lines += ["### Reconstruction quality", "",
                  "| Metric | Value |", "|---|---|"]
        for key in ("mse_mean", "mse_median", "mse_std", "mse_min", "mse_max",
                    "mae_mean", "mae_median", "rmse_mean",
                    "psnr_mean", "psnr_median", "r2_mean", "r2_median"):
            val = recon.get(key)
            if val is not None:
                lines.append(f"| {key} | {val:.6e} |")
        lines.append("")
        lines.append(f"![Error histogram]({Path(summary.get('error_histogram_path', '')).relative_to(self.run_directory) if summary.get('error_histogram_path') else ''})")
        lines.append("")

        lines += ["### Embedding quality", "",
                  "| Metric | Value |", "|---|---|",
                  f"| latent_dim | {embed.get('latent_dim', '—')} |",
                  f"| active_dimensions (std>1e-2) | {embed.get('active_dimensions', '—')} |",
                  f"| participation_ratio | {embed.get('participation_ratio', float('nan')):.4f} |",
                  f"| effective_rank | {embed.get('effective_rank', float('nan')):.4f} |",
                  f"| components_for_90pct | {embed.get('components_for_90pct', '—')} |",
                  f"| components_for_95pct | {embed.get('components_for_95pct', '—')} |",
                  ""]

        eig = embed.get("covariance_eigenvalues") or []
        if eig:
            top = ", ".join(f"{v:.3e}" for v in eig[:min(8, len(eig))])
            lines.append(f"- Top eigenvalues: {top}")
            lines.append("")

        lines += ["### Reconstruction galleries", ""]
        for tag, path in summary.get("gallery_paths", {}).items():
            try:
                rel = Path(path).relative_to(self.run_directory)
            except ValueError:
                rel = Path(path)
            lines.append(f"- **{tag}**: ![{split_name} {tag}]({rel})")
        lines.append("")

        if summary.get("spectrum_path"):
            try:
                rel = Path(summary["spectrum_path"]).relative_to(self.run_directory)
            except ValueError:
                rel = Path(summary["spectrum_path"])
            lines += [f"![{split_name} spectrum]({rel})", ""]

        if summary.get("pca_path"):
            try:
                rel = Path(summary["pca_path"]).relative_to(self.run_directory)
            except ValueError:
                rel = Path(summary["pca_path"])
            lines += [f"![{split_name} PCA]({rel})", ""]

        if summary.get("umap_path"):
            try:
                rel = Path(summary["umap_path"]).relative_to(self.run_directory)
            except ValueError:
                rel = Path(summary["umap_path"])
            lines += [f"![{split_name} UMAP]({rel})", ""]

        return lines

    def _section_checkpoints(self, checkpoint_paths: dict[str, str]) -> list[str]:
        lines = ["## Checkpoints", "", "| Tag | File |", "|---|---|"]
        for tag, path in checkpoint_paths.items():
            lines.append(f"| {tag} | `{path}` |")
        lines.append("")
        return lines

    def _section_artifacts(self, inference_results: dict[str, dict[str, Any]]) -> list[str]:
        lines = ["## Artifacts", ""]
        for split_name, summary in inference_results.items():
            lines.append(f"- `{split_name}` HDF5 latents: `{summary.get('h5_path', '—')}`")
        lines.append("")
        return lines
