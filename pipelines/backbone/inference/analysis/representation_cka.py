from __future__ import annotations

from itertools import combinations
from pathlib   import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from configuration.diagnostics                          import CkaConfig
from pipelines.backbone.inference.analysis.layer_probes import FeatureSampler
from pipelines.backbone.inference.loader                import RunLoader
from pipelines.backbone.inference.probes                import ModelDevice
from tools.data.io                                      import FileIO
from tools.diagnostics.activation_recorder              import ActivationRecorder
from tools.reporting.markdown                           import MarkdownDoc, MarkdownTable
from tools.reporting.plotting                           import PlotBase
from tools.runtime.run_selector                         import RunSelector
from tools.runtime.run_tag                              import RunTag


class CkaComputation:

    DIVERGENCE_THRESHOLD = 0.5

    @staticmethod
    def _unbiased_hsic(X: np.ndarray, Y: np.ndarray) -> float:
        n = X.shape[0]

        cross      = X.T @ Y
        norm_x     = (X * X).sum(axis=1)
        norm_y     = (Y * Y).sum(axis=1)
        sum_x      = X.sum(axis=0)
        sum_y      = Y.sum(axis=0)
        row_kx     = X @ sum_x
        row_ly     = Y @ sum_y

        trace_term = float((cross * cross).sum()) - float((norm_x * norm_y).sum())
        ones_kx    = float(sum_x @ sum_x) - float(norm_x.sum())
        ones_ly    = float(sum_y @ sum_y) - float(norm_y.sum())
        mixed_term = float((row_kx * row_ly).sum()) - float((row_kx * norm_y).sum()) - float((norm_x * row_ly).sum()) + float((norm_x * norm_y).sum())

        return (trace_term + ones_kx * ones_ly / ((n - 1.0) * (n - 2.0)) - 2.0 * mixed_term / (n - 2.0)) / (n * (n - 3.0))

    @classmethod
    def linear_cka(cls, features_a: np.ndarray, features_b: np.ndarray) -> float:
        X = np.asarray(features_a, dtype=np.float64)
        Y = np.asarray(features_b, dtype=np.float64)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"CKA needs matched samples, got {X.shape[0]} and {Y.shape[0]}")
        if X.shape[0] < 4:
            raise ValueError(f"Debiased CKA needs at least 4 samples, got {X.shape[0]}")

        hsic_xy = cls._unbiased_hsic(X, Y)
        hsic_xx = cls._unbiased_hsic(X, X)
        hsic_yy = cls._unbiased_hsic(Y, Y)

        if hsic_xx <= 0.0 or hsic_yy <= 0.0:
            return 0.0

        return max(0.0, hsic_xy / np.sqrt(hsic_xx * hsic_yy))

    @classmethod
    def cross_matrix(cls, layers_a: dict[str, np.ndarray], layers_b: dict[str, np.ndarray]) -> np.ndarray:
        names_a = list(layers_a)
        names_b = list(layers_b)

        matrix = np.zeros((len(names_a), len(names_b)))
        for i, name_a in enumerate(names_a):
            for j, name_b in enumerate(names_b):
                matrix[i, j] = cls.linear_cka(layers_a[name_a], layers_b[name_b])

        return matrix

    @staticmethod
    def alignment_score(matrix: np.ndarray) -> float:
        return float((matrix.max(axis=1).mean() + matrix.max(axis=0).mean()) / 2.0)

    @staticmethod
    def best_match_profiles(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return matrix.max(axis=1), matrix.max(axis=0)

    @classmethod
    def divergence_depth(cls, profile: np.ndarray) -> float | None:
        below = np.where(profile < cls.DIVERGENCE_THRESHOLD)[0]
        if below.size == 0:
            return None
        if profile.size == 1:
            return 0.0

        return float(below[0] / (profile.size - 1))


class CkaPlots(PlotBase):

    def pair_heatmap(self, matrix: np.ndarray, name_a: str, name_b: str, score: float, path: Path) -> Path:
        return self._imshow_figure(
            matrix,
            x_label        = f"{name_b} layers (forward order)",
            y_label        = f"{name_a} layers (forward order)",
            title          = f"Debiased linear CKA: {name_a} vs {name_b}",
            cmap           = self._cmap_with_bad("magma"),
            vmin           = 0.0,
            vmax           = 1.0,
            colorbar_label = "CKA",
            text_overlay   = f"alignment = {score:.3f}",
            path           = path,
        )

    def self_heatmap(self, matrix: np.ndarray, name: str, path: Path) -> Path:
        return self._imshow_figure(
            matrix,
            x_label        = "Layers (forward order)",
            y_label        = "Layers (forward order)",
            title          = f"Within-run layer similarity: {name}",
            cmap           = self._cmap_with_bad("magma"),
            vmin           = 0.0,
            vmax           = 1.0,
            colorbar_label = "CKA",
            path           = path,
        )

    def best_match_profile(self, forward: np.ndarray, backward: np.ndarray, name_a: str, name_b: str, path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH))

        depth_a = np.linspace(0.0, 1.0, forward.size) if forward.size > 1 else np.array([0.0])
        depth_b = np.linspace(0.0, 1.0, backward.size) if backward.size > 1 else np.array([0.0])

        ax.plot(depth_a, forward, marker="o", color=self.OKABE_ITO[0], linewidth=1.4, label=f"{name_a} → best match in {name_b}")
        ax.plot(depth_b, backward, marker="s", color=self.OKABE_ITO[1], linewidth=1.4, label=f"{name_b} → best match in {name_a}")
        ax.axhline(CkaComputation.DIVERGENCE_THRESHOLD, color="0.45", linestyle="--", linewidth=1.0, label=f"divergence threshold ({CkaComputation.DIVERGENCE_THRESHOLD})")

        ax.set_xlabel("Relative depth")
        ax.set_ylabel("Best-match CKA")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"Where representations diverge: {name_a} vs {name_b}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        return self._save(fig, path)

    def summary_heatmap(self, matrix: np.ndarray, names: list[str], path: Path) -> Path:
        self._apply_style()

        fig, ax = plt.subplots(figsize=self.figsize(self.FULL_WIDTH, aspect=0.9))
        im      = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)

        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title("Representation alignment across runs")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("Mean best-match CKA (debiased)")

        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix[i, j] < 0.6 else "black")

        fig.tight_layout()
        return self._save(fig, path)


class CkaComparison:

    def __init__(self, config: CkaConfig, logger) -> None:
        self.config = config
        self.logger = logger

        self.output_dir = Path(config.output_dir) / RunTag.now()

    def _select_runs(self) -> list[Path]:
        selector = RunSelector(self.config.runs_dir, self.config.checkpoint_name, self.logger, action="compare")
        run_dirs = selector.resolve(self.config.run_filter)

        if len(run_dirs) < 2:
            raise ValueError(f"CKA needs at least two runs, got {len(run_dirs)}")

        return run_dirs

    def _load(self, run_dir: Path):
        return RunLoader(run_dir, logger=self.logger).load(
            split           = self.config.split,
            batch_size      = self.config.batch_size,
            num_workers     = 0,
            device          = self.config.device,
            checkpoint_name = self.config.checkpoint_name,
            load_tomogram   = False,
        )

    @staticmethod
    def _sample_grid(run) -> tuple:
        return (run.split_region.as_tuple(), tuple(run.dataset_config.patch.size), tuple(run.dataset_config.patch.stride))

    def _validate_alignment(self, grids: list[tuple]) -> None:
        regions = {grid[0] for grid in grids}
        patches = {grid[1] for grid in grids}
        strides = {grid[2] for grid in grids}

        if len(regions) != 1 or len(patches) != 1 or len(strides) != 1:
            raise ValueError(f"Runs are not sample-aligned: regions {sorted(regions)}, patches {sorted(patches)}, strides {sorted(strides)}; CKA needs identical split regions and patch grids")

    def _probe_layers(self, run) -> list[str]:
        names = ActivationRecorder(run.model.module).leaf_names()

        if len(names) <= self.config.max_layers:
            return names

        keep = np.linspace(0, len(names) - 1, self.config.max_layers).astype(int)
        return [names[index] for index in sorted(set(keep))]

    def _collect(self, run) -> dict[str, np.ndarray]:
        layers   = self._probe_layers(run)
        sampler  = FeatureSampler(self.config.samples_per_batch, seed=self.config.sample_seed)
        recorder = ActivationRecorder(run.model.module)
        recorder.attach_store(layers)

        chunks: dict[str, list] = {layer: [] for layer in layers}

        for index, batch in enumerate(run.loader):
            if index >= self.config.max_batches:
                break

            images = batch[0]
            with torch.no_grad():
                run.model.module(images.to(ModelDevice.of(run.model.module)))

            stored     = recorder.stored()
            B, _, H, W = images.shape
            b, i, j    = sampler.sample_coords(B, H, W)

            for layer in layers:
                tensor = stored.get(layer)
                if tensor is None or not sampler.is_feature_map(tensor, B, H, W):
                    chunks.pop(layer, None)
                    continue
                if layer in chunks:
                    chunks[layer].append(sampler.features_at(tensor, b, i, j, B, H, W))

        recorder.detach()

        if not chunks:
            raise ValueError(f"Run {run.backbone_name} produced no (B, C, H, W) feature map on the input grid to compare")

        return {layer: np.concatenate(parts, axis=0) for layer, parts in chunks.items()}

    def _write_report(self, names: list[str], summary: np.ndarray, pairs: dict, figures: dict[str, Path]) -> Path:
        doc = MarkdownDoc(title="Representation similarity (debiased linear CKA)")
        doc.paragraph(
            f"Debiased linear CKA (unbiased HSIC estimator) between {len(names)} runs on identical sampled pixels of the '{self.config.split}' split. "
            "The summary cell is the mean best-match CKA across layers (1 = every layer of one run has a near-identical counterpart in the other). "
            "Per-pair heatmaps show the full cross-layer structure; the divergence profiles track each layer's best match by relative depth, and the "
            f"divergence depth is the first relative depth whose best match falls below {CkaComputation.DIVERGENCE_THRESHOLD}. "
            "Within-run self-similarity heatmaps expose the block structure of each network's stages."
        )

        table = MarkdownTable(("Run", *[name for name in names]))
        for i, name in enumerate(names):
            table.add_row(f"`{name}`", *[f"{summary[i, j]:.3f}" for j in range(len(names))])
        doc.table(table)

        doc.heading("Pair diagnostics", level=2)
        pair_table = MarkdownTable(("Pair", "Alignment", "Divergence depth →", "Divergence depth ←"))
        for (name_a, name_b), entry in pairs.items():
            forward  = f"{entry['divergence_forward']:.2f}" if entry["divergence_forward"] is not None else "never"
            backward = f"{entry['divergence_backward']:.2f}" if entry["divergence_backward"] is not None else "never"
            pair_table.add_row(f"`{name_a}` vs `{name_b}`", f"{entry['score']:.3f}", forward, backward)
        doc.table(pair_table)

        doc.heading("Figures", level=2)
        for name, path in figures.items():
            doc.image(name, str(path.relative_to(self.output_dir)))

        return doc.save(self.output_dir / "cka_report.md")

    def run(self) -> dict:
        FileIO.ensure_dirs(self.output_dir)
        PlotBase.use_style(self.config.figure_style)

        run_dirs = self._select_runs()
        names    = ["/".join(run_dir.relative_to(self.config.runs_dir).parts) if run_dir.is_relative_to(self.config.runs_dir) else run_dir.name for run_dir in run_dirs]

        grids    = []
        features = []
        for run_dir in run_dirs:
            run = self._load(run_dir)

            grids.append(self._sample_grid(run))
            self._validate_alignment(grids)

            features.append(self._collect(run))
            del run

        plots   = CkaPlots()
        figures = {}

        for i, name in enumerate(names):
            safe = name.replace("/", "_")
            figures[f"self_{safe}"] = plots.self_heatmap(CkaComputation.cross_matrix(features[i], features[i]), name, self.output_dir / "self" / f"{i}.png")

        summary = np.eye(len(names))
        pairs   = {}

        for (i, j) in combinations(range(len(names)), 2):
            matrix            = CkaComputation.cross_matrix(features[i], features[j])
            score             = CkaComputation.alignment_score(matrix)
            forward, backward = CkaComputation.best_match_profiles(matrix)

            summary[i, j] = summary[j, i] = score
            pairs[(names[i], names[j])]   = {
                "score"               : score,
                "divergence_forward"  : CkaComputation.divergence_depth(forward),
                "divergence_backward" : CkaComputation.divergence_depth(backward),
            }

            figures[f"pair_{i}_{j}"]    = plots.pair_heatmap(matrix, names[i], names[j], score, self.output_dir / "pairs" / f"{i}_{j}.png")
            figures[f"profile_{i}_{j}"] = plots.best_match_profile(forward, backward, names[i], names[j], self.output_dir / "pairs" / f"{i}_{j}_profile.png")

        figures["alignment"] = plots.summary_heatmap(summary, names, self.output_dir / "alignment.png")

        payload = {
            "runs"        : names,
            "split"       : self.config.split,
            "alignment"   : summary.tolist(),
            "pair_scores" : {f"{name_a}|{name_b}": entry["score"] for (name_a, name_b), entry in pairs.items()},
            "divergence"  : {f"{name_a}|{name_b}": {"forward": entry["divergence_forward"], "backward": entry["divergence_backward"]} for (name_a, name_b), entry in pairs.items()},
        }
        FileIO.save_json(payload, self.output_dir / "cka.json")

        report_path = self._write_report(names, summary, pairs, figures)
        self.logger.ok(f"CKA over {len(names)} runs -> {report_path}")

        return payload
