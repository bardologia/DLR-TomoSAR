from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import fields
from pathlib     import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuration.dataset                import AugmentationConfig
from configuration.training               import LossConfig
from models                               import BACKBONE_MODEL_REGISTRY
from pipelines.shared.training.run_naming import RunNaming
from tools.monitoring.logger              import Logger


class RunRenamer:

    PATCH_FILES  = (Path("meta") / "run_summary.json", Path("docs") / "trainer_config.json", Path("docs") / "resolved_entry_config.json")
    RESULT_FILES = ("training_results.json", "inference_results.json")

    def __init__(self, roots: list[Path], apply: bool) -> None:
        self.roots  = [Path(root) for root in roots]
        self.apply  = apply
        self.logger = Logger(log_dir="logs", name="rename_runs")

    def _process_root(self, root: Path) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a directory")

        if (root / "training").is_dir():
            self._process_benchmark_tag(root)
            return

        if (root / "meta" / "run_summary.json").is_file():
            self._process_runs_root(root.parent, [root])
            return

        children = [child for child in sorted(root.iterdir()) if child.is_dir()]
        tags     = [child for child in children if (child / "training").is_dir()]
        runs     = [child for child in children if (child / "meta" / "run_summary.json").is_file()]

        if not tags and not runs:
            raise FileNotFoundError(f"{root} holds no benchmark run tags (*/training) and no training runs (*/meta/run_summary.json)")

        for tag_dir in tags:
            self._process_benchmark_tag(tag_dir)

        if runs:
            self._process_runs_root(root, runs)

    def _process_benchmark_tag(self, tag_dir: Path) -> None:
        self.logger.section(f"[{self._mode()}] benchmark tag {tag_dir}")

        run_dirs = sorted(path for path in (tag_dir / "training").iterdir() if path.is_dir())
        mapping  = self._plan(run_dirs)
        if not mapping:
            self.logger.info("nothing to rename")
            return

        self._rename_runs(tag_dir / "training", mapping)

        if self.apply:
            for name in self.RESULT_FILES:
                self._patch_file(tag_dir / "pipeline" / name, mapping)

    def _process_runs_root(self, root: Path, run_dirs: list[Path]) -> None:
        self.logger.section(f"[{self._mode()}] runs root {root}")

        mapping = self._plan(run_dirs)
        if not mapping:
            self.logger.info("nothing to rename")
            return

        self._rename_runs(root, mapping)

    def _mode(self) -> str:
        return "APPLY" if self.apply else "DRY RUN"

    def _plan(self, run_dirs: list[Path]) -> dict[str, str]:
        mapping = {}
        for run_dir in run_dirs:
            new_name = self._plan_run(run_dir)
            if new_name is not None:
                mapping[run_dir.name] = new_name

        return mapping

    def _plan_run(self, run_dir: Path) -> str | None:
        summary = self._read_json(run_dir / "meta" / "run_summary.json")
        if summary is None:
            self.logger.info(f"skip {run_dir.name}: no meta/run_summary.json (not a training run)")
            return None

        model = summary["model_name"]
        if model not in BACKBONE_MODEL_REGISTRY:
            self.logger.info(f"skip {run_dir.name}: model '{model}' is not a backbone (autoencoder or foreign run)")
            return None

        model_config = self._read_json(run_dir / "meta" / "model_config.json")
        trainer      = self._read_json(run_dir / "docs" / "trainer_config.json")
        dataset      = self._read_json(run_dir / "meta" / "dataset_creation_config.json")
        if model_config is None or trainer is None or dataset is None:
            self.logger.warning(f"skip {run_dir.name}: missing model_config/trainer_config/dataset_creation_config metadata")
            return None

        payload = self._loss_payload(trainer)
        if payload is None:
            self.logger.warning(f"skip {run_dir.name}: trainer config has neither a loss curriculum nor a param_loss")
            return None

        head         = model_config["config"].get("head", "conv")
        loss         = self._loss_config(payload)
        augmentation = self._augmentation(dataset["augmentation"])
        n_gaussians  = trainer["gaussian"]["n_default_gaussians"]

        old_stem = f"{model}_{head}_{self._matching(payload)}"
        old_tag  = f"{old_stem}_{self._old_loss_tag(loss)}"
        new_stem = RunNaming.stem(model, head, loss, n_gaussians, augmentation)
        new_tag  = RunNaming.tag(model, head, loss, n_gaussians, augmentation)

        name = run_dir.name
        if name == new_tag or name.startswith(f"{new_tag}_") or name.startswith(f"{new_stem}__"):
            self.logger.info(f"skip {name}: already in the new naming")
            return None

        if "__" in name:
            base, rest = name.split("__", 1)
            if base in (old_stem, model):
                return f"{new_stem}__{rest}"

            self.logger.warning(f"skip {name}: unit prefix '{base}' matches neither '{old_stem}' nor '{model}'")
            return None

        if name == old_tag:
            return new_tag

        if name.startswith(f"{old_tag}_"):
            return f"{new_tag}{name[len(old_tag):]}"

        self.logger.warning(f"skip {name}: does not start with the expected old tag '{old_tag}'")
        return None

    def _read_json(self, path: Path) -> dict | None:
        if not path.is_file():
            return None

        return json.loads(path.read_text())

    def _loss_payload(self, trainer: dict) -> dict | None:
        if "curriculum" in trainer:
            return trainer["curriculum"]["complete"]

        return trainer.get("param_loss")

    def _loss_config(self, payload: dict) -> LossConfig:
        known = {spec.name for spec in fields(LossConfig)}
        return LossConfig(**{key: value for key, value in payload.items() if key in known})

    def _augmentation(self, payload: dict) -> AugmentationConfig:
        known = {spec.name for spec in fields(AugmentationConfig)}
        return AugmentationConfig(**{key: value for key, value in payload.items() if key in known})

    def _matching(self, payload: dict) -> str:
        return str(payload["param_matching"]).split(".")[-1].lower()

    def _old_loss_tag(self, loss: LossConfig) -> str:
        names = [term.name for term in RunNaming.NAMING_ORDER if getattr(loss, term.use_flag)]
        return "-".join(names)

    def _rename_runs(self, parent: Path, mapping: dict[str, str]) -> None:
        for old, new in mapping.items():
            source = parent / old
            target = parent / new
            if target.exists():
                raise FileExistsError(f"{target} already exists; refusing to overwrite")

            self.logger.info(f"{old} -> {new}")
            if not self.apply:
                continue

            source.rename(target)
            for relative in self.PATCH_FILES:
                self._patch_file(target / relative, {old: new})

    def _patch_file(self, path: Path, mapping: dict[str, str]) -> None:
        if not path.exists():
            return

        original = path.read_text()
        patched  = self._boundary_replace(original, mapping)
        if patched != original:
            path.write_text(patched)
            self.logger.info(f"patched {path}")

    def _boundary_replace(self, text: str, mapping: dict[str, str]) -> str:
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
            text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}", new, text)
        return text

    def run(self) -> None:
        for root in self.roots:
            self._process_root(root)

        if not self.apply:
            self.logger.info("dry run only; re-run with --apply to rename")

        self.logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename existing runs to the model-head-K_N-aug-presence-loss_weight naming, rebuilding each name from the run's persisted metadata (run_summary, model_config, trainer_config, dataset_creation_config)")
    parser.add_argument("roots", nargs="+", type=Path, help="run directories, roots holding training runs, benchmark run-tag directories (containing training/), or a benchmarks root holding them")
    parser.add_argument("--apply", action="store_true", help="perform the renames; without it the script only prints the plan")
    args = parser.parse_args()

    RunRenamer(args.roots, args.apply).run()
