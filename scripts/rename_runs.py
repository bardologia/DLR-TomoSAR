from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import BACKBONE_MODEL_REGISTRY
from tools.monitoring.logger import Logger


class BenchmarkRunRenamer:

    HEAD         = "conv"
    UNIT_PATTERN = re.compile(r"^(?P<model>[a-z0-9_]+?)__(?P<rest>.+)$")
    RESULT_FILES = ("training_results.json", "inference_results.json")
    PATCH_FILES  = (Path("meta") / "run_summary.json", Path("docs") / "trainer_config.json")

    def __init__(self, roots: list[Path], apply: bool) -> None:
        self.roots  = [Path(root) for root in roots]
        self.apply  = apply
        self.logger = Logger(log_dir="logs", name="rename_benchmark_runs")

    def _tag_dirs(self) -> list[Path]:
        tags = []
        for root in self.roots:
            if not root.is_dir():
                raise FileNotFoundError(f"{root} is not a directory")

            if (root / "training").is_dir():
                tags.append(root)
                continue

            children = [child for child in sorted(root.iterdir()) if (child / "training").is_dir()]
            if not children:
                raise FileNotFoundError(f"{root} contains no benchmark run tags (no */training directories)")
            tags.extend(children)

        return tags

    def _assert_conv_head(self, run_dir: Path) -> None:
        payload = json.loads((run_dir / "meta" / "model_config.json").read_text())
        head    = payload["config"].get("head", self.HEAD)
        if head != self.HEAD:
            raise ValueError(f"{run_dir.name}: persisted head is '{head}', expected '{self.HEAD}'; this run was not trained with the standard head")

    def _matching(self, run_dir: Path) -> str:
        payload = json.loads((run_dir / "docs" / "trainer_config.json").read_text())
        value   = payload["curriculum"]["complete"]["param_matching"]
        return str(value).split(".")[-1].lower()

    def _new_name(self, run_dir: Path, model: str, rest: str) -> str:
        return f"{model}_{self.HEAD}_{self._matching(run_dir)}__{rest}"

    def _plan_tag(self, tag_dir: Path) -> dict[str, str]:
        mapping = {}
        for run_dir in sorted(path for path in (tag_dir / "training").iterdir() if path.is_dir()):
            match = self.UNIT_PATTERN.match(run_dir.name)
            if match is None or match.group("model") not in BACKBONE_MODEL_REGISTRY:
                self.logger.warning(f"skip {tag_dir.name}/training/{run_dir.name}: not an old-format backbone unit (already canonical or foreign)")
                continue

            self._assert_conv_head(run_dir)
            mapping[run_dir.name] = self._new_name(run_dir, match.group("model"), match.group("rest"))

        return mapping

    def _boundary_replace(self, text: str, mapping: dict[str, str]) -> str:
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
            text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}", new, text)
        return text

    def _patch_file(self, path: Path, mapping: dict[str, str]) -> None:
        if not path.exists():
            return

        original = path.read_text()
        patched  = self._boundary_replace(original, mapping)
        if patched != original:
            path.write_text(patched)
            self.logger.info(f"patched {path}")

    def _rename_tag(self, tag_dir: Path, mapping: dict[str, str]) -> None:
        for old, new in mapping.items():
            source = tag_dir / "training" / old
            target = tag_dir / "training" / new
            if target.exists():
                raise FileExistsError(f"{target} already exists; refusing to overwrite")

            self.logger.info(f"{tag_dir.name}: {old} -> {new}")
            if not self.apply:
                continue

            source.rename(target)
            for relative in self.PATCH_FILES:
                self._patch_file(target / relative, {old: new})

        if self.apply:
            for name in self.RESULT_FILES:
                self._patch_file(tag_dir / "pipeline" / name, mapping)

    def run(self) -> None:
        mode = "APPLY" if self.apply else "DRY RUN"

        for tag_dir in self._tag_dirs():
            self.logger.section(f"[{mode}] {tag_dir}")

            mapping = self._plan_tag(tag_dir)
            if not mapping:
                self.logger.info("nothing to rename")
                continue

            self._rename_tag(tag_dir, mapping)

        if not self.apply:
            self.logger.info("dry run only; re-run with --apply to rename")

        self.logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename pre-2026-07-06 benchmark runs to the canonical model_head_matching__loss naming (all runs assumed conv head)")
    parser.add_argument("roots", nargs="+", type=Path, help="benchmark run-tag directories (containing training/), or a benchmarks root holding them")
    parser.add_argument("--apply", action="store_true", help="perform the renames; without it the script only prints the plan")
    args = parser.parse_args()

    BenchmarkRunRenamer(args.roots, args.apply).run()
