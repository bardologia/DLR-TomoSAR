from __future__ import annotations

from pathlib import Path


class RunSelector:
    def __init__(self, runs_dir: Path, checkpoint_filename: str, logger) -> None:
        self.runs_dir            = Path(runs_dir)
        self.checkpoint_filename = checkpoint_filename
        self.logger              = logger

    def _discover(self) -> list[Path]:
        if not self.runs_dir.is_dir():
            raise FileNotFoundError(f"Runs directory does not exist: {self.runs_dir}")

        checkpoints = self.runs_dir.rglob(self.checkpoint_filename)
        run_dirs    = sorted({path.parent for path in checkpoints if path.is_file()})

        if not run_dirs:
            raise FileNotFoundError(f"No '{self.checkpoint_filename}' found in any directory under {self.runs_dir}")

        return run_dirs

    def _present(self, run_dirs: list[Path]) -> None:
        rows = []
        for index, run_dir in enumerate(run_dirs, start=1):
            checkpoint = run_dir / self.checkpoint_filename
            size_mb    = checkpoint.stat().st_size / (1024 * 1024)
            rows.append({
                "#"          : index,
                "Run"        : str(run_dir.relative_to(self.runs_dir)),
                "Checkpoint" : f"{size_mb:,.1f} MB",
            })

        self.logger.metrics_table(rows, columns=["#", "Run", "Checkpoint"], title=f"Runs under {self.runs_dir}")

    def _prompt(self, run_dirs: list[Path]) -> list[Path]:
        raw       = input(f"Select run(s) to x-ray [1-{len(run_dirs)}, comma-separated, or 'all']: ").strip()
        selection = self._parse(raw, run_dirs)

        self.logger.ok(f"Selected {len(selection)} run(s): {', '.join(run_dir.name for run_dir in selection)}")
        return selection

    def _parse(self, raw: str, run_dirs: list[Path]) -> list[Path]:
        if raw == "" or raw.lower() in ("all", "*"):
            return run_dirs

        indices = []
        for token in raw.replace(",", " ").split():
            if not token.isdigit():
                raise ValueError(f"Invalid selection token '{token}'; expected run numbers 1-{len(run_dirs)} or 'all'")

            index = int(token)
            if index < 1 or index > len(run_dirs):
                raise ValueError(f"Selection {index} is out of range 1-{len(run_dirs)}")

            indices.append(index)

        ordered = sorted(dict.fromkeys(indices))
        return [run_dirs[index - 1] for index in ordered]

    def select(self) -> list[Path]:
        run_dirs = self._discover()
        self._present(run_dirs)
        return self._prompt(run_dirs)
