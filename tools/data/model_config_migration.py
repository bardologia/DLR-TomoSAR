from __future__ import annotations

from pathlib import Path

from tools.data.io import FileIO, BackboneModelConfigIO


class ModelConfigKeyMigrator:
    OLD_NAME_KEY = "model_name"

    def __init__(self, runs_dir: Path, dry_run: bool, logger) -> None:
        self.runs_dir = Path(runs_dir)
        self.dry_run  = dry_run
        self.logger   = logger

        self.new_key  = BackboneModelConfigIO.NAME_KEY
        self.filename = BackboneModelConfigIO.FILENAME

    def discover(self) -> list:
        return sorted(self.runs_dir.rglob(self.filename))

    def migrate_file(self, path: Path) -> bool:
        payload = FileIO.load_json(path)

        if self.new_key in payload:
            return False

        if self.OLD_NAME_KEY not in payload:
            raise KeyError(f"{path} has neither '{self.new_key}' nor '{self.OLD_NAME_KEY}'; not a recognizable model config")

        migrated = {self.new_key: payload.pop(self.OLD_NAME_KEY), **payload}

        if not self.dry_run:
            FileIO.save_json(migrated, path)

        return True

    def run(self) -> None:
        paths = self.discover()

        migrated = []
        skipped  = []

        for path in paths:
            if self.migrate_file(path):
                migrated.append(path)
                self.logger.info(f"{'would migrate' if self.dry_run else 'migrated'}: {path}")
            else:
                skipped.append(path)

        self.logger.kv_table({
            "Runs dir"    : str(self.runs_dir),
            "Files found" : len(paths),
            "Migrated"    : len(migrated),
            "Already new" : len(skipped),
            "Dry run"     : self.dry_run,
        }, title="Migration summary")
