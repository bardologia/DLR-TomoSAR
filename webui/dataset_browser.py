from __future__ import annotations

from pathlib import Path

from webui.web_logger import WebLogger


class DatasetBrowser:

    DATA_MARKER     = "data"
    PARAMS_DIR      = "params"
    PARAM_SUFFIX    = ".npy"
    MAX_DEPTH       = 3
    CHECKPOINT_NAME = "best_model.pt"
    INFERENCE_DIR   = "inference"

    def __init__(self, logger: WebLogger) -> None:
        self.logger = logger

    def datasets(self, raw_base: str) -> dict:
        base = self._directory(raw_base)
        if base is None:
            return {"ok": False, "error": f"not a directory: {raw_base}"}

        entries = []
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            has_data   = (entry / self.DATA_MARKER).is_dir()
            params_dir = entry / self.PARAMS_DIR
            has_params = params_dir.is_dir() and self._has_param_files(params_dir)

            entries.append({
                "name"       : entry.name,
                "path"       : str(entry),
                "is_dataset" : has_data,
                "has_params" : has_params,
            })

        self.logger.info(f"datasets: listed {len(entries)} under {base}")
        return {"ok": True, "base": str(base), "datasets": entries}

    def runs(self, raw_base: str) -> dict:
        base = self._directory(raw_base)
        if base is None:
            return {"ok": False, "error": f"not a directory: {raw_base}"}

        entries = []
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue

            entries.append({
                "name"           : entry.name,
                "path"           : str(entry),
                "has_checkpoint" : (entry / self.CHECKPOINT_NAME).is_file(),
                "has_inference"  : self._has_inference(entry),
            })

        self.logger.info(f"runs: listed {len(entries)} under {base}")
        return {"ok": True, "base": str(base), "runs": entries}

    def params(self, raw_dataset: str) -> dict:
        dataset = self._directory(raw_dataset)
        if dataset is None:
            return {"ok": False, "error": f"not a directory: {raw_dataset}"}

        params_dir = dataset / self.PARAMS_DIR
        if not params_dir.is_dir():
            return {"ok": True, "dataset": str(dataset), "params_root": str(params_dir), "files": []}

        files = []
        for path in sorted(self._param_files(params_dir)):
            files.append({
                "name" : str(path.relative_to(params_dir)),
                "path" : str(path),
            })

        self.logger.info(f"params: found {len(files)} under {params_dir}")
        return {"ok": True, "dataset": str(dataset), "params_root": str(params_dir), "files": files}

    def _has_inference(self, run_dir: Path) -> bool:
        inference_dir = run_dir / self.INFERENCE_DIR
        if not inference_dir.is_dir():
            return False

        for entry in inference_dir.iterdir():
            if entry.is_dir():
                return True
        return False

    def _directory(self, raw: str) -> Path | None:
        if not raw or not raw.strip():
            return None

        path = Path(raw).expanduser()
        if not path.is_absolute():
            return None

        path = path.resolve()
        return path if path.is_dir() else None

    def _has_param_files(self, params_dir: Path) -> bool:
        for _ in self._param_files(params_dir):
            return True
        return False

    def _param_files(self, params_dir: Path):
        yield from self._walk(params_dir, 0)

    def _walk(self, directory: Path, depth: int):
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return

        for entry in entries:
            if entry.is_file() and entry.suffix.lower() == self.PARAM_SUFFIX:
                yield entry
            elif entry.is_dir() and not entry.name.startswith(".") and depth < self.MAX_DEPTH:
                yield from self._walk(entry, depth + 1)
