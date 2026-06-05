from __future__ import annotations

import argparse
import ast
import json
from dataclasses import fields, is_dataclass
from pathlib import Path

from .detach import Detacher

_SUPPORTED_TYPES = (bool, int, float, str, Path, list, tuple, dict)


class ConfigCli:
    def __init__(self, config, description: str | None = None) -> None:
        self.config    = config
        self.overrides : dict = {}
        self.parser    = argparse.ArgumentParser(description=description, add_help=False)

        self.parser.add_argument("--help-config", action="store_true", dest="_help_config")
        self.parser.add_argument("--detach", "--nohup", action="store_true", dest="_detach")

        for path, value in self._leaves(config):
            if value is not None and not isinstance(value, _SUPPORTED_TYPES):
                continue

            options = [f"--{path}"]
            dashed  = f"--{path.replace('_', '-')}"
            if dashed not in options:
                options.append(dashed)

            self.parser.add_argument(*options, dest=path, type=str, default=None)

    def apply(self, argv: list[str] | None = None):
        args, _ = self.parser.parse_known_args(argv)

        if getattr(args, "_help_config", False):
            self._print_config_help()
            raise SystemExit(0)

        if getattr(args, "_detach", False):
            Detacher().ensure()

        for path, current in list(self._leaves(self.config)):
            raw = getattr(args, path, None)
            if raw is None:
                continue

            value = self._coerce(raw, current)
            self.set_path(self.config, path, value)
            self.overrides[path] = value

        return self.config

    @classmethod
    def _leaves(cls, config, prefix: str = ""):
        for f in fields(config):
            value = getattr(config, f.name)
            path  = f"{prefix}{f.name}"

            if is_dataclass(value):
                yield from cls._leaves(value, prefix=f"{path}.")
            else:
                yield path, value

    def _coerce(self, raw: str, current):
        if isinstance(current, bool):
            lowered = raw.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"Cannot parse boolean from '{raw}'")

        if isinstance(current, int):
            return int(raw)
        if isinstance(current, float):
            return float(raw)
        if isinstance(current, Path):
            return Path(raw)
        if isinstance(current, (list, dict)):
            return ast.literal_eval(raw)

        if isinstance(current, tuple):
            parsed = ast.literal_eval(raw)
            return tuple(parsed) if isinstance(parsed, (list, tuple)) else (parsed,)

        if current is None:
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return raw

        return raw

    def _print_config_help(self) -> None:
        rows = [(path, type(value).__name__ if value is not None else "any", repr(value)) for path, value in self._leaves(self.config)]
        width = max(len(path) for path, _, _ in rows)

        print(f"Configuration overrides for {type(self.config).__name__} (pass as --<path> <value>):")
        for path, type_name, default in rows:
            print(f"  --{path:<{width}}  {type_name:<6}  default: {default}")
        print("Execution flags:")
        print("  --detach (alias --nohup)  relaunch detached from the terminal, output to logs/<script>_<stamp>.out")

    @staticmethod
    def set_path(config, path: str, value) -> None:
        parts  = path.split(".")
        target = config
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)

    @classmethod
    def apply_overrides(cls, config, overrides: dict):
        for path, value in overrides.items():
            cls.set_path(config, path, value)
        return config

    @classmethod
    def to_mapping(cls, config) -> dict:
        mapping = {}
        for path, value in cls._leaves(config):
            if isinstance(value, Path):
                mapping[path] = str(value)
            elif isinstance(value, tuple):
                mapping[path] = list(value)
            elif value is None or isinstance(value, _SUPPORTED_TYPES):
                mapping[path] = value
        return mapping

    @classmethod
    def save_resolved(cls, config, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cls.to_mapping(config), f, indent=2)
        return path

    @classmethod
    def load_resolved(cls, config, path: Path):
        if not Path(path).exists():
            return config

        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        for leaf, current in list(cls._leaves(config)):
            if leaf not in mapping:
                continue

            value = mapping[leaf]
            if isinstance(current, Path) and isinstance(value, str):
                value = Path(value)
            elif isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)

            cls.set_path(config, leaf, value)

        return config

    @staticmethod
    def to_argv(overrides: dict) -> list[str]:
        argv = []
        for path, value in overrides.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, tuple):
                rendered = str(list(value))
            else:
                rendered = str(value)
            argv += [f"--{path}", rendered]
        return argv
