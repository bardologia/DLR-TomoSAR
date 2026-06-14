from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib     import Path


class FileIO:
    @staticmethod
    def ensure_dir(path: Path) -> Path:
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)

    @staticmethod
    def ensure_dirs(*paths: Path) -> None:
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_json(payload: dict, path: Path, indent: int = 4, atomic: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        target = path.with_name(path.name + ".tmp") if atomic else path
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, default=str)

        if atomic:
            os.replace(target, path)

        return path

    @staticmethod
    def load_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_text_metadata(entries: dict, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for key, value in entries.items():
                f.write(f"{key}: {value}\n")

        return path


class ModelConfigIO:
    FILENAME      = "model_config.json"
    EXCLUDED      = {"shape_logger_types"}

    @staticmethod
    def _normalize(name: str) -> str:
        return name.lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def save(cls, model_config, model_name: str, meta_directory: Path) -> Path:
        from dataclasses import fields

        serializable = {f.name: getattr(model_config, f.name) for f in fields(model_config) if f.name not in cls.EXCLUDED}

        payload = {
            "model_name" : model_name,
            "config_type": type(model_config).__name__,
            "config"     : serializable,
        }

        return FileIO.save_json(payload, Path(meta_directory) / cls.FILENAME)

    @classmethod
    def load(cls, meta_directory: Path):
        from models import CONFIG_REGISTRY

        path = Path(meta_directory) / cls.FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"No {cls.FILENAME} under {meta_directory}; the model architecture was not persisted at training time and the checkpoint cannot be reconstructed faithfully. Retrain to regenerate it.")

        payload    = FileIO.load_json(path)
        model_name = cls._normalize(str(payload["model_name"]))

        if model_name not in CONFIG_REGISTRY:
            raise ValueError(f"Persisted model_name '{model_name}' is not a known architecture: {list(CONFIG_REGISTRY.keys())}")

        config = CONFIG_REGISTRY[model_name]()

        for key, value in payload["config"].items():
            if not hasattr(config, key):
                raise ValueError(f"Persisted field '{key}' is not an attribute of {type(config).__name__}; the architecture definition changed since this checkpoint was trained")

            current = getattr(config, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)

            setattr(config, key, value)

        return config, payload["model_name"]


class AutoencoderConfigIO:
    FILENAME = "autoencoder_config.json"

    @classmethod
    def save(cls, config, meta_directory: Path) -> Path:
        return FileIO.save_json(asdict(config), Path(meta_directory) / cls.FILENAME)

    @classmethod
    def load(cls, meta_directory: Path):
        from configuration.models_config import AutoencoderConfig

        return AutoencoderConfig(**FileIO.load_json(Path(meta_directory) / cls.FILENAME))

    @classmethod
    def exists(cls, meta_directory: Path) -> bool:
        return (Path(meta_directory) / cls.FILENAME).is_file()
