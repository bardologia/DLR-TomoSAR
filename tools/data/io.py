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


class ConfigIO:
    FILENAME     : str
    NAME_KEY     : str
    MISSING_NOUN : str
    UNKNOWN_NOUN : str

    @staticmethod
    def _normalize(name: str) -> str:
        return name.lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _registry(cls) -> dict:
        raise NotImplementedError

    @classmethod
    def _serialize(cls, config) -> dict:
        raise NotImplementedError

    @classmethod
    def exists(cls, meta_directory: Path) -> bool:
        return (Path(meta_directory) / cls.FILENAME).is_file()

    @classmethod
    def save(cls, config, model_name: str, meta_directory: Path) -> Path:
        payload = {
            cls.NAME_KEY  : model_name,
            "config_type" : type(config).__name__,
            "config"      : cls._serialize(config),
        }

        return FileIO.save_json(payload, Path(meta_directory) / cls.FILENAME)

    @classmethod
    def load(cls, meta_directory: Path):
        path = Path(meta_directory) / cls.FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"No {cls.FILENAME} under {meta_directory}; the {cls.MISSING_NOUN} architecture was not persisted at training time and the checkpoint cannot be reconstructed faithfully. Retrain to regenerate it.")

        payload  = FileIO.load_json(path)
        raw_name = payload[cls.NAME_KEY]
        name     = cls._normalize(str(raw_name))

        registry = cls._registry()
        if name not in registry:
            raise ValueError(f"Persisted {cls.NAME_KEY} '{name}' is not a known {cls.UNKNOWN_NOUN}: {list(registry.keys())}")

        config = registry[name]()

        for key, value in payload["config"].items():
            if not hasattr(config, key):
                raise ValueError(f"Persisted field '{key}' is not an attribute of {type(config).__name__}; the architecture definition changed since this checkpoint was trained")

            current = getattr(config, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)

            setattr(config, key, value)

        return config, raw_name


class ModelConfigIO(ConfigIO):
    FILENAME     = "model_config.json"
    NAME_KEY     = "model_name"
    MISSING_NOUN = "model"
    UNKNOWN_NOUN = "architecture"
    EXCLUDED     = {"shape_logger_types"}

    @classmethod
    def _registry(cls) -> dict:
        from models import CONFIG_REGISTRY
        return CONFIG_REGISTRY

    @classmethod
    def _serialize(cls, config) -> dict:
        from dataclasses import fields
        return {f.name: getattr(config, f.name) for f in fields(config) if f.name not in cls.EXCLUDED}


class AutoencoderConfigIO(ConfigIO):
    FILENAME     = "autoencoder_config.json"
    NAME_KEY     = "ae_model_name"
    MISSING_NOUN = "autoencoder"
    UNKNOWN_NOUN = "autoencoder"

    @classmethod
    def _registry(cls) -> dict:
        from models.autoencoder import AE_CONFIG_REGISTRY
        return AE_CONFIG_REGISTRY

    @classmethod
    def _serialize(cls, config) -> dict:
        return asdict(config)


class ImageAutoencoderConfigIO(ConfigIO):
    FILENAME     = "image_autoencoder_config.json"
    NAME_KEY     = "image_ae_model_name"
    MISSING_NOUN = "image autoencoder"
    UNKNOWN_NOUN = "image autoencoder"

    @classmethod
    def _registry(cls) -> dict:
        from models.image_autoencoder import IMAGE_AE_CONFIG_REGISTRY
        return IMAGE_AE_CONFIG_REGISTRY

    @classmethod
    def _serialize(cls, config) -> dict:
        return asdict(config)
