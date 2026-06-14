from __future__ import annotations

import sys
from pathlib import Path


class ProjectPaths:

    PRIORITY        = ["conda:dlr-cu12", "conda:Dune", "conda:nazaria"]
    SCRIPT_PRIORITY = {
        "generate_tomogram"        : ["conda:stetools"],
        "generate_interferograms"  : ["conda:stetools"],
    }

    ENTRY_OVERRIDES = {
        "train_backbone": {
            "file"          : "train",
            "args"          : ["backbone"],
            "config_module" : "configuration.training.backbone_config",
            "config_class"  : "BackboneEntryConfig",
        },
        "train_autoencoder": {
            "file"          : "train",
            "args"          : ["autoencoder"],
            "config_module" : "configuration.training.autoencoder_config",
            "config_class"  : "ProfileAeEntryConfig",
        },
        "train_jepa": {
            "file"          : "train",
            "args"          : ["jepa"],
            "config_module" : "configuration.training.jepa_config",
            "config_class"  : "JepaEntryConfig",
        },
    }

    def __init__(self) -> None:
        self.webui_root  = Path(__file__).resolve().parent
        self.repo_root   = self.webui_root.parent
        self.main_dir    = self.repo_root / "main"
        self.scripts_dir = self.repo_root / "scripts"
        self.config_dir  = self.repo_root / "configuration"
        self.static_dir  = self.webui_root / "static"

    def script_entry(self, key: str) -> dict:
        override  = self.ENTRY_OVERRIDES.get(key, {})
        file_stem = override.get("file", key)

        return {
            "path"          : self.main_dir / f"{file_stem}.py",
            "rel"           : f"main/{file_stem}.py",
            "args"          : list(override.get("args", [])),
            "config_module" : override.get("config_module"),
            "config_class"  : override.get("config_class"),
        }

    def discover_interpreters(self) -> list[dict]:
        found   = []
        seen    = set()
        current = str(Path(sys.executable))

        home  = Path.home()
        bases = [home / "miniconda3", home / "anaconda3", home / ".conda"]
        for base in bases:
            base_py = base / "bin" / "python"
            if base_py.exists() and str(base_py) not in seen:
                found.append({"label": "conda:base", "path": str(base_py), "current": str(base_py) == current})
                seen.add(str(base_py))

            envs_dir = base / "envs"
            if envs_dir.is_dir():
                for env in sorted(envs_dir.iterdir()):
                    env_py = env / "bin" / "python"
                    if env_py.exists() and str(env_py) not in seen:
                        found.append({"label": f"conda:{env.name}", "path": str(env_py), "current": str(env_py) == current})
                        seen.add(str(env_py))

        if current not in seen and Path(current).exists():
            found.insert(0, {"label": "current", "path": current, "current": True})

        return found

    def preferred_interpreter(self, interpreters: list[dict], script_key: str = "") -> str:
        priority = self.SCRIPT_PRIORITY.get(script_key, []) + self.PRIORITY
        for wanted in priority:
            for item in interpreters:
                if item["label"] == wanted:
                    return item["path"]
        return interpreters[0]["path"] if interpreters else sys.executable
