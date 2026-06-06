from __future__ import annotations

from project_paths import ProjectPaths
from script_config_resolver import ScriptConfigResolver


class ScriptCatalog:

    META = {
        "pre_process": {
            "title"   : "Pre-process",
            "category": "Data",
            "purpose" : "Ingest raw F-SAR products, beamform the tomogram, and form interferograms.",
        },
        "extract_params": {
            "title"   : "Extract Parameters",
            "category": "Data",
            "purpose" : "Fit per-pixel Gaussian mixtures to build the supervised parameter targets.",
        },
        "train": {
            "title"   : "Train",
            "category": "Training",
            "purpose" : "Train one model end to end, or fan out loss-curriculum trials (warmup x complete combinations) across GPUs.",
        },
        "infer": {
            "title"   : "Inference",
            "category": "Inference",
            "purpose" : "Run sliding-window prediction, stitch cubes, and generate reports for one or more trained runs.",
        },
        "benchmark": {
            "title"   : "Benchmark",
            "category": "Analysis",
            "purpose" : "Benchmark inference speed and capacity-matched architecture trade-offs.",
        },
        "physics_check": {
            "title"   : "Physics Check",
            "category": "Analysis",
            "purpose" : "Compare physical quantities between Gaussian fits and the Capon tomogram to establish physics loss floors.",
        },
        "tune": {
            "title"   : "Tune",
            "category": "Tuning",
            "purpose" : "Run the Optuna two-phase hyperparameter search.",
        },
    }

    ORDER = [
        "pre_process",
        "extract_params",
        "train",
        "infer",
        "benchmark",
        "physics_check",
        "tune",
    ]

    def __init__(self, paths: ProjectPaths, resolver: ScriptConfigResolver) -> None:
        self.paths    = paths
        self.resolver = resolver

    def list_scripts(self) -> list[dict]:
        entries = []
        for key in self.ORDER:
            path = self.paths.main_dir / f"{key}.py"
            if not path.exists():
                continue

            meta  = self.META.get(key, {"title": key, "category": "Other", "purpose": ""})
            entry = self.resolver.entry_config(key)

            entries.append({
                "key"          : key,
                "file"         : f"main/{key}.py",
                "title"        : meta["title"],
                "category"     : meta["category"],
                "purpose"      : meta["purpose"],
                "config_class" : entry["class"] if entry else None,
            })
        return entries

    def get_script(self, key: str) -> dict | None:
        path = self.paths.main_dir / f"{key}.py"
        if not path.exists():
            return None

        meta   = self.META.get(key, {"title": key, "category": "Other", "purpose": ""})
        source = path.read_text(encoding="utf-8")
        entry  = self.resolver.entry_config(key)

        return {
            "key"          : key,
            "file"         : f"main/{key}.py",
            "title"        : meta["title"],
            "category"     : meta["category"],
            "purpose"      : meta["purpose"],
            "source"       : source,
            "language"     : "python",
            "config_class" : entry["class"] if entry else None,
            "command"      : f"python main/{key}.py",
        }
