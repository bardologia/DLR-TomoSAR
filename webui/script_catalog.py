from __future__ import annotations

from project_paths          import ProjectPaths
from script_config_resolver import ScriptConfigResolver


class ScriptCatalog:

    META = {
        "pre_process": {
            "title"     : "Pre-process",
            "category"  : "Data",
            "purpose"   : "Ingest raw F-SAR products, beamform the tomogram, and form interferograms.",
            "essentials": ["dataset_name", "effort", "azimuth_start", "azimuth_end", "range_start", "range_end", "polarisation", "base_directory"],
        },
        "extract_params": {
            "title"     : "Extract Parameters",
            "category"  : "Data",
            "purpose"   : "Fit per-pixel Gaussian mixtures to build the supervised parameter targets.",
            "essentials": ["dataset_base_path", "dataset_filter", "gpu_device_ids", "output_prefix", "fit_k_max", "fit_sigma_init_divisor"],
        },
        "train_backbone": {
            "title"     : "Train",
            "category"  : "Training",
            "purpose"   : "Train one model end to end, or fan out trials across GPUs: loss-curriculum combinations, warmup-only losses, or secondary-track selections.",
            "essentials": ["run_name", "model_name", "gpu", "logdir", "paths.dataset_path", "paths.parameters_path"],
        },
        "infer": {
            "title"     : "Inference",
            "category"  : "Inference",
            "purpose"   : "Run sliding-window prediction, stitch cubes, and generate reports for one or more trained runs.",
            "essentials": ["logs_dir", "run_filter", "gpu"],
        },
        "benchmark": {
            "title"     : "Benchmark",
            "category"  : "Analysis",
            "purpose"   : "Benchmark inference speed and capacity-matched architecture trade-offs.",
            "essentials": ["run_tag", "gpus"],
        },
        "physics_check": {
            "title"     : "Physics Check",
            "category"  : "Analysis",
            "purpose"   : "Compare physical quantities between Gaussian fits and the Capon tomogram to establish physics loss floors.",
            "essentials": ["dataset_path", "device"],
        },
        "tune": {
            "title"     : "Tune",
            "category"  : "Tuning",
            "purpose"   : "Run the Optuna hyperparameter search, resumable in chunks.",
            "essentials": ["run_tag", "gpus"],
        },
    }

    ORDER = [
        "pre_process",
        "extract_params",
        "train_backbone",
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
            spec = self.paths.script_entry(key)
            if not spec["path"].exists():
                continue

            meta  = self.META.get(key, {"title": key, "category": "Other", "purpose": ""})
            entry = self.resolver.entry_config(key)

            entries.append({
                "key"          : key,
                "file"         : spec["rel"],
                "title"        : meta["title"],
                "category"     : meta["category"],
                "purpose"      : meta["purpose"],
                "config_class" : entry["class"] if entry else None,
            })
        return entries

    def get_script(self, key: str) -> dict | None:
        spec = self.paths.script_entry(key)
        if not spec["path"].exists():
            return None

        meta    = self.META.get(key, {"title": key, "category": "Other", "purpose": ""})
        source  = spec["path"].read_text(encoding="utf-8")
        entry   = self.resolver.entry_config(key)
        command = " ".join(["python", spec["rel"], *spec["args"]])

        return {
            "key"          : key,
            "file"         : spec["rel"],
            "title"        : meta["title"],
            "category"     : meta["category"],
            "purpose"      : meta["purpose"],
            "essentials"   : meta.get("essentials", []),
            "source"       : source,
            "language"     : "python",
            "config_class" : entry["class"] if entry else None,
            "command"      : command,
            "preferred"    : self.paths.preferred_interpreter(self.paths.discover_interpreters(), key),
        }
