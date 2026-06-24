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
            "title"     : "Train Backbone",
            "category"  : "Training",
            "purpose"   : "Train one supervised backbone end to end, or fan out trials across GPUs: loss-curriculum combinations, warmup-only losses, secondary-track selections, or input-channel ablations.",
            "essentials": ["run_name", "backbone_name", "gpu", "logdir", "paths.dataset_path", "paths.parameters_path"],
        },
        "train_profile_autoencoder": {
            "title"     : "Train Profile Autoencoder",
            "category"  : "Training",
            "purpose"   : "Train the per-pixel profile autoencoder that learns the latent embedding targets consumed by JEPA.",
            "essentials": ["run_name", "ae_model_name", "gpu", "logdir", "paths.dataset_path", "paths.parameters_path"],
        },
        "train_image_autoencoder": {
            "title"     : "Train Image Autoencoder",
            "category"  : "Training",
            "purpose"   : "Train the 2D image autoencoder that learns the latent input embedding consumed as a JEPA front-end.",
            "essentials": ["run_name", "ae_model_name", "gpu", "logdir", "paths.dataset_path", "paths.parameters_path"],
        },
        "train_jepa": {
            "title"     : "Train JEPA",
            "category"  : "Training",
            "purpose"   : "Train the JEPA predictor in latent space. Operates in three modes depending on which autoencoder runs are selected: backbone + profile autoencoder, image autoencoder + backbone, or image autoencoder + backbone + profile autoencoder. Each autoencoder is imported pretrained and either frozen or fine-tuned.",
            "essentials": ["run_name", "backbone_name", "profile_autoencoder_mode", "profile_autoencoder_logdir", "profile_autoencoder_run", "image_autoencoder_mode", "image_autoencoder_logdir", "image_autoencoder_run", "gpu", "logdir", "paths.dataset_path", "paths.parameters_path"],
        },
        "infer": {
            "title"     : "Inference",
            "category"  : "Inference",
            "purpose"   : "Run sliding-window prediction, stitch cubes, and generate reports for one or more trained runs.",
            "essentials": ["logs_dir", "run_filter", "gpus"],
        },
        "infer_profile_autoencoder": {
            "title"     : "Profile AE Inference",
            "category"  : "Inference",
            "purpose"   : "Reconstruct held-out profiles with a trained profile autoencoder and score reconstruction quality (no spatial cube). Dataset paths and splits are read from each run's metadata. Select profile-autoencoder runs only.",
            "essentials": ["logs_dir", "run_filter", "gpus"],
        },
        "benchmark": {
            "title"     : "Benchmark",
            "category"  : "Experiments",
            "purpose"   : "Benchmark inference speed and capacity-matched architecture trade-offs.",
            "essentials": ["run_tag", "gpus", "jepa.profile_autoencoder_mode", "jepa.profile_autoencoder_run", "paths.dataset_path", "paths.parameters_path"],
        },
        "cross_validate": {
            "title"     : "Cross-validate",
            "category"  : "Experiments",
            "purpose"   : "Run K-fold cross-validation for a model across azimuth folds, training and inferring each fold across GPUs.",
            "essentials": ["backbone_name", "run_tag", "gpus", "jepa.profile_autoencoder_mode", "jepa.profile_autoencoder_run", "paths.dataset_path", "paths.parameters_path"],
        },
        "compare_trials": {
            "title"     : "Compare Trials",
            "category"  : "Analysis",
            "purpose"   : "Compare inference results across multiple training runs: metrics leaderboard, side-by-side figures, and optional GIF comparison.",
            "essentials": ["runs_dir", "run_tags"],
        },
        "tune": {
            "title"     : "Tune",
            "category"  : "Experiments",
            "purpose"   : "Run the Optuna hyperparameter search, resumable in chunks.",
            "essentials": ["run_tag", "gpus", "jepa.profile_autoencoder_mode", "jepa.profile_autoencoder_run"],
        },
        "tune_dataloader": {
            "title"     : "Feed Tuner",
            "category"  : "Experiments",
            "purpose"   : "Sweep DataLoader settings (batch size, workers, prefetch, pin-memory) per training mode and recommend the configuration that keeps the GPU fed, ending data starvation.",
            "essentials": ["mode", "gpu", "batch_sizes", "worker_counts", "prefetch_factors", "timed_batches", "paths.dataset_path", "paths.parameters_path"],
        },
    }

    ORDER = [
        "pre_process",
        "extract_params",
        "train_backbone",
        "train_profile_autoencoder",
        "train_image_autoencoder",
        "train_jepa",
        "infer",
        "infer_profile_autoencoder",
        "benchmark",
        "cross_validate",
        "tune",
        "tune_dataloader",
        "compare_trials",
    ]

    GROUPS = {
        "train": {
            "title"    : "Train",
            "category" : "Training",
            "purpose"  : "Train one model end to end. Pick the stage to train: the supervised backbone, the profile autoencoder, the image autoencoder, or the JEPA predictor.",
            "members"  : [
                ("train_backbone",            "Backbone"),
                ("train_profile_autoencoder", "Profile AE"),
                ("train_image_autoencoder",   "Image AE"),
                ("train_jepa",                "JEPA"),
            ],
        },
    }

    def __init__(self, paths: ProjectPaths, resolver: ScriptConfigResolver) -> None:
        self.paths    = paths
        self.resolver = resolver

    def _group_of(self, key: str) -> tuple[str | None, dict | None, str | None]:
        for group_key, group in self.GROUPS.items():
            for member_key, label in group["members"]:
                if member_key == key:
                    return group_key, group, label
        return None, None, None

    def _variants(self, group: dict) -> list[dict]:
        variants = []
        for member_key, label in group["members"]:
            if self.paths.script_entry(member_key)["path"].exists():
                variants.append({"key": member_key, "label": label})
        return variants

    def list_scripts(self) -> list[dict]:
        entries = []
        for key in self.ORDER:
            spec = self.paths.script_entry(key)
            if not spec["path"].exists():
                continue

            meta  = self.META.get(key, {"title": key, "category": "Other", "purpose": ""})
            entry = self.resolver.entry_config(key)

            group_key, group, label = self._group_of(key)

            entries.append({
                "key"            : key,
                "file"           : spec["rel"],
                "title"          : meta["title"],
                "category"       : meta["category"],
                "purpose"        : meta["purpose"],
                "config_class"   : entry["class"] if entry else None,
                "group"          : group_key,
                "variant"        : label,
                "group_title"    : group["title"] if group else None,
                "group_category" : group["category"] if group else None,
                "group_purpose"  : group["purpose"] if group else None,
                "variants"       : self._variants(group) if group else [],
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

        group_key, group, label = self._group_of(key)

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
            "group"        : group_key,
            "group_title"  : group["title"] if group else None,
            "variant"      : label,
            "variants"     : self._variants(group) if group else [],
        }
