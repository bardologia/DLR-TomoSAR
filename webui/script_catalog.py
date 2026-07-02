from __future__ import annotations

from project_paths          import ProjectPaths
from script_config_resolver import ScriptConfigResolver


class ScriptCatalog:

    META = {
        "pre_process": {
            "title"     : "Pre-process",
            "category"  : "Data",
            "purpose"   : "Ingest raw F-SAR products, beamform the tomogram, and form interferograms.",
        },
        "extract_params": {
            "title"     : "Extract Parameters",
            "category"  : "Data",
            "purpose"   : "Fit per-pixel Gaussian mixtures to build the supervised parameter targets. Sweeps every permutation of the selected datasets, K values, lambda values, and fit modes.",
        },
        "train_backbone": {
            "title"     : "Train Backbone",
            "category"  : "Training",
            "purpose"   : "Train one supervised backbone end to end, or fan out trials across GPUs: loss-curriculum combinations, warmup-only losses, secondary-track selections, or input-channel ablations.",
        },
        "train_profile_autoencoder": {
            "title"     : "Train Profile Autoencoder",
            "category"  : "Training",
            "purpose"   : "Train the per-pixel profile autoencoder that learns the latent embedding targets consumed by JEPA.",
        },
        "train_image_autoencoder": {
            "title"     : "Train Image Autoencoder",
            "category"  : "Training",
            "purpose"   : "Train the 2D image autoencoder that learns the latent input embedding consumed as a JEPA front-end.",
        },
        "train_jepa": {
            "title"     : "Train JEPA",
            "category"  : "Training",
            "purpose"   : "Train the JEPA predictor in latent space. Operates in three modes depending on which autoencoder runs are selected: backbone + profile autoencoder, image autoencoder + backbone, or image autoencoder + backbone + profile autoencoder. Each autoencoder is imported pretrained and either frozen or fine-tuned.",
        },
        "infer_backbone": {
            "title"     : "Infer Backbone",
            "category"  : "Inference",
            "purpose"   : "Backbone and JEPA inference: sliding-window prediction, stitched cubes, and reports. Sweeps every run root and runs only backbone/JEPA runs.",
        },
        "infer_profile_autoencoder": {
            "title"     : "Infer Profile AE",
            "category"  : "Inference",
            "purpose"   : "Profile-autoencoder inference: reconstruction scoring. Sweeps every run root and runs only standalone profile-autoencoder runs.",
        },
        "infer_image_autoencoder": {
            "title"     : "Infer Image AE",
            "category"  : "Inference",
            "purpose"   : "Image-autoencoder inference: reconstruction scoring. Sweeps every run root and runs only standalone image-autoencoder runs.",
        },
        "benchmark": {
            "title"     : "Benchmark",
            "category"  : "Experiments",
            "purpose"   : "Benchmark capacity-matched architecture trade-offs, sweeping every permutation of architecture and selected loss component (one architecture + one loss component per run).",
        },
        "cross_validate": {
            "title"     : "Cross-validate",
            "category"  : "Experiments",
            "purpose"   : "Run K-fold cross-validation for a model across azimuth folds, training and inferring each fold across GPUs.",
        },
        "tune": {
            "title"     : "Tune",
            "category"  : "Experiments",
            "purpose"   : "Run the Optuna hyperparameter search, resumable in chunks.",
        },
        "tune_dataloader": {
            "title"     : "Feed Tuner",
            "category"  : "Experiments",
            "purpose"   : "Sweep DataLoader settings (batch size, workers, prefetch, pin-memory) per training mode and recommend the configuration that keeps the GPU fed, ending data starvation.",
        },
        "analyze_preprocessing": {
            "title"     : "Analyze Preprocessing",
            "category"  : "Analysis",
            "purpose"   : "Render the stack-overview plots (SLC amplitudes, flattened interferograms, DEM) for one or more preprocessing trials, decoupled from the tomogram/interferogram generation step.",
        },
        "analyze_param_extraction": {
            "title"     : "Analyze Param Extraction",
            "category"  : "Analysis",
            "purpose"   : "Recompute the Gaussian-fit metrics, summary, and diagnostic plots for one or more parameter-extraction trials, decoupled from the GPU fitting step.",
        },
        "compare_trials": {
            "title"     : "Compare Trials",
            "category"  : "Analysis",
            "purpose"   : "Compare inference results across multiple training runs: metrics leaderboard, side-by-side figures, and optional GIF comparison.",
        },
        "compare_preprocessing_trials": {
            "title"     : "Compare Preprocessing",
            "category"  : "Analysis",
            "purpose"   : "Compare preprocessing trials that differ by multilook window size. Surfaces the bias-variance trade-off per window (contrast, residual speckle, spurious peaks, azimuth correlation length) as descriptive tables and plots, without forcing a single winner.",
        },
        "compare_param_extraction_trials": {
            "title"     : "Compare Param Extraction",
            "category"  : "Analysis",
            "purpose"   : "Compare Gaussian-fit parameter-extraction trials grouped by number of Gaussians K. Ranks within each K family on complexity-penalised BIC, variance explained, spatial coherence, and selection decisiveness, and exposes slot-collapse diagnostics. The K families are treated as separate deliverables.",
        },
        "xray_weights": {
            "title"     : "X-Ray Weights",
            "category"  : "Analysis",
            "purpose"   : "Scan a runs directory, select one or more checkpoints, and diagnose each: dead weights, near-uniform layers, rank collapse, dead neurons, exploded or non-finite values, normalisation-scale collapse, and initialisation anomalies. Writes a console report, a markdown report with per-tensor plots, and a JSON of all metrics inside each run directory.",
        },
    }

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
        "infer": {
            "title"    : "Infer",
            "category" : "Inference",
            "purpose"  : "Run inference end to end. Pick the stage to infer: the supervised backbone (and JEPA), the profile autoencoder, or the image autoencoder.",
            "members"  : [
                ("infer_backbone",            "Backbone"),
                ("infer_profile_autoencoder", "Profile AE"),
                ("infer_image_autoencoder",   "Image AE"),
            ],
        },
        "analyze": {
            "title"    : "Analyze",
            "category" : "Analysis",
            "purpose"  : "Re-render the diagnostic artifacts for a family of trials without re-running the heavy generation step. Pick the stage to analyze: preprocessing stack overviews or Gaussian-fit parameter extraction.",
            "members"  : [
                ("analyze_preprocessing",     "Preprocessing"),
                ("analyze_param_extraction",  "Param Extraction"),
            ],
        },
        "compare": {
            "title"    : "Compare",
            "category" : "Analysis",
            "purpose"  : "Compare a family of trials side by side. Pick the stage to compare: preprocessing windows, Gaussian-fit parameter extraction, or inference results across training runs.",
            "members"  : [
                ("compare_preprocessing_trials",    "Preprocessing"),
                ("compare_param_extraction_trials", "Param Extraction"),
                ("compare_trials",                  "Inference Trials"),
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
        for key in self.META:
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
