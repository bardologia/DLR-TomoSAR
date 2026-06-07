from __future__ import annotations

import gc
from pathlib import Path
from typing  import Literal, NamedTuple, Tuple

from configuration.processing_config             import ProcessingConfiguration, TomogramConfiguration
from pipelines.processing_pipeline.artifacts     import ArtifactRegistry, MetadataManager
from pipelines.processing_pipeline.interferogram import InterferogramBuilder
from pipelines.processing_pipeline.tomogram      import TomogramProcessor
from tools.logger                                import Logger


class TomogramVariant(NamedTuple):
    stack_identifier : str
    tomogram_config  : TomogramConfiguration
    identifier_tag   : str


class ProcessingPipeline:
    def __init__(self, config: ProcessingConfiguration, logger: Logger | None = None) -> None:
        self.config = config
        run_dir     = Path(config.paths.run_directory)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir     = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger                = logger or Logger(log_dir = str(log_dir), name = "preprocessing", level = "INFO")
        self.artifact_registry     = ArtifactRegistry (config,   logger=self.logger)
        self.metadata_manager      = MetadataManager  (config,    logger=self.logger)
        self.tomogram_processor    = TomogramProcessor (config,   logger=self.logger)
        self.interferogram_builder = InterferogramBuilder(config, logger=self.logger)

        self.logger.section("[Pre-Processing Pipeline Initialized]")
        self.logger.subsection(f"Full Stack ID    : {config.full_stack_identifier}")
        self.logger.subsection(f"Reduced Stack ID : {config.reduced_stack_identifier}")
        self.logger.subsection(f"Separate X/Y     : {config.has_split_configs}")

    def _resolve_variant(self, variant: Literal["full", "reduced"]) -> TomogramVariant:
        if variant == "full":
            return TomogramVariant(self.config.full_stack_identifier, self.config.output_config, self.config.parameter_tag)

        return TomogramVariant(self.config.reduced_stack_identifier, self.config.input_configs, self.config.tomogram_tag)

    def _stage_tomogram(self, variant : Literal["full", "reduced"]) -> Tuple[Path, Path]:
        tomogram_key  = f"tomogram_{variant}"
        dem_key       = f"dem_{variant}"

        resolved      = self._resolve_variant(variant)

        tomogram_path = self.artifact_registry.artifact_path(tomogram_key)
        dem_path      = self.artifact_registry.artifact_path(dem_key)

        self.logger.subsection(f"[Active] Generating {variant} tomogram")
        self.tomogram_processor.run(
            tomogram_path    = tomogram_path,
            dem_path         = dem_path,
            stack_identifier = resolved.stack_identifier,
            tomogram_config  = resolved.tomogram_config,
        )

        self.metadata_manager.save_stage_metadata(
            stage_name       = tomogram_key,
            identifier_tag   = resolved.identifier_tag,
            metadata_entries = self.metadata_manager.build_tomogram_metadata(variant, tomogram_path, resolved.stack_identifier, resolved.tomogram_config),
        )

        gc.collect()

        return tomogram_path, dem_path

    def _stage_inputs(self) -> Tuple[Path, Path, Path]:
        primary_path        = self.artifact_registry.artifact_path("primary_reduced")
        secondaries_path    = self.artifact_registry.artifact_path("secondaries_reduced")
        interferograms_path = self.artifact_registry.artifact_path("interferograms_reduced")

        self.logger.subsection("[Active] Building interferometric stack")
        primary_shape, secondaries_shape, interferograms_shape = self.interferogram_builder.run(
            crop_tuple          = self.config.crop.as_tuple(),
            primary_path        = primary_path,
            secondaries_path    = secondaries_path,
            interferograms_path = interferograms_path,
        )

        self.metadata_manager.save_stage_metadata(
            stage_name       = "inputs",
            identifier_tag   = self.config.tomogram_tag,
            metadata_entries = self.metadata_manager.build_inputs_metadata(primary_path, secondaries_path, interferograms_path, primary_shape, secondaries_shape, interferograms_shape),
        )

        if self.interferogram_builder.track_baselines is not None:
            self.metadata_manager.save_baselines(self.interferogram_builder.track_baselines)

        if self.interferogram_builder.track_profiles is not None:
            self.metadata_manager.save_track_profiles(self.interferogram_builder.track_profiles)

        gc.collect()

        return primary_path, secondaries_path, interferograms_path

    def run(self) -> dict[str, Path]:
        self.logger.section("[Pre-Processing Pipeline Execution]")

        self.metadata_manager.save_pipeline_configuration()

        full_tomo_path,    full_dem_path    = self._stage_tomogram("full")
        reduced_tomo_path, reduced_dem_path = self._stage_tomogram("reduced")
        
        primary_path, secondaries_path, interferograms_path = self._stage_inputs()

        self.metadata_manager.save_dataset_layout()

        self.logger.section("[Pre-Processing Execution Completed]")
       
        return {
            "tomogram_full"          : full_tomo_path,
            "tomogram_reduced"       : reduced_tomo_path,
            "dem_full"               : full_dem_path,
            "dem_reduced"            : reduced_dem_path,
            "primary_reduced"        : primary_path,
            "secondaries_reduced"    : secondaries_path,
            "interferograms_reduced" : interferograms_path,
            "run_directory"          : self.config.paths.run_directory,
        }


