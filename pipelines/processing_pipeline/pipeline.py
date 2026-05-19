from __future__ import annotations

import gc
from pathlib import Path
from typing  import Literal, Tuple

from configuration.processing_config                 import ProcessingConfiguration
from pipelines.pre_processing_pipeline.interferogram import InterferogramBuilder
from pipelines.pre_processing_pipeline.metadata      import MetadataManager
from pipelines.pre_processing_pipeline.tomogram      import TomogramProcessor
from tools.logger                                    import Logger


class ProcessingPipeline:
    def __init__(self, config: ProcessingConfiguration, logger: Logger | None = None) -> None:
        self.config = config
        run_dir     = Path(config.paths.run_directory)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir     = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger                = logger or Logger(log_dir = str(log_dir), name = "preprocessing", level = "INFO")
        self.metadata_manager      = MetadataManager  (config,    logger=self.logger)
        self.tomogram_processor    = TomogramProcessor (config,   logger=self.logger)
        self.interferogram_builder = InterferogramBuilder(config, logger=self.logger)

        self.logger.section("[Pre-Processing Pipeline Initialized]")
        self.logger.subsection(f"Full Stack ID    : {config.full_stack_identifier}")
        self.logger.subsection(f"Reduced Stack ID : {config.reduced_stack_identifier}")
        self.logger.subsection(f"Separate X/Y     : {config.has_split_configs}")

    def _clear_memory(self) -> None:
        gc.collect()

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

    def _stage_tomogram(
        self,
        variant          : Literal["full", "reduced"],
    ) -> Tuple[Path, Path]:
        tomogram_key     = f"tomogram_{variant}"
        dem_key          = f"dem_{variant}"
        stack_identifier = self.config.full_stack_identifier    if variant == "full" else self.config.reduced_stack_identifier
        tomogram_config  = self.config.output_config            if variant == "full" else self.config.input_configs
        identifier_tag   = self.config.parameter_tag            if variant == "full" else self.config.tomogram_tag

        tomogram_path = self.metadata_manager.artifact_path(tomogram_key)
        dem_path      = self.metadata_manager.artifact_path(dem_key)

        self.logger.subsection(f"[Active] Generating {variant} tomogram...")
        self.tomogram_processor.run(
            tomogram_path    = tomogram_path,
            dem_path         = dem_path,
            stack_identifier = stack_identifier,
            tomogram_config  = tomogram_config,
        )

        self.metadata_manager.save_stage_metadata(
            stage_name       = tomogram_key,
            identifier_tag   = identifier_tag,
            metadata_entries = self._tomogram_metadata(tomogram_path, stack_identifier, tomogram_config),
        )

        self._clear_memory()
        return tomogram_path, dem_path

    def _stage_inputs(self) -> Tuple[Path, Path, Path]:
        primary_path        = self.metadata_manager.artifact_path("primary_reduced")
        secondaries_path    = self.metadata_manager.artifact_path("secondaries_reduced")
        interferograms_path = self.metadata_manager.artifact_path("interferograms_reduced")

        self.logger.subsection("[Active] Building interferometric stack...")
        primary_shape, secondaries_shape, interferograms_shape = self.interferogram_builder.run(
            crop_tuple          = self.config.crop.as_tuple(),
            primary_path        = primary_path,
            secondaries_path    = secondaries_path,
            interferograms_path = interferograms_path,
        )

        cfg = self.config.input_configs
        self.metadata_manager.save_stage_metadata(
            stage_name       = "inputs",
            identifier_tag   = self.config.tomogram_tag,
            metadata_entries = {
                "primary_path"         : str(primary_path),
                "secondaries_path"     : str(secondaries_path),
                "interferograms_path"  : str(interferograms_path),
                "primary_shape"        : f"[{', '.join(str(v) for v in primary_shape)}]",
                "secondaries_shape"    : f"[{', '.join(str(v) for v in secondaries_shape)}]",
                "interferograms_shape" : f"[{', '.join(str(v) for v in interferograms_shape)}]",
                "crop"                 : f"[{', '.join(str(v) for v in self.config.crop.as_tuple())}]",
                "FuSARproject"         : cfg.fusar_project_path,
                "id"                   : self.config.reduced_stack_identifier,
                "basedir"              : cfg.base_directory,
                "polarisation"         : cfg.polarisation,
                "select"               : cfg.track_selection,
                "data_type"            : self.config.dataset_type,
            },
        )
       
        self._clear_memory()
        return primary_path, secondaries_path, interferograms_path

    def _tomogram_metadata(self, output_path: Path, stack_identifier: str, cfg) -> dict[str, str]:
        return {
            "tomofull"     : str(output_path),
            "crop"         : f"[{', '.join(str(v) for v in self.config.crop.as_tuple())}]",
            "FuSARproject" : cfg.fusar_project_path,
            "id"           : stack_identifier,
            "basedir"      : cfg.base_directory,
            "polarisation" : cfg.polarisation,
            "select"       : cfg.track_selection,
            "range"        : f"[{', '.join(str(v) for v in cfg.height_range)}]",
            "filter"       : cfg.filter_method,
            "method"       : cfg.beamforming_method,
            "win"          : f"[{', '.join(str(v) for v in cfg.filter_arguments.get('win', []))}]",
        }


