from __future__ import annotations

import gc
from pathlib import Path

from configuration.preprocessing_config import PreProcessingConfiguration
from pipelines.pre_processing_pipeline.interferogram import InterferogramBuilder
from pipelines.pre_processing_pipeline.metadata      import MetadataManager
from pipelines.pre_processing_pipeline.tomogram      import TomogramProcessor
from tools.logger                                    import Logger


class PreProcessingPipeline:
    def __init__(self, config: PreProcessingConfiguration, logger: Logger | None = None) -> None:
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

        full_tomo_path  = self._stage_full_tomogram()
        input_tomo_path = self._stage_input_tomogram()
        inputs_path     = self._stage_inputs()

        self.metadata_manager.save_dataset_layout()

        self.logger.section("[Pre-Processing Execution Completed]")
        return {
            "full_tomogram"  : full_tomo_path,
            "input_tomogram" : input_tomo_path,
            "inputs"         : inputs_path,
            "run_directory"  : self.config.paths.run_directory,
        }

    def _stage_full_tomogram(self) -> Path:
        output_path = self.metadata_manager.artifact_path("full_tomogram")
        self.logger.subsection("[Active] Generating full tomogram...")
        
        self.tomogram_processor.run(
            output_path      = output_path,
            stack_identifier = self.config.full_stack_identifier,
            tomogram_config  = self.config.output_config,
        )

        cfg = self.config.output_config
        self.metadata_manager.save_stage_metadata(
            stage_name       = "tomofull_full",
            identifier_tag   = self.config.parameter_tag,
            metadata_entries = self._tomogram_metadata(output_path, self.config.full_stack_identifier, cfg),
        )
        
        self._clear_memory()
        return output_path

    def _stage_input_tomogram(self) -> Path:
        output_path = self.metadata_manager.artifact_path("input_tomogram")
      
        self.logger.subsection("[Active] Generating input tomogram...")
        self.tomogram_processor.run(
            output_path      = output_path,
            stack_identifier = self.config.reduced_stack_identifier,
            tomogram_config  = self.config.input_configs,
        )

        cfg = self.config.input_configs
        self.metadata_manager.save_stage_metadata(
            stage_name       = "tomofull_input",
            identifier_tag   = self.config.tomogram_tag,
            metadata_entries = self._tomogram_metadata(output_path, self.config.reduced_stack_identifier, cfg),
        )
       
        self._clear_memory()
        return output_path

    def _stage_inputs(self) -> Path:
        output_path = self.metadata_manager.artifact_path("inputs")

        self.logger.subsection("[Active] Building interferometric stack...")
        saved_shape = self.interferogram_builder.run(
            crop_tuple  = self.config.crop.as_tuple(),
            output_path = output_path,
        )

        cfg = self.config.input_configs
        self.metadata_manager.save_stage_metadata(
            stage_name       = "inputs",
            identifier_tag   = self.config.tomogram_tag,
            
            metadata_entries = {
                "tomofull"     : str(output_path),
                "crop"         : f"[{', '.join(str(v) for v in self.config.crop.as_tuple())}]",
                "saved_shape"  : f"[{', '.join(str(v) for v in saved_shape)}]",
                "FuSARproject" : cfg.fusar_project_path,
                "id"           : self.config.reduced_stack_identifier,
                "basedir"      : cfg.base_directory,
                "polarisation" : cfg.polarisation,
                "select"       : cfg.track_selection,
                "data_type"    : self.config.dataset_type,
            },
        )
       
        self._clear_memory()
        return output_path

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


