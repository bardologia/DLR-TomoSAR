from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

from configuration.processing_config           import ProcessingConfiguration, TomogramConfiguration
from pipelines.processing_pipeline.artifacts   import ArtifactRegistry
from tools.logger                              import Logger


class MetadataManager:
    def __init__(self, config: ProcessingConfiguration, logger: Logger) -> None:
        self.config   = config
        self.logger   = logger
        self.registry = ArtifactRegistry(config, logger)

        self.logger.section("[MetadataManager Initialization]")
        self.logger.subsection(f"Run Directory : {config.paths.run_directory}")
        self.logger.subsection(f"Tomogram Tag  : {config.tomogram_tag}")
        self.logger.subsection(f"Parameter Tag : {config.parameter_tag}")

    def build_tomogram_metadata(self, variant: str, output_path: Path, stack_identifier: str, cfg: TomogramConfiguration) -> dict[str, str]:
        return {
            f"tomo_{variant}" : str(output_path),
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

    def build_inputs_metadata(self, primary_path: Path, secondaries_path: Path, interferograms_path: Path, primary_shape: Tuple[int, ...], secondaries_shape: Tuple[int, ...], interferograms_shape: Tuple[int, ...]) -> dict[str, str]:
        cfg = self.config.input_configs
        return {
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
        }

    def save_stage_metadata(self, stage_name: str, identifier_tag: str, metadata_entries: dict[str, str]) -> Path:
        self.registry.ensure_directory_structure()
        meta_filename = f"meta_{stage_name}_{identifier_tag}.txt"
        meta_path     = self.config.paths.metadata_directory / meta_filename

        self.logger.section(f"[Saving Metadata] Stage: {stage_name} | Tag: {identifier_tag}")
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            for key, value in metadata_entries.items():
                meta_file.write(f"{key}: {value}\n")
        
        self.logger.subsection(f"Metadata written: {meta_path}")
        return meta_path

    def save_pipeline_configuration(self) -> Path:
        self.registry.ensure_directory_structure()

        run_tag       = self.config.tomogram_tag
        dump_filename = f"config_state_{run_tag}.json"
        dump_path     = self.config.paths.metadata_directory / dump_filename

        config_dict = asdict(self.config)

        self.logger.section("[Saving Configuration State]")
        with open(dump_path, "w", encoding="utf-8") as file:
            json.dump(config_dict, file, indent=4, default=str)
        
        self.logger.subsection(f"Configuration preserved at: {dump_path}")
        return dump_path

    def save_dataset_layout(self) -> Path:
        self.registry.ensure_directory_structure()
        
        layout = {
            "global_crop"   : list(self.config.crop.as_tuple()),
            "dataset_type"  : self.config.dataset_type,
            "tomogram_tag"  : self.config.tomogram_tag,
            "parameter_tag" : self.config.parameter_tag,
            "artifacts"     : self.registry.artifact_filenames(),
        }
        
        out_path = self.config.paths.data_directory / "dataset.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2)
        
        self.logger.section(f"[Dataset Layout Saved] {out_path}")
        return out_path
