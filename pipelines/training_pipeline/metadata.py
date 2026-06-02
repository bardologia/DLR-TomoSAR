from __future__ import annotations

import json
from dataclasses import asdict
from datetime    import datetime
from pathlib     import Path

import torch


from tensorboard.summary.writer.event_file_writer import EventFileWriter as _  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

from configuration.training_config import TrainerConfig
from tools.logger                  import Logger


class TrainingRunMetadata:
    def __init__(self, trainer_config : TrainerConfig, model_name : str, base_logdir : Path, run_name : str | None = None, logger : Logger | None = None) -> None:
        self.trainer_config = trainer_config
        self.model_name     = model_name
        self.base_logdir    = Path(base_logdir)

        timestamp           = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_name       = run_name or f"run_{model_name}_{timestamp}"
       
        self.run_directory       = self.base_logdir / resolved_name
        self.tensorboard_dir     = self.run_directory / "tensorboard"
        self.docs_directory      = self.run_directory / "docs"
        self.logs_directory      = self.run_directory / "logs"
        self.metadata_directory  = self.run_directory / "meta"
        self.checkpoint_dir      = self.run_directory / "checkpoints"

        for directory in (
            self.run_directory, self.tensorboard_dir, self.docs_directory,
            self.logs_directory, self.metadata_directory,
            self.checkpoint_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        trainer_config.io.logdir     = str(self.run_directory)
        trainer_config.io.tb_dir     = str(self.tensorboard_dir)
        trainer_config.io.docs_dir   = str(self.docs_directory)
        trainer_config.io.logs_dir   = str(self.logs_directory)
        trainer_config.io.writer     = self.writer

        if hasattr(trainer_config, "resources"):
            trainer_config.resources.logs_dir = str(self.logs_directory)

        self.logger = logger or Logger(log_dir = str(self.logs_directory), name = f"{model_name}_metadata", level = "INFO",)

        self.logger.section("[Training RunMetadata Initialized]")
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.logger.kv_table({
            "Run Directory" : self.run_directory,
            "Model"         : self.model_name,
            "Backend"       : "PyTorch",
            "Devices"       : f"{devices} -> {[torch.cuda.get_device_name(i) for i in range(devices)]}",
        })

    def save_trainer_config(self) -> Path:
        out_path                     = self.docs_directory / "trainer_config.json"
        serializable                 = asdict(self.trainer_config)
        serializable["io"]["writer"] = None
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4, default=str)
        
        self.logger.info(f"Trainer config saved: {out_path}")
        
        return out_path

    def save_run_summary(self, model_name: str, in_channels: int, out_channels: int, x_axis_length: int, param_match: str = "none") -> Path:
        out_path = self.metadata_directory / "run_summary.json"
        
        payload  = {
            "model_name"    : model_name,
            "in_channels"   : in_channels,
            "out_channels"  : out_channels,
            "x_axis_length" : x_axis_length,
            "run_directory" : str(self.run_directory),
            "framework"     : "pytorch",
            "n_devices"     : torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "param_match"   : param_match,
        }
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        
        self.logger.info(f"Run summary saved: {out_path}")
        
        return out_path

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
