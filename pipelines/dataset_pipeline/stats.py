from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib     import Path
from typing      import Optional

from configuration.norm_config import ChannelStats
from tools.logger              import Logger


@dataclass
class Stats:
    input_stats  : Optional[ChannelStats]  = None
    output_stats : Optional[ChannelStats]  = None

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out_path  = directory / "normalization_stats.json"

        payload = {
            "input_stats"  : self.input_stats.as_dict()  if self.input_stats  else None,
            "output_stats" : self.output_stats.as_dict() if self.output_stats else None,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        return out_path

    @classmethod
    def load(cls, directory: Path, logger: Logger) -> "Stats":
        path = Path(directory) / "normalization_stats.json"
        if not path.exists():
            raise FileNotFoundError(f"Normalization stats not found at '{path}'.")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        input_stats  = ChannelStats.from_dict(payload["input_stats"])
        output_stats = ChannelStats.from_dict(payload["output_stats"])

        logger.section(f"[Normalization stats loaded]")
        logger.kv_table({
            "Stats path":      path,
            "Input channels":  input_stats.n_channels,
            "Output channels": output_stats.n_channels,
        })

        return cls(input_stats = input_stats, output_stats = output_stats)

    @classmethod
    def merge(cls, input_only: "Stats", output_only: "Stats") -> "Stats":
        return cls(
            input_stats  = input_only.input_stats,
            output_stats = output_only.output_stats,
        )
