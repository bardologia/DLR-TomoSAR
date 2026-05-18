from __future__ import annotations

from dataclasses import dataclass, field
from enum        import Enum
from pathlib     import Path
from typing      import Literal, Optional, Sequence, Tuple

import numpy as np

from tools.representation  import Representation
from tools.split_regions   import SplitRegions 


@dataclass
class ChannelStats:
    mean  : list[float]
    std   : list[float]
    names : Optional[list[str]] = None

    @property
    def n_channels(self) -> int:
        return len(self.mean)

    def as_dict(self) -> dict:
        entries = []
        for i, (m, s) in enumerate(zip(self.mean, self.std)):
            entry: dict = {"name": self.names[i] if self.names else f"ch{i}", "mean": m, "std": s}
            entries.append(entry)
        return {"channels": entries}

    @classmethod
    def from_dict(cls, payload: dict) -> "ChannelStats":
        if "channels" in payload:
            entries = payload["channels"]
            return cls(
                mean  = [e["mean"]  for e in entries],
                std   = [e["std"]   for e in entries],
                names = [e["name"]  for e in entries],
            )
        return cls(mean=list(payload["mean"]), std=list(payload["std"]))


class InputNormalizationMode(Enum):
    PER_CHANNEL = "per_channel"
    GROUPED     = "grouped"


class OutputNormalizationMode(Enum):
    DISABLED    = "disabled"
    PER_CHANNEL = "per_channel"
    GROUPED     = "grouped"


@dataclass
class InputConfig:
    use_master                    : bool           = True
    master_representation         : Representation = Representation.MAG_ONLY
    use_slaves                    : bool           = False
    slaves_representation         : Representation = Representation.MAG_ONLY
    use_interferograms            : bool           = True
    interferograms_representation : Representation = Representation.ANGLE_ONLY

    @property
    def master_channels_per_pass(self) -> int:
        return self.master_representation.channels_per_pass if self.use_master else 0

    @property
    def slaves_channels_per_pass(self) -> int:
        return self.slaves_representation.channels_per_pass if self.use_slaves else 0

    @property
    def interferograms_channels_per_pass(self) -> int:
        return self.interferograms_representation.channels_per_pass if self.use_interferograms else 0

    def total_channels(self, n_slaves: int) -> int:
        n = self.master_channels_per_pass
        if n_slaves > 0:
            n += n_slaves * (self.slaves_channels_per_pass + self.interferograms_channels_per_pass)
        return n

    def build_tensor(self, complex_data: np.ndarray) -> np.ndarray:

        n_samples, n_passes, h, w = complex_data.shape
       
        master_data = complex_data[:, :1]
        slave_data  = complex_data[:, 1:] if n_passes > 1 else None

        parts: list[np.ndarray] = []
        if self.use_master:
            parts.append(self.master_representation.convert(master_data))

        if slave_data is not None and slave_data.shape[1] > 0:
            if self.use_slaves:
                parts.append(self.slaves_representation.convert(slave_data))
            if self.use_interferograms:
                interferograms = slave_data * np.conj(master_data)
                parts.append(self.interferograms_representation.convert(interferograms))

        if not parts:
            raise ValueError("InputConfig produced no channels (no slaves and master disabled).")
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    def as_dict(self) -> dict:
        return {
            "use_master"                    : self.use_master,
            "master_representation"         : self.master_representation.value,
            "use_slaves"                    : self.use_slaves,
            "slaves_representation"         : self.slaves_representation.value,
            "use_interferograms"            : self.use_interferograms,
            "interferograms_representation" : self.interferograms_representation.value,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InputConfig":
        if "master" in payload:
            return cls(
                use_master                    = bool(payload["master"]["use"]),
                master_representation         = Representation(payload["master"]["representation"]),
                use_slaves                    = bool(payload["slaves"]["use"]),
                slaves_representation         = Representation(payload["slaves"]["representation"]),
                use_interferograms            = bool(payload["interferograms"]["use"]),
                interferograms_representation = Representation(payload["interferograms"]["representation"]),
            )
        return cls(
            use_master                    = bool(payload["use_master"]),
            master_representation         = Representation(payload["master_representation"]),
            use_slaves                    = bool(payload["use_slaves"]),
            slaves_representation         = Representation(payload["slaves_representation"]),
            use_interferograms            = bool(payload["use_interferograms"]),
            interferograms_representation = Representation(payload["interferograms_representation"]),
        )


@dataclass
class PatchConfiguration:
    size                   : Tuple[int, int] = (64, 64)
    stride                 : int             = 32
    use_reflective_padding : bool            = True


@dataclass
class DatasetCreationConfiguration:
    preprocessing_run_directory : Path
    split_regions               : SplitRegions
    parameters_path             : Optional[Path]          = None
    patch                       : PatchConfiguration      = field(default_factory=PatchConfiguration)
    input_config                : InputConfig             = field(default_factory=InputConfig)      
    batch_size                  : int                     = 8
    num_workers                 : int                     = 16
    shuffle_train               : bool                    = True
    pin_memory                  : bool                    = True
    input_normalization_mode    : InputNormalizationMode  = InputNormalizationMode.PER_CHANNEL
    output_normalization_mode   : OutputNormalizationMode = OutputNormalizationMode.DISABLED
    x_axis                      : Optional[np.ndarray]    = field(default=None, repr=False)
    n_gaussians                 : int                     = 1
