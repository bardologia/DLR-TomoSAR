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
    DISABLED    = "disabled"


class OutputNormalizationMode(Enum):
    DISABLED    = "disabled"
    PER_CHANNEL = "per_channel"
    GROUPED     = "grouped"


@dataclass
class InputConfig:
    use_primary                   : bool           = True
    primary_representation        : Representation = Representation.MAG_ONLY
    use_secondaries               : bool           = False
    secondaries_representation    : Representation = Representation.MAG_ONLY
    use_interferograms            : bool           = True
    interferograms_representation : Representation = Representation.ANGLE_ONLY

    @property
    def primary_channels_per_pass(self) -> int:
        return self.primary_representation.channels_per_pass if self.use_primary else 0

    @property
    def secondaries_channels_per_pass(self) -> int:
        return self.secondaries_representation.channels_per_pass if self.use_secondaries else 0

    @property
    def interferograms_channels_per_pass(self) -> int:
        return self.interferograms_representation.channels_per_pass if self.use_interferograms else 0

    def total_channels(self, n_secondaries: int) -> int:
        n = self.primary_channels_per_pass
        if n_secondaries > 0:
            n += n_secondaries * (self.secondaries_channels_per_pass + self.interferograms_channels_per_pass)
      
        return n

    def as_dict(self) -> dict:
        return {
            "use_primary"                   : self.use_primary,
            "primary_representation"        : self.primary_representation.value,
            "use_secondaries"               : self.use_secondaries,
            "secondaries_representation"    : self.secondaries_representation.value,
            "use_interferograms"            : self.use_interferograms,
            "interferograms_representation" : self.interferograms_representation.value,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InputConfig":
        if "primary" in payload:
            return cls(
                use_primary                   = bool(payload["primary"]["use"]),
                primary_representation        = Representation(payload["primary"]["representation"]),
                use_secondaries               = bool(payload["secondaries"]["use"]),
                secondaries_representation    = Representation(payload["secondaries"]["representation"]),
                use_interferograms            = bool(payload["interferograms"]["use"]),
                interferograms_representation = Representation(payload["interferograms"]["representation"]),
            )
        else:
            return cls(
                use_primary                   = bool(payload.get("use_primary", True)),
                primary_representation        = Representation(payload.get("primary_representation", Representation.MAG_ONLY.value)),
                use_secondaries               = bool(payload.get("use_secondaries", False)),
                secondaries_representation    = Representation(payload.get("secondaries_representation", Representation.MAG_ONLY.value)),
                use_interferograms            = bool(payload.get("use_interferograms", True)),
                interferograms_representation = Representation(payload.get("interferograms_representation", Representation.ANGLE_ONLY.value)),
            )

      
@dataclass
class OutputConfig:
    use_amplitude : bool = True
    use_mu        : bool = True
    use_sigma     : bool = True

    @property
    def role_names(self) -> list[str]:
        names: list[str] = []
        if self.use_amplitude:
            names.append("a")
        if self.use_mu:
            names.append("mu")
        if self.use_sigma:
            names.append("sig")
        return names

    @property
    def params_per_gaussian(self) -> int:
        return len(self.role_names)

    def selected_indices(self, n_gaussians: int) -> list[int]:
        role_to_idx = {"a": 0, "mu": 1, "sig": 2}
        local        = [role_to_idx[name] for name in self.role_names]

        indices: list[int] = []
        for g in range(n_gaussians):
            base = g * 3
            for i in local:
                indices.append(base + i)
     
        return indices

    def total_channels(self, n_gaussians: int) -> int:
        return n_gaussians * self.params_per_gaussian

    def as_dict(self) -> dict:
        return {
            "use_amplitude" : self.use_amplitude,
            "use_mu"        : self.use_mu,
            "use_sigma"     : self.use_sigma,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OutputConfig":
        if "amplitude" in payload:
            return cls(
                use_amplitude = bool(payload["amplitude"]["use"]),
                use_mu        = bool(payload["mu"]["use"]),
                use_sigma     = bool(payload["sigma"]["use"]),
            )
        else:
            return cls(
                use_amplitude = bool(payload.get("use_amplitude", True)),
                use_mu        = bool(payload.get("use_mu", True)),
                use_sigma     = bool(payload.get("use_sigma", True)),
            )

   
@dataclass
class PatchConfiguration:
    size                   : Tuple[int, int] = (64, 64)
    stride                 : int             = 32
    use_reflective_padding : bool            = True


@dataclass
class AugmentationConfig:
    p_flip_h        : float               = 0.5
    p_flip_v        : float               = 0.5
    p_rot90         : float               = 0.0
    amp_scale_range : Tuple[float, float] = (0.8, 1.2)
    p_amp_scale     : float               = 0.5
    noise_std       : float               = 0.05
    p_noise         : float               = 0.5


@dataclass
class DatasetConfiguration:
    preprocessing_run_directory : Path
    split_regions               : SplitRegions
    parameters_path             : Optional[Path]          = None
    patch                       : PatchConfiguration      = field(default_factory=PatchConfiguration)
    input_config                : InputConfig             = field(default_factory=InputConfig)
    output_config               : OutputConfig            = field(default_factory=OutputConfig)
    augmentation                : AugmentationConfig      = field(default_factory=AugmentationConfig)
    batch_size                  : int                     = 8
    num_workers                 : int                     = 16
    shuffle_train               : bool                    = True
    pin_memory                  : bool                    = True
    input_normalization_mode    : InputNormalizationMode  = InputNormalizationMode.PER_CHANNEL
    output_normalization_mode   : OutputNormalizationMode = OutputNormalizationMode.DISABLED
    x_axis                      : Optional[np.ndarray]    = field(default=None, repr=False)
    n_gaussians                 : int                     = 1

