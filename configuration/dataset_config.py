from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Literal, Optional, Sequence, Tuple

import numpy as np

from configuration.norm_config    import ChannelStrategy
from tools.representation         import Representation
from tools.split_regions          import SplitRegions


@dataclass
class InputConfig:
    use_primary                   : bool           = True
    primary_representation        : Representation = Representation.MAG_ONLY
 
    use_secondaries               : bool           = False
    secondaries_representation    : Representation = Representation.MAG_ONLY
 
    use_interferograms            : bool           = True
    interferograms_representation : Representation = Representation.ANGLE_ONLY

    use_dem                       : bool           = False

    @property
    def primary_channels_per_pass(self) -> int:
        return self.primary_representation.channels_per_pass if self.use_primary else 0

    @property
    def secondaries_channels_per_pass(self) -> int:
        return self.secondaries_representation.channels_per_pass if self.use_secondaries else 0

    @property
    def interferograms_channels_per_pass(self) -> int:
        return self.interferograms_representation.channels_per_pass if self.use_interferograms else 0

    def total_channels(self, n_secondaries: int, n_interferograms: int) -> int:
        n  = self.primary_channels_per_pass
        n += n_secondaries    * self.secondaries_channels_per_pass
        n += n_interferograms * self.interferograms_channels_per_pass
        if self.use_dem:
            n += 1
        return n

    def channel_group_keys(self, n_secondaries: int, n_interferograms: int) -> list[str]:
        keys: list[str] = []

        if self.use_primary:
            slot_kinds = self.primary_representation.slot_kinds
            cpp        = len(slot_kinds)
            keys.extend(f"pass/{slot_kinds[i % cpp]}" for i in range(1 * cpp))

        if self.use_secondaries:
            slot_kinds = self.secondaries_representation.slot_kinds
            cpp        = len(slot_kinds)
            keys.extend(f"pass/{slot_kinds[i % cpp]}" for i in range(n_secondaries * cpp))

        if self.use_interferograms:
            slot_kinds = self.interferograms_representation.slot_kinds
            cpp        = len(slot_kinds)
            keys.extend(f"ifg/{slot_kinds[i % cpp]}" for i in range(n_interferograms * cpp))

        if self.use_dem:
            keys.append("dem/elevation")

        return keys

    def as_dict(self) -> dict:
        return {
            "use_primary"                   : self.use_primary,
            "primary_representation"        : self.primary_representation.value,
            "use_secondaries"               : self.use_secondaries,
            "secondaries_representation"    : self.secondaries_representation.value,
            "use_interferograms"            : self.use_interferograms,
            "interferograms_representation" : self.interferograms_representation.value,
            "use_dem"                       : self.use_dem,
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
                use_dem                       = bool(payload.get("dem", {}).get("use", False)),
            )
        else:
            return cls(
                use_primary                   = bool(payload.get("use_primary", True)),
                primary_representation        = Representation(payload.get("primary_representation", Representation.MAG_ONLY.value)),
                use_secondaries               = bool(payload.get("use_secondaries", False)),
                secondaries_representation    = Representation(payload.get("secondaries_representation", Representation.MAG_ONLY.value)),
                use_interferograms            = bool(payload.get("use_interferograms", True)),
                interferograms_representation = Representation(payload.get("interferograms_representation", Representation.ANGLE_ONLY.value)),
                use_dem                       = bool(payload.get("use_dem", False)),
            )

      
@dataclass
class OutputConfig:
    use_amplitude      : bool                              = True
    use_mu             : bool                              = True
    use_sigma          : bool                              = True
    
    output_strategies  : dict[str, ChannelStrategy] = field(default_factory=lambda: {
        "out/amp"   : ChannelStrategy.from_slot("out/amp"),
        "out/mu"    : ChannelStrategy.from_slot("out/mu"),
        "out/sigma" : ChannelStrategy.from_slot("out/sigma"),
    })

    def strategy_for(self, role_key: str) -> ChannelStrategy:
        return self.output_strategies.get(role_key, ChannelStrategy.from_slot(role_key))

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
            "use_amplitude"     : self.use_amplitude,
            "use_mu"            : self.use_mu,
            "use_sigma"         : self.use_sigma,
            "output_strategies" : {
                k: v.as_dict() for k, v in self.output_strategies.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OutputConfig":
        raw_strats = payload.get("output_strategies", {})
        strategies = {
            k: ChannelStrategy.from_dict(v)
            for k, v in raw_strats.items()
        } if raw_strats else {
            "out/amp"   : ChannelStrategy.from_slot("out/amp"),
            "out/mu"    : ChannelStrategy.from_slot("out/mu"),
            "out/sigma" : ChannelStrategy.from_slot("out/sigma"),
        }

        if "amplitude" in payload:
            return cls(
                use_amplitude     = bool(payload["amplitude"]["use"]),
                use_mu            = bool(payload["mu"]["use"]),
                use_sigma         = bool(payload["sigma"]["use"]),
                output_strategies = strategies,
            )
        return cls(
            use_amplitude     = bool(payload.get("use_amplitude", True)),
            use_mu            = bool(payload.get("use_mu", True)),
            use_sigma         = bool(payload.get("use_sigma", True)),
            output_strategies = strategies,
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
    noise_std       : float               = 0.01
    p_noise         : float               = 0.25


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
    x_axis                      : Optional[np.ndarray]    = field(default=None, repr=False)
    n_gaussians                 : int                     = 1

