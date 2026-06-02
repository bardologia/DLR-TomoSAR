from __future__ import annotations

from dataclasses import dataclass, field
from enum        import Enum
from typing      import Optional

import numpy as np


class NormMethod(Enum):
    MIN_MAX_P999 = "min_max_p999"
    ROBUST_IQR   = "robust_iqr"
    FIXED_DIV_PI = "fixed_div_pi"
    ZSCORE       = "zscore"


@dataclass
class ChannelStrategy:
    norm_method : NormMethod
    apply_log1p : bool = False

    def fit(self, flat: np.ndarray) -> tuple[float, float]:
        if self.norm_method is NormMethod.FIXED_DIV_PI:
            return 0.0, float(np.pi)

        data = (
            np.log1p(np.maximum(flat.astype(np.float64), 0.0))
            if self.apply_log1p
            else flat.astype(np.float64)
        )

        if self.norm_method is NormMethod.MIN_MAX_P999:
            lo = float(np.percentile(data, 0.1))
            hi = float(np.percentile(data, 99.9))
            return lo, max(hi - lo, 1e-8)

        if self.norm_method is NormMethod.ROBUST_IQR:
            med = float(np.percentile(data, 50))
            iqr = float(np.percentile(data, 75)) - float(np.percentile(data, 25))
            return med, max(iqr, 1e-8)

        if self.norm_method is NormMethod.ZSCORE:
            m = float(data.mean())
            s = float(data.std())
            return m, max(s, 1e-8)

        raise ValueError(f"Unknown norm method: {self.norm_method}")

    def as_dict(self) -> dict:
        return {"norm_method": self.norm_method.value, "apply_log1p": self.apply_log1p}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelStrategy":
        return cls(
            norm_method = NormMethod(d.get("norm_method", d.get("strategy", NormMethod.ZSCORE.value))),
            apply_log1p = bool(d.get("apply_log1p", False)),
        )

    @classmethod
    def from_slot(cls, full_key: str) -> "ChannelStrategy":
        return _SLOT_STRATEGIES[full_key]


class Presets:
    MIN_MAX          = ChannelStrategy(NormMethod.MIN_MAX_P999, apply_log1p=False)
    MIN_MAX_LOG1P    = ChannelStrategy(NormMethod.MIN_MAX_P999, apply_log1p=True)
    ROBUST_IQR       = ChannelStrategy(NormMethod.ROBUST_IQR,   apply_log1p=False)
    ROBUST_IQR_LOG1P = ChannelStrategy(NormMethod.ROBUST_IQR,   apply_log1p=True)
    FIXED_DIV_PI     = ChannelStrategy(NormMethod.FIXED_DIV_PI, apply_log1p=False)
    ZSCORE           = ChannelStrategy(NormMethod.ZSCORE,       apply_log1p=False)
    ZSCORE_LOG1P     = ChannelStrategy(NormMethod.ZSCORE,       apply_log1p=True)


_SLOT_STRATEGIES: dict[str, ChannelStrategy] = {
    "pass/mag"        : Presets.MIN_MAX_LOG1P,
    "pass/raw_re_im"  : Presets.MIN_MAX,
    "pass/norm_re_im" : Presets.ROBUST_IQR,
    "pass/phase"      : Presets.FIXED_DIV_PI,

    "ifg/mag"         : Presets.MIN_MAX_LOG1P,
    "ifg/raw_re_im"   : Presets.MIN_MAX,
    "ifg/norm_re_im"  : Presets.ROBUST_IQR,
    "ifg/phase"       : Presets.FIXED_DIV_PI,

    "out/amp"         : Presets.MIN_MAX_LOG1P,
    "out/mu"          : Presets.MIN_MAX,
    "out/sigma"       : Presets.MIN_MAX_LOG1P,

    "dem/elevation"   : Presets.ZSCORE,
}


@dataclass
class ChannelStats:
    loc            : list[float]
    scale          : list[float]
    names          : Optional[list[str]]             = None
    strategies     : Optional[list[ChannelStrategy]] = None
    log1p_channels : list[int]                       = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return len(self.loc)

    def as_dict(self) -> dict:
        entries = []
        for i, (m, s) in enumerate(zip(self.loc, self.scale)):
            entry: dict = {"name": self.names[i], "loc": m, "scale": s}
            if self.strategies and i < len(self.strategies):
                entry.update(self.strategies[i].as_dict())
           
            entries.append(entry)
       
        return {"channels": entries, "log1p_channels": self.log1p_channels}

    @classmethod
    def from_dict(cls, payload: dict) -> "ChannelStats":
        entries    = payload["channels"]
        strategies = [ChannelStrategy.from_dict(e) for e in entries]
      
        return cls(
            loc            = [e.get("loc",   e.get("mean")) for e in entries],
            scale          = [e.get("scale", e.get("std"))  for e in entries],
            names          = [e["name"] for e in entries],
            strategies     = strategies,
            log1p_channels = payload.get("log1p_channels", []),
        )
