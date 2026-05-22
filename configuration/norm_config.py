from __future__ import annotations

from dataclasses import dataclass, field
from enum        import Enum
from typing      import Optional

import numpy as np


class NormStrategy(Enum):
    MIN_MAX_P999 = "min_max_p999"
    ROBUST_IQR   = "robust_iqr"
    FIXED_DIV_PI = "fixed_div_pi"
    ZSCORE       = "zscore"


@dataclass
class ChannelTransformStrategy:
    strategy    : NormStrategy
    apply_log1p : bool = False

    def fit(self, flat: np.ndarray) -> tuple[float, float]:
        if self.strategy is NormStrategy.FIXED_DIV_PI:
            return 0.0, float(np.pi)

        data = (
            np.log1p(np.maximum(flat.astype(np.float64), 0.0))
            if self.apply_log1p
            else flat.astype(np.float64)
        )

        if self.strategy is NormStrategy.MIN_MAX_P999:
            lo = float(np.percentile(data, 0.1))
            hi = float(np.percentile(data, 99.9))
            return lo, max(hi - lo, 1e-8)

        if self.strategy is NormStrategy.ROBUST_IQR:
            med = float(np.percentile(data, 50))
            iqr = float(np.percentile(data, 75)) - float(np.percentile(data, 25))
            return med, max(iqr, 1e-8)

        if self.strategy is NormStrategy.ZSCORE:
            m = float(data.mean())
            s = float(data.std())
            return m, max(s, 1e-8)

        raise ValueError(f"Unknown strategy: {self.strategy}")

    def as_dict(self) -> dict:
        return {"strategy": self.strategy.value, "apply_log1p": self.apply_log1p}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelTransformStrategy":
        return cls(
            strategy    = NormStrategy(d.get("strategy", NormStrategy.ZSCORE.value)),
            apply_log1p = bool(d.get("apply_log1p", False)),
        )

    @classmethod
    def from_slot(cls, full_key: str) -> "ChannelTransformStrategy":
        return _SLOT_STRATEGIES[full_key]


class Strat:
    MIN_MAX          = ChannelTransformStrategy(NormStrategy.MIN_MAX_P999, apply_log1p=False)
    MIN_MAX_LOG1P    = ChannelTransformStrategy(NormStrategy.MIN_MAX_P999, apply_log1p=True)
    ROBUST_IQR       = ChannelTransformStrategy(NormStrategy.ROBUST_IQR,   apply_log1p=False)
    ROBUST_IQR_LOG1P = ChannelTransformStrategy(NormStrategy.ROBUST_IQR,   apply_log1p=True)
    FIXED_DIV_PI     = ChannelTransformStrategy(NormStrategy.FIXED_DIV_PI, apply_log1p=False)
    ZSCORE           = ChannelTransformStrategy(NormStrategy.ZSCORE,       apply_log1p=False)
    ZSCORE_LOG1P     = ChannelTransformStrategy(NormStrategy.ZSCORE,       apply_log1p=True)


_SLOT_STRATEGIES: dict[str, ChannelTransformStrategy] = {
    "pass/mag"        : Strat.MIN_MAX_LOG1P,  
    "pass/raw_re_im"  : Strat.MIN_MAX,         
    "pass/norm_re_im" : Strat.ROBUST_IQR,      
    "pass/phase"      : Strat.FIXED_DIV_PI,    

    "ifg/mag"         : Strat.MIN_MAX_LOG1P,
    "ifg/raw_re_im"   : Strat.MIN_MAX,
    "ifg/norm_re_im"  : Strat.ROBUST_IQR,
    "ifg/phase"       : Strat.FIXED_DIV_PI,

    "out/amp"         : Strat.MIN_MAX_LOG1P,   
    "out/mu"          : Strat.MIN_MAX,         
    "out/sigma"       : Strat.MIN_MAX_LOG1P,   
}


@dataclass
class ChannelStats:
    loc            : list[float]
    scale          : list[float]
    names          : Optional[list[str]]                       = None
    strategies     : Optional[list[ChannelTransformStrategy]]  = None
    log1p_channels : list[int]                                 = field(default_factory=list)

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
        strategies = [ChannelTransformStrategy.from_dict(e) for e in entries]
        return cls(
            loc            = [e.get("loc",   e.get("mean")) for e in entries],
            scale          = [e.get("scale", e.get("std"))  for e in entries],
            names          = [e["name"] for e in entries],
            strategies     = strategies,
            log1p_channels = payload.get("log1p_channels", []),
        )
