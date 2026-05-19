from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ModelWrapper:
    def __init__(
        self,
        model,
        device,
        *,
        params_per_gaussian: int = 3,
        normalizer=None,
    ) -> None:

        self._model               = model
        self._device              = device
        self._params_per_gaussian = params_per_gaussian
        self._normalizer          = normalizer

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self._device)

        with torch.no_grad():
            out = self._model(x_t)

        out = self.denormalize_output(out)
        out = self.apply_constrains(out)

        return out.cpu().numpy()

    def denormalize_output(self, out: torch.Tensor) -> torch.Tensor:
        if self._normalizer is not None:
            out = self._normalizer.denormalize_output(out)
        return out

    def apply_constrains(self, out: torch.Tensor) -> torch.Tensor:
        ppg = self._params_per_gaussian
        for i in range(0, out.shape[1], ppg):
            out[:, i]     = F.softplus(out[:, i])
            if i + 2 < out.shape[1]:
                out[:, i + 2] = torch.abs(out[:, i + 2])
       
        return out
