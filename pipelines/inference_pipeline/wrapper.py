from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class ModelWrapper:
    def __init__(
        self,
        apply_fn,
        params,
        batch_stats,
        *,
        params_per_gaussian: int = 3,
        normalizer=None,
    ) -> None:

        self._apply_fn            = apply_fn
        self._params              = params
        self._batch_stats         = batch_stats
        self._params_per_gaussian = params_per_gaussian
        self._normalizer          = normalizer

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_jax = jnp.asarray(x, dtype=jnp.float32)

        variables: dict = {"params": self._params}
        if self._batch_stats is not None:
            variables["batch_stats"] = self._batch_stats

        out_jax = self._apply_fn(variables, x_jax, training=False)

        if self._normalizer is not None:
            out_jax = self._normalizer.denormalize_output_jax(out_jax)

        ppg = self._params_per_gaussian
        for i in range(0, out_jax.shape[1], ppg):
            out_jax = out_jax.at[:, i].set(jax.nn.softplus(out_jax[:, i]))      # amplitude → strictly positive
            if i + 2 < out_jax.shape[1]:
                out_jax = out_jax.at[:, i + 2].set(jnp.abs(out_jax[:, i + 2])) # sigma     → non-negative

        return np.asarray(out_jax)
