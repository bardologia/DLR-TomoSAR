from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


class SigmaScan:
    @staticmethod
    def per_pixel_loss(sigmas, height_axis, profile, amps, mus):
        safe_s2 = 2.0 * jnp.maximum(sigmas, 1e-6) ** 2
        diff    = height_axis[None, :] - mus[:, None]
        expon   = jnp.clip(-(diff ** 2) / safe_s2[:, None], -100.0, 0.0)
        pred    = (amps[:, None] * jnp.exp(expon)).sum(axis=0)
        mse     = jnp.mean((pred - profile) ** 2)

        return mse

    @staticmethod
    def adam_scan(
        batched_vg  ,
        sigmas_init  : jnp.ndarray,
        height_axis  : jnp.ndarray,
        profiles     : jnp.ndarray,
        amps         : jnp.ndarray,
        mus          : jnp.ndarray,
        sigma_lower  : jnp.ndarray,
        sigma_upper  : jnp.ndarray,
        n_steps      : int,
        lr           : float,
        b1           : float,
        b2           : float,
    ) -> jnp.ndarray:
        b1_ = jnp.float32(b1)
        b2_ = jnp.float32(b2)
        eps = jnp.float32(1e-8)
        lr_ = jnp.float32(lr)
        s   = jnp.clip(sigmas_init.astype(jnp.float32), sigma_lower, sigma_upper)
        m   = jnp.zeros_like(s)
        v   = jnp.zeros_like(s)

        def _step(carry, t):
            s_, m_, v_ = carry
            _, g       = batched_vg(s_, height_axis, profiles, amps, mus)
            m_ = b1_ * m_ + (1.0 - b1_) * g
            v_ = b2_ * v_ + (1.0 - b2_) * g * g
            tf = t.astype(jnp.float32) + 1.0
            s_ = s_ - lr_ * (m_ / (1.0 - b1_ ** tf)) / (jnp.sqrt(v_ / (1.0 - b2_ ** tf)) + eps)
            s_ = jnp.clip(s_, sigma_lower, sigma_upper)
            return (s_, m_, v_), None

        (s_final, _, _), _ = jax.lax.scan(_step, (s, m, v), jnp.arange(n_steps))

        return s_final


class SigmaAdamKernel:
    def __init__(self) -> None:
        batched_vg = jax.vmap(jax.value_and_grad(SigmaScan.per_pixel_loss), in_axes=(0, None, 0, 0, 0))
        self._run  = self._build(batched_vg)

    @staticmethod
    def _build(batched_vg):
        @partial(jax.jit, static_argnames=("n_steps", "lr", "b1", "b2"))
        def _run(
            sigmas_init : jnp.ndarray,
            height_axis : jnp.ndarray,
            profiles    : jnp.ndarray,
            amps        : jnp.ndarray,
            mus         : jnp.ndarray,
            sigma_lower : jnp.ndarray,
            sigma_upper : jnp.ndarray,
            n_steps     : int   = 2000,
            lr          : float = 1e-2,
            b1          : float = 0.9,
            b2          : float = 0.999,
        ) -> jnp.ndarray:
            return SigmaScan.adam_scan(batched_vg, sigmas_init, height_axis, profiles, amps, mus, sigma_lower, sigma_upper, n_steps, lr, b1, b2)
        return _run

    def __call__(
        self,
        sigmas_init, height_axis, profiles, amps, mus,
        sigma_lower, sigma_upper,
        n_steps=2000, lr=1e-2, b1=0.9, b2=0.999,
    ):
        return self._run(
            sigmas_init, height_axis, profiles, amps, mus,
            sigma_lower, sigma_upper,
            n_steps, lr, b1, b2,
        )


class PmapSigmaAdamKernel:
    def __init__(self, devices: list) -> None:
        self._n_devices = len(devices)
        batched_vg      = jax.vmap(jax.value_and_grad(SigmaScan.per_pixel_loss), in_axes=(0, None, 0, 0, 0))
        self._run       = self._build(batched_vg, devices)

    @staticmethod
    def _build(batched_vg, devices):
        def _run_on_device(
            sigmas_init : jnp.ndarray,
            height_axis : jnp.ndarray,
            profiles    : jnp.ndarray,
            amps        : jnp.ndarray,
            mus         : jnp.ndarray,
            sigma_lower : jnp.ndarray,
            sigma_upper : jnp.ndarray,
            n_steps     : int   = 2000,
            lr          : float = 1e-2,
            b1          : float = 0.9,
            b2          : float = 0.999,
        ) -> jnp.ndarray:
            return SigmaScan.adam_scan(batched_vg, sigmas_init, height_axis, profiles, amps, mus, sigma_lower, sigma_upper, n_steps, lr, b1, b2)

        return jax.pmap(
            _run_on_device,
            in_axes                    = (0, None, 0, 0, 0, None, None),
            static_broadcasted_argnums = (7, 8, 9, 10),
            devices                    = devices,
        )

    def __call__(
        self,
        sigmas_init, height_axis, profiles, amps, mus,
        sigma_lower, sigma_upper,
        n_steps=2000, lr=1e-2, b1=0.9, b2=0.999,
    ):
        n, K = sigmas_init.shape
        H   = profiles.shape[1]
        D   = self._n_devices
        pad = (-n) % D

        if pad > 0:
            z_K         = jnp.zeros((pad, K), dtype=jnp.float32)
            z_H         = jnp.zeros((pad, H), dtype=jnp.float32)
            sigmas_init = jnp.concatenate([sigmas_init, z_K], axis=0)
            profiles    = jnp.concatenate([profiles,    z_H], axis=0)
            amps        = jnp.concatenate([amps,        z_K], axis=0)
            mus         = jnp.concatenate([mus,         z_K], axis=0)

        n_pad = n + pad
        shard = n_pad // D

        out_s = self._run(
            sigmas_init.reshape(D, shard, K),
            height_axis,
            profiles   .reshape(D, shard, H),
            amps       .reshape(D, shard, K),
            mus        .reshape(D, shard, K),
            sigma_lower,
            sigma_upper,
            n_steps, lr, b1, b2,
        )

        return out_s.reshape(n_pad, K)[:n]
