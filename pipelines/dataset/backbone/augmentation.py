from __future__ import annotations

import numpy as np

from configuration.data.dataset_config import AugmentationConfig


class SpatialAugmenter:
    def __init__(self, config: AugmentationConfig, logger, seed: int = 0):
        self.config = config
        self.logger = logger
        self.seed   = int(seed)
        self._rng   = np.random.default_rng(self.seed)

        self.logger.section("[Data Augmentation]")
        self.logger.kv_table(
            {
                "Flip Horizontal" : self.config.p_flip_h,
                "Flip Vertical"   : self.config.p_flip_v,
                "Rotate 90°"      : self.config.p_rot90,
                "Noise"           : f"std={self.config.noise_std} (normalized units) p={self.config.p_noise}",
            },
            title="Augmentation Config",
        )

    def _flip_horizontal(self, input_tensor: np.ndarray, gt_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._rng.random() >= self.config.p_flip_h:
            return input_tensor, gt_params

        sl = (Ellipsis, slice(None), slice(None, None, -1))

        return input_tensor[sl], gt_params[sl]

    def _flip_vertical(self, input_tensor: np.ndarray, gt_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._rng.random() >= self.config.p_flip_v:
            return input_tensor, gt_params

        sl = (Ellipsis, slice(None, None, -1), slice(None))

        return input_tensor[sl], gt_params[sl]

    def _rotate90(self, input_tensor: np.ndarray, gt_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.config.p_rot90 <= 0.0 or self._rng.random() >= self.config.p_rot90:
            return input_tensor, gt_params

        k = int(self._rng.integers(1, 4))

        return np.rot90(input_tensor, k=k, axes=(-2, -1)), np.rot90(gt_params, k=k, axes=(-2, -1))

    def add_noise(self, normalized_input: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self.config.p_noise:
            return normalized_input

        noise = self._rng.normal(0.0, self.config.noise_std, normalized_input.shape).astype(normalized_input.dtype)

        return normalized_input + noise

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, input_tensor: np.ndarray, gt_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        input_tensor, gt_params = self._flip_horizontal(input_tensor, gt_params)
        input_tensor, gt_params = self._flip_vertical(input_tensor,   gt_params)
        input_tensor, gt_params = self._rotate90(input_tensor,        gt_params)

        input_tensor = np.ascontiguousarray(input_tensor)
        gt_params    = np.ascontiguousarray(gt_params)

        return input_tensor, gt_params
