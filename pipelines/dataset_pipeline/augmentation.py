import numpy as np

from configuration.dataset_config import AugmentationConfig


class SpatialAugmenter:
    def __init__(self, config: AugmentationConfig, logger):
        self.config = config
        self.logger = logger
        self._rng   = np.random.default_rng()

        self.logger.section("[Data Augmentation]")
        self.logger.kv_table(
            {
                "Flip Horizontal" : self.config.p_flip_h,
                "Flip Vertical"   : self.config.p_flip_v,
                "Rotate 90°"      : self.config.p_rot90,
                "Noise"           : f"std={self.config.noise_std} p={self.config.p_noise}",
            },
            title="Augmentation Config",
        )

    def __call__(self, input_tensor: np.ndarray, gt_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        flip_h  = self._rng.random() < self.config.p_flip_h
        flip_v  = self._rng.random() < self.config.p_flip_v
        rotate  = self.config.p_rot90 > 0.0 and self._rng.random() < self.config.p_rot90
        k       = int(self._rng.integers(1, 4)) if rotate else 0
        noise   = self._rng.random() < self.config.p_noise

        sl_h    = slice(None, None, -1) if flip_h else slice(None)
        sl_v    = slice(None, None, -1) if flip_v else slice(None)
        sl      = (Ellipsis, sl_v, sl_h)

        input_view = input_tensor[sl]
        gt_view    = gt_params[sl]

        if k:
            input_view = np.rot90(input_view, k=k, axes=(-2, -1))
            gt_view    = np.rot90(gt_view, k=k, axes=(-2, -1))

        input_tensor = np.ascontiguousarray(input_view)
        gt_params    = np.ascontiguousarray(gt_view)

        if noise:
            input_tensor += self._rng.normal(0.0, self.config.noise_std, input_tensor.shape).astype(input_tensor.dtype)

        return input_tensor, gt_params
