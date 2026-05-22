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
     
        if self._rng.random() < self.config.p_flip_h:
            input_tensor = np.flip(input_tensor, axis=-1).copy()
            gt_params    = np.flip(gt_params, axis=-1).copy()
            
        if self._rng.random() < self.config.p_flip_v:
            input_tensor = np.flip(input_tensor, axis=-2).copy()
            gt_params    = np.flip(gt_params, axis=-2).copy()

        if self.config.p_rot90 > 0.0 and self._rng.random() < self.config.p_rot90:
            k = int(self._rng.integers(1, 4))
            input_tensor = np.rot90(input_tensor, k=k, axes=(-2, -1)).copy()
            gt_params    = np.rot90(gt_params, k=k, axes=(-2, -1)).copy()

        if self._rng.random() < self.config.p_noise:
            noise = self._rng.normal(0.0, self.config.noise_std, input_tensor.shape).astype(input_tensor.dtype)
            input_tensor = input_tensor + noise

        return input_tensor, gt_params
