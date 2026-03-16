from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .base import (
    BaseInterpreter,
    normalize_to_unit_range,
    overlay_heatmap_on_image,
    save_figure,
    plot_image_grid,
)


class SaliencyMap(BaseInterpreter):

    @torch.enable_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        use_smooth_grad: bool = False,
        num_smooth_samples: int = 25,
        noise_standard_deviation: float = 0.1,
    ) -> np.ndarray:
        prepared_input = self._prepare_input(input_tensor)

        if use_smooth_grad:
            return self._compute_smooth_gradient(
                prepared_input, target_class, num_smooth_samples, noise_standard_deviation,
            )
        return self._compute_vanilla_gradient(prepared_input, target_class)

    def _compute_vanilla_gradient(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        input_with_grad = input_tensor.clone().requires_grad_(True)
        model_output = self.model(input_with_grad)
        score = model_output[0, target_class].sum()

        self.model.zero_grad()
        score.backward()

        saliency = input_with_grad.grad[0].abs().max(dim=0).values
        return self._to_numpy(normalize_to_unit_range(saliency))

    def _compute_smooth_gradient(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        num_samples: int,
        noise_standard_deviation: float,
    ) -> np.ndarray:
        accumulated_gradient = torch.zeros(input_tensor.shape[2:], device=self.device)

        for _ in range(num_samples):
            noisy_input = input_tensor + torch.randn_like(input_tensor) * noise_standard_deviation
            noisy_input.requires_grad_(True)
            model_output = self.model(noisy_input)
            score = model_output[0, target_class].sum()
            self.model.zero_grad()
            score.backward()
            accumulated_gradient += noisy_input.grad[0].abs().max(dim=0).values

        averaged_gradient = accumulated_gradient / num_samples
        return self._to_numpy(normalize_to_unit_range(averaged_gradient))

    def visualise(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        use_smooth_grad: bool = False,
        opacity: float = 0.5,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        saliency = self.generate(input_tensor, target_class, use_smooth_grad=use_smooth_grad)
        input_image = self._to_numpy(self._prepare_input(input_tensor)[0, 0])
        overlay = overlay_heatmap_on_image(input_image, saliency, opacity=opacity, colormap="hot")

        method_name = "SmoothGrad" if use_smooth_grad else "Saliency"
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(input_image, cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(saliency, cmap="hot")
        axes[1].set_title(f"{method_name} (class {target_class})")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()

        if save_path:
            save_figure(figure, save_path)
        return figure

    def compare_methods(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        vanilla_gradient = self.generate(input_tensor, target_class, use_smooth_grad=False)
        smooth_gradient = self.generate(input_tensor, target_class, use_smooth_grad=True)
        return plot_image_grid(
            [vanilla_gradient, smooth_gradient],
            titles=["Vanilla gradient", "SmoothGrad"],
            columns=2,
            main_title=f"Saliency comparison — class {target_class}",
            save_path=save_path,
        )
