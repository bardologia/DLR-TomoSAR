from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base import (
    BaseInterpreter,
    normalize_to_unit_range,
    overlay_heatmap_on_image,
    plot_image_grid,
    save_figure,
)


class GradCAM(BaseInterpreter):
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__(model, device)
        self.target_layer = target_layer

        self._stored_activations: torch.Tensor | None = None
        self._stored_gradients: torch.Tensor | None = None

        self._forward_hook = target_layer.register_forward_hook(self._on_forward)
        self._backward_hook = target_layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, module, layer_input, layer_output):
        self._stored_activations = layer_output.detach()

    def _on_backward(self, module, gradient_input, gradient_output):
        self._stored_gradients = gradient_output[0].detach()

    @torch.enable_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        prepared_input = self._prepare_input(input_tensor).requires_grad_(True)
        model_output = self.model(prepared_input)

        class_activation = model_output[0, target_class]
        if spatial_mask is not None:
            spatial_mask = spatial_mask.to(class_activation.device).float()
            class_activation = class_activation * spatial_mask

        score = class_activation.sum()
        self.model.zero_grad()
        score.backward()

        channel_weights = self._stored_gradients[0].mean(dim=(1, 2))
        class_activation_map = (channel_weights[:, None, None] * self._stored_activations[0]).sum(dim=0)
        class_activation_map = F.relu(class_activation_map)

        upsampled_map = F.interpolate(
            class_activation_map.unsqueeze(0).unsqueeze(0),
            size=prepared_input.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        return self._to_numpy(normalize_to_unit_range(upsampled_map))

    def visualise(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        spatial_mask: Optional[torch.Tensor] = None,
        opacity: float = 0.5,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        heatmap = self.generate(input_tensor, target_class, spatial_mask)
        input_image = self._to_numpy(self._prepare_input(input_tensor)[0, 0])
        overlay = overlay_heatmap_on_image(input_image, heatmap, opacity=opacity)

        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(input_image, cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title(f"Grad-CAM (class {target_class})")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()

        if save_path:
            save_figure(figure, save_path)
        return figure

    def compare_classes(
        self,
        input_tensor: torch.Tensor,
        class_indices: list[int],
        class_names: Optional[list[str]] = None,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        heatmaps = [self.generate(input_tensor, class_index) for class_index in class_indices]
        display_names = class_names or [f"Class {class_index}" for class_index in class_indices]
        return plot_image_grid(
            heatmaps,
            titles=display_names,
            main_title="Grad-CAM comparison",
            save_path=save_path,
        )

    def remove_hooks(self) -> None:
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self):
        self.remove_hooks()
