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
    save_figure,
    plot_image_grid,
)


class OcclusionSensitivity(BaseInterpreter):

    @torch.no_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        patch_size: int = 16,
        stride: int = 8,
        occlusion_value: float = 0.0,
    ) -> np.ndarray:
        prepared_input = self._prepare_input(input_tensor)
        _, num_channels, height, width = prepared_input.shape

        baseline_output = self.model(prepared_input)
        baseline_probability = F.softmax(baseline_output, dim=1)[0, target_class]
        baseline_score = baseline_probability.mean().item()

        sensitivity_map = torch.zeros(height, width, device=self.device)
        overlap_count = torch.zeros(height, width, device=self.device)

        for top in range(0, height, stride):
            for left in range(0, width, stride):
                bottom = min(top + patch_size, height)
                right = min(left + patch_size, width)

                occluded_input = prepared_input.clone()
                occluded_input[:, :, top:bottom, left:right] = occlusion_value

                occluded_output = self.model(occluded_input)
                occluded_probability = F.softmax(occluded_output, dim=1)[0, target_class]
                confidence_drop = baseline_score - occluded_probability.mean().item()

                sensitivity_map[top:bottom, left:right] += confidence_drop
                overlap_count[top:bottom, left:right] += 1.0

        overlap_count = overlap_count.clamp(min=1.0)
        sensitivity_map /= overlap_count

        return self._to_numpy(normalize_to_unit_range(sensitivity_map))

    def visualise(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        patch_size: int = 16,
        stride: int = 8,
        occlusion_value: float = 0.0,
        opacity: float = 0.5,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        sensitivity = self.generate(input_tensor, target_class, patch_size, stride, occlusion_value)
        input_image = self._to_numpy(self._prepare_input(input_tensor)[0, 0])
        overlay = overlay_heatmap_on_image(input_image, sensitivity, opacity=opacity)

        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(input_image, cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(sensitivity, cmap="hot")
        axes[1].set_title(f"Occlusion sensitivity (class {target_class})")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()

        if save_path:
            save_figure(figure, save_path)
        return figure

    def compare_patch_sizes(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        patch_sizes: list[int] | None = None,
        stride_ratio: float = 0.5,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        if patch_sizes is None:
            patch_sizes = [8, 16, 32]

        sensitivity_maps = []
        display_titles = []
        for current_patch_size in patch_sizes:
            current_stride = max(1, int(current_patch_size * stride_ratio))
            sensitivity_maps.append(
                self.generate(input_tensor, target_class, patch_size=current_patch_size, stride=current_stride)
            )
            display_titles.append(f"Patch {current_patch_size}x{current_patch_size}")

        return plot_image_grid(
            sensitivity_maps,
            titles=display_titles,
            columns=len(patch_sizes),
            main_title=f"Occlusion sensitivity — class {target_class}",
            save_path=save_path,
        )
