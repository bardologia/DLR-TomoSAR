from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .base import BaseInterpreter, normalize_to_unit_range, plot_image_grid, save_figure


class FeatureMapInspector(BaseInterpreter):
    def __init__(self, model: nn.Module, layers: dict[str, nn.Module], device: Optional[str | torch.device] = None) -> None:
        super().__init__(model, device)
        self.layer_names = list(layers.keys())
        self._registered_hooks: list = []
        self._captured_activations: dict[str, torch.Tensor] = {}

        for layer_name, layer_module in layers.items():
            hook = layer_module.register_forward_hook(self._create_capture_hook(layer_name))
            self._registered_hooks.append(hook)

    def _create_capture_hook(self, layer_name: str):
        def hook_function(module, layer_input, layer_output):
            self._captured_activations[layer_name] = layer_output.detach()
        return hook_function

    @torch.no_grad()
    def extract(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        prepared_input = self._prepare_input(input_tensor)
        self._captured_activations.clear()
        self.model(prepared_input)
        return dict(self._captured_activations)

    def compute_statistics(self, input_tensor: torch.Tensor) -> dict[str, dict[str, float]]:
        activations      = self.extract(input_tensor)
        layer_statistics = {}

        for layer_name, activation_tensor in activations.items():
            float_tensor = activation_tensor.float()
            layer_statistics[layer_name] = {
                "mean"     : float_tensor.mean().item(),
                "std"      : float_tensor.std().item(),
                "min"      : float_tensor.min().item(),
                "max"      : float_tensor.max().item(),
                "sparsity" : (float_tensor.abs() < 1e-6).float().mean().item(),
            }

        return layer_statistics

    def visualise_channels(self, input_tensor: torch.Tensor, layer_name: str, max_channels: int = 16, save_path: Optional[str | Path] = None) -> plt.Figure:
        activations          = self.extract(input_tensor)
        feature_maps         = activations[layer_name][0]
        num_channels_to_show = min(feature_maps.shape[0], max_channels)

        channel_images = [
            self._to_numpy(normalize_to_unit_range(feature_maps[channel_index]))
            for channel_index in range(num_channels_to_show)
        ]
        channel_titles = [f"Channel {channel_index}" for channel_index in range(num_channels_to_show)]

        return plot_image_grid(
            channel_images,
            titles=channel_titles,
            main_title=f"Feature maps — {layer_name}",
            save_path=save_path,
        )

    def visualise(
        self,
        input_tensor: torch.Tensor,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        activations = self.extract(input_tensor)
        mean_activation_images = []
        layer_titles = []

        for layer_name in self.layer_names:
            mean_activation = activations[layer_name][0].mean(dim=0)
            mean_activation_images.append(self._to_numpy(normalize_to_unit_range(mean_activation)))
            layer_titles.append(layer_name)

        return plot_image_grid(
            mean_activation_images,
            titles=layer_titles,
            main_title="Mean activation per layer",
            save_path=save_path,
        )

    def plot_statistics(
        self,
        input_tensor: torch.Tensor,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        statistics = self.compute_statistics(input_tensor)
        layer_names = list(statistics.keys())
        mean_values = [statistics[name]["mean"] for name in layer_names]
        std_values = [statistics[name]["std"] for name in layer_names]
        sparsity_values = [statistics[name]["sparsity"] for name in layer_names]

        figure, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].barh(layer_names, mean_values, color="steelblue")
        axes[0].set_title("Mean activation")

        axes[1].barh(layer_names, std_values, color="coral")
        axes[1].set_title("Standard deviation")

        axes[2].barh(layer_names, sparsity_values, color="mediumseagreen")
        axes[2].set_title("Sparsity")
        axes[2].set_xlim(0, 1)

        figure.tight_layout()
        if save_path:
            save_figure(figure, save_path)
        return figure

    def remove_hooks(self) -> None:
        for hook in self._registered_hooks:
            hook.remove()
        self._registered_hooks.clear()

    def __del__(self):
        self.remove_hooks()
