from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .base import BaseInterpreter, normalize_to_unit_range, plot_image_grid, save_figure


class ParameterSensitivity(BaseInterpreter):

    def weight_summary(self) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}

        for parameter_name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                weight_data = parameter.data.float()
                summary[parameter_name] = {
                    "mean"                : weight_data.mean().item(),
                    "std"                 : weight_data.std().item(),
                    "min"                 : weight_data.min().item(),
                    "max"                 : weight_data.max().item(),
                    "num_params"          : parameter.numel(),
                    "fraction_near_zero"  : (weight_data.abs() < 1e-3).float().mean().item(),
                }

        return summary

    def plot_weight_distributions(
        self,
        include_bias: bool = False,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        collected_layers: list[tuple[str, np.ndarray]] = []

        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight_values = module.weight.data.cpu().float().numpy().ravel()
                collected_layers.append((module_name, weight_values))
                if include_bias and module.bias is not None:
                    bias_values = module.bias.data.cpu().float().numpy().ravel()
                    collected_layers.append((f"{module_name}.bias", bias_values))

        num_layers = len(collected_layers)
        columns = min(4, num_layers)
        rows = max(1, (num_layers + columns - 1) // columns)
        figure, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3 * rows))
        axes = np.asarray(axes).flatten()

        for index, axis in enumerate(axes):
            if index < num_layers:
                layer_name, weight_values = collected_layers[index]
                axis.hist(weight_values, bins=60, color="steelblue", edgecolor="white", density=True)
                axis.set_title(layer_name, fontsize=7)
                axis.tick_params(labelsize=6)
            else:
                axis.axis("off")

        figure.suptitle("Weight distributions per Conv2d layer", fontsize=11)
        figure.tight_layout()

        if save_path:
            save_figure(figure, save_path)
        return figure

    def plot_filter_grid(
        self,
        layer_name: str,
        max_filters: int = 32,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        target_module = dict(self.model.named_modules()).get(layer_name)

        if target_module is None:
            available_conv_layers = [
                name for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d)
            ]
            raise KeyError(f"Layer '{layer_name}' not found. Available Conv2d layers: {available_conv_layers}")

        if not isinstance(target_module, nn.Conv2d):
            raise TypeError(f"'{layer_name}' is {type(target_module).__name__}, expected Conv2d")

        filter_weights = target_module.weight.data.cpu().float()
        num_filters_to_show = min(filter_weights.shape[0], max_filters)

        filter_images = []
        for filter_index in range(num_filters_to_show):
            single_filter = filter_weights[filter_index]
            if single_filter.shape[0] > 1:
                averaged_filter = single_filter.mean(dim=0)
            else:
                averaged_filter = single_filter[0]
            filter_images.append(normalize_to_unit_range(averaged_filter).numpy())

        filter_titles = [f"Filter {filter_index}" for filter_index in range(num_filters_to_show)]
        return plot_image_grid(
            filter_images,
            titles=filter_titles,
            main_title=f"Filters — {layer_name}",
            save_path=save_path,
        )

    @torch.enable_grad()
    def compute_layer_importance(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> dict[str, float]:
        prepared_input = self._prepare_input(input_tensor).requires_grad_(False)

        for parameter in self.model.parameters():
            parameter.requires_grad_(True)

        model_output = self.model(prepared_input)
        score = model_output[0, target_class].sum()
        self.model.zero_grad()
        score.backward()

        importance_scores: dict[str, float] = {}
        for parameter_name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                importance_scores[parameter_name] = parameter.grad.abs().mean().item()

        sorted_importance = dict(
            sorted(importance_scores.items(), key=lambda key_value: key_value[1], reverse=True)
        )
        return sorted_importance

    def plot_layer_importance(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        top_k: int = 20,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        importance_ranking = self.compute_layer_importance(input_tensor, target_class)
        top_layer_names = list(importance_ranking.keys())[:top_k]
        top_scores = [importance_ranking[name] for name in top_layer_names]

        figure, axis = plt.subplots(figsize=(8, max(4, 0.35 * top_k)))
        vertical_positions = np.arange(len(top_layer_names))
        axis.barh(vertical_positions, top_scores, color="coral")
        axis.set_yticks(vertical_positions)
        axis.set_yticklabels(top_layer_names, fontsize=7)
        axis.invert_yaxis()
        axis.set_xlabel("Mean |gradient|")
        axis.set_title(f"Layer importance — class {target_class}")
        figure.tight_layout()

        if save_path:
            save_figure(figure, save_path)
        return figure
