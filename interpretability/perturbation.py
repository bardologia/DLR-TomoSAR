from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base import BaseInterpreter, normalize_to_unit_range, save_figure, plot_image_grid


class PerturbationExperiment(BaseInterpreter):
    @torch.no_grad()
    def _compute_class_iou(self, prediction: torch.Tensor, reference: torch.Tensor, target_class: int) -> float:
        predicted_mask = prediction.argmax(dim=1) == target_class
        reference_mask = reference.argmax(dim=1) == target_class
        intersection   = (predicted_mask & reference_mask).sum().float()
        union = (predicted_mask | reference_mask).sum().float()
        if union < 1:
            return 1.0
        return (intersection / union).item()

    @torch.no_grad()
    def _compute_mean_probability(self, logits: torch.Tensor, target_class: int) -> float:
        probabilities = F.softmax(logits, dim=1)
        return probabilities[0, target_class].mean().item()

    @staticmethod
    def gaussian_noise(input_tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
        return input_tensor + torch.randn_like(input_tensor) * noise_level

    @staticmethod
    def gaussian_blur(input_tensor: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        num_channels = input_tensor.shape[1]
        coordinates  = torch.arange(kernel_size, dtype=input_tensor.dtype, device=input_tensor.device) - kernel_size // 2
        
        gaussian_1d = torch.exp(-0.5 * (coordinates / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        kernel_2d   = gaussian_1d[:, None] @ gaussian_1d[None, :]
        kernel_2d   = kernel_2d.expand(num_channels, 1, -1, -1)
        
        padding = kernel_size // 2
        return F.conv2d(input_tensor, kernel_2d, padding=padding, groups=num_channels)

    @staticmethod
    def drop_channels(input_tensor: torch.Tensor, channel_indices_to_drop: list[int]) -> torch.Tensor:
        result = input_tensor.clone()
        for channel_index in channel_indices_to_drop:
            result[:, channel_index] = 0.0
        return result

    @torch.enable_grad()
    def fgsm_perturbation(self, input_tensor: torch.Tensor, target_class: int, epsilon: float = 0.01) -> torch.Tensor:
        prepared_input = self._prepare_input(input_tensor).clone().requires_grad_(True)
        model_output  = self.model(prepared_input)
        
        loss = model_output[0, target_class].sum()
        self.model.zero_grad()
        loss.backward()
        
        adversarial_perturbation = epsilon * prepared_input.grad.sign()
        return (prepared_input - adversarial_perturbation).detach()

    @torch.no_grad()
    def noise_sweep(self, input_tensor: torch.Tensor, target_class: int, noise_levels: list[float] | None = None) -> dict[str, list]:
        prepared_input   = self._prepare_input(input_tensor)
        reference_output = self.model(prepared_input)

        iou_scores, mean_probabilities = [], []
        for noise_level in noise_levels:
            noisy_input  = self.gaussian_noise(prepared_input, noise_level)
            noisy_output = self.model(noisy_input)
            iou_scores.append(self._compute_class_iou(noisy_output, reference_output, target_class))
            mean_probabilities.append(self._compute_mean_probability(noisy_output, target_class))

        return {"noise_level": noise_levels, "iou": iou_scores, "mean_probability": mean_probabilities}

    @torch.no_grad()
    def blur_sweep(self, input_tensor: torch.Tensor, target_class: int, kernel_sizes: list[int] | None = None) -> dict[str, list]:
        prepared_input   = self._prepare_input(input_tensor)
        reference_output = self.model(prepared_input)

        iou_scores, mean_probabilities = [], []
        for current_kernel_size in kernel_sizes:
            if current_kernel_size <= 1:
                blurred_input = prepared_input
            else:
                blurred_input = self.gaussian_blur(
                    prepared_input, kernel_size=current_kernel_size, sigma=current_kernel_size / 3.0,
                )
            blurred_output = self.model(blurred_input)
            iou_scores.append(self._compute_class_iou(blurred_output, reference_output, target_class))
            mean_probabilities.append(self._compute_mean_probability(blurred_output, target_class))

        return {"kernel_size": kernel_sizes, "iou": iou_scores, "mean_probability": mean_probabilities}

    @torch.no_grad()
    def fgsm_sweep(self, input_tensor: torch.Tensor, target_class: int, epsilon_values: list[float] | None = None) -> dict[str, list]:

        prepared_input = self._prepare_input(input_tensor)
        with torch.no_grad():
            reference_output = self.model(prepared_input)

        iou_scores, mean_probabilities = [], []
        for epsilon in epsilon_values:
            if epsilon == 0:
                perturbed_input = prepared_input
            else:
                perturbed_input = self.fgsm_perturbation(prepared_input, target_class, epsilon=epsilon)
            
            with torch.no_grad():
                perturbed_output = self.model(perturbed_input)
            
            iou_scores.append(self._compute_class_iou(perturbed_output, reference_output, target_class))
            mean_probabilities.append(self._compute_mean_probability(perturbed_output, target_class))

        return {"epsilon": epsilon_values, "iou": iou_scores, "mean_probability": mean_probabilities}

    def plot_noise_sweep(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        noise_levels: list[float] | None = None,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        results = self.noise_sweep(input_tensor, target_class, noise_levels)
        return self._plot_sweep_results(
            results["noise_level"], results["iou"], results["mean_probability"],
            horizontal_label="Noise level",
            title=f"Gaussian noise robustness — class {target_class}",
            save_path=save_path,
        )

    def plot_blur_sweep(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        kernel_sizes: list[int] | None = None,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        results = self.blur_sweep(input_tensor, target_class, kernel_sizes)
        return self._plot_sweep_results(
            results["kernel_size"], results["iou"], results["mean_probability"],
            horizontal_label="Blur kernel size",
            title=f"Gaussian blur robustness — class {target_class}",
            save_path=save_path,
        )

    def plot_fgsm_sweep(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        epsilon_values: list[float] | None = None,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        results = self.fgsm_sweep(input_tensor, target_class, epsilon_values)
        return self._plot_sweep_results(
            results["epsilon"], results["iou"], results["mean_probability"],
            horizontal_label="FGSM epsilon",
            title=f"Adversarial robustness — class {target_class}",
            save_path=save_path,
        )

    def plot_qualitative_comparison(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        noise_level: float = 0.1,
        blur_kernel_size: int = 7,
        fgsm_epsilon: float = 0.01,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        prepared_input = self._prepare_input(input_tensor)

        perturbation_variants = {
            "Clean": prepared_input,
            f"Noise level={noise_level}": self.gaussian_noise(prepared_input, noise_level),
            f"Blur kernel={blur_kernel_size}": self.gaussian_blur(
                prepared_input, blur_kernel_size, blur_kernel_size / 3,
            ),
            f"FGSM epsilon={fgsm_epsilon}": self.fgsm_perturbation(
                prepared_input, target_class, fgsm_epsilon,
            ),
        }

        prediction_images, variant_titles = [], []
        with torch.no_grad():
            for variant_label, variant_input in perturbation_variants.items():
                predicted_segmentation = self.model(variant_input).argmax(dim=1)[0]
                prediction_images.append(self._to_numpy(predicted_segmentation.float()))
                variant_titles.append(variant_label)

        return plot_image_grid(
            prediction_images,
            titles=variant_titles,
            columns=len(prediction_images),
            main_title=f"Perturbation effects — class {target_class}",
            save_path=save_path,
        )

    @staticmethod
    def _plot_sweep_results(
        horizontal_values,
        iou_scores,
        mean_probabilities,
        horizontal_label: str,
        title: str,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        figure, left_axis = plt.subplots(figsize=(7, 4))

        iou_color = "steelblue"
        probability_color = "coral"

        left_axis.plot(horizontal_values, iou_scores, "o-", color=iou_color, label="IoU vs. clean")
        left_axis.set_xlabel(horizontal_label)
        left_axis.set_ylabel("IoU", color=iou_color)
        left_axis.set_ylim(-0.05, 1.05)
        left_axis.tick_params(axis="y", labelcolor=iou_color)

        right_axis = left_axis.twinx()
        right_axis.plot(horizontal_values, mean_probabilities, "s--", color=probability_color, label="Mean probability")
        right_axis.set_ylabel("Mean probability", color=probability_color)
        right_axis.set_ylim(-0.05, 1.05)
        right_axis.tick_params(axis="y", labelcolor=probability_color)

        left_handles, left_labels = left_axis.get_legend_handles_labels()
        right_handles, right_labels = right_axis.get_legend_handles_labels()
        left_axis.legend(left_handles + right_handles, left_labels + right_labels, loc="lower left")

        left_axis.set_title(title)
        figure.tight_layout()

        if save_path:
            save_figure(figure, save_path)
        return figure
