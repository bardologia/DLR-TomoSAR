from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .base import BaseInterpreter, save_figure


class LatentSpaceAnalyzer(BaseInterpreter):
    def __init__(
        self,
        model: nn.Module,
        bottleneck_layer: nn.Module,
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__(model, device)
        self.bottleneck_layer = bottleneck_layer

        self._captured_activation: torch.Tensor | None = None
        self._hook = bottleneck_layer.register_forward_hook(self._on_forward)

        self.embeddings: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def _on_forward(self, module, layer_input, layer_output):
        self._captured_activation = layer_output.detach()

    @torch.no_grad()
    def extract_single(self, input_tensor: torch.Tensor) -> np.ndarray:
        prepared_input = self._prepare_input(input_tensor)
        self.model(prepared_input)
        pooled_vector = self._captured_activation.mean(dim=(2, 3))
        return self._to_numpy(pooled_vector[0])

    @torch.no_grad()
    def fit(
        self,
        dataloader: DataLoader,
        max_samples: int = 2000,
        label_key: Optional[str] = None,
    ) -> None:
        all_embedding_vectors: list[np.ndarray] = []
        all_batch_labels: list[np.ndarray] = []
        collected_count = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                batch_labels = batch[1] if len(batch) > 1 else None
            elif isinstance(batch, dict):
                images = batch["image"]
                batch_labels = batch.get(label_key)
            else:
                images = batch
                batch_labels = None

            images = images.to(self.device)
            self.model(images)
            pooled_vectors = self._captured_activation.mean(dim=(2, 3))
            all_embedding_vectors.append(self._to_numpy(pooled_vectors))

            if batch_labels is not None:
                if isinstance(batch_labels, torch.Tensor):
                    batch_labels = self._to_numpy(batch_labels)
                all_batch_labels.append(np.asarray(batch_labels))

            collected_count += images.shape[0]
            if collected_count >= max_samples:
                break

        self.embeddings = np.concatenate(all_embedding_vectors, axis=0)[:max_samples]
        if all_batch_labels:
            self.labels = np.concatenate(all_batch_labels, axis=0)[:max_samples]
        else:
            self.labels = None

    def reduce_dimensions(
        self,
        method: Literal["pca", "tsne", "umap"] = "pca",
        num_components: int = 2,
        **reduction_kwargs,
    ) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Call .fit(dataloader) first.")

        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=num_components, **reduction_kwargs)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=num_components, **reduction_kwargs)
        elif method == "umap":
            import umap
            reducer = umap.UMAP(n_components=num_components, **reduction_kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'")

        return reducer.fit_transform(self.embeddings)

    def plot_projection(
        self,
        method: Literal["pca", "tsne", "umap"] = "pca",
        num_components: int = 2,
        save_path: Optional[str | Path] = None,
        **reduction_kwargs,
    ) -> plt.Figure:
        projected_embeddings = self.reduce_dimensions(method, num_components=num_components, **reduction_kwargs)

        figure, axis = plt.subplots(figsize=(7, 6))
        scatter_options = {"s": 8, "alpha": 0.7}

        if self.labels is not None:
            scatter_plot = axis.scatter(
                projected_embeddings[:, 0],
                projected_embeddings[:, 1],
                c=self.labels,
                cmap="tab10",
                **scatter_options,
            )
            figure.colorbar(scatter_plot, ax=axis, label="Label")
        else:
            axis.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], **scatter_options)

        axis.set_title(f"Latent space — {method.upper()}")
        axis.set_xlabel("Component 1")
        axis.set_ylabel("Component 2")
        figure.tight_layout()

        if save_path:
            save_figure(figure, save_path)
        return figure

    def plot_embedding_norms(
        self,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        if self.embeddings is None:
            raise RuntimeError("Call .fit(dataloader) first.")

        embedding_norms = np.linalg.norm(self.embeddings, axis=1)

        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(embedding_norms, bins=50, color="steelblue", edgecolor="white")
        axis.set_xlabel("L2 norm")
        axis.set_ylabel("Count")
        axis.set_title("Distribution of bottleneck embedding norms")
        figure.tight_layout()

        if save_path:
            save_figure(figure, save_path)
        return figure

    def plot_channel_distribution(
        self,
        top_k: int = 16,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        if self.embeddings is None:
            raise RuntimeError("Call .fit(dataloader) first.")

        channel_standard_deviations = self.embeddings.std(axis=0)
        most_variable_indices = np.argsort(channel_standard_deviations)[-top_k:][::-1]

        figure, axis = plt.subplots(figsize=(10, 4))
        channel_data = [self.embeddings[:, channel_index] for channel_index in most_variable_indices]
        axis.boxplot(channel_data, labels=[str(index) for index in most_variable_indices], vert=True)
        axis.set_xlabel("Channel index")
        axis.set_ylabel("Activation value")
        axis.set_title(f"Top-{top_k} most variable bottleneck channels")
        figure.tight_layout()

        if save_path:
            save_figure(figure, save_path)
        return figure

    def remove_hooks(self) -> None:
        self._hook.remove()

    def __del__(self):
        self.remove_hooks()
