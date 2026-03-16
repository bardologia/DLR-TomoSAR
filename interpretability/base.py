from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device != device:
        return tensor.to(device)
    return tensor


def ensure_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected 3-D or 4-D tensor, got {tensor.ndim}-D")
    return tensor


def normalize_to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    if tensor_max - tensor_min < 1e-8:
        return torch.zeros_like(tensor)
    return (tensor - tensor_min) / (tensor_max - tensor_min)


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    opacity: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    color_mapper = plt.get_cmap(colormap)
    colored_heatmap = color_mapper(heatmap)[..., :3]

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.max() > 1.0:
        image = image / 255.0

    blended = opacity * colored_heatmap + (1 - opacity) * image
    return np.clip(blended, 0.0, 1.0)


def save_figure(figure: Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(figure)


def plot_image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    columns: int = 4,
    cell_size: tuple[float, float] = (3.0, 3.0),
    main_title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    num_images = len(images)
    rows = max(1, (num_images + columns - 1) // columns)
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(cell_size[0] * columns, cell_size[1] * rows),
    )
    axes = np.asarray(axes).flatten()

    for index, axis in enumerate(axes):
        if index < num_images:
            current_image = images[index]
            axis.imshow(current_image, cmap="viridis" if current_image.ndim == 2 else None)
            if titles and index < len(titles):
                axis.set_title(titles[index], fontsize=9)
        axis.axis("off")

    if main_title:
        figure.suptitle(main_title, fontsize=12)
    figure.tight_layout()

    if save_path:
        save_figure(figure, save_path)
    return figure


class BaseInterpreter:
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str | torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def _prepare_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return ensure_batch_dimension(move_to_device(input_tensor, self.device))

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()
