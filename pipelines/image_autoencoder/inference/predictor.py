from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.monitoring.logger import Logger


@dataclass
class ImageAeResult:
    gt         : np.ndarray
    pred       : np.ndarray
    embeddings : np.ndarray


class ImageAePredictor:
    def __init__(self, run, device: str, logger: Logger) -> None:
        self.run        = run
        self.device     = device
        self.logger     = logger
        self.model      = run.model
        self.normalizer = run.normalizer

    def _reconstruct_batch(self, image_n: torch.Tensor):
        image_n = image_n.to(self.device)

        with torch.no_grad():
            image_hat_n, z = self.model.reconstruct(image_n)

        gt   = self.normalizer.denormalize_input(image_n.cpu().numpy())
        pred = self.normalizer.denormalize_input(image_hat_n.cpu().numpy())
        emb  = z.mean(dim=tuple(range(2, z.ndim))) if z.ndim > 2 else z

        return gt, pred, emb.cpu().numpy()

    def run_inference(self) -> ImageAeResult:
        self.logger.section("[Image AE Inference: Reconstruct]")

        gt_chunks, pred_chunks, emb_chunks = [], [], []

        for batch in self.run.loader:
            gt, pred, emb = self._reconstruct_batch(batch[0])
            gt_chunks.append(gt)
            pred_chunks.append(pred)
            emb_chunks.append(emb)

        result = ImageAeResult(
            gt         = np.concatenate(gt_chunks,   axis=0).astype(np.float32),
            pred       = np.concatenate(pred_chunks, axis=0).astype(np.float32),
            embeddings = np.concatenate(emb_chunks,  axis=0).astype(np.float32),
        )

        self.logger.kv_table({
            "Patches"       : result.gt.shape[0],
            "Channels"      : result.gt.shape[1],
            "Patch size"    : f"{result.gt.shape[2]} x {result.gt.shape[3]}",
            "Embedding dim" : result.embeddings.shape[1],
        })

        return result
