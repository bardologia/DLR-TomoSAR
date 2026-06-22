from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.monitoring.logger import Logger


@dataclass
class ProfileAeResult:
    gt         : np.ndarray
    pred       : np.ndarray
    embeddings : np.ndarray


class ProfileAePredictor:
    def __init__(self, run, device: str, logger: Logger) -> None:
        self.run        = run
        self.device     = device
        self.logger     = logger
        self.model      = run.model
        self.normalizer = run.normalizer

    def _reconstruct_batch(self, curve_n: torch.Tensor):
        x = curve_n.to(self.device).unsqueeze(-1).unsqueeze(-1)

        with torch.no_grad():
            curve_hat_n, z = self.model.reconstruct(x)

        gt   = self.normalizer.denormalize(x).squeeze(-1).squeeze(-1)
        pred = self.normalizer.denormalize(curve_hat_n).squeeze(-1).squeeze(-1)
        emb  = z.squeeze(-1).squeeze(-1)

        return gt.cpu().numpy(), pred.cpu().numpy(), emb.cpu().numpy()

    def run_inference(self) -> ProfileAeResult:
        self.logger.section("[Profile AE Inference: Reconstruct]")

        gt_chunks, pred_chunks, emb_chunks = [], [], []

        for batch in self.run.loader:
            gt, pred, emb = self._reconstruct_batch(batch)
            gt_chunks.append(gt)
            pred_chunks.append(pred)
            emb_chunks.append(emb)

        result = ProfileAeResult(
            gt         = np.concatenate(gt_chunks,   axis=0).astype(np.float32),
            pred       = np.concatenate(pred_chunks, axis=0).astype(np.float32),
            embeddings = np.concatenate(emb_chunks,  axis=0).astype(np.float32),
        )

        self.logger.kv_table({
            "Curves"        : result.gt.shape[0],
            "Profile length": result.gt.shape[1],
            "Embedding dim" : result.embeddings.shape[1],
        })

        return result
