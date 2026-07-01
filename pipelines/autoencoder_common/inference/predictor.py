from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.monitoring.logger import Logger


@dataclass
class AeResult:
    gt         : np.ndarray
    pred       : np.ndarray
    embeddings : np.ndarray


class AeReconstructionPredictor:
    SECTION      = "[AE Inference: Reconstruct]"
    RESULT_CLASS = AeResult

    def __init__(self, run, device: str, logger: Logger) -> None:
        self.run        = run
        self.device     = device
        self.logger     = logger
        self.model      = run.model
        self.normalizer = run.normalizer

    def _batch_input(self, batch):
        raise NotImplementedError

    def _reconstruct_batch(self, batch_input):
        raise NotImplementedError

    def _summary(self, result) -> dict:
        raise NotImplementedError

    def run_inference(self):
        self.logger.section(self.SECTION)

        gt_chunks, pred_chunks, emb_chunks = [], [], []

        for batch in self.run.loader:
            gt, pred, emb = self._reconstruct_batch(self._batch_input(batch))
            gt_chunks.append(gt)
            pred_chunks.append(pred)
            emb_chunks.append(emb)

        result = self.RESULT_CLASS(
            gt         = np.concatenate(gt_chunks,   axis=0).astype(np.float32),
            pred       = np.concatenate(pred_chunks, axis=0).astype(np.float32),
            embeddings = np.concatenate(emb_chunks,  axis=0).astype(np.float32),
        )

        self.logger.kv_table(self._summary(result))

        return result
