from __future__ import annotations

from contextlib import contextmanager
from typing     import Any, Mapping, Optional

import numpy as np


class Tracker:
    def __init__(self, writer=None, debug: bool = False) -> None:
        self.writer  = writer
        self.debug   = debug
        self._step   = 0
        self._scopes = []

    @property
    def active(self) -> bool:
        return self.writer is not None

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def advance(self, n: int = 1) -> int:
        self._step += n
        return self._step

    @contextmanager
    def scope(self, name: str):
        self._scopes.append(str(name))
        try:
            yield self
        finally:
            self._scopes.pop()

    def _tag(self, tag: str) -> str:
        return "/".join([*self._scopes, str(tag)])

    def _resolve(self, step: Optional[int]) -> int:
        return self._step if step is None else int(step)

    def _emit(self, method: str, tag: str, payload: Any, step: Optional[int], **kwargs: Any) -> None:
        if self.writer is None:
            return
        getattr(self.writer, method)(self._tag(tag), payload, self._resolve(step), **kwargs)

    def log_scalar(self, tag, value, step=None) -> None:
        self._emit("add_scalar", tag, float(value), step)

    def log_metrics(self, prefix, values: Mapping[str, Any], step=None) -> None:
        for k, v in values.items():
            try:
                self._emit("add_scalar", f"{prefix}/{k}", float(v), step)
            except (TypeError, ValueError):
                continue

    def log_histogram(self, tag, values, step=None, bins="auto") -> None:
        v = np.asarray(values).ravel().astype(np.float32)
        try:
            self._emit("add_histogram", tag, v, step, bins=bins)
        except (ValueError, RuntimeError):
            pass

    def log_image(self, tag, img, step=None, dataformats="CHW") -> None:
        self._emit("add_image", tag, np.asarray(img), step, dataformats=dataformats)

    def log_images(self, tag, imgs, step=None, dataformats="NCHW") -> None:
        self._emit("add_images", tag, np.asarray(imgs), step, dataformats=dataformats)

    def log_figure(self, tag, fig, step=None, close=True) -> None:
        if self.writer is None:
            if close:
                import matplotlib.pyplot as plt
                plt.close(fig)
            return
        self._emit("add_figure", tag, fig, step, close=close)

    def log_text(self, tag, text, step=None) -> None:
        self._emit("add_text", tag, str(text), step)

    def log_pr_curve(self, tag, labels, predictions, step=None) -> None:
        if self.writer is None:
            return
        self.writer.add_pr_curve(self._tag(tag), labels, predictions, self._resolve(step))

    def log_hparams(self, hparams: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
        if self.writer is None:
            return

        clean_h = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in hparams.items()}
        clean_m = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

        try:
            self.writer.add_hparams(clean_h, clean_m)
        except (ValueError, TypeError):
            pass

    def log_graph(self, model, input_to_model) -> None:
        if self.writer is None:
            return
        try:
            self.writer.add_graph(model, input_to_model)
        except Exception:
            pass

    def log_param_stats(self, name, tensor, step=None) -> None:
        t = np.asarray(tensor, dtype=np.float32).ravel()
        if t.size == 0:
            return

        stats = {
            "mean" : float(t.mean()),
            "std"  : float(t.std()),
            "min"  : float(t.min()),
            "max"  : float(t.max()),
            "norm" : float(np.linalg.norm(t)),
        }
        self.log_metrics(name, stats, step)

    def log_memory(self, step=None, device=None) -> None:
        import torch

        if self.writer is None or not torch.cuda.is_available():
            return

        dev = device if device is not None else torch.cuda.current_device()
        try:
            memory = {
                "gpu_mem_alloc_GB"      : torch.cuda.memory_allocated(dev)     / 1024 ** 3,
                "gpu_mem_reserved_GB"   : torch.cuda.memory_reserved(dev)      / 1024 ** 3,
                "gpu_mem_peak_alloc_GB" : torch.cuda.max_memory_allocated(dev) / 1024 ** 3,
            }
            self.log_metrics("system", memory, step)
        except Exception:
            pass

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class NullTracker(Tracker):
    def __init__(self, debug: bool = False) -> None:
        super().__init__(writer=None, debug=debug)
