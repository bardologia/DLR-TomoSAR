import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')


class Tracker:    
    def __init__(self, writer=None, debug: bool = False):
        self.writer = writer
        self.debug  = debug

    def log_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_metrics(self, prefix, dict_values, step):
        if self.writer is None:
            return
        for k, v in dict_values.items():
            try:
                self.writer.add_scalar(f"{prefix}/{k}", float(v), step)
            except (TypeError, ValueError):
                continue

    def log_dict(self, tag, dict_values, step):
        self.log_metrics(tag, dict_values, step)

    def log_scalars_flat(self, prefix, dict_values, step):
        self.log_metrics(prefix, dict_values, step)

    def log_histogram(self, tag, values, step, bins="auto"):
        if self.writer is None:
            return
        v = np.asarray(values).ravel().astype(np.float32)
        try:
            self.writer.add_histogram(tag, v, step, bins=bins)
        except (ValueError, RuntimeError):
            pass

    def log_image(self, tag, img, step, dataformats="CHW"):
        if self.writer is None:
            return
        self.writer.add_image(tag, np.asarray(img), step, dataformats=dataformats)

    def log_images(self, tag, imgs, step, dataformats="NCHW"):
        if self.writer is None:
            return
        self.writer.add_images(tag, np.asarray(imgs), step, dataformats=dataformats)

    def log_figure(self, tag, fig, step, close=True):
        if self.writer is None:
            if close:
                plt.close(fig)
            return
        self.writer.add_figure(tag, fig, step, close=close)

    def log_text(self, tag, text, step=0):
        if self.writer is not None:
            self.writer.add_text(tag, str(text), step)

    def log_hparams(self, hparams: dict, metrics: dict):
        if self.writer is None:
            return
        clean_h = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in hparams.items()}
        clean_m = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        try:
            self.writer.add_hparams(clean_h, clean_m)
        except (ValueError, TypeError):
            pass

    def log_pr_curve(self, tag, labels, predictions, step):
        if self.writer is not None:
            self.writer.add_pr_curve(tag, labels, predictions, step)

    def log_graph(self, model, input_to_model):
        if self.writer is None:
            return
        try:
            self.writer.add_graph(model, input_to_model)
        except Exception:
            pass

    def log_optimizer(self, lr: float, step, name: str = "lr"):
        if self.writer is None or not self.debug:
            return
        self.writer.add_scalar(f"debug/optimizer/{name}", lr, step)

    def log_activations(self, model, step_box):
        return []

    def log_param_stats(self, name, tensor, step):
        if self.writer is None:
            return
        t = np.asarray(tensor, dtype=np.float32).ravel()
        if t.size == 0:
            return
        self.writer.add_scalar(f"{name}/mean", float(t.mean()), step)
        self.writer.add_scalar(f"{name}/std",  float(t.std()),  step)
        self.writer.add_scalar(f"{name}/min",  float(t.min()),  step)
        self.writer.add_scalar(f"{name}/max",  float(t.max()),  step)
        self.writer.add_scalar(f"{name}/norm", float(np.linalg.norm(t)), step)

    def log_memory(self, step, device=None):
        import torch
        if self.writer is None:
            return
        if not torch.cuda.is_available():
            return
        dev = device if device is not None else torch.cuda.current_device()
        try:
            alloc_gb    = torch.cuda.memory_allocated(dev) / 1024 ** 3
            reserved_gb = torch.cuda.memory_reserved(dev)  / 1024 ** 3
            peak_gb     = torch.cuda.max_memory_allocated(dev) / 1024 ** 3
            self.writer.add_scalar("system/gpu_mem_alloc_GB",     alloc_gb,    step)
            self.writer.add_scalar("system/gpu_mem_reserved_GB",  reserved_gb, step)
            self.writer.add_scalar("system/gpu_mem_peak_alloc_GB", peak_gb,    step)
        except Exception:
            pass

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
