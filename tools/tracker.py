import torch
from torch import nn

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')


class Tracker:
    """
    Thin TensorBoard writer wrapper.

    Logging conventions (grouped for easy comparison):
        loss/train_step, loss/train_epoch, loss/train_eval, loss/val
        curve_{mse,mae,rmse}/<stage>    — train/val on same graph
        r2/<stage>, r2_{overall,median,min}/<stage>
        pixel_{mse,mae}_max/<stage>
        cos_sim/<stage>, spectral_coh/<stage>, gt_param_{mse,mae}/<stage>
        
        train/grad_norm, train/grad_clip_thr, train/components/<name>, train/loss_total
        lr/<group>, lr/warmup_factor
        early_stop/*, system/*, params/<stage>/<name>
        debug/* — only if log_debug=True
    """
    
    def __init__(self, writer=None, debug: bool = False):
        self.writer = writer
        self.debug  = debug

    def log_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_metrics(self, prefix, dict_values, step):
        """Flat-log a dict of scalars under `prefix/<key>`. Preferred over add_scalars."""
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
        if isinstance(values, torch.Tensor):
            v = values.detach().float().cpu()
        else:
            v = values
        try:
            self.writer.add_histogram(tag, v, step, bins=bins)
        except (ValueError, RuntimeError):
            pass

    def log_image(self, tag, img, step, dataformats="CHW"):
        if self.writer is None:
            return
        if isinstance(img, torch.Tensor):
            img = img.detach().float().cpu()
        self.writer.add_image(tag, img, step, dataformats=dataformats)

    def log_images(self, tag, imgs, step, dataformats="NCHW"):
        if self.writer is None:
            return
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.detach().float().cpu()
        self.writer.add_images(tag, imgs, step, dataformats=dataformats)

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

    def log_gradients(self, model, step, max_grad_norm=None, prefix="debug/grads"):
        if self.writer is None or not self.debug:
            return
        total_norm  = 0.0
        max_abs     = 0.0
        n_with_grad = 0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            n_with_grad += 1
            g = param.grad.detach()
            self.writer.add_histogram(f"{prefix}/{name}", g, step)
            total_norm += g.norm(2).item() ** 2
            max_abs    = max(max_abs, g.abs().max().item())

        total_norm = total_norm ** 0.5
        self.writer.add_scalar(f"{prefix}/total_norm", total_norm, step)
        self.writer.add_scalar(f"{prefix}/global_max", max_abs,    step)
        if max_grad_norm is not None:
            self.writer.add_scalar(f"{prefix}/clip_ratio",
                                   total_norm / max(max_grad_norm, 1e-12), step)

    def log_optimizer(self, optimizer, step):
        if self.writer is None or not self.debug:
            return
        for i, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group_{i}")
            self.writer.add_scalar(f"debug/optimizer/lr_{name}", group["lr"], step)
            if "weight_decay" in group:
                self.writer.add_scalar(f"optimizer/wd_{name}", group["weight_decay"], step)
            if "betas" in group:
                b1, b2 = group["betas"]
                self.writer.add_scalar(f"optimizer/beta1_{name}", b1, step)
                self.writer.add_scalar(f"optimizer/beta2_{name}", b2, step)
            if "eps" in group:
                self.writer.add_scalar(f"optimizer/eps_{name}", group["eps"], step)

    def log_activations(self, model, step_box):
        hooks = []
        if self.writer is None or not self.debug:
            return hooks

        def _step():
            if isinstance(step_box, int):
                return step_box
            if isinstance(step_box, (list, tuple)) and len(step_box) > 0:
                return int(step_box[0])
            if isinstance(step_box, dict):
                return int(step_box.get("step", 0))
            return 0

        def hook_fn(name):
            def fn(module, _input, output):
                if isinstance(output, torch.Tensor):
                    s = _step()
                    self.writer.add_histogram(f"debug/activations/{name}", output.detach(), s)
                    self.writer.add_scalar(f"debug/activations_mean/{name}", output.mean().item(), s)
                    self.writer.add_scalar(f"debug/activations_std/{name}",  output.std().item(),  s)
            return fn

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.GELU, nn.SiLU)):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        return hooks

    def log_weights(self, model, step):
        if self.writer is None or not self.debug:
            return
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"debug/weights/{name}", param.detach(), step)


    def log_param_stats(self, name, tensor, step):
        if self.writer is None or tensor.numel() == 0:
            return
        t = tensor.detach().float()
        self.writer.add_scalar(f"{name}/mean", t.mean().item(), step)
        self.writer.add_scalar(f"{name}/std",  t.std().item(),  step)
        self.writer.add_scalar(f"{name}/min",  t.min().item(),  step)
        self.writer.add_scalar(f"{name}/max",  t.max().item(),  step)
        self.writer.add_scalar(f"{name}/norm", t.norm(2).item(), step)

    def log_memory(self, step, device=None):
        if self.writer is None or not torch.cuda.is_available():
            return
        dev = device if device is not None else torch.cuda.current_device()
        self.writer.add_scalar("system/gpu_mem_alloc_GB",     torch.cuda.memory_allocated(dev)     / 1024 ** 3, step)
        self.writer.add_scalar("system/gpu_mem_reserved_GB",  torch.cuda.memory_reserved(dev)      / 1024 ** 3, step)
        self.writer.add_scalar("system/gpu_mem_max_alloc_GB", torch.cuda.max_memory_allocated(dev) / 1024 ** 3, step)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
