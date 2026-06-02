from __future__ import annotations

import select
import subprocess
import sys
import tempfile
import atexit
from pathlib import Path

gpu_ids = [0, 1, 3]

def _make_experiments() -> list[dict]:
    extra_losses = [
        ("complete_use_mse_curve",          "complete_weight_mse_curve"),
        ("complete_use_l1_curve",           "complete_weight_l1_curve"),
        ("complete_use_huber_curve",        "complete_weight_huber_curve"),
        ("complete_use_charbonnier_curve",  "complete_weight_charbonnier_curve"),
        ("complete_use_cosine_curve",       "complete_weight_cosine_curve"),
        ("complete_use_spectral_coherence", "complete_weight_spectral_coh"),
        ("complete_use_ssim_curve",         "complete_weight_ssim_curve"),
    ]
    weights = [0.01, 0.05, 0.02]

    exps = []
    for use_key, weight_key in extra_losses:
        for w in weights:
            exps.append({
                "curriculum_enabled"              : True,
                "curriculum_swap_epoch"           : 30,
                "curriculum_reset_early_stopping" : True,
                "curriculum_reset_lr"             : True,
                "curriculum_reset_warmup"         : True,
                "curriculum_reset_optimizer"      : False,
                "warmup_use_param_l1"             : True,  "warmup_weight_param_l1"  : 1.0,
                "complete_use_param_l1"           : True,  "complete_weight_param_l1": 1.0,
                use_key                           : True,  weight_key                : w,
            })
    
    return exps

experiments = _make_experiments()

train_script = Path(__file__).resolve().parent / "single_train.py"

warmup_loss_labels: dict[str, str] = {
    "warmup_use_mse_curve"          : "mse",
    "warmup_use_l1_curve"           : "l1",
    "warmup_use_huber_curve"        : "huber",
    "warmup_use_charbonnier_curve"  : "charb",
    "warmup_use_cosine_curve"       : "cos",
    "warmup_use_spectral_coherence" : "spec",
    "warmup_use_ssim_curve"         : "ssim",
    "warmup_use_param_huber"        : "pHub",
    "warmup_use_smoothness_tv"      : "tv",
    "warmup_use_param_l1"           : "pL1",
}

complete_loss_labels: dict[str, str] = {
    "complete_use_mse_curve"          : "mse",
    "complete_use_l1_curve"           : "l1",
    "complete_use_huber_curve"        : "huber",
    "complete_use_charbonnier_curve"  : "charb",
    "complete_use_cosine_curve"       : "cos",
    "complete_use_spectral_coherence" : "spec",
    "complete_use_ssim_curve"         : "ssim",
    "complete_use_param_l1"           : "pL1",
    "complete_use_param_huber"        : "pHub",
    "complete_use_smoothness_tv"      : "tv",
}

warmup_loss_weight_key: dict[str, str] = {
    "warmup_use_mse_curve"          : "warmup_weight_mse_curve",
    "warmup_use_l1_curve"           : "warmup_weight_l1_curve",
    "warmup_use_huber_curve"        : "warmup_weight_huber_curve",
    "warmup_use_charbonnier_curve"  : "warmup_weight_charbonnier_curve",
    "warmup_use_cosine_curve"       : "warmup_weight_cosine_curve",
    "warmup_use_spectral_coherence" : "warmup_weight_spectral_coh",
    "warmup_use_ssim_curve"         : "warmup_weight_ssim_curve",
    "warmup_use_param_l1"           : "warmup_weight_param_l1",
    "warmup_use_param_huber"        : "warmup_weight_param_huber",
    "warmup_use_smoothness_tv"      : "warmup_weight_smoothness_tv",
}

complete_loss_weight_key: dict[str, str] = {
    "complete_use_mse_curve"          : "complete_weight_mse_curve",
    "complete_use_l1_curve"           : "complete_weight_l1_curve",
    "complete_use_huber_curve"        : "complete_weight_huber_curve",
    "complete_use_charbonnier_curve"  : "complete_weight_charbonnier_curve",
    "complete_use_cosine_curve"       : "complete_weight_cosine_curve",
    "complete_use_spectral_coherence" : "complete_weight_spectral_coh",
    "complete_use_ssim_curve"         : "complete_weight_ssim_curve",
    "complete_use_param_l1"           : "complete_weight_param_l1",
    "complete_use_param_huber"        : "complete_weight_param_huber",
    "complete_use_smoothness_tv"      : "complete_weight_smoothness_tv",
}

loss_weight_key = warmup_loss_weight_key

bool_vars = {
    "warmup_use_mse_curve",         "warmup_use_l1_curve",             "warmup_use_huber_curve",          "warmup_use_charbonnier_curve",
    "warmup_use_cosine_curve",      "warmup_use_spectral_coherence",   "warmup_use_ssim_curve",           "warmup_use_param_l1",
    "warmup_use_param_huber",       "warmup_use_smoothness_tv",        "warmup_use_spectral_coherence",   "warmup_use_ssim_curve",  
    "warmup_use_param_l1",          "warmup_use_param_huber",          "warmup_use_smoothness_tv",
    
    "complete_use_mse_curve",       "complete_use_l1_curve",           "complete_use_huber_curve",        "complete_use_charbonnier_curve",
    "complete_use_cosine_curve",    "complete_use_spectral_coherence", "complete_use_ssim_curve",         "complete_use_param_l1",
    "complete_use_param_huber",     "complete_use_smoothness_tv",      "complete_use_spectral_coherence", "complete_use_ssim_curve", 
    "complete_use_param_l1",        "complete_use_param_huber",        "complete_use_smoothness_tv",
 
    "curriculum_enabled",           "curriculum_reset_early_stopping", "curriculum_reset_lr",      "curriculum_reset_warmup",      "curriculum_reset_optimizer",
}

float_vars = {
    "warmup_weight_mse_curve",      "warmup_weight_l1_curve",        "warmup_weight_huber_curve",    "warmup_weight_charbonnier_curve",
    "warmup_weight_cosine_curve",   "warmup_weight_spectral_coh",    "warmup_weight_ssim_curve",     "warmup_weight_param_l1",
    "warmup_weight_param_huber",    "warmup_weight_smoothness_tv",   "warmup_weight_spectral_coh",   "warmup_weight_ssim_curve",   
    "warmup_weight_param_l1",       "warmup_weight_param_huber",     "warmup_weight_smoothness_tv"
    
    "complete_weight_mse_curve",    "complete_weight_l1_curve",      "complete_weight_huber_curve",   "complete_weight_charbonnier_curve",
    "complete_weight_cosine_curve", "complete_weight_spectral_coh",  "complete_weight_ssim_curve",    "complete_weight_param_l1",
    "complete_weight_param_huber",  "complete_weight_smoothness_tv", "complete_weight_spectral_coh",  "complete_weight_ssim_curve",  
    "complete_weight_param_l1",     "complete_weight_param_huber",   "complete_weight_smoothness_tv",
}

int_vars = {
    "seed", "batch_size", "num_workers", "n_gaussians",
    "epochs", "validation_frequency", "early_stopping_patience",
    "probe_n_batches", "curriculum_swap_epoch",
}

str_vars = {"run_name", "model_name", "logdir", "probe_reference"}

def _read_model_name() -> str:
    src = train_script.read_text(encoding="utf-8")
    
    for line in src.splitlines():
        s = line.lstrip()
        if s.startswith("model_name"):
            return s.split("=")[1].strip().strip('"').strip("'")
    
    return "model"


def _build_run_name(exp: dict, model_name: str) -> str:
    def _phase_segment(label_map: dict[str, str], weight_map: dict[str, str], prefix: str) -> str:
        tokens: list[str] = []
        for use_key, label in label_map.items():
            if exp.get(use_key, False):
                w_key = weight_map[use_key]
                w     = exp.get(w_key, "")
                w_str = f"{w:g}" if isinstance(w, float) else str(w)
                tokens.append(f"{label}{w_str}")
        return (prefix + "-" + "-".join(tokens)) if tokens else ""

    warmup_seg   = _phase_segment(warmup_loss_labels,   warmup_loss_weight_key,   "w")
    complete_seg = _phase_segment(complete_loss_labels, complete_loss_weight_key, "c")

    parts = [model_name]
    if warmup_seg:   parts.append(warmup_seg)
    if complete_seg: parts.append(complete_seg)

    return "_".join(parts)


def _patch_script(src: str, overrides: dict, gpu_id: int, exp_name: str) -> str:
    lines = src.splitlines()
    out   = []
    for line in lines:
        stripped = line.lstrip()

        if stripped.startswith("GPU_ID"):
            out.append(f"GPU_ID = {gpu_id}")
            continue

        matched = False
        if not stripped.rstrip().endswith(","):
            for key in bool_vars | float_vars | int_vars | str_vars:
                if stripped.startswith(key) and key in overrides:
                    indent = line[: len(line) - len(stripped)]
                    val    = overrides[key]
                    if key in str_vars:
                        out.append(f'{indent}{key:<28} = "{val}"')
                    else:
                        out.append(f"{indent}{key:<28} = {val}")
                    matched = True
                    break
        if not matched:
            out.append(line)

    return "\n".join(out)


def _launch(exp: dict, gpu_id: int, model_name: str, logger) -> tuple[str, subprocess.Popen]:
    """Patch single_train.py and spawn it on the given GPU."""
    name    = _build_run_name(exp, model_name)
    src     = train_script.read_text(encoding="utf-8")
    patched = _patch_script(src, {**exp, "run_name": name}, gpu_id, name)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"train_{name}_",
        dir=train_script.parent, delete=False, encoding="utf-8",
    )
    tmp.write(patched)
    tmp.close()
    tmp_path = Path(tmp.name)
    atexit.register(tmp_path.unlink, missing_ok=True)

    proc = subprocess.Popen(
        [sys.executable, str(tmp_path)],
        cwd     = str(train_script.parent),
        stdout  = subprocess.PIPE,
        stderr  = subprocess.STDOUT,
        text    = True,
        bufsize = 1,
    )

    logger.info(f"[GPU {gpu_id}] Started  → {name}")
    return name, proc


def main() -> None:
    repo_root = train_script.resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tools.logger import Logger
    logger = Logger(log_dir="", name="batch_train")

    model_name = _read_model_name()

    logger.section("Batch train")
    logger.kv_table({
        "Model"       : model_name,
        "Experiments" : len(experiments),
        "GPUs"        : gpu_ids,
    }, title="Configuration")

    # --- scheduler state -------------------------------------------------------
    queue        = list(experiments)           # experiments still waiting
    free_gpus    = list(gpu_ids)               # GPUs available right now
    # fd  → (name, proc, gpu_id)
    fd_info:  dict[int, tuple[str, subprocess.Popen, int]] = {}
    all_procs: list[tuple[str, subprocess.Popen]]          = []  # for final summary

    logger.section("Scheduling")

    # Seed: fill every GPU with an experiment from the queue
    while queue and free_gpus:
        gpu_id = free_gpus.pop(0)
        exp    = queue.pop(0)
        name, proc = _launch(exp, gpu_id, model_name, logger)
        fd_info[proc.stdout.fileno()] = (name, proc, gpu_id)
        all_procs.append((name, proc))

    # Main loop: drain output and schedule next experiment whenever a GPU is free
    while fd_info:
        readable, _, _ = select.select(list(fd_info.keys()), [], [], 0.5)

        for fd in readable:
            name, proc, gpu_id = fd_info[fd]
            line = proc.stdout.readline()
            if line:
                logger.info(f"[{name}] {line.rstrip()}")
            elif proc.poll() is not None:
                # Drain any remaining output
                for remaining in proc.stdout:
                    logger.info(f"[{name}] {remaining.rstrip()}")

                rc = proc.returncode
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                logger.info(f"[GPU {gpu_id}] Finished → {name}  [{status}]")

                del fd_info[fd]

                # Launch the next queued experiment on the freed GPU
                if queue:
                    next_exp = queue.pop(0)
                    n2, p2   = _launch(next_exp, gpu_id, model_name, logger)
                    fd_info[p2.stdout.fileno()] = (n2, p2, gpu_id)
                    all_procs.append((n2, p2))
                else:
                    free_gpus.append(gpu_id)

    logger.section("Summary")
    rows = [
        {"Experiment": name, "Status": "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"}
        for name, proc in all_procs
    ]
    logger.metrics_table(rows, columns=["Experiment", "Status"])

    logger.close()


if __name__ == "__main__":
    main()
