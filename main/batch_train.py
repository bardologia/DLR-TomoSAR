from __future__ import annotations

import select
import subprocess
import sys
import tempfile
import atexit
from pathlib import Path

gpu_ids = [0, 1, 2, 3]

experiments = [
    {
        "curriculum_enabled"  : False,
        "warmup_use_param_l1" : True,  "warmup_weight_param_l1"  : 1.0,
        "complete_use_param_l1": True, "complete_weight_param_l1": 1.0,
    },
    {
        "curriculum_enabled"   : False,
        "warmup_use_mse_curve" : True,  "warmup_weight_mse_curve" : 0.5,
        "warmup_use_param_l1"  : True,  "warmup_weight_param_l1"  : 1.0,
        "complete_use_param_l1": True,  "complete_weight_param_l1": 1.0,
    },
    {
        "curriculum_enabled"       : False,
        "warmup_use_mse_curve"     : True,  "warmup_weight_mse_curve"    : 0.5,
        "warmup_use_cosine_curve"  : True,  "warmup_weight_cosine_curve" : 0.1,
        "warmup_use_param_l1"      : True,  "warmup_weight_param_l1"     : 1.0,
        "complete_use_param_l1"    : True,  "complete_weight_param_l1"   : 1.0,
    },
    {
        "curriculum_enabled"            : False,
        "warmup_use_mse_curve"          : True,  "warmup_weight_mse_curve"    : 0.5,
        "warmup_use_cosine_curve"       : True,  "warmup_weight_cosine_curve" : 0.1,
        "warmup_use_spectral_coherence" : True,  "warmup_weight_spectral_coh" : 0.4,
        "warmup_use_param_l1"           : True,  "warmup_weight_param_l1"     : 1.0,
        "complete_use_param_l1"         : True,  "complete_weight_param_l1"   : 1.0,
    },
]

train_script = Path(__file__).resolve().parent / "single_train.py"

loss_labels: dict[str, str] = {
    "warmup_use_mse_curve"          : "mse",
    "warmup_use_l1_curve"           : "l1",
    "warmup_use_huber_curve"        : "huber",
    "warmup_use_charbonnier_curve"  : "charb",
    "warmup_use_cosine_curve"       : "cos",
    "warmup_use_spectral_coherence" : "spec",
    "warmup_use_ssim_curve"         : "ssim",
    "warmup_use_param_l1"           : "pL1",
    "warmup_use_param_huber"        : "pHub",
    "warmup_use_smoothness_tv"      : "tv",
}

loss_weight_key: dict[str, str] = {
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

bool_vars = {
    "warmup_use_mse_curve",         "warmup_use_l1_curve",            "warmup_use_huber_curve",   "warmup_use_charbonnier_curve",
    "warmup_use_cosine_curve",      "warmup_use_spectral_coherence",  "warmup_use_ssim_curve",    "warmup_use_param_l1",
    "warmup_use_param_huber",       "warmup_use_smoothness_tv",
    
    "complete_use_mse_curve",       "complete_use_l1_curve",           "complete_use_huber_curve", "complete_use_charbonnier_curve",
    "complete_use_cosine_curve",    "complete_use_spectral_coherence", "complete_use_ssim_curve",  "complete_use_param_l1",
    "complete_use_param_huber",     "complete_use_smoothness_tv",
 
    "curriculum_enabled",           "curriculum_reset_early_stopping", "curriculum_reset_lr",      "curriculum_reset_warmup",
}

float_vars = {
    "warmup_weight_mse_curve",      "warmup_weight_l1_curve",        "warmup_weight_huber_curve",  "warmup_weight_charbonnier_curve",
    "warmup_weight_cosine_curve",   "warmup_weight_spectral_coh",    "warmup_weight_ssim_curve",   "warmup_weight_param_l1",
    "warmup_weight_param_huber",    "warmup_weight_smoothness_tv",
    
    "complete_weight_mse_curve",    "complete_weight_l1_curve",      "complete_weight_huber_curve", "complete_weight_charbonnier_curve",
    "complete_weight_cosine_curve", "complete_weight_spectral_coh",  "complete_weight_ssim_curve",  "complete_weight_param_l1",
    "complete_weight_param_huber",  "complete_weight_smoothness_tv",
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
    parts = [model_name]
   
    for use_key, label in loss_labels.items():
        if exp.get(use_key, False):
            w_key = loss_weight_key[use_key]
            w     = exp.get(w_key, "")
            w_str = f"{w:g}" if isinstance(w, float) else str(w)
            parts.append(f"{label}{w_str}")
   
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


def main() -> None:
    repo_root = train_script.resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tools.logger import Logger
    logger = Logger(log_dir="", name="batch_train")

    assert len(experiments) <= len(gpu_ids), (f"Not enough GPU IDs ({len(gpu_ids)}) for {len(experiments)} experiments.")

    model_name = _read_model_name()
    processes: list[tuple[str, subprocess.Popen]] = []

    logger.section("Batch train")
    logger.kv_table({
        "Model"       : model_name,
        "Experiments" : len(experiments),
        "GPUs"        : gpu_ids[: len(experiments)],
    }, title="Configuration")

    logger.section("Launching")
    for gpu_id, exp in zip(gpu_ids, experiments):
        name    = _build_run_name(exp, model_name)
        src     = train_script.read_text(encoding="utf-8")
        patched = _patch_script(src, {**exp, "run_name": name}, gpu_id, name)

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix=f"train_{name}_", dir=train_script.parent, delete=False, encoding="utf-8",)
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
       
        processes.append((name, proc))
        logger.info(f"[GPU {gpu_id}] {name}")

    fds   = {p.stdout.fileno(): (name, p) for name, p in processes}
    alive = set(fds.keys())

    while alive:
        readable, _, _ = select.select(list(alive), [], [], 0.5)
        for fd in readable:
            name, proc = fds[fd]
            line = proc.stdout.readline()
            if line:
                logger.info(f"[{name}] {line.rstrip()}")
            elif proc.poll() is not None:
                alive.discard(fd)

    logger.section("Summary")
    rows = [{"Experiment": name, "Status": "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"} for name, proc in processes]
    logger.metrics_table(rows, columns=["Experiment", "Status"])

    logger.close()


if __name__ == "__main__":
    main()
