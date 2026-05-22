from __future__ import annotations

import subprocess
import sys
import tempfile
import atexit
from pathlib import Path

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"

LOSS_LABELS: dict[str, str] = {
    "use_mse_curve":          "mse",
    "use_l1_curve":           "l1",
    "use_huber_curve":        "huber",
    "use_charbonnier_curve":  "charb",
    "use_cosine_curve":       "cos",
    "use_spectral_coherence": "spec",
    "use_ssim_curve":         "ssim",
    "use_param_l1":           "pL1",
    "use_param_huber":        "pHub",
    "use_smoothness_tv":      "tv",
}

LOSS_WEIGHT_KEY: dict[str, str] = {
    "use_mse_curve":          "weight_mse_curve",
    "use_l1_curve":           "weight_l1_curve",
    "use_huber_curve":        "weight_huber_curve",
    "use_charbonnier_curve":  "weight_charbonnier_curve",
    "use_cosine_curve":       "weight_cosine_curve",
    "use_spectral_coherence": "weight_spectral_coh",
    "use_ssim_curve":         "weight_ssim_curve",
    "use_param_l1":           "weight_param_l1",
    "use_param_huber":        "weight_param_huber",
    "use_smoothness_tv":      "weight_smoothness_tv",
}


def _read_model_name() -> str:
    src = TRAIN_SCRIPT.read_text(encoding="utf-8")
    for line in src.splitlines():
        s = line.lstrip()
        if s.startswith("model_name"):
            return s.split("=")[1].strip().strip('"').strip("'")
    return "model"


def _build_run_name(exp: dict, model_name: str) -> str:
    parts = [model_name]
    for use_key, label in LOSS_LABELS.items():
        if exp.get(use_key, False):
            w_key = LOSS_WEIGHT_KEY[use_key]
            w     = exp.get(w_key, "")
            w_str = f"{w:g}" if isinstance(w, float) else str(w)
            parts.append(f"{label}{w_str}")
    return "_".join(parts)


EXPERIMENTS = [
    {
        "use_mse_curve":            False, "weight_mse_curve":         0.0,
        "use_l1_curve":             False, "weight_l1_curve":          0.0,
        "use_huber_curve":          False, "weight_huber_curve":       0.0,
        "use_charbonnier_curve":    False,  "weight_charbonnier_curve": 0.0,
        "use_cosine_curve":         False, "weight_cosine_curve":      0.0,
        "use_spectral_coherence":   False, "weight_spectral_coh":      0.0,
        "use_ssim_curve":           False, "weight_ssim_curve":        0.0,
        "use_param_l1":             True,  "weight_param_l1":          1.0,
        "use_param_huber":          False, "weight_param_huber":       0.0,
        "use_smoothness_tv":        False , "weight_smoothness_tv":    0.0,
    },
    {
        "use_mse_curve":            False, "weight_mse_curve":         0.0,
        "use_l1_curve":             False, "weight_l1_curve":          0.0,
        "use_huber_curve":          False, "weight_huber_curve":       0.0,
        "use_charbonnier_curve":    True,  "weight_charbonnier_curve": 1.0,
        "use_cosine_curve":         False, "weight_cosine_curve":      0.0,
        "use_spectral_coherence":   False, "weight_spectral_coh":      0.0,
        "use_ssim_curve":           False, "weight_ssim_curve":        0.0,
        "use_param_l1":             True,  "weight_param_l1":          1.0,
        "use_param_huber":          False, "weight_param_huber":       0.0,
        "use_smoothness_tv":        False , "weight_smoothness_tv":    0.0,
    },
     {
        "use_mse_curve":            False, "weight_mse_curve":         0.0,
        "use_l1_curve":             False, "weight_l1_curve":          0.0,
        "use_huber_curve":          False, "weight_huber_curve":       0.0,
        "use_charbonnier_curve":    True,  "weight_charbonnier_curve": 0.5,
        "use_cosine_curve":         False, "weight_cosine_curve":      0.0,
        "use_spectral_coherence":   False, "weight_spectral_coh":      0.0,
        "use_ssim_curve":           False, "weight_ssim_curve":        0.0,
        "use_param_l1":             True,  "weight_param_l1":          1.0,
        "use_param_huber":          False, "weight_param_huber":       0.0,
        "use_smoothness_tv":        False , "weight_smoothness_tv":    0.0,
    },
    {
        "use_mse_curve":            False, "weight_mse_curve":         0.0,
        "use_l1_curve":             False, "weight_l1_curve":          0.0,
        "use_huber_curve":          False, "weight_huber_curve":       0.0,
        "use_charbonnier_curve":    False, "weight_charbonnier_curve": 0.0,
        "use_cosine_curve":         True,  "weight_cosine_curve":      0.2,
        "use_spectral_coherence":   False, "weight_spectral_coh":      0.0,
        "use_ssim_curve":           False, "weight_ssim_curve":        0.0,
        "use_param_l1":             True,  "weight_param_l1":          1.0,
        "use_param_huber":          False, "weight_param_huber":       0.0,
        "use_smoothness_tv":        False , "weight_smoothness_tv":    0.0,
    },
]

BOOL_VARS = {
    "use_mse_curve",
    "use_l1_curve",
    "use_huber_curve",
    "use_charbonnier_curve",
    "use_cosine_curve",
    "use_spectral_coherence",
    "use_ssim_curve",
    "use_param_l1",
    "use_param_huber",
    "use_smoothness_tv",
}

FLOAT_VARS = {
    "weight_mse_curve",
    "weight_l1_curve",
    "weight_huber_curve",       "huber_delta",
    "weight_charbonnier_curve", "charbonnier_eps",
    "weight_cosine_curve",
    "weight_spectral_coh",      "spectral_coh_window",
    "weight_ssim_curve",        "ssim_window_size",   "ssim_sigma",
    "ssim_data_range",          "ssim_k1",            "ssim_k2",
    "weight_param_l1",
    "weight_param_huber",       "param_huber_delta",
    "weight_smoothness_tv",
}

STR_VARS = {"ssim_axis", "param_match", "run_name"}


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
            for key in BOOL_VARS | FLOAT_VARS | STR_VARS:
                if stripped.startswith(key) and key in overrides:
                    indent = line[: len(line) - len(stripped)]
                    val    = overrides[key]
                    if key in STR_VARS:
                        out.append(f'{indent}{key:<28} = "{val}"')
                    else:
                        out.append(f"{indent}{key:<28} = {val}")
                    matched = True
                    break
        if not matched:
            out.append(line)

    return "\n".join(out)


GPU_IDS = [0, 1, 2, 3]


def main() -> None:
    assert len(EXPERIMENTS) <= len(GPU_IDS), (
        f"Not enough GPU IDs ({len(GPU_IDS)}) for {len(EXPERIMENTS)} experiments."
    )

    model_name = _read_model_name()
    processes: list[tuple[str, subprocess.Popen]] = []

    for gpu_id, exp in zip(GPU_IDS, EXPERIMENTS):
        name = _build_run_name(exp, model_name)
        print(f"[LAUNCH] GPU {gpu_id}  →  {name}")

        src     = TRAIN_SCRIPT.read_text(encoding="utf-8")
        patched = _patch_script(src, {**exp, "run_name": name}, gpu_id, name)

        # Write to a real temp file so __file__ is defined inside the script
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix=f"train_{name}_",
            dir=TRAIN_SCRIPT.parent, delete=False, encoding="utf-8",
        )
        tmp.write(patched)
        tmp.close()
        tmp_path = Path(tmp.name)
        atexit.register(tmp_path.unlink, missing_ok=True)

        proc = subprocess.Popen(
            [sys.executable, str(tmp_path)],
            cwd    = str(TRAIN_SCRIPT.parent),
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text   = True,
            bufsize= 1,
        )
        processes.append((name, proc))

    import select, io
    fds = {p.stdout.fileno(): (name, p) for name, p in processes}
    alive = set(fds.keys())

    while alive:
        readable, _, _ = select.select(list(alive), [], [], 0.5)
        for fd in readable:
            name, proc = fds[fd]
            line = proc.stdout.readline()
            if line:
                print(f"[{name}] {line}", end="")
            elif proc.poll() is not None:
                alive.discard(fd)

    print(f"\n{'='*60}")
    for name, proc in processes:
        rc = proc.returncode
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {name:<30}  {status}")
    print("="*60)


if __name__ == "__main__":
    main()
