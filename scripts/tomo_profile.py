import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def load_tomogram(tomo_file_path: str, dataset_key: str = 'tomogram'):
    tomo_file_path = Path(tomo_file_path)
    h5f  = h5py.File(tomo_file_path, 'r')
    tomo = h5f[dataset_key]
    n_height, n_azimuth, n_range = tomo.shape
    print(f"Loaded tomogram from: {tomo_file_path}")
    print(f"  shape: height={n_height}, azimuth={n_azimuth}, range={n_range}")
    return h5f, tomo


def estimate_color_limits(tomo, n_samples: int = 20):
    n_range = tomo.shape[2]
    sample_indices = np.linspace(0, n_range - 1, min(n_samples, n_range), dtype=int)
    global_min =  np.inf
    global_max = -np.inf
    for idx in sample_indices:
        amp = np.abs(tomo[:, :, int(idx)])
        global_min = min(global_min, np.nanmin(amp))
        global_max = max(global_max, np.nanmax(amp))
    
    return float(global_min), float(global_max)


def compute_sar_image(tomo, n_h_samples: int = 40):
    n_height, n_azimuth, n_range = tomo.shape
    n_h_samples = min(n_h_samples, n_height)
    h_indices   = np.linspace(0, n_height - 1, n_h_samples, dtype=int)
    sar_image   = np.zeros((n_azimuth, n_range), dtype=np.float64)
    for h_i in h_indices:
        sar_image += np.abs(tomo[int(h_i), :, :])
    sar_image /= n_h_samples
    return sar_image


def plot_single_slice(
    tomo,
    range_idx : int,
    cmap      : str   = 'jet',
    vmin      : float = None,
    vmax      : float = None,
    title     : str   = None,
    ax        : plt.Axes = None,
    figsize   : tuple = (8, 6),
):
    n_height, n_azimuth, _ = tomo.shape
    slide = np.abs(tomo[:, :, range_idx])

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if vmin is None or vmax is None:
        _vmin, _vmax = estimate_color_limits(tomo)
        vmin = vmin if vmin is not None else _vmin
        vmax = vmax if vmax is not None else _vmax

    im = ax.imshow(
        slide.T, origin='lower', cmap=cmap,
        vmin=vmin, vmax=vmax, aspect='auto',
        extent=[0, n_height, 0, n_azimuth],
    )
    ax.set_xlabel("Height")
    ax.set_ylabel("Azimuth")
    ax.set_title(title or f"Range slice {range_idx}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig, ax


def plot_slice_with_sar(
    tomo,
    range_idx   : int,
    sar_image   : np.ndarray = None,
    cmap        : str   = 'jet',
    vmin        : float = None,
    vmax        : float = None,
    figsize     : tuple = (15, 6),
):
    n_height, n_azimuth, n_range = tomo.shape

    if sar_image is None:
        sar_image = compute_sar_image(tomo)

    if vmin is None or vmax is None:
        _vmin, _vmax = estimate_color_limits(tomo)
        vmin = vmin if vmin is not None else _vmin
        vmax = vmax if vmax is not None else _vmax

    sar_vmin = np.nanpercentile(sar_image, 2)
    sar_vmax = np.nanpercentile(sar_image, 98)

    fig, (ax_sar, ax_slice) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1.2, 1]},
    )
    plt.subplots_adjust(wspace=0.30)

    # SAR overview
    ax_sar.imshow(
        sar_image, origin='lower', cmap=cmap, aspect='auto',
        vmin=sar_vmin, vmax=sar_vmax,
        extent=[0, n_range, 0, n_azimuth],
    )
    ax_sar.set_title("SAR image (azimuth × range)", fontsize=12)
    ax_sar.set_xlabel("Range")
    ax_sar.set_ylabel("Azimuth")
    ax_sar.axvline(range_idx, color='white', lw=1.5, ls='--')

    # Tomo slice
    slide = np.abs(tomo[:, :, range_idx])
    im = ax_slice.imshow(
        slide.T, origin='lower', cmap=cmap,
        vmin=vmin, vmax=vmax, aspect='auto',
        extent=[0, n_height, 0, n_azimuth],
    )
    ax_slice.set_title(f"Range slice {range_idx}/{n_range - 1}", fontsize=13)
    ax_slice.set_xlabel("Height")
    ax_slice.set_ylabel("Azimuth")
    fig.colorbar(im, ax=ax_slice, fraction=0.046, pad=0.04)

    return fig, (ax_sar, ax_slice)


def make_tomo_gif(
    tomo_file_path : str,
    output_dir     : str   = None,
    gif_name       : str   = None,
    slice_step     : int   = 1,
    fps            : int   = 10,
    dpi            : int   = 100,
    cmap           : str   = 'jet',
    vmin           : float = None,
    vmax           : float = None,
    keep_frames    : bool  = False,
):
    tomo_file_path = Path(tomo_file_path)
    if output_dir is None:
        output_dir = tomo_file_path.parent
    else:
        output_dir = Path(output_dir)

    frames_dir = output_dir / f"{tomo_file_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if gif_name is None:
        gif_name = f"{tomo_file_path.stem}_slices.gif"
    gif_path = output_dir / gif_name

    # ---- load ----
    h5f, tomo = load_tomogram(tomo_file_path)
    n_height, n_azimuth, n_range = tomo.shape

    # ---- colour limits ----
    if vmin is None or vmax is None:
        _vmin, _vmax = estimate_color_limits(tomo)
        vmin = vmin if vmin is not None else _vmin
        vmax = vmax if vmax is not None else _vmax
        print(f"Colour scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

    # ---- SAR overview ----
    sar_image = compute_sar_image(tomo)
    sar_vmin  = np.nanpercentile(sar_image, 2)
    sar_vmax  = np.nanpercentile(sar_image, 98)

    # ---- frames ----
    slice_indices = list(range(0, n_range, slice_step))
    n_frames = len(slice_indices)
    print(f"Generating {n_frames} frames (step={slice_step})…")

    frame_paths = []
    fig, (ax_sar, ax_slice) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.2, 1]},
    )
    plt.subplots_adjust(wspace=0.30)

    ax_sar.imshow(
        sar_image, origin='lower', cmap=cmap, aspect='auto',
        vmin=sar_vmin, vmax=sar_vmax,
        extent=[0, n_range, 0, n_azimuth],
    )
    ax_sar.set_title("SAR image (azimuth × range)", fontsize=12)
    ax_sar.set_xlabel("Range")
    ax_sar.set_ylabel("Azimuth")
    sweep_line = ax_sar.axvline(0, color='white', lw=1.5, ls='--')

    for i, rn_idx in enumerate(slice_indices):
        ax_slice.clear()
        slide = np.abs(tomo[:, :, rn_idx])
        im = ax_slice.imshow(
            slide.T, origin='lower', cmap=cmap,
            vmin=vmin, vmax=vmax, aspect='auto',
            extent=[0, n_height, 0, n_azimuth],
        )
        ax_slice.set_title(f"Range slice {rn_idx}/{n_range - 1}", fontsize=13)
        ax_slice.set_xlabel("Height")
        ax_slice.set_ylabel("Azimuth")

        if i == 0:
            fig.colorbar(im, ax=ax_slice, fraction=0.046, pad=0.04)
        else:
            im.set_clim(vmin, vmax)

        sweep_line.set_xdata([rn_idx, rn_idx])

        frame_path = frames_dir / f"frame_{rn_idx:05d}.png"
        fig.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        frame_paths.append(frame_path)

    plt.close(fig)
    h5f.close()

    # ---- assemble GIF ----
    print(f"Assembling GIF → {gif_path}")
    duration_ms = int(1000 / fps)
    frames = [Image.open(fp) for fp in frame_paths]
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0,
    )
    print(f"GIF saved: {gif_path}  ({len(frames)} frames, {fps} fps)")

    if not keep_frames:
        for fp in frame_paths:
            fp.unlink()
        frames_dir.rmdir()
        print("Temporary frames removed.")
    else:
        print(f"Frames kept in: {frames_dir}")

    return str(gif_path)


if __name__ == "__main__":
    tomo_path = "/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/tomo_17sartom0102_Lhv_MSF_Test_withVictor2.hd5"

    # --- Example 1: single profile ------------------------------------------
    h5f, tomo = load_tomogram(tomo_path)
    fig, ax = plot_single_slice(tomo, range_idx=100, cmap='jet')
    fig.savefig("single_slice.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Example 2: profile with SAR context --------------------------------
    fig, _ = plot_slice_with_sar(tomo, range_idx=100, cmap='jet')
    fig.savefig("slice_with_sar.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    h5f.close()

    # --- Example 3: full GIF ------------------------------------------------
    make_tomo_gif(
        tomo_file_path = tomo_path,
        slice_step     = 1,
        fps            = 50,
        dpi            = 100,
        cmap           = 'jet',
        keep_frames    = False,
    )
