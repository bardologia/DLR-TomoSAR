import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# ── Academic figure style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':   'dejavuserif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'axes.linewidth':     0.8,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.top':          True,
    'ytick.right':        True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})


_GIF_TOMO = None
_GIF_SAR = None
_GIF_CMAP = None
_GIF_VMIN = None
_GIF_VMAX = None
_GIF_SAR_VMIN = None
_GIF_SAR_VMAX = None
_GIF_N_AZ = None
_GIF_N_RG = None
_GIF_N_H = None
_GIF_DPI = None
_GIF_FRAMES_DIR = None


def _init_gif_worker(tomo_abs, sar_image, cmap, vmin, vmax,
                     sar_vmin, sar_vmax, n_azimuth, n_range, n_height,
                     dpi, frames_dir):
    global _GIF_TOMO, _GIF_SAR, _GIF_CMAP, _GIF_VMIN, _GIF_VMAX
    global _GIF_SAR_VMIN, _GIF_SAR_VMAX, _GIF_N_AZ, _GIF_N_RG, _GIF_N_H
    global _GIF_DPI, _GIF_FRAMES_DIR

    _GIF_TOMO = tomo_abs
    _GIF_SAR = sar_image
    _GIF_CMAP = cmap
    _GIF_VMIN = vmin
    _GIF_VMAX = vmax
    _GIF_SAR_VMIN = sar_vmin
    _GIF_SAR_VMAX = sar_vmax
    _GIF_N_AZ = n_azimuth
    _GIF_N_RG = n_range
    _GIF_N_H = n_height
    _GIF_DPI = dpi
    _GIF_FRAMES_DIR = frames_dir


def _render_gif_frame(rn_idx):
    fig, (ax_sar, ax_slice) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1]}
    )
    plt.subplots_adjust(wspace=0.30)

    ax_sar.imshow(_GIF_SAR.T, origin='lower', cmap=_GIF_CMAP, aspect='auto',
                  vmin=_GIF_SAR_VMIN, vmax=_GIF_SAR_VMAX,
                  extent=[0, _GIF_N_AZ, 0, _GIF_N_RG])
    ax_sar.set_xlabel('Azimuth (pixels)')
    ax_sar.set_ylabel('Range (pixels)')
    ax_sar.axhline(rn_idx, color='white', lw=1.5, ls='--')

    slide = _GIF_TOMO[:, :, rn_idx]
    im = ax_slice.imshow(slide, origin='lower', cmap=_GIF_CMAP, vmin=_GIF_VMIN, vmax=_GIF_VMAX, aspect='auto', extent=[0, _GIF_N_AZ, 0, _GIF_N_H])
    ax_slice.set_xlabel('Azimuth (pixels)')
    ax_slice.set_ylabel('Height (bins)')
    divider = make_axes_locatable(ax_slice)
    cax = divider.append_axes('right', size='3.5%', pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Amplitude (a.u.)')

    frame_path = Path(_GIF_FRAMES_DIR) / f"frame_{rn_idx:05d}.png"
    fig.savefig(frame_path, dpi=_GIF_DPI)
    plt.close(fig)
    return str(frame_path)


def make_tomo_gif(
    tomo_file_path : str,
    output_dir     : str = None,
    gif_name       : str = None,
    slice_step     : int = 1,
    fps            : int = 10,
    dpi            : int = 100,
    cmap           : str = 'jet',
    vmin           : float = None,
    vmax           : float = None,
    keep_frames    : bool = False,
    n_workers      : int = None,
):

    tomo_file_path = Path(tomo_file_path)
    tomo_name = tomo_file_path.stem
    if output_dir is None:
        # Automatic output location:
        # - if input is in a tomogram folder, save outputs one level above
        # - otherwise, save next to the input file
        input_parent = tomo_file_path.parent
        if input_parent.name.lower() in {'tomograms', 'tomogram'}:
            output_dir = input_parent.parent
        else:
            output_dir = input_parent
    else:
        output_dir = Path(output_dir)

    save_dir = output_dir / 'gifs' / tomo_name
    save_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = save_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if gif_name is None:
        gif_name = "range_sweep.gif"
    gif_path = save_dir / gif_name

    print(f"Loading tomogram from: {tomo_file_path}")
    h5f = h5py.File(tomo_file_path, 'r')
    tomo = np.abs(h5f['tomogram'][:])
    h5f.close()

    n_height, n_azimuth, n_range = tomo.shape
    print(f"Tomogram shape: height={n_height}, azimuth={n_azimuth}, range={n_range}")

    if vmin is None or vmax is None:
        sample_indices = np.linspace(0, n_range - 1, min(20, n_range), dtype=int)
        sample_vals = []
        for idx in sample_indices:
            sample_vals.append(tomo[:, :, int(idx)].ravel())
        sample_vals = np.concatenate(sample_vals)
        if vmin is None:
            vmin = float(np.nanpercentile(sample_vals, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(sample_vals, 98))
        print(f"Colour scale: vmin = {vmin:.4f}, vmax = {vmax:.4f} (a.u.)")

    n_h_samples = min(40, n_height)
    h_samples   = np.linspace(0, n_height - 1, n_h_samples, dtype=int)
    sar_image   = np.zeros((n_azimuth, n_range), dtype=np.float64)
    
    for h_i in h_samples:
        sar_image += tomo[int(h_i), :, :]
    sar_image /= n_h_samples

    sar_vmin = np.nanpercentile(sar_image, 2)
    sar_vmax = np.nanpercentile(sar_image, 98)

    slice_indices = list(range(0, n_range, slice_step))
    n_frames = len(slice_indices)
    print(f"Generating {n_frames} frames (step={slice_step})…")

    if n_workers is None:
        n_workers_eff = max(1, (os.cpu_count() or 1) - 1)
    else:
        n_workers_eff = max(1, int(n_workers))
    n_workers_eff = min(n_workers_eff, n_frames)
    print(f"Rendering frames with {n_workers_eff} worker(s)")

    frame_paths = []
    if n_workers_eff == 1:
        _init_gif_worker(
            tomo, sar_image, cmap, vmin, vmax,
            sar_vmin, sar_vmax, n_azimuth, n_range, n_height,
            dpi, str(frames_dir)
        )
        for rn_idx in slice_indices:
            frame_paths.append(Path(_render_gif_frame(rn_idx)))
    else:
        ctx = mp.get_context('spawn')
        chunksize = max(1, n_frames // (n_workers_eff * 4))
        with ProcessPoolExecutor(
            max_workers=n_workers_eff,
            mp_context=ctx,
            initializer=_init_gif_worker,
            initargs=(
                tomo, sar_image, cmap, vmin, vmax,
                sar_vmin, sar_vmax, n_azimuth, n_range, n_height,
                dpi, str(frames_dir)
            ),
        ) as ex:
            frame_paths = [Path(p) for p in ex.map(_render_gif_frame, slice_indices, chunksize=chunksize)]

    frame_paths.sort(key=lambda p: p.name)

    print(f"Assembling GIF → {gif_path}")
    duration_ms = int(1000 / fps)
    frames = [Image.open(fp) for fp in frame_paths]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
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
    tomo_path = "/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/tomograms/tomo_17sartom0102_Lhv_MSF_w10_20.hd5"

    make_tomo_gif(
        tomo_file_path = tomo_path,
        slice_step     = 1,       
        fps            = 50,             
        dpi            = 100,
        cmap           = 'jet',
        keep_frames    = False,  
        n_workers      = 32,
    )
