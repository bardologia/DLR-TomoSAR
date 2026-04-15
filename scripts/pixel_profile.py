import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from pathlib import Path


def multi_gaussian(x, *p):
    y       = np.zeros_like(x, dtype=np.float64)
    n_gauss = len(p) // 3
    
    for k in range(n_gauss):
        mu, sigma, A  = p[3*k], p[3*k+1], p[3*k+2]
        y            += A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    
    return y


def single_gaussian(x, mu, sigma, A):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def estimate_initial_params(x, y, n_gaussians):
    y_smooth    = np.convolve(y, np.ones(5) / 5, mode='same')
    sigma_guess = (x[-1] - x[0]) / (4.0 * n_gaussians)
    y_work      = y_smooth.copy()
    p0 = []

    for _ in range(n_gaussians):
        idx = np.argmax(y_work)
        mu  = x[idx]
        A   = y_work[idx]
        p0.extend([mu, sigma_guess, max(A, 1e-10)])
        mask = np.abs(x - mu) < 2 * sigma_guess
        y_work[mask] = 0.0

    return p0


def fit_profile(x, profile, n_gaussians=2, max_iter=5000):
    n_params = 3 * n_gaussians
    y        = np.abs(profile).astype(np.float64)
    p0       = estimate_initial_params(x, y, n_gaussians)

    x_lo, x_hi = x[0], x[-1]
    lower = []
    upper = []
    for _ in range(n_gaussians):
        lower.extend([x_lo, 1e-6, 0.0])
        upper.extend([x_hi, (x_hi - x_lo), np.inf])

    try:
        popt, _ = curve_fit(
            multi_gaussian, x, y, p0=p0,
            bounds=(lower, upper),
            maxfev=max_iter,
        )
        return popt, True
    except (RuntimeError, ValueError):
        return np.full(n_params, np.nan), False


def load_tomogram_slice(tomo_file, rn_idx, height_axis_range):
    print(f"Loading tomogram slice rn_idx={rn_idx}:")
    with h5py.File(tomo_file, 'r') as h5f:
        slide = h5f['tomogram'][:, :, rn_idx]
    
    slide_abs = np.abs(slide)
    n_height, n_azimuth = slide_abs.shape
    print(f"  Slice shape: height={n_height}, azimuth={n_azimuth}")
    x = np.linspace(height_axis_range[0], height_axis_range[1], n_height)
    
    return slide_abs, x


def fit_all_profiles(slide_abs, x, n_gaussians):
    n_height, n_azimuth = slide_abs.shape
    n_params = 3 * n_gaussians
    params = np.full((n_azimuth, n_params), np.nan)
    fit_success = np.zeros(n_azimuth, dtype=bool)

    print(f"Fitting {n_gaussians}-Gaussian to each pixel")
    for az in range(n_azimuth):
        popt, ok = fit_profile(x, slide_abs[:, az], n_gaussians=n_gaussians)
        params[az] = popt
        fit_success[az] = ok
        if (az + 1) % 200 == 0 or az == n_azimuth - 1:
            print(f"  [{az+1}/{n_azimuth}] fitted  "
                  f"(success: {fit_success[:az+1].sum()}/{az+1})")

    n_ok = fit_success.sum()
    print(f"Fitting complete: {n_ok}/{n_azimuth} profiles converged "
          f"({100*n_ok/n_azimuth:.1f}%)")
    return params, fit_success


def save_fitted_params(output_dir, rn_idx, n_gaussians, params, fit_success, x, tomo_file):
    out_h5 = output_dir / f'pixel_profiles_rn{rn_idx}_{n_gaussians}g.hd5'
    print(f"Saving fitted parameters → {out_h5}")
    param_names = ', '.join(f'mu{k+1}, sigma{k+1}, A{k+1}' for k in range(n_gaussians))
    with h5py.File(out_h5, 'w') as hf:
        hf.create_dataset('params', data=params)
        hf.create_dataset('fit_success', data=fit_success)
        hf.create_dataset('height_axis', data=x)
        hf.attrs['rn_idx'] = rn_idx
        hf.attrs['n_gaussians'] = n_gaussians
        hf.attrs['source_file'] = tomo_file
        hf.attrs['param_order'] = param_names


def reconstruct_slice(slide_abs, x, params, fit_success, n_gaussians):
    print(f"Reconstructing slice from fitted {n_gaussians}-Gaussians ...")
    slide_recon = np.zeros_like(slide_abs)
    for az in range(slide_abs.shape[1]):
        if fit_success[az]:
            slide_recon[:, az] = multi_gaussian(x, *params[az])
    
    return slide_recon


def plot_slice_comparison(slide_abs, slide_recon, x, rn_idx, output_dir):
    n_azimuth = slide_abs.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    im0 = axes[0].imshow(slide_abs, origin='lower', cmap='jet', aspect='auto', extent=[0, n_azimuth, x[0], x[-1]])
    axes[0].set_title('Original Tomogram Slice')
    axes[0].set_xlabel('Azimuth pixel')
    axes[0].set_ylabel('Height [m]')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(slide_recon, origin='lower', cmap='jet', aspect='auto', extent=[0, n_azimuth, x[0], x[-1]])
    axes[1].set_title(f'Reconstructed ({slide_recon.shape[1]}-Gauss)')
    axes[1].set_xlabel('Azimuth pixel')
    axes[1].set_ylabel('Height [m]')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    residual = slide_abs - slide_recon
    im2 = axes[2].imshow(residual, origin='lower', cmap='RdBu_r', aspect='auto', extent=[0, n_azimuth, x[0], x[-1]])
    axes[2].set_title('Residual (Original − Reconstructed)')
    axes[2].set_xlabel('Azimuth pixel')
    axes[2].set_ylabel('Height [m]')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(f'Range index = {rn_idx}', fontsize=14)
    fig.savefig(output_dir / f'slice_comparison_rn{rn_idx}.png', dpi=150)
    return fig


def plot_parameter_maps(params, fit_success, n_gaussians, rn_idx, gauss_colors, output_dir):
    n_azimuth = params.shape[0]
    az_axis   = np.arange(n_azimuth)

    fig, axes = plt.subplots(n_gaussians, 3, figsize=(16, 3.2 * n_gaussians), constrained_layout=True)
    if n_gaussians == 1:
        axes = axes[np.newaxis, :]

    for k in range(n_gaussians):
        for j, (label, unit) in enumerate([
            (fr'$\mu_{{{k+1}}}$', '[m]'),
            (fr'$\sigma_{{{k+1}}}$', '[m]'),
            (fr'$A_{{{k+1}}}$', ''),
        ]):
            ax = axes[k, j]
            data = params[:, 3*k + j].copy()
            data[~fit_success] = np.nan
            ax.plot(az_axis, data, '.', markersize=1.5, alpha=0.7, color=gauss_colors[k])
            ax.set_title(f'{label} {unit}', fontsize=13)
            ax.set_xlabel('Azimuth pixel')
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'{n_gaussians}-Gaussian parameters per pixel (range idx={rn_idx})', fontsize=14)
    fig.savefig(output_dir / f'param_maps_rn{rn_idx}_{n_gaussians}g.png', dpi=150)
    
    return fig


def plot_peak_heights(params, fit_success, n_gaussians, rn_idx, gauss_colors, output_dir):
    n_azimuth = params.shape[0]
    az_axis   = np.arange(n_azimuth)
    valid     = fit_success

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for k in range(n_gaussians):
        mu_k = params[:, 3*k]
        ax.scatter(az_axis[valid], mu_k[valid], s=3, alpha=0.6, color=gauss_colors[k], label=fr'$\mu_{{{k+1}}}$ (peak {k+1})')

    ax.set_xlabel('Azimuth pixel')
    ax.set_ylabel('Height [m]')
    ax.set_title(f'Scatterer heights from {n_gaussians}-Gaussian fit (rn={rn_idx})')
    ax.legend(markerscale=4)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / f'peak_heights_rn{rn_idx}_{n_gaussians}g.png', dpi=150)
    return fig


def plot_example_fits(slide_abs, x, params, fit_success, n_gaussians,rn_idx, gauss_colors, output_dir, n_example_profiles=4):
    ok_indices = np.where(fit_success)[0]
    example_idx = np.linspace(0, len(ok_indices) - 1, n_example_profiles, dtype=int)
    example_pixels = ok_indices[example_idx]

    n_cols = 4
    n_rows = int(np.ceil(n_example_profiles / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows), constrained_layout=True)
    axes_flat = axes.flat if n_example_profiles > 1 else [axes]

    x_fine = np.linspace(x[0], x[-1], 500)
    for ax, px in zip(axes_flat, example_pixels):
        profile = slide_abs[:, px]
        fitted = multi_gaussian(x_fine, *params[px])

        ax.plot(x, profile, 'k-', lw=0.8, label='data')
        ax.plot(x_fine, fitted, 'r-', lw=1.5, label='fit (sum)')
        for k in range(n_gaussians):
            gk = single_gaussian(x_fine, *params[px, 3*k:3*k+3])
            ax.fill_between(x_fine, gk, alpha=0.20, color=gauss_colors[k], label=f'Gauss {k+1}')
        
        ax.set_title(f'az pixel {px}', fontsize=10)
        ax.set_xlabel('Height [m]')
        ax.set_ylabel('|Amplitude|')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    for ax in list(axes_flat)[len(example_pixels):]:
        ax.set_visible(False)

    fig.suptitle(f'Example {n_gaussians}-Gaussian fits (range idx={rn_idx})', fontsize=14)
    fig.savefig(output_dir / f'example_fits_rn{rn_idx}_{n_gaussians}g.png', dpi=150)
    return fig


def generate_gif(slide_abs, x, params, fit_success, n_gaussians, rn_idx, gauss_colors, output_dir, gif_fps=20):
    print("Generating animated GIF (sweeping through azimuth pixels) ...")
    n_azimuth = slide_abs.shape[1]

    gif_step = max(1, n_azimuth // 300)
    gif_pixels = np.arange(0, n_azimuth, gif_step)
    gif_path = output_dir / f'profile_sweep_rn{rn_idx}_{n_gaussians}g.gif'
    x_fine = np.linspace(x[0], x[-1], 500)

    global_ymax = np.nanmax(slide_abs) * 1.1

    fig, (ax_prof, ax_slice) = plt.subplots(
        1, 2, figsize=(13, 6), sharey=True,
        gridspec_kw={'width_ratios': [1, 1.4]},
    )
    plt.subplots_adjust(wspace=0.05)

    # Left panel – profile
    line_data, = ax_prof.plot([], [], 'k-', lw=0.9, label='data')
    line_fit,  = ax_prof.plot([], [], 'r-', lw=1.8, label='fit (sum)')
    fills = {k: None for k in range(n_gaussians)}
    ax_prof.set_ylim(x[0], x[-1])
    ax_prof.set_xlim(global_ymax, 0)
    ax_prof.set_ylabel('Height [m]')
    ax_prof.set_xlabel('|Amplitude|')
    ax_prof.grid(True, alpha=0.3)
    ax_prof.legend(loc='upper left', fontsize=8)
    title_prof = ax_prof.set_title('')

    # Right panel – slice with marker
    ax_slice.imshow(slide_abs, origin='lower', cmap='jet', aspect='auto', extent=[0, n_azimuth, x[0], x[-1]])
    vline = ax_slice.axvline(0, color='white', lw=1.2, ls='--')
    ax_slice.set_xlabel('Azimuth pixel')
    ax_slice.yaxis.set_visible(False)
    ax_slice.set_title(f'Tomogram slice  (range idx={rn_idx})')

    def _update(frame_idx):
        px = gif_pixels[frame_idx]
        profile = slide_abs[:, px]
        line_data.set_data(profile, x)

        for k in range(n_gaussians):
            if fills[k] is not None:
                fills[k].remove()

        if fit_success[px]:
            fitted = multi_gaussian(x_fine, *params[px])
            line_fit.set_data(fitted, x_fine)
            mu_texts = []
            for k in range(n_gaussians):
                gk = single_gaussian(x_fine, *params[px, 3*k:3*k+3])
                fills[k] = ax_prof.fill_betweenx(
                    x_fine, gk, alpha=0.20,
                    color=gauss_colors[k], label='_')
                mu_texts.append(fr'$\mu_{{{k+1}}}$={params[px, 3*k]:.1f}')
            title_prof.set_text(
                f'az pixel {px}   ' + '  '.join(mu_texts) + ' m')
        else:
            line_fit.set_data([], [])
            for k in range(n_gaussians):
                fills[k] = ax_prof.fill_betweenx([], [], alpha=0)
            title_prof.set_text(f'az pixel {px}  (fit failed)')

        vline.set_xdata([px, px])
        return ([line_data, line_fit, vline, title_prof]
                + [fills[k] for k in range(n_gaussians)])

    anim = animation.FuncAnimation(
        fig, _update, frames=len(gif_pixels),
        interval=1000 // gif_fps, blit=False,
    )
    anim.save(str(gif_path), writer='pillow', fps=gif_fps, dpi=100)
    plt.close(fig)
    print(f"GIF saved → {gif_path}  ({len(gif_pixels)} frames, {gif_fps} fps)")


if __name__ == "__main__":
    rn_idx             = 500
    n_gaussians        = 3
    tomo_file          = ('/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/tomo_17sartom0102_Lhv_Capon_Test_withVictor2.hd5')
    output_dir         = Path('/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/')
    height_axis_range  = (-20, 80)
    n_example_profiles = 4
    save_results       = True

    gauss_colors = [cm.tab10(i) for i in range(n_gaussians)]

    slide_abs, x        = load_tomogram_slice(tomo_file, rn_idx, height_axis_range)
    params, fit_success = fit_all_profiles(slide_abs, x, n_gaussians)

    if save_results:
        save_fitted_params(output_dir, rn_idx, n_gaussians, params, fit_success, x, tomo_file)

    slide_recon = reconstruct_slice(slide_abs, x, params, fit_success, n_gaussians)
    
    plot_slice_comparison(slide_abs, slide_recon, x, rn_idx, output_dir)
    plot_parameter_maps(params, fit_success, n_gaussians, rn_idx, gauss_colors, output_dir)
    plot_peak_heights(params, fit_success, n_gaussians, rn_idx, gauss_colors, output_dir)
    plot_example_fits(slide_abs, x, params, fit_success, n_gaussians, rn_idx, gauss_colors, output_dir, n_example_profiles)
    generate_gif(slide_abs, x, params, fit_success, n_gaussians, rn_idx, gauss_colors, output_dir)

    plt.show()
   
