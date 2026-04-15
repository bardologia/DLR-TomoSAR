import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from time import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tomo_pixel_profiles import (
    multi_gaussian, single_gaussian, fit_profile,
    load_tomogram_slice, fit_all_profiles, reconstruct_slice,
    save_fitted_params,
)

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
    'legend.fontsize':    9,
    'axes.linewidth':     0.8,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.top':          True,
    'ytick.right':        True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
})


def fit_and_evaluate(slide_abs, x, ng, verbose=True):
    t0 = time()
    params, fit_success = fit_all_profiles(slide_abs, x, ng)
    recon = reconstruct_slice(slide_abs, x, params, fit_success, ng)

    residual = slide_abs - recon
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))

    elapsed = time() - t0
    n_ok = fit_success.sum()
    n_azimuth = slide_abs.shape[1]
    if verbose:
        print(f"  Done in {elapsed:.1f} s — converged {n_ok}/{n_azimuth} "
              f"({100*n_ok/n_azimuth:.1f} %)  RMSE = {rmse:.6f}  MAE = {mae:.6f}")

    return dict(params=params, fit_success=fit_success, recon=recon, rmse=rmse, mae=mae)


def run_all_fits(slide_abs, x, n_gauss_range, output_dir, rn_idx, tomo_file, save_results=True):
    results = {}
    for ng in n_gauss_range:
        results[ng] = fit_and_evaluate(slide_abs, x, ng)
        if save_results:
            save_fitted_params(output_dir, rn_idx, ng, results[ng]['params'], results[ng]['fit_success'], x, tomo_file)
    
    return results


def pick_example_pixels(results, n_gauss_range, n_example_profiles, fallback_ng=2):
    n_azimuth = results[next(iter(n_gauss_range))]['params'].shape[0]
    all_ok    = np.ones(n_azimuth, dtype=bool)
    for ng in n_gauss_range:
        all_ok &= results[ng]['fit_success']

    ok_indices = np.where(all_ok)[0]
    if len(ok_indices) < n_example_profiles:
        ok_indices = np.where(results[fallback_ng]['fit_success'])[0]

    sample_idx = np.linspace(0, len(ok_indices) - 1, n_example_profiles, dtype=int)
    return ok_indices[sample_idx]


def plot_slice_comparison_multi(slide_abs, x, results, n_gauss_range, rn_idx, output_dir):
    print("\nPlotting slice comparison")
    n_azimuth = slide_abs.shape[1]
    n_ng = len(n_gauss_range)
    vmin_abs = np.nanmin(slide_abs)
    vmax_abs = np.nanmax(slide_abs)

    fig, axes = plt.subplots(2, 1 + n_ng, figsize=(4.2 * (1 + n_ng), 8), constrained_layout=True)

    im = axes[0, 0].imshow(slide_abs, origin='lower', cmap='viridis', aspect='auto',
                           vmin=vmin_abs, vmax=vmax_abs, extent=[0, n_azimuth, x[0], x[-1]])
    axes[0, 0].set_title('Original tomogram')
    axes[0, 0].set_ylabel('Height (m)')
    cb = plt.colorbar(im, ax=axes[0, 0], shrink=0.75)
    cb.set_label('Amplitude (a.u.)')
    axes[1, 0].set_visible(False)

    res_max = max(np.nanmax(np.abs(slide_abs - results[ng]['recon'])) for ng in n_gauss_range)

    for col, ng in enumerate(n_gauss_range, start=1):
        recon = results[ng]['recon']
        res = slide_abs - recon

        im_r = axes[0, col].imshow(recon, origin='lower', cmap='viridis', aspect='auto', vmin=vmin_abs, vmax=vmax_abs, extent=[0, n_azimuth, x[0], x[-1]])
        axes[0, col].set_title(rf'$N_g = {ng}$  (RMSE$\,=${results[ng]["rmse"]:.4f})')
        axes[0, col].set_xlabel('Azimuth (pixels)')
        cb_r = plt.colorbar(im_r, ax=axes[0, col], shrink=0.75)
        cb_r.set_label('Amplitude (a.u.)')

        im_d = axes[1, col].imshow(res, origin='lower', cmap='RdBu_r', aspect='auto', vmin=-res_max, vmax=res_max, extent=[0, n_azimuth, x[0], x[-1]])
        axes[1, col].set_title(rf'Residual ($N_g = {ng}$)')
        axes[1, col].set_xlabel('Azimuth (pixels)')
        axes[1, col].set_ylabel('Height (m)')
        cb_d = plt.colorbar(im_d, ax=axes[1, col], shrink=0.75)
        cb_d.set_label('Residual (a.u.)')

    fig.suptitle(rf'Tomographic Slice Reconstruction Comparison (range index$\,={rn_idx}$)', fontsize=14)
    fig.savefig(output_dir / f'slice_comparison_rn{rn_idx}_1to5g.png')
    return fig


def plot_peak_heights_multi(results, n_gauss_range, rn_idx, base_colors, output_dir):
    print("Plotting peak-height maps")
    n_ng = len(n_gauss_range)
    n_azimuth = results[next(iter(n_gauss_range))]['params'].shape[0]
    az_axis = np.arange(n_azimuth)

    fig, axes = plt.subplots(n_ng, 1, figsize=(14, 3.5 * n_ng), constrained_layout=True, sharex=True)
    if n_ng == 1:
        axes = [axes]

    for row, ng in enumerate(n_gauss_range):
        ax = axes[row]
        valid = results[ng]['fit_success']
        params_ng = results[ng]['params']

        for k in range(ng):
            mu_k = params_ng[:, 3 * k]
            ax.scatter(az_axis[valid], mu_k[valid], s=4, alpha=0.55, marker='o', edgecolors='none', color=base_colors[k], label=rf'$\mu_{{{k+1}}}$')

        ax.set_ylabel('Height (m)')
        ax.set_title(rf'$N_g = {ng}$')
        ax.legend(markerscale=3, ncol=ng, loc='upper right', framealpha=0.85, edgecolor='gray')
        ax.grid(True, alpha=0.25, linewidth=0.5)

    axes[-1].set_xlabel('Azimuth (pixels)')
    fig.suptitle(rf'Estimated Scatterer Heights per Gaussian Order (range index$\,={rn_idx}$)', fontsize=14)
    fig.savefig(output_dir / f'peak_heights_rn{rn_idx}_1to5g.png')
    return fig


def plot_example_fits_multi(slide_abs, x, results, n_gauss_range, example_pixels, rn_idx, base_colors, output_dir):
    n_ng      = len(n_gauss_range)
    n_example = len(example_pixels)
    x_fine    = np.linspace(x[0], x[-1], 500)

    fig, axes = plt.subplots(n_ng, n_example, figsize=(3.5 * n_example, 3.2 * n_ng),
                              constrained_layout=True, sharex=True, sharey=True)

    for row, ng in enumerate(n_gauss_range):
        params_ng = results[ng]['params']
        fs_ng = results[ng]['fit_success']
        for col, px in enumerate(example_pixels):
            ax = axes[row, col]
            profile = slide_abs[:, px]
            ax.plot(x, profile, 'k-', lw=0.8, label='Observed')

            if fs_ng[px]:
                fitted = multi_gaussian(x_fine, *params_ng[px])
                ax.plot(x_fine, fitted, color='tab:red', ls='-', lw=1.4, label='Fitted')
                for k in range(ng):
                    gk = single_gaussian(x_fine, *params_ng[px, 3*k:3*k+3])
                    ax.fill_between(x_fine, gk, alpha=0.18, color=base_colors[k],
                                    label=rf'$g_{{{k+1}}}$')
            else:
                ax.text(0.5, 0.5, 'No convergence', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, fontstyle='italic',
                        color='red')

            if row == 0:
                ax.set_title(rf'Azimuth$\,={px}$')
            if col == 0:
                ax.set_ylabel(rf'$N_g = {ng}$' + '\nAmplitude (a.u.)')
            if row == n_ng - 1:
                ax.set_xlabel('Height (m)')
            ax.grid(True, alpha=0.25, linewidth=0.5)
            if row == 0 and col == n_example - 1:
                ax.legend(fontsize=7, loc='upper right',
                          framealpha=0.85, edgecolor='gray')

    fig.suptitle(rf'Vertical Profile Fits: $N_g = 1$ to $N_g = 5$ '
                 rf'(range index$\,={rn_idx}$)', fontsize=14)
    fig.savefig(output_dir / f'example_fits_rn{rn_idx}_1to5g.png')
    return fig


def plot_residual_metrics(results, n_gauss_range, rn_idx, output_dir):
    print("Plotting residual metrics")
    ng_list = list(n_gauss_range)
    n_azimuth = results[ng_list[0]]['params'].shape[0]

    rmses = [results[ng]['rmse'] for ng in ng_list]
    maes  = [results[ng]['mae']  for ng in ng_list]
    conv  = [results[ng]['fit_success'].sum() / n_azimuth * 100 for ng in ng_list]

    fig, (ax_err, ax_conv) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    bar_x = np.arange(len(ng_list))
    width = 0.35

    ax_err.bar(bar_x - width / 2, rmses, width, label='RMSE', color='tab:blue',
               edgecolor='black', linewidth=0.5)
    ax_err.bar(bar_x + width / 2, maes, width, label='MAE', color='tab:orange',
               edgecolor='black', linewidth=0.5)
    ax_err.set_xticks(bar_x)
    ax_err.set_xticklabels([rf'$N_g = {ng}$' for ng in ng_list])
    ax_err.set_xlabel(r'Number of Gaussian components ($N_g$)')
    ax_err.set_ylabel('Error (a.u.)')
    ax_err.set_title('Reconstruction Error')
    ax_err.legend(framealpha=0.85, edgecolor='gray')
    ax_err.grid(True, alpha=0.25, linewidth=0.5, axis='y')

    ax_conv.bar(bar_x, conv, color='tab:green', alpha=0.8,
                edgecolor='black', linewidth=0.5)
    ax_conv.set_xticks(bar_x)
    ax_conv.set_xticklabels([rf'$N_g = {ng}$' for ng in ng_list])
    ax_conv.set_xlabel(r'Number of Gaussian components ($N_g$)')
    ax_conv.set_ylabel('Convergence rate (%)')
    ax_conv.set_title(r'Fit Convergence vs.\ $N_g$')
    ax_conv.set_ylim(0, 105)
    ax_conv.grid(True, alpha=0.25, linewidth=0.5, axis='y')

    fig.suptitle(rf'Fitting Quality Summary (range index$\,={rn_idx}$)', fontsize=14)
    fig.savefig(output_dir / f'residual_metrics_rn{rn_idx}_1to5g.png')
    return fig



if __name__ == "__main__":
    rn_idx             = 500
    tomo_file          = ('/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/tomo_17sartom0102_Lhv_Capon_Test_withVictor2.hd5')
    output_dir         = Path('/ste/rnd/User/vice_vi/Pruebas/TOMO/TOMO-SR/')
    height_axis_range  = (-20, 80)
    n_gauss_range      = range(1, 6)
    n_example_profiles = 4
    save_results       = True
    base_colors        = [cm.tab10(i) for i in range(10)]

    slide_abs, x = load_tomogram_slice(tomo_file, rn_idx, height_axis_range)

    results = run_all_fits(slide_abs, x, n_gauss_range, output_dir, rn_idx, tomo_file, save_results=save_results)

    example_pixels = pick_example_pixels(results, n_gauss_range, n_example_profiles)

    plot_slice_comparison_multi(slide_abs, x, results, n_gauss_range, rn_idx, output_dir)
    plot_peak_heights_multi(results, n_gauss_range, rn_idx, base_colors, output_dir)
    plot_example_fits_multi(slide_abs, x, results, n_gauss_range, example_pixels, rn_idx, base_colors, output_dir)
    plot_residual_metrics(results, n_gauss_range, rn_idx, output_dir)

    plt.show()
