import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import uniform_filter
from itertools import combinations

sys.path.append('/ste/rnd/User/vice_vi/pyrat')
from pyrat import *
from pyrat.dlr.tomo.tools import fusar_readproject, split_flps

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,
    'image.interpolation': 'nearest',
})


CONFIG = dict(
    project_csv  = '/ste/rnd/FuSAR/Projects/17sartom-traun_L.csv',
    project_id   = "1",
    basedir      = "/ste/rnd/",
    polarisation = 'hv',
    select       = [0, 5, 7, 10,11, 15, 18, 20],  # slave indices to use (see list_passes)
    crop         = (4149, 5027, 905, 1945),
    output_dir   = '/ste/rnd/User/vice_vi/Pruebas/interferograms/',
    coh_win      = (5, 5),
    ref          = 0,       # index into loaded stack used as reference (0 = project primary)
    pairs_mode   = 'star',  # 'star' = ref vs each, 'all' = every combination, (i,j) = single pair
)


def build_paths(cfg):
    project = fusar_readproject(cfg['project_csv'], ids=cfg['project_id'])
    band    = project['bands'].split(',')[0]
    base    = os.path.join(cfg['basedir'], project['campaign'].upper())
    tries   = 'T' + project['tries']

    flps_m = split_flps(project['master'])
    master_dir    = os.path.join(base, flps_m[0], flps_m[1], tries)
    primary_name  = project['master'].strip()

    slaves_raw = project['slaves'].split(',')
    if isinstance(cfg['select'], list):
        slaves_raw = [slaves_raw[i] for i in cfg['select']]

    slave_dirs      = []
    secondary_names = []
    for s in slaves_raw:
        flps = split_flps(s)
        slave_dirs.append(os.path.join(base, flps[0], flps[1], tries))
        secondary_names.append(s.strip())

    labels = [primary_name] + secondary_names

    opts = [o.split('=') for o in project['options'].split(',')]
    opts = {o[0].lower().strip(): (True if len(o) == 1 else o[1].strip()) for o in opts}
    suffix = opts.get('suffix', '')
    project_name = project['campaign'].strip()

    return master_dir, slave_dirs, band, suffix, labels, project_name


def list_passes(cfg):
    project = fusar_readproject(cfg['project_csv'], ids=cfg['project_id'])
    secondaries = project['slaves'].split(',')
    print(f'Primary   : {project["master"].strip()}')
    print(f'Secondary : {len(secondaries)} available')
    print('-' * 35)
    for i, s in enumerate(secondaries):
        print(f'  [{i:2d}]  {s.strip()}')
    n = len(secondaries)
    print(f'\nWith select="*" → {n+1} passes → {(n+1)*n//2} pairs')
    print('Pick a subset, e.g. select=[0, 5, 10, 15, 20]')


def load_slc_stack(master_dir, slave_dirs, band, suffix, cfg):
    pol, crop = cfg['polarisation'], cfg['crop']

    master_layer = pyrat.load.fsar(master_dir, product='RGI-SLC', polarisations=pol, bands=band, crop=crop, sym=True)

    slave_layers = []
    for sdir in slave_dirs:
        slc_l = pyrat.load.fsar(sdir, product='INF-SLC', polarisations=pol, bands=band, crop=crop, suffix=suffix, sym=True)
        ph_l = pyrat.load.fsar_phadem(sdir, bands=band, crop=crop, suffix=suffix)
        flat_l = pyrat.tomo.fsarflatten(layer=[slc_l, ph_l])
        pyrat.delete([slc_l, ph_l], silent=True)
        slave_layers.append(flat_l)

    arrays = [pyrat.getdata(master_layer)]
    for sl in slave_layers:
        arrays.append(pyrat.getdata(sl))

    for i in range(len(arrays)):
        if arrays[i].ndim == 3:
            arrays[i] = arrays[i][0]

    return arrays


def compute_interferograms(slc_arrays, ref=0, mode='star'):
    n = len(slc_arrays)
    if isinstance(mode, (tuple, list)):
        pairs = [tuple(mode)]
    elif mode == 'star':
        pairs = [(ref, j) for j in range(n) if j != ref]
    else:
        pairs = list(combinations(range(n), 2))
    ifgs = {(i, j): slc_arrays[i] * np.conj(slc_arrays[j]) for i, j in pairs}
    return pairs, ifgs


def compute_coherence(slc_arrays, ifgs, pairs, win=(5, 5)):
    coherences = {}
    for i, j in pairs:
        num = np.abs(uniform_filter(ifgs[(i, j)], size=win, mode='constant'))
        den = np.sqrt(
            uniform_filter(np.abs(slc_arrays[i])**2, size=win, mode='constant') *
            uniform_filter(np.abs(slc_arrays[j])**2, size=win, mode='constant')
        )
        coherences[(i, j)] = np.clip(num / (den + 1e-10), 0, 1)
    return coherences


def cmap_phase():
    r = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,5,7,9,13,15,17,21,25,29,33,37,43,47,53,59,65,73,79,85,91,97,101,107,113,117,121,127,131,135,141,145,151,155,159,163,167,171,173,177,181,185,189,193,197,201,205,209,211,215,217,219,221,223,223,223,223,223,223,221,219,217,215,213,211,207,205,201,199,195,193,191,189,187,185,183,181,179,179,179,179,179,179,181,183,185,187,189,191,193,195,199,201,205,209,213,215,219,223,227,229,233,235,237,239,243,245,245,247,249,251,251,253,253,253,253,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,253,253,251,249,245,243,239,233,229,223,215,209,201,193,185,175,165,155,145,133,121,109,99,87,79,69,61,53,45,37,31,25,21,17,13,9,5,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    g = [249,249,249,247,245,245,243,239,237,235,233,229,227,223,219,215,211,205,201,197,193,189,185,181,177,173,171,167,163,159,155,151,145,141,135,131,127,121,117,113,107,101,97,91,85,79,73,65,59,53,47,43,37,33,29,25,21,17,15,13,9,7,5,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,5,5,7,9,11,13,15,17,19,23,25,29,33,35,41,45,49,53,57,59,63,67,71,75,79,81,85,89,93,97,101,103,107,111,113,117,119,123,125,129,131,135,137,141,145,147,151,155,159,163,167,171,173,177,181,185,189,193,197,201,205,207,211,213,217,217,219,219,219,219,219,217,215,213,211,209,205,203,199,195,193,189,187,185,181,179,179,177,175,175,175,175,177,177,179,181,183,187,189,193,195,199,201,205,207,211,215,217,221,223,227,229,233,235,237,239,241,243,245,245,247,249,249,249,251,251]
    b = [253,253,253,253,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,253,253,251,251,249,247,245,241,239,235,231,227,223,217,211,207,201,193,185,179,169,161,153,145,137,129,121,113,105,99,91,85,77,71,63,57,51,45,39,35,31,27,23,21,17,15,13,9,7,7,5,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,5,7,9,13,15,17,21,25,29,33,37,43,47,53,59,65,73,79,85,91,97,101,107,113,117,121,127,131,135,141,145,151,155,159,163,167,171,173,177,181,185,189,193,197,201,205,211,215,219,223,227,229,233,235,237,239,243,245,245,247,249,251,251]
    a = [255] * len(r)
    rgba = list(np.array(list(zip(r, g, b, a))) / 255)
    return ListedColormap(rgba)


def _setup_grid(n_panels, col_width=3.2, row_height=2.8, max_cols=4):
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(col_width * ncols + 0.8, row_height * nrows + 0.6), constrained_layout=True, squeeze=False)
    return fig, axes, nrows, ncols


def _pair_label(labels, i, j):
    return f'({labels[i]}, {labels[j]})'


def plot_interferograms(pairs, ifgs, labels, output_dir, project_name='', ref_label=''):
    fig, axes, nrows, ncols = _setup_grid(len(pairs))
    im = None

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        phase = np.angle(ifgs[(i, j)])
        im = ax.imshow(phase, cmap=cmap_phase(), vmin=-np.pi, vmax=np.pi, aspect='auto')
        ax.set_title(_pair_label(labels, i, j))
        ax.set_xlabel('Range [px]')
        ax.set_ylabel('Azimuth [px]')
        ax.tick_params(direction='in', top=True, right=True)

    for idx in range(len(pairs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'{project_name} — Interferometric phase (primary {ref_label})', fontsize=11)

    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.75, pad=0.02)
    cbar.set_label(r'Interferometric phase $\phi$ [rad]')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

    os.makedirs(output_dir, exist_ok=True)
    fname = f'{project_name}_primary-{ref_label}_interferograms.png'
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath)
    print(f'Saved: {fpath}')
    plt.show()
    return fig


def plot_coherence(pairs, coherences, labels, output_dir, project_name='', ref_label=''):
    fig, axes, nrows, ncols = _setup_grid(len(pairs))
    im = None

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        im = ax.imshow(coherences[(i, j)], cmap='inferno', vmin=0, vmax=1, aspect='auto')
        ax.set_title(_pair_label(labels, i, j))
        ax.set_xlabel('Range [px]')
        ax.set_ylabel('Azimuth [px]')
        ax.tick_params(direction='in', top=True, right=True)

    for idx in range(len(pairs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'{project_name} — Coherence (primary {ref_label})', fontsize=11)

    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.75, pad=0.02)
    cbar.set_label(r'Coherence $|\gamma|$')

    os.makedirs(output_dir, exist_ok=True)
    fname = f'{project_name}_primary-{ref_label}_coherence.png'
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath)
    print(f'Saved: {fpath}')
    plt.show()
    return fig


def main(cfg=CONFIG):
    list_passes(cfg)
    print()
    pyrat_init(debug=False, nthreads=4)

    master_dir, slave_dirs, band, suffix, labels, project_name = build_paths(cfg)
    slc_arrays = load_slc_stack(master_dir, slave_dirs, band, suffix, cfg)

    ref = cfg['ref']
    ref_label = labels[ref]
    print(f'Project: {project_name}')
    print(f'Passes: {len(slc_arrays)},  SLC shape: {slc_arrays[0].shape}')
    print(f'Reference: [{ref}] {ref_label}')
    print(f'Pairs mode: {cfg["pairs_mode"]}')

    pairs, ifgs = compute_interferograms(slc_arrays, ref=ref, mode=cfg['pairs_mode'])
    coherences = compute_coherence(slc_arrays, ifgs, pairs, win=cfg['coh_win'])

    print(f'Interferogram pairs: {len(pairs)}')

    plot_interferograms(pairs, ifgs, labels, cfg['output_dir'], project_name, ref_label)
    plot_coherence(pairs, coherences, labels, cfg['output_dir'], project_name, ref_label)


if __name__ == '__main__':
    main()
