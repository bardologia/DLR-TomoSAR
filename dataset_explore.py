#!/usr/bin/env python3
"""Dataset exploration script for DLR-TomoSAR.

Converts the earlier notebook into a standalone script that:
- Imports and inspects `core.dataset` classes
- Searches the repo for `.rat`, `.npy`, `.h5`, `.npz` files
- Attempts a safe, lightweight inspection of a sample `.rat` (via STEtools) if available
- Inspects `.npy` files with mmap to avoid heavy IO

Run: python DLR-TomoSAR/dataset_explore.py
"""

from pathlib import Path
import os
import sys
from pprint import pprint

repo_root = Path('/ste/rnd/User/vice_vi/DLR-TomoSAR')
print('Repository root set to', repo_root)
# Ensure repo root is on sys.path so we can import `core`
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# --- Inspect core.dataset ---
try:
    from core import dataset as ds
    print('\nImported core.dataset as ds')
    names = [n for n in dir(ds) if not n.startswith('_')]
    print('\nAvailable names in core.dataset:')
    pprint(names)

    for name in ('SLCPatchDataset', 'FeatureMapDataset', 'ParameterEstimationDataset', 'TensorPairDataset', 'DataPipeline', 'Representation'):
        if hasattr(ds, name):
            obj = getattr(ds, name)
            print(f'\n{name}:')
            doc = getattr(obj, '__doc__', None)
            if doc:
                print('  ', doc.strip().splitlines()[0])
            else:
                print('  (no docstring)')
except Exception as e:
    print('\nCould not import core.dataset:', e)

# --- Search for raw/feature files in the repo ---
extensions = ('.rat', '.npy', '.h5', '.npz')
found = {ext: [] for ext in extensions}
for root, dirs, files in os.walk(repo_root):
    for f in files:
        lf = f.lower()
        for ext in extensions:
            if lf.endswith(ext):
                found[ext].append(Path(root) / f)

for ext, paths in found.items():
    print(f"\n{ext} — found {len(paths)} files. Examples:")
    for p in paths[:10]:
        print(' -', p)

# Also show the external path used by training scripts (if present)
external_path = Path('/ste/rnd/User/sera_se')
print('\nExternal training path referenced in scripts:', external_path)
if external_path.exists():
    print('External path exists; listing a few files:')
    for i, p in enumerate(sorted(external_path.iterdir())):
        print(' -', p)
        if i >= 20:
            break
else:
    print('External path does not exist or is not accessible from this environment.')

# --- Attempt a safe inspection of a .rat file using STEtools.rrat if available ---
sample_rat = None
if found.get('.rat'):
    sample_rat = found['.rat'][0]
else:
    # If no .rat in repo, look under the external path used by training scripts.
    if external_path.exists():
        # Prefer known filenames used by training scripts, otherwise pick the first .rat under external_path
        for name in ('Elsa96x96ref0s.rat', 'Ana96x96ref0s.rat', 'Nani_small_96x96.rat'):
            p = external_path / name
            if p.exists():
                sample_rat = p
                break
        if sample_rat is None:
            candidates = list(external_path.rglob('*.rat'))
            if candidates:
                sample_rat = candidates[0]

print('\nSample .rat chosen for inspection:', sample_rat)
if sample_rat is None:
    print('No .rat file found to inspect. If your raw data is outside this repo, update the external path or copy a sample file into the environment.')
else:
    # Try to import STEtools.rrat and numpy; if unavailable, skip heavy inspections.
    try:
        from STEtools.ste_io import rrat
    except Exception:
        rrat = None
    try:
        import numpy as np
        HAVE_NUMPY = True
    except Exception:
        np = None
        HAVE_NUMPY = False

    if rrat is None or not HAVE_NUMPY:
        print('Could not import STEtools or numpy; skipping .rat content inspection.')
    else:
        try:
            print('rrat available; attempting a lightweight header read...')
            arr = rrat(str(sample_rat))  # note: may load considerable memory depending on file
            print('rrat loaded array shape:', getattr(arr, 'shape', None))
            print('dtype:', getattr(arr, 'dtype', None))
            if hasattr(arr, 'size') and arr.size > 0:
                s = arr.reshape(-1)[:1000]
                print('min, max, mean, std =', float(np.min(s)), float(np.max(s)), float(np.mean(s)), float(np.std(s)))
        except Exception as e:
            print('Could not read .rat with rrat:', e)
            print('As an alternative, inspect .npy feature files below.')

# --- Inspect .npy feature files safely ---
npy_paths = found.get('.npy', [])
print('\nFound .npy files:', len(npy_paths))
if not HAVE_NUMPY:
    print('numpy is not available in this Python environment; skipping .npy inspections.')
else:
    for p in npy_paths[:10]:
        try:
            print('\nLoading', p)
            a = np.load(p, mmap_mode='r')
            print(' - shape:', getattr(a, 'shape', None), 'dtype:', getattr(a, 'dtype', None))
            flat = a.reshape(-1)
            sample = flat[:1000] if flat.size else flat
            if sample.size:
                print(' - min/max/mean (sample):', float(sample.min()), float(sample.max()), float(sample.mean()))
        except Exception as e:
            print(' - could not read', p, '->', e)

print('\nDone.')
