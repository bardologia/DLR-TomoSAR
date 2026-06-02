from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PyRatJob:
    pyrat_root_path       : str
    crop_tuple            : Tuple[int, int, int, int]
    suffix                : str
    fusar_project_path    : str
    stack_identifier      : str
    base_directory        : str
    polarisation          : str
    track_selection       : str
    height_range          : Tuple[float, float]
    filter_method         : str
    filter_arguments      : Dict
    beamforming_method    : str
    beamforming_arguments : List
    output_directory      : str
    apply_resampling      : bool
    apply_presumming      : bool
    pyrat_threads         : int
    parent_sys_path       : Optional[list] = None


def run_pyrat(job: PyRatJob) -> int:
    import os as _os

    _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    _conda_lib = _os.path.join(sys.prefix, "lib")
    _ldpath = _os.environ.get("LD_LIBRARY_PATH", "")
    if _conda_lib not in _ldpath.split(":"):
        _os.environ["LD_LIBRARY_PATH"] = _conda_lib + (":" + _ldpath if _ldpath else "")

    if job.parent_sys_path is not None:
        sys.path[:] = job.parent_sys_path

    if job.pyrat_root_path not in sys.path:
        sys.path.insert(0, job.pyrat_root_path)

    from pyrat import pyrat_init, tomo
    pyrat_init(debug=True, nthreads=job.pyrat_threads, silent=True)

    tomo.fusartomo(
        FuSARproject = job.fusar_project_path,
        id           = job.stack_identifier,
        basedir      = job.base_directory,
        polarisation = job.polarisation,
        select       = job.track_selection,
        presum       = job.apply_presumming,
        crop         = job.crop_tuple,
        range        = list(job.height_range),
        filter       = job.filter_method,
        filargs      = job.filter_arguments,
        method       = job.beamforming_method,
        args         = job.beamforming_arguments,
        suffix       = job.suffix,
        dir          = job.output_directory,
        resampling   = job.apply_resampling,
    )

    gc.collect()
    return 0
