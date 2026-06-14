class FlowLibrary:

    def _processing(self) -> dict:
        nodes = [
            {"id": "s0",      "tex": r"s_0",                   "role": "measured",     "kind": "matrix", "shape": "A_z x R_g",   "desc": "master (primary) SLC, complex (PyRat RGI-SLC)",          "sample": [["0.8+0.2j", "0.6-0.4j", "1.1+0.0j"], ["0.5+0.5j", "0.9-0.1j", "0.7+0.3j"], ["1.0+0.2j", "0.4-0.6j", "0.8+0.1j"]]},
            {"id": "si",      "tex": r"s_i",                   "role": "measured",     "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "co-registered secondary SLC stack (PyRat INF-SLC)",   "sample": [["0.7+0.3j", "0.5-0.3j"], ["0.6+0.4j", "0.8-0.2j"]]},
            {"id": "phidem",  "tex": r"\phi_{\mathrm{DEM},i}", "role": "measured",     "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "DEM-predicted phase per pass (rad)",                  "sample": [["0.12", "0.31"], ["0.27", "0.44"]]},
            {"id": "trk",     "tex": r"\mathbf{t}_i",          "role": "measured",     "kind": "tensor", "shape": "N_s x 4 x A_z", "desc": "per-pass track position rows (horizontal, vertical)",   "sample": [["...h...", "...v..."]]},
            {"id": "stilde",  "tex": r"\tilde{s}_i",           "role": "intermediate", "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "DEM-deramped secondary SLC",                          "sample": [["0.6+0.1j", "0.5-0.1j"], ["0.7+0.2j", "0.6-0.0j"]]},
            {"id": "cross",   "tex": r"c_i",                   "role": "intermediate", "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "raw master-secondary cross-product",                  "sample": [["0.45+0.30j", "0.41-0.12j"], ["0.55+0.18j", "0.48-0.05j"]]},
            {"id": "pi",      "tex": r"p_i",                   "role": "intermediate", "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "unit-magnitude interferometric phasor",               "sample": [["0.83+0.55j", "0.96-0.28j"], ["0.95+0.31j", "0.99-0.10j"]]},
            {"id": "Ai",      "tex": r"A_i",                   "role": "intermediate", "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "clipped secondary amplitude weight",                  "sample": [["0.71", "0.93"], ["1.25", "0.58"]]},
            {"id": "phii",    "tex": r"\tilde{\phi}_i",        "role": "calculated",   "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "amplitude-weighted complex interferogram",            "sample": [["0.51+0.20j", "0.44-0.12j"], ["0.63+0.18j", "0.55-0.05j"]]},
            {"id": "bv",      "tex": r"b^{\mathrm{v}}_i",      "role": "calculated",   "kind": "vector", "shape": "N_s",         "desc": "vertical baseline relative to reference (m)",            "sample": ["0.00", "12.4", "-8.1", "20.7"]},
            {"id": "bh",      "tex": r"b^{\mathrm{h}}_i",      "role": "calculated",   "kind": "vector", "shape": "N_s",         "desc": "horizontal baseline relative to reference (m)",          "sample": ["0.00", "5.2", "-3.4", "9.8"]},
            {"id": "bstd",    "tex": r"\sigma^{\mathrm{pos}}_i","role": "intermediate","kind": "vector", "shape": "N_s",         "desc": "per-track position std, validation gate (<= 5 m)",       "sample": ["0.4", "0.8", "1.1", "0.6"]},
            {"id": "M",       "tex": r"M",                     "role": "calculated",   "kind": "scalar", "shape": "1",           "desc": "number of azimuth subsections",                          "sample": "15"},
            {"id": "PT",      "tex": r"(P,T)",                 "role": "calculated",   "kind": "scalar", "shape": "1",           "desc": "resolved (workers, threads-per-worker) plan",            "sample": "(8, 2)"},
            {"id": "Rhat",    "tex": r"\hat{\mathbf{R}}",      "role": "intermediate", "kind": "matrix", "shape": "N_s x N_s",   "desc": "Boxcar-windowed sample covariance",                      "sample": [["1.0", "0.7"], ["0.7", "1.0"]]},
            {"id": "xi",      "tex": r"x_h",                   "role": "intermediate", "kind": "vector", "shape": "H",           "desc": "elevation axis samples (m)",                             "sample": ["-20.0", "-19.0", "...", "80.0"]},
            {"id": "asteer",  "tex": r"\mathbf{a}(\xi)",       "role": "intermediate", "kind": "vector", "shape": "N_s",         "desc": "steering vector (PyRat-internal model)",                 "sample": ["1", "e^{j..}", "e^{j..}"]},
            {"id": "Tm",      "tex": r"T_m",                   "role": "calculated",   "kind": "tensor", "shape": "H x W_m x R_g", "desc": "Capon beamformed tomogram of subsection m",            "sample": [["0.2", "0.9", "0.3"], ["0.1", "0.7", "0.8"], ["0.0", "0.4", "1.0"]]},
            {"id": "DEMm",    "tex": r"D_m",                   "role": "calculated",   "kind": "matrix", "shape": "W_m x R_g",   "desc": "DEM of subsection m (PyRat byproduct)",                  "sample": [["31.2", "30.8"], ["32.1", "31.5"]]},
            {"id": "Tcomb",   "tex": r"T_{\mathrm{comb}}",     "role": "final",        "kind": "tensor", "shape": "H x A_z x R_g", "desc": "combined beamformed tomogram, model reference",        "sample": [["0.2", "0.9", "0.3"], ["0.1", "0.7", "0.8"], ["0.0", "0.4", "1.0"]]},
            {"id": "DEM",     "tex": r"D",                     "role": "final",        "kind": "matrix", "shape": "A_z x R_g",   "desc": "full-stack concatenated DEM",                            "sample": [["31.2", "30.8"], ["32.1", "31.5"]]},
        ]
        steps = [
            {
                "id": "slc_load", "title": "SLC loading and co-registration", "phase": "A - Ingest",
                "note": "The master is read as the range-Doppler image; each secondary is the interferometric product already co-registered to the master, carrying its own DEM-predicted phase.",
                "inputs": [], "outputs": ["s0", "si", "phidem"],
                "lines": [
                    [{"id": "s0", "tex": r"s_0", "role": "measured"}, {"tex": "="}, {"tex": r"\mathrm{load}\big(\mathrm{master};\ \mathtt{RGI\text{-}SLC}\big)"}],
                    [{"id": "si", "tex": r"s_i", "role": "measured"}, {"tex": "="}, {"tex": r"\mathrm{load}\big(\mathrm{sec}_i;\ \mathtt{INF\text{-}SLC}\big),\quad"}, {"id": "phidem", "tex": r"\phi_{\mathrm{DEM},i}", "role": "measured"}, {"tex": r"=\mathrm{phadem}_i"}],
                ],
            },
            {
                "id": "baselines", "title": "Track baseline extraction", "phase": "A - Ingest",
                "note": "Per pass, horizontal and vertical track positions are averaged over the azimuth window and re-expressed relative to the reference track; a per-track position std above 5 m aborts the run.",
                "inputs": ["trk"], "outputs": ["bstd", "bv", "bh"],
                "lines": [
                    [{"id": "bv", "tex": r"b^{\mathrm{v}}_i", "role": "calculated"}, {"tex": "="}, {"tex": r"\overline{t^{\mathrm{v}}_i}-\overline{t^{\mathrm{v}}_0},\qquad"}, {"id": "bh", "tex": r"b^{\mathrm{h}}_i", "role": "calculated"}, {"tex": "="}, {"tex": r"\overline{t^{\mathrm{h}}_i}-\overline{t^{\mathrm{h}}_0}"}],
                    [{"id": "bstd", "tex": r"\sigma^{\mathrm{pos}}_i", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max\!\big(\mathrm{std}\,t^{\mathrm{h}}_i,\ \mathrm{std}\,t^{\mathrm{v}}_i\big)\ \le\ 5\,\mathrm{m}"}],
                ],
            },
            {
                "id": "deramp", "title": "DEM-phase deramping", "phase": "B - Interferogram",
                "note": "Each secondary is multiplied by the DEM phasor; after the later conjugation this subtracts the DEM-predicted phase, removing terrain topography and leaving sub-resolution elevation structure.",
                "inputs": ["si", "phidem"], "outputs": ["stilde"],
                "lines": [
                    [{"id": "stilde", "tex": r"\tilde{s}_i", "role": "intermediate"}, {"tex": "="}, {"id": "si", "tex": r"s_i", "role": "measured"}, {"tex": r"\cdot"}, {"tex": r"\exp\!\left(j\,"}, {"id": "phidem", "tex": r"\phi_{\mathrm{DEM},i}", "role": "measured"}, {"tex": r"\right)"}],
                ],
            },
            {
                "id": "crossprod", "title": "Master-secondary cross-product", "phase": "B - Interferogram",
                "note": "Conjugating the deramped secondary against the master leaves the phase difference minus the DEM phase; the conjugation flips the DEM sign, so it is effectively subtracted.",
                "inputs": ["s0", "stilde"], "outputs": ["cross"],
                "lines": [
                    [{"id": "cross", "tex": r"c_i", "role": "intermediate"}, {"tex": "="}, {"id": "s0", "tex": r"s_0", "role": "measured"}, {"tex": r"\cdot"}, {"id": "stilde", "tex": r"\overline{\tilde{s}_i}", "role": "intermediate"}],
                    [{"tex": r"\arg(c_i)"}, {"tex": "="}, {"tex": r"\psi_0 - \psi_i - \phi_{\mathrm{DEM},i}"}],
                ],
            },
            {
                "id": "phasor", "title": "Unit-phasor normalisation", "phase": "B - Interferogram",
                "note": "Dividing by the cross-product magnitude (floored at 1e-30) removes inter-pass amplitude differences while preserving coherence; null pixels collapse to zero instead of NaN.",
                "inputs": ["cross"], "outputs": ["pi"],
                "lines": [
                    [{"id": "pi", "tex": r"p_i", "role": "intermediate"}, {"tex": "="}, {"tex": r"\dfrac{"}, {"id": "cross", "tex": r"c_i", "role": "intermediate"}, {"tex": r"}{\left|c_i\right| + \epsilon},\qquad \epsilon = 10^{-30}"}],
                ],
            },
            {
                "id": "clip", "title": "Amplitude clipping", "phase": "B - Interferogram",
                "note": "The secondary amplitude is clipped at c_max = 1.25 so bright corner reflectors or artefacts cannot dominate the per-pass weight.",
                "inputs": ["si"], "outputs": ["Ai"],
                "lines": [
                    [{"id": "Ai", "tex": r"A_i", "role": "intermediate"}, {"tex": "="}, {"tex": r"\min\!\big(\left|"}, {"id": "si", "tex": r"s_i", "role": "measured"}, {"tex": r"\right|,\ 1.25\big)"}],
                ],
            },
            {
                "id": "interf", "title": "Amplitude-weighted interferogram", "phase": "B - Interferogram",
                "note": "Re-introducing the clipped amplitude as the modulus gives an interferogram whose phase is the residual elevation phase and whose magnitude is a bounded signal-to-noise proxy.",
                "inputs": ["Ai", "pi"], "outputs": ["phii"],
                "lines": [
                    [{"id": "phii", "tex": r"\tilde{\phi}_i", "role": "calculated"}, {"tex": "="}, {"id": "Ai", "tex": r"A_i", "role": "intermediate"}, {"tex": r"\cdot"}, {"id": "pi", "tex": r"p_i", "role": "intermediate"}, {"tex": "="}, {"id": "Ai", "tex": r"A_i", "role": "intermediate"}, {"tex": r"\dfrac{s_0\,\overline{\tilde{s}_i}}{\left|s_0\,\overline{\tilde{s}_i}\right| + \epsilon}"}],
                ],
            },
            {
                "id": "subdivide", "title": "Azimuth subdivision and worker plan", "phase": "C - Beamforming",
                "note": "When the crop exceeds W_max (1000 lines) it is split into M non-overlapping subsections; the worker/thread plan is resolved from an effort-based core budget B = floor(C f_effort).",
                "inputs": [], "outputs": ["M", "PT"],
                "lines": [
                    [{"id": "M", "tex": r"M", "role": "calculated"}, {"tex": "="}, {"tex": r"\left\lceil \dfrac{W_{az}}{W_{\max}} \right\rceil,\qquad W_{\max}=1000"}],
                    [{"id": "PT", "tex": r"(P,T)", "role": "calculated"}, {"tex": "="}, {"tex": r"\mathrm{resolve}\big(M;\ B\big),\quad B = \lfloor C\,f_{\mathrm{effort}} \rfloor"}],
                ],
            },
            {
                "id": "covariance", "title": "Boxcar sample covariance", "phase": "C - Beamforming",
                "note": "Inside each PyRat subprocess the interferometric stack is averaged over a 20x10 px Boxcar window to estimate the per-pixel sample covariance the Capon estimator inverts.",
                "inputs": ["phii"], "outputs": ["Rhat"],
                "lines": [
                    [{"id": "Rhat", "tex": r"\hat{\mathbf{R}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\big\langle\,"}, {"id": "phii", "tex": r"\boldsymbol{\phi}\,\boldsymbol{\phi}^{H}", "role": "calculated"}, {"tex": r"\,\big\rangle_{W_{\mathrm{box}}},\qquad W_{\mathrm{box}} = 20\times 10"}],
                ],
            },
            {
                "id": "capon", "title": "Capon beamforming over elevation", "phase": "C - Beamforming",
                "note": "Over a uniform elevation grid spanning [x_min, x_max] the minimum-variance estimator evaluates 1 / (a^H R^-1 a). The steering vector a(xi) is PyRat's internal model, not computed by this stage.",
                "inputs": ["Rhat", "xi", "asteer"], "outputs": ["DEMm", "Tm"],
                "lines": [
                    [{"id": "xi", "tex": r"x_h", "role": "intermediate"}, {"tex": "="}, {"tex": r"x_{\min} + h\,\dfrac{x_{\max}-x_{\min}}{H-1}"}],
                    [{"id": "asteer", "tex": r"\mathbf{a}(\xi)\big|_i", "role": "intermediate"}, {"tex": "="}, {"tex": r"\exp\!\Big(j\,\tfrac{4\pi}{\lambda}\,b_i\,\xi / r_0\Big)"}],
                    [{"id": "Tm", "tex": r"T_m(\xi)", "role": "calculated"}, {"tex": r"\;\propto\;"}, {"tex": r"\dfrac{1}{\mathbf{a}^{H}(\xi)\,"}, {"id": "Rhat", "tex": r"\hat{\mathbf{R}}^{-1}", "role": "intermediate"}, {"tex": r"\,\mathbf{a}(\xi)}"}],
                ],
            },
            {
                "id": "concat", "title": "Subsection concatenation", "phase": "C - Beamforming",
                "note": "Each worker writes an HDF5 file; outputs are reassembled along azimuth, the DEM on axis 0 and the tomogram on axis 1, then saved for downstream stages.",
                "inputs": ["Tm", "DEMm"], "outputs": ["DEM", "Tcomb"],
                "lines": [
                    [{"id": "DEM", "tex": r"D", "role": "final"}, {"tex": "="}, {"tex": r"\mathrm{concat}\!\left[\,"}, {"id": "DEMm", "tex": r"D_0, \dots, D_{M-1}", "role": "calculated"}, {"tex": r"\,\right]_{\mathrm{axis}=0}"}],
                    [{"id": "Tcomb", "tex": r"T_{\mathrm{comb}}", "role": "final"}, {"tex": "="}, {"tex": r"\mathrm{concat}\!\left[\,"}, {"id": "Tm", "tex": r"T_0, \dots, T_{M-1}", "role": "calculated"}, {"tex": r"\,\right]_{\mathrm{axis}=1}"}],
                ],
            },
        ]
        return {
            "key"   : "processing",
            "name"  : "Processing (Pre-process)",
            "blurb" : "From F-SAR SLC passes to the beamformed tomogram. Load and co-register, extract track baselines, deramp against the DEM, form the amplitude-weighted interferogram, then beamform per azimuth subsection and reassemble.",
            "nodes" : nodes,
            "steps" : steps,
        }

    def _param_extraction(self) -> dict:
        nodes = [
            {"id": "T",       "tex": r"T",                "role": "measured",     "kind": "tensor", "shape": "H x A x R", "desc": "beamformed tomogram from processing",        "sample": [["0.2", "0.9", "0.3"], ["0.1", "0.7", "0.8"], ["0.0", "0.4", "1.0"]]},
            {"id": "P",       "tex": r"P_h",              "role": "intermediate", "kind": "vector", "shape": "H",         "desc": "thresholded, truncated magnitude profile",   "sample": ["0.00", "0.62", "0.31", "0.88", "0.00"]},
            {"id": "active",  "tex": r"\mathbb{1}_{\mathrm{act}}", "role": "intermediate", "kind": "scalar", "shape": "1", "desc": "active-pixel gate (max above threshold)",  "sample": "1"},
            {"id": "scale",   "tex": r"s",                "role": "intermediate", "kind": "scalar", "shape": "1",         "desc": "per-pixel scale = profile max (1 if inactive)", "sample": "0.88"},
            {"id": "gtilde",  "tex": r"\tilde{\gamma}_h", "role": "intermediate", "kind": "vector", "shape": "H",         "desc": "peak-normalised profile",                    "sample": ["0.00", "0.70", "0.35", "1.00", "0.00"]},
            {"id": "peaks",   "tex": r"\mathcal{P}",      "role": "intermediate", "kind": "set",    "shape": "P",         "desc": "prominence-and-distance gated peak indices", "sample": ["97", "34", "151"]},
            {"id": "sigbase", "tex": r"\sigma_{\mathrm{base}}", "role": "intermediate", "kind": "scalar", "shape": "1",   "desc": "span-derived width scale",                   "sample": "2.50"},
            {"id": "sig0",    "tex": r"\sigma^{(0)}",     "role": "intermediate", "kind": "vector", "shape": "K",         "desc": "shared initial width guess",                 "sample": ["0.62", "0.62", "0.62"]},
            {"id": "idxs",    "tex": r"\mathcal{I}",      "role": "intermediate", "kind": "set",    "shape": "K",         "desc": "final K seed indices (peaks + residual fill)", "sample": ["97", "34", "151", "12", "180"]},
            {"id": "mu0",     "tex": r"\mu",              "role": "intermediate", "kind": "vector", "shape": "K",         "desc": "component means (frozen)",                   "sample": ["12.4", "31.8", "47.0"]},
            {"id": "a0",      "tex": r"a",                "role": "intermediate", "kind": "vector", "shape": "K",         "desc": "normalised component amplitudes (frozen)",   "sample": ["0.92", "0.62", "0.34"]},
            {"id": "loss",    "tex": r"\mathcal{L}",      "role": "intermediate", "kind": "scalar", "shape": "1",         "desc": "per-pixel MSE on the normalised profile",    "sample": "3.4e-2"},
            {"id": "grad",    "tex": r"g_t",              "role": "intermediate", "kind": "vector", "shape": "K",         "desc": "width gradient at step t",                   "sample": ["-0.04", "0.01", "-0.02"]},
            {"id": "sigstar", "tex": r"\sigma^{*}",       "role": "calculated",   "kind": "vector", "shape": "K",         "desc": "fitted, clamped component widths",            "sample": ["1.84", "2.55", "3.10"]},
            {"id": "mseK",    "tex": r"\mathrm{MSE}_K",   "role": "intermediate", "kind": "vector", "shape": "K_max",     "desc": "per-order residual MSE (normalised)",         "sample": ["0.18", "0.04", "0.05", "0.07"]},
            {"id": "penK",    "tex": r"\mathcal{L}_K",    "role": "intermediate", "kind": "vector", "shape": "K_max",     "desc": "penalised score per model order K",           "sample": ["0.21", "0.06", "0.08", "0.11"]},
            {"id": "Kstar",   "tex": r"K^{*}",            "role": "calculated",   "kind": "scalar", "shape": "1",         "desc": "selected number of active components",        "sample": "2"},
            {"id": "aout",    "tex": r"a^{\mathrm{out}}", "role": "calculated",   "kind": "vector", "shape": "K",         "desc": "winner amplitudes rescaled to raw units",    "sample": ["0.81", "0.55"]},
            {"id": "theta",   "tex": r"\theta",           "role": "final",        "kind": "vector", "shape": "3K",        "desc": "ordered (a, mu, sigma) supervised target",   "sample": ["a_1", "mu_1", "sig_1", "a_2", "..."]},
            {"id": "r2",      "tex": r"R^2",              "role": "final",        "kind": "matrix", "shape": "A x R",     "desc": "per-pixel fit-quality map",                  "sample": [["0.97", "0.91"], ["0.88", "0.99"]]},
            {"id": "mrel",    "tex": r"m_{\mathrm{rel}}", "role": "final",        "kind": "matrix", "shape": "A x R",     "desc": "relative K-selection margin diagnostic",     "sample": [["0.31", "0.04"], ["0.12", "0.55"]]},
        ]
        steps = [
            {
                "id": "threshold", "title": "Profile floor and truncation", "phase": "0 - Conditioning",
                "note": "The elevation profile is the tomogram magnitude; samples below a relative floor are zeroed and the upper axis is truncated past index H_tr, the same preprocessing the loss and R2 see.",
                "inputs": ["T"], "outputs": ["P"],
                "lines": [
                    [{"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": "="}, {"id": "T", "tex": r"\left|T_h\right|", "role": "measured"}, {"tex": r"\cdot\,\mathbb{1}\!\left[\left|T_h\right| > t_f \max_h \left|T_h\right|\right]"}],
                    [{"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": r"\leftarrow 0\quad \text{for } h \ge H_{\mathrm{tr}}"}],
                ],
            },
            {
                "id": "activity", "title": "Active-pixel gate", "phase": "0 - Conditioning",
                "note": "A pixel is fitted only if its profile maximum exceeds the activity threshold; inactive pixels are skipped (parameters stay zero) and take scale 1.",
                "inputs": ["P"], "outputs": ["active", "scale"],
                "lines": [
                    [{"id": "active", "tex": r"\mathbb{1}_{\mathrm{act}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\mathbb{1}\!\left[\max_h "}, {"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": r" > \tau_a\right],\qquad \tau_a = 10^{-3}"}],
                    [{"id": "scale", "tex": r"s", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max_h P_h \ \ \text{if active, else } 1"}],
                ],
            },
            {
                "id": "pnorm", "title": "Peak normalisation", "phase": "0 - Conditioning",
                "note": "Dividing by the per-pixel maximum decouples the loss surface from absolute backscatter, so the MSE and penalty are comparable across pixels.",
                "inputs": ["P", "scale"], "outputs": ["gtilde"],
                "lines": [
                    [{"id": "gtilde", "tex": r"\tilde{\gamma}_h", "role": "intermediate"}, {"tex": "="}, {"tex": r"\dfrac{1}{"}, {"id": "scale", "tex": r"s", "role": "intermediate"}, {"tex": r"}\;"}, {"id": "P", "tex": r"P_h", "role": "intermediate"}],
                ],
            },
            {
                "id": "peakfind", "title": "Prominence-gated peak detection", "phase": "1 - Init (CPU)",
                "note": "find_peaks runs directly on the raw (un-normalised) profile, no smoothing, keeping a peak only when its topographic prominence clears a fraction of the maximum and it is at least d_min bins from every other peak.",
                "inputs": ["P"], "outputs": ["peaks"],
                "lines": [
                    [{"id": "peaks", "tex": r"\mathcal{P}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\left\{\,p : \mathrm{prom}(p) \ge p_{\mathrm{frac}}\,\max_h"}, {"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": r",\ \ |p_i-p_j|\ge d_{\min}\right\}"}],
                ],
            },
            {
                "id": "geometry", "title": "Width scales and clamp bounds", "phase": "1 - Init (CPU)",
                "note": "The span-derived sigma_base sets the residual-suppression distance and, via the divisor, the initial width; the Adam clamp bounds are one elevation bin and half the span.",
                "inputs": [], "outputs": ["sigbase", "sig0"],
                "lines": [
                    [{"id": "sigbase", "tex": r"\sigma_{\mathrm{base}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max\!\left(2\Delta\xi,\ \tfrac{x_{\max}-x_{\min}}{8K}\right)"}],
                    [{"id": "sig0", "tex": r"\sigma^{(0)}", "role": "intermediate"}, {"tex": "="}, {"id": "sigbase", "tex": r"\sigma_{\mathrm{base}}", "role": "intermediate"}, {"tex": r"/D_\sigma,\quad [\sigma_{\mathrm{lo}},\sigma_{\mathrm{hi}}] = \big[\Delta\xi,\ \tfrac{x_{\max}-x_{\min}}{2}\big]"}],
                ],
            },
            {
                "id": "residfill", "title": "Residual fill of remaining slots", "phase": "1 - Init (CPU)",
                "note": "If fewer than K peaks are found, a window of half-width d_min is zeroed around each detected peak and the remaining slots are filled by repeated argmax of the residual, guaranteeing every seed is d_min apart.",
                "inputs": ["peaks", "P"], "outputs": ["idxs"],
                "lines": [
                    [{"id": "idxs", "tex": r"\mathcal{I}", "role": "intermediate"}, {"tex": "="}, {"id": "peaks", "tex": r"\mathcal{P}", "role": "intermediate"}, {"tex": r"\ \cup\ \big\{\arg\max_h\,\rho_h\big\}^{\times (K-|\mathcal{P}|)}"}],
                    [{"tex": r"\rho_h"}, {"tex": "="}, {"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": r"\cdot\,\mathbb{1}\!\left[\min_{p\in\mathcal{P}}|h-p| > d_{\min}\right]"}],
                ],
            },
            {
                "id": "seed", "title": "Freeze amplitudes and means", "phase": "1 - Init (CPU)",
                "note": "Amplitude and mean are read off the raw-profile peaks and held fixed through Phase 2, reducing the fit to a well-conditioned 1D problem per component; an amplitude floor keeps every slot weakly active.",
                "inputs": ["idxs", "P", "scale"], "outputs": ["mu0", "a0"],
                "lines": [
                    [{"id": "mu0", "tex": r"\mu_k", "role": "intermediate"}, {"tex": "="}, {"tex": r"x_{\,\mathcal{I}_k},\qquad"}, {"id": "a0", "tex": r"a_k", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max\!\big(P_{\mathcal{I}_k},\,10^{-10}\big)\big/"}, {"id": "scale", "tex": r"s", "role": "intermediate"}],
                ],
            },
            {
                "id": "objective", "title": "Sigma-fit objective", "phase": "2 - Sigma fit (GPU)",
                "note": "With amplitudes and means frozen, only the K widths are optimised against the normalised profile. The exponent is clipped and sigma floored before squaring; vectorised with vmap, differentiated with value_and_grad.",
                "inputs": ["gtilde", "mu0", "a0", "sig0"], "outputs": ["loss"],
                "lines": [
                    [{"id": "loss", "tex": r"\mathcal{L}(\sigma)", "role": "intermediate"}, {"tex": "="}, {"tex": r"\dfrac{1}{H}\sum_{h}\left(\sum_{k}"}, {"id": "a0", "tex": r"a_k", "role": "intermediate"}, {"tex": r"e^{-\frac{(x_h-"}, {"id": "mu0", "tex": r"\mu_k", "role": "intermediate"}, {"tex": r")^2}{2\max(\sigma_k,10^{-6})^2}} -"}, {"id": "gtilde", "tex": r"\tilde{\gamma}_h", "role": "intermediate"}, {"tex": r"\right)^{2}"}],
                ],
            },
            {
                "id": "adam", "title": "Bias-corrected Adam width update", "phase": "2 - Sigma fit (GPU)",
                "note": "Hand-written bias-corrected Adam as a lax.scan over T = 3000 steps compiled into one XLA program; the widths are clamped to [sigma_lo, sigma_hi] before the loop and after every step.",
                "inputs": ["loss", "sig0"], "outputs": ["sigstar"],
                "iterative": {"var": "sigstar", "steps": 3000, "unit": "t", "symbol": "σ",
                              "trace": ["0.62", "0.81", "1.10", "1.45", "1.70", "1.82", "1.84"]},
                "lines": [
                    [{"id": "grad", "tex": r"g_t", "role": "intermediate"}, {"tex": "="}, {"tex": r"\nabla_{\sigma}"}, {"id": "loss", "tex": r"\mathcal{L}", "role": "intermediate"}, {"tex": r",\quad \hat m_t = \tfrac{m_t}{1-\beta_1^{t}},\ \ \hat v_t = \tfrac{v_t}{1-\beta_2^{t}}"}],
                    [{"id": "sigstar", "tex": r"\sigma^{*}_t", "role": "calculated"}, {"tex": "="}, {"tex": r"\mathrm{clip}\!\Big("}, {"id": "sigstar", "tex": r"\sigma^{*}_{t-1}", "role": "calculated"}, {"tex": r"-\eta\,\tfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},\ \sigma_{\mathrm{lo}},\ \sigma_{\mathrm{hi}}\Big)"}],
                ],
            },
            {
                "id": "scoreK", "title": "Per-order penalised score", "phase": "3 - Best-K",
                "note": "Every order K is scored on the normalised profile with a complexity penalty lambda_K K a-bar_K, where a-bar_K is the mean normalised amplitude, so the budget is spent only when peaks are genuinely present.",
                "inputs": ["gtilde", "sigstar", "a0"], "outputs": ["mseK", "penK"],
                "lines": [
                    [{"id": "mseK", "tex": r"\mathrm{MSE}_K", "role": "intermediate"}, {"tex": "="}, {"tex": r"\tfrac{1}{H}\sum_h\big(\hat\gamma_K(x_h) -"}, {"id": "gtilde", "tex": r"\tilde{\gamma}_h", "role": "intermediate"}, {"tex": r"\big)^2"}],
                    [{"id": "penK", "tex": r"\mathcal{L}_K", "role": "intermediate"}, {"tex": "="}, {"id": "mseK", "tex": r"\mathrm{MSE}_K", "role": "intermediate"}, {"tex": r"+\ \lambda_K\,K\,\bar{a}_K,\qquad \bar a_K = \tfrac{1}{K}\sum_{k\le K} a_k"}],
                ],
            },
            {
                "id": "selectK", "title": "Best-K argmin selection", "phase": "3 - Best-K",
                "note": "The penalised score is minimised over model order; on exact ties the smaller K wins, reinforcing parsimony. The choice is never overridden by ambiguity diagnostics.",
                "inputs": ["penK"], "outputs": ["Kstar"],
                "lines": [
                    [{"id": "Kstar", "tex": r"K^{*}", "role": "calculated"}, {"tex": "="}, {"tex": r"\operatorname*{arg\,min}_{K\in\{1,\dots,K_{\max}\}}"}, {"id": "penK", "tex": r"\mathcal{L}_K", "role": "intermediate"}],
                ],
            },
            {
                "id": "rescale", "title": "Rescale winner to raw amplitude", "phase": "3 - Best-K",
                "note": "Scoring lives on the normalised scale, but the saved amplitudes return to raw backscatter units; means and widths are written unchanged.",
                "inputs": ["Kstar", "a0", "scale"], "outputs": ["aout"],
                "lines": [
                    [{"id": "aout", "tex": r"a^{\mathrm{out}}_k", "role": "calculated"}, {"tex": "="}, {"id": "a0", "tex": r"a_k", "role": "intermediate"}, {"tex": r"\cdot"}, {"id": "scale", "tex": r"s", "role": "intermediate"}, {"tex": r",\qquad k \le"}, {"id": "Kstar", "tex": r"K^{*}", "role": "calculated"}],
                ],
            },
            {
                "id": "assemble", "title": "Order and pack the target", "phase": "3 - Best-K",
                "note": "Active components are sorted by ascending mean elevation, inactive slots keyed to infinity and pushed last, then written into the interleaved 3K target; slots beyond K* are exact zeros.",
                "inputs": ["Kstar", "aout", "mu0", "sigstar"], "outputs": ["theta"],
                "lines": [
                    [{"id": "theta", "tex": r"\theta", "role": "final"}, {"tex": "="}, {"tex": r"\Big[\,"}, {"id": "aout", "tex": r"a^{\mathrm{out}}", "role": "calculated"}, {"tex": ","}, {"id": "mu0", "tex": r"\mu", "role": "intermediate"}, {"tex": ","}, {"id": "sigstar", "tex": r"\sigma^{*}", "role": "calculated"}, {"tex": r"\,\Big]_{\pi}^{\,k \le"}, {"id": "Kstar", "tex": r"K^{*}", "role": "calculated"}, {"tex": r"},\ \ \pi=\operatorname{argsort}_k \mu_k"}],
                ],
            },
            {
                "id": "quality", "title": "Fit-quality R-squared map", "phase": "4 - Diagnostics",
                "note": "Per-pixel coefficient of determination over elevation, computed against the thresholded and truncated profile (the same preprocessing as the fit) with a small stabiliser on the total sum of squares.",
                "inputs": ["theta", "T"], "outputs": ["r2"],
                "lines": [
                    [{"id": "r2", "tex": r"R^2", "role": "final"}, {"tex": "="}, {"tex": r"1 - \dfrac{\sum_h \big(\hat{\gamma}_h("}, {"id": "theta", "tex": r"\theta", "role": "final"}, {"tex": r") - \gamma_h\big)^2}{\sum_h (\gamma_h - \bar{\gamma})^2 + \delta},\quad \delta = 10^{-12}"}],
                ],
            },
            {
                "id": "diagnostics", "title": "K-margin and peak contrast", "phase": "4 - Diagnostics",
                "note": "Post-hoc only and never altering selection: the relative selection margin flags ambiguous pixels, and the uncalibrated peak-to-floor contrast uses the lowest-quartile bins as the noise floor.",
                "inputs": ["penK", "T"], "outputs": ["mrel"],
                "lines": [
                    [{"id": "mrel", "tex": r"m_{\mathrm{rel}}", "role": "final"}, {"tex": "="}, {"tex": r"\dfrac{\mathcal{L}_{2\mathrm{nd}} - \mathcal{L}_{K^{*}}}{\max\!\big(|\mathcal{L}_{K^{*}}|,\,10^{-12}\big)}"}],
                    [{"tex": r"C_{\mathrm{dB}}"}, {"tex": "="}, {"tex": r"10\log_{10}\dfrac{\max_h |T_h|}{\frac{1}{|\mathcal{N}|}\sum_{h\in\mathcal{N}} |T_h|},\quad |\mathcal{N}|=\lceil 0.25 H\rceil"}],
                ],
            },
        ]
        return {
            "key"   : "param",
            "name"  : "Parameter Extraction (Fitting)",
            "blurb" : "Per-pixel multi-phase fit. Condition and normalise each elevation profile, seed a K-Gaussian mixture from prominence peaks on the CPU, fit the widths with clamped Adam on the GPU, select the penalised best order, then score and diagnose the fit.",
            "nodes" : nodes,
            "steps" : steps,
        }

    def _dataset(self) -> dict:
        nodes = [
            {"id": "Gcrop",  "tex": r"\Omega_G",          "role": "measured",     "kind": "set",    "shape": "(az0,az1,rg0,rg1)",  "desc": "global crop region in absolute pixel coords",                "sample": "(0, 4096, 0, 3000)"},
            {"id": "Rsplit", "tex": r"\Omega_s",          "role": "intermediate", "kind": "set",    "shape": "(az0,az1,rg0,rg1)",  "desc": "one split region (train/val/test)",                          "sample": "(0, 2867, 0, 3000)"},
            {"id": "azsl",   "tex": r"\sigma_a,\sigma_r", "role": "calculated",   "kind": "set",    "shape": "slices",             "desc": "zero-based local slices into the mmap arrays",               "sample": "[0:2867], [0:3000]"},
            {"id": "sel",    "tex": r"\pi",               "role": "calculated",   "kind": "vector", "shape": "N_s",                "desc": "positional indices of secondaries selected by label",        "sample": ["3", "5", "7", "25"]},
            {"id": "s0",     "tex": r"\mathbf{s}_0",      "role": "measured",     "kind": "matrix", "shape": "Az x Rg",            "desc": "primary (reference) complex SLC",                            "sample": [["0.8+0.2j", "0.6-0.4j"], ["0.5+0.5j", "0.9-0.1j"]]},
            {"id": "Smat",   "tex": r"S",                 "role": "measured",     "kind": "tensor", "shape": "N_s x Az x Rg",      "desc": "selected secondary complex SLC passes",                      "sample": [["0.7+0.1j"]]},
            {"id": "Imat",   "tex": r"I",                 "role": "measured",     "kind": "tensor", "shape": "N_i x Az x Rg",      "desc": "selected interferogram passes",                              "sample": [["0.9+0.4j"]]},
            {"id": "X",      "tex": r"\mathbf{X}",        "role": "measured",     "kind": "tensor", "shape": "(1+N_s+N_i) x Az x Rg", "desc": "stacked complex buffer: primary, secondaries, ifgs",       "sample": [["0.8+0.2j", "0.6-0.4j"], ["0.5+0.5j", "0.9-0.1j"]]},
            {"id": "dem",    "tex": r"\mathbf{D}",        "role": "measured",     "kind": "matrix", "shape": "Az x Rg",            "desc": "DEM elevation channel, real (optional)",                     "sample": [["112.4", "113.1"], ["110.8", "111.9"]]},
            {"id": "grid",   "tex": r"\mathcal{G}",       "role": "calculated",   "kind": "set",    "shape": "n_v x n_h",          "desc": "patch grid counts and padding",                              "sample": "n_v=89, n_h=92"},
            {"id": "cpatch", "tex": r"\mathbf{p}",        "role": "intermediate", "kind": "tensor", "shape": "C x P x P",          "desc": "one extracted complex patch (padded copy)",                  "sample": [["0.8+0.2j", "0.6-0.4j", "1.1+0.0j"], ["0.5+0.5j", "0.9-0.1j", "0.7+0.3j"], ["1.0+0.2j", "0.4-0.6j", "0.8+0.1j"]]},
            {"id": "rep",    "tex": r"\mathbf{r}",        "role": "intermediate", "kind": "tensor", "shape": "c x P x P",          "desc": "real channels of one complex pass",                          "sample": ["|s|", "ang", "re", "im"]},
            {"id": "x",      "tex": r"\mathbf{x}",        "role": "calculated",   "kind": "tensor", "shape": "C_in x P x P",       "desc": "assembled real-valued input tensor",                         "sample": [["0.91", "0.62", "1.10"], ["0.55", "0.88", "0.71"]]},
            {"id": "y",      "tex": r"\mathbf{y}",        "role": "calculated",   "kind": "tensor", "shape": "C_out x P x P",      "desc": "selected Gaussian-parameter target patch",                   "sample": ["a_1", "mu_1", "sig_1", "..."]},
            {"id": "xgeo",   "tex": r"\mathbf{x}'",       "role": "intermediate", "kind": "tensor", "shape": "C_in x P x P",       "desc": "geometrically augmented input (flip/rotation)",              "sample": [["0.93", "0.60", "1.08"], ["0.57", "0.86", "0.70"]]},
            {"id": "ygeo",   "tex": r"\mathbf{y}'",       "role": "intermediate", "kind": "tensor", "shape": "C_out x P x P",      "desc": "augmented target (same transform as input)",                 "sample": ["a_1", "mu_1", "sig_1", "..."]},
            {"id": "gkeys",  "tex": r"g_c",               "role": "intermediate", "kind": "vector", "shape": "C_in",               "desc": "per-channel slot keys (pass/mag, ifg/phase, ...)",            "sample": ["pass/mag", "ifg/phase", "..."]},
            {"id": "stats",  "tex": r"(\mu_c, s_c)",      "role": "calculated",   "kind": "vector", "shape": "C_in",               "desc": "per-slot location and scale, fitted on the train split",      "sample": ["0.12", "0.94", "..."]},
            {"id": "xhat",   "tex": r"\hat{\mathbf{x}}",  "role": "final",        "kind": "tensor", "shape": "C_in x P x P",       "desc": "normalised network input",                                   "sample": [["0.41", "-0.20", "0.88"], ["-0.33", "0.27", "0.05"]]},
            {"id": "yhat",   "tex": r"\hat{\mathbf{y}}",  "role": "final",        "kind": "tensor", "shape": "C_out x P x P",      "desc": "normalised target",                                          "sample": ["0.30", "-0.11", "0.42"]},
            {"id": "xrec",   "tex": r"\tilde{\mathbf{x}}","role": "calculated",   "kind": "tensor", "shape": "C_in x P x P",       "desc": "denormalised tensor (inverse, with expm1 ceiling)",          "sample": [["0.90", "0.61"], ["0.55", "0.88"]]},
        ]
        steps = [
            {
                "id": "splitgeom", "title": "Scene split by azimuth", "phase": "Crop & split",
                "note": "The global crop is partitioned along azimuth into contiguous train/val/test bands; the standard ratio split gives 70/15/15 with no overlap and the full range extent shared.",
                "inputs": ["Gcrop"], "outputs": ["Rsplit"],
                "lines": [
                    [{"tex": r"A = \mathtt{az}_1 - \mathtt{az}_0,\quad \mathtt{az}^{\mathrm{tr}}_1 = \mathtt{az}_0 + \lfloor 0.70\,A\rfloor"}],
                    [{"id": "Rsplit", "tex": r"\Omega_s", "role": "intermediate"}, {"tex": r"\in\ \big\{[\mathtt{az}_0,\mathtt{az}^{\mathrm{tr}}_1),\ [\mathtt{az}^{\mathrm{tr}}_1,\mathtt{az}^{\mathrm{val}}_1),\ [\mathtt{az}^{\mathrm{val}}_1,\mathtt{az}_1)\big\}\times[\mathtt{rg}_0,\mathtt{rg}_1)"}],
                ],
            },
            {
                "id": "localslice", "title": "Global-to-local indexing", "phase": "Crop & split",
                "note": "Each split region's absolute bounds are shifted by the global-crop origin to give zero-based slices into the memory-mapped artifact arrays.",
                "inputs": ["Rsplit", "Gcrop"], "outputs": ["azsl"],
                "lines": [
                    [{"id": "azsl", "tex": r"\sigma_a", "role": "calculated"}, {"tex": "="}, {"tex": r"\big[\,\mathtt{az}^S_0 - \mathtt{az}^G_0,\ \ \mathtt{az}^S_1 - \mathtt{az}^G_0\,\big),\qquad A_z = \mathtt{az}^S_1-\mathtt{az}^S_0"}],
                ],
            },
            {
                "id": "secselect", "title": "Secondary selection by label", "phase": "Crop & split",
                "note": "The processed stack carries every pass; the secondary subset is chosen by flight-qualified label, mapping labels to positional indices applied to both secondaries and interferograms.",
                "inputs": ["Smat", "Imat"], "outputs": ["sel"],
                "lines": [
                    [{"id": "sel", "tex": r"\pi", "role": "calculated"}, {"tex": "="}, {"tex": r"\{\,i : \ell_i \in L_{\mathrm{req}}\,\},\qquad \ell_0 \notin L_{\mathrm{req}}"}],
                    [{"id": "Smat", "tex": r"S", "role": "measured"}, {"tex": r"\leftarrow S[\pi],\quad"}, {"id": "Imat", "tex": r"I", "role": "measured"}, {"tex": r"\leftarrow I[\pi],\quad N_s = N_i = |\pi|"}],
                ],
            },
            {
                "id": "stack", "title": "Stacked complex input buffer", "phase": "Crop & split",
                "note": "Primary, selected secondaries, and selected interferograms are written by pass-index into one pre-allocated complex buffer; the DEM and parameters are cropped to the same window but kept separate.",
                "inputs": ["s0", "Smat", "Imat"], "outputs": ["X"],
                "lines": [
                    [{"id": "X", "tex": r"\mathbf{X}[0]", "role": "measured"}, {"tex": "="}, {"id": "s0", "tex": r"\mathbf{s}_0", "role": "measured"}, {"tex": r",\quad \mathbf{X}[1:1{+}N_s] = "}, {"id": "Smat", "tex": r"S", "role": "measured"}, {"tex": r",\quad \mathbf{X}[1{+}N_s:] = "}, {"id": "Imat", "tex": r"I", "role": "measured"}],
                ],
            },
            {
                "id": "patchgrid", "title": "Sliding-window grid", "phase": "Patch extraction",
                "note": "A regular strided grid of PxP patches tiles the region; the row and column counts use a ceiling so the last window still covers the border.",
                "inputs": ["X"], "outputs": ["grid"],
                "lines": [
                    [{"tex": r"n_v = \Big\lceil \tfrac{A_z - P_H}{s}\Big\rceil + 1,\qquad n_h = \Big\lceil \tfrac{R_g - P_W}{s}\Big\rceil + 1"}],
                    [{"id": "grid", "tex": r"N_p", "role": "calculated"}, {"tex": "="}, {"tex": r"n_v \cdot n_h"}],
                ],
            },
            {
                "id": "padgeom", "title": "Symmetric boundary padding", "phase": "Patch extraction",
                "note": "The deficit between the grid's covered extent and the region is split symmetrically, with the extra pixel going to the bottom or right on odd deficits; border patches reflect-pad the overhang.",
                "inputs": ["grid"], "outputs": ["grid"],
                "lines": [
                    [{"tex": r"p_v = P_H + (n_v - 1)\,s - A_z,\qquad p_h = P_W + (n_h - 1)\,s - R_g"}],
                    [{"tex": r"p_{\mathrm{top}} = \lfloor p_v/2\rfloor,\quad p_{\mathrm{bot}} = p_v - p_{\mathrm{top}}"}],
                ],
            },
            {
                "id": "extract", "title": "Patch extraction (contiguous copy)", "phase": "Patch extraction",
                "note": "The clipped read window is copied, never aliased, then reflect-padded in one pass; the same routine serves the complex stack, the parameters, and the DEM.",
                "inputs": ["X", "dem", "grid"], "outputs": ["cpatch"],
                "lines": [
                    [{"id": "cpatch", "tex": r"\mathbf{p}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\mathrm{pad}\!\big("}, {"id": "X", "tex": r"\mathbf{X}", "role": "measured"}, {"tex": r"[:,\ v_0^c{:}v_1^c,\ h_0^c{:}h_1^c]\big)"}],
                ],
            },
            {
                "id": "represent", "title": "Complex to real channels", "phase": "Patch extraction",
                "note": "Each complex pass is converted to real channels by its representation mode; the default keeps magnitude for SLCs and phase for interferograms. Magnitude-normalised modes guard zero magnitude by substituting one.",
                "inputs": ["cpatch"], "outputs": ["rep"],
                "lines": [
                    [{"tex": r"\texttt{MAG}:\ (\left|\mathbf{p}\right|)\quad \texttt{ANG}:\ (\angle\mathbf{p})\quad \texttt{RI}:\ (\Re\mathbf{p},\ \Im\mathbf{p})"}],
                    [{"id": "rep", "tex": r"\mathbf{r}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\Big(\left|"}, {"id": "cpatch", "tex": r"\mathbf{p}", "role": "intermediate"}, {"tex": r"\right|,\ \tfrac{\Re\mathbf{p}}{m},\ \tfrac{\Im\mathbf{p}}{m},\ \angle\mathbf{p}\Big),\quad m=\max(\left|\mathbf{p}\right|,1)"}],
                ],
            },
            {
                "id": "assemble_in", "title": "Input tensor assembly", "phase": "Patch extraction",
                "note": "Per patch, every enabled source is represented and concatenated along the channel axis: primary, then secondaries, then interferograms, with the optional DEM channel last; the secondary and ifg counts are independent.",
                "inputs": ["rep", "dem"], "outputs": ["x"],
                "lines": [
                    [{"id": "x", "tex": r"\mathbf{x}", "role": "calculated"}, {"tex": "="}, {"tex": r"\big[\ "}, {"id": "rep", "tex": r"\mathbf{r}_0", "role": "intermediate"}, {"tex": r"\mid \mathbf{r}_S \mid \mathbf{r}_I \mid"}, {"id": "dem", "tex": r"\mathbf{D}", "role": "measured"}, {"tex": r"\ \big]"}],
                    [{"tex": r"C_{\mathrm{in}} = c_{\mathrm{prim}} + N_s\,c_{\mathrm{sec}} + N_i\,c_{\mathrm{ifg}} + c_{\mathrm{dem}}"}],
                ],
            },
            {
                "id": "target", "title": "Target channel selection", "phase": "Patch extraction",
                "note": "The configured subset of Gaussian parameters is selected from the interleaved ground-truth layout; with all three roles enabled it keeps every channel.",
                "inputs": [], "outputs": ["y"],
                "lines": [
                    [{"id": "y", "tex": r"\mathbf{y}", "role": "calculated"}, {"tex": "="}, {"tex": r"\Theta\big[\{3g + r : g<n_g,\ r\in\mathcal{R}\}\big],\qquad C_{\mathrm{out}} = n_g\,|\mathcal{R}|"}],
                ],
            },
            {
                "id": "augment_geo", "title": "Joint geometric augmentation", "phase": "Augmentation",
                "note": "On the train split only, flips and an optional 90-degree rotation are applied with the identical transform to input and target, preserving spatial correspondence; this runs on the raw patch before normalisation.",
                "inputs": ["x", "y"], "outputs": ["xgeo", "ygeo"],
                "lines": [
                    [{"id": "xgeo", "tex": r"\mathbf{x}'", "role": "intermediate"}, {"tex": "="}, {"tex": r"\mathcal{T}\big("}, {"id": "x", "tex": r"\mathbf{x}", "role": "calculated"}, {"tex": r"\big),\qquad"}, {"id": "ygeo", "tex": r"\mathbf{y}'", "role": "intermediate"}, {"tex": "="}, {"tex": r"\mathcal{T}\big("}, {"id": "y", "tex": r"\mathbf{y}", "role": "calculated"}, {"tex": r"\big)"}],
                    [{"tex": r"\mathcal{T} \in \big\{\mathrm{flip}_H,\ \mathrm{flip}_V,\ \operatorname{rot90}^{k}\big\},\quad k\sim\mathcal{U}\{1,2,3\}"}],
                ],
            },
            {
                "id": "slotkeys", "title": "Per-channel slot assignment", "phase": "Normalization",
                "note": "Each input channel is labelled with a slot key by the same strided layout used to build the tensor, fixing which normalisation strategy each channel receives.",
                "inputs": ["x"], "outputs": ["gkeys"],
                "lines": [
                    [{"id": "gkeys", "tex": r"g_c", "role": "intermediate"}, {"tex": "="}, {"tex": r"\mathrm{family}\,/\,\mathrm{slot}\big[\,c \bmod c_{pp}\,\big]\quad (\texttt{pass/mag},\ \texttt{ifg/phase},\ \texttt{dem/elev})"}],
                ],
            },
            {
                "id": "fitstats", "title": "Fit normalisation statistics", "phase": "Normalization",
                "note": "Statistics are fitted on the train split only, per slot, in float64; the live mapping is z-score with an optional log1p compression applied before fitting, and the scale floored at 1e-8.",
                "inputs": ["xgeo", "gkeys"], "outputs": ["stats"],
                "lines": [
                    [{"tex": r"f(x_c) = \log\!\big(1 + \max(x_c,0)\big)\ \ \text{(log1p slots)},\ \ \text{else } x_c"}],
                    [{"id": "stats", "tex": r"(\mu_c, s_c)", "role": "calculated"}, {"tex": "="}, {"tex": r"\Big(\operatorname{mean} f(x_c),\ \ \max\!\big(\operatorname{std} f(x_c),\ 10^{-8}\big)\Big)"}],
                ],
            },
            {
                "id": "normalise", "title": "Forward normalisation", "phase": "Normalization",
                "note": "The fitted statistics are applied identically to every split, yielding the dimensionless tensors the network consumes; the target is normalised the same way when output stats exist.",
                "inputs": ["xgeo", "ygeo", "stats"], "outputs": ["xhat", "yhat"],
                "lines": [
                    [{"id": "xhat", "tex": r"\hat{\mathbf{x}}", "role": "final"}, {"tex": "="}, {"tex": r"\dfrac{f("}, {"id": "xgeo", "tex": r"\mathbf{x}'", "role": "intermediate"}, {"tex": r") - "}, {"id": "stats", "tex": r"\mu_c", "role": "calculated"}, {"tex": r"}{"}, {"id": "stats", "tex": r"s_c", "role": "calculated"}, {"tex": r"},\qquad"}, {"id": "yhat", "tex": r"\hat{\mathbf{y}}", "role": "final"}, {"tex": "="}, {"tex": r"\dfrac{f(\mathbf{y}') - \mu_c}{s_c}"}],
                ],
            },
            {
                "id": "noise", "title": "Additive noise", "phase": "Normalization",
                "note": "On the train split only, Gaussian noise is added to the already-normalised input with probability p_N, so the noise std is in normalised units; noise never touches the target.",
                "inputs": ["xhat"], "outputs": ["xhat"],
                "lines": [
                    [{"id": "xhat", "tex": r"\hat{\mathbf{x}}", "role": "final"}, {"tex": r"\leftarrow"}, {"id": "xhat", "tex": r"\hat{\mathbf{x}}", "role": "final"}, {"tex": r"+\ \varepsilon,\quad \varepsilon \sim \mathcal{N}\!\big(0,\ \sigma_{\mathrm{noise}}^2 \mathbf{I}\big),\quad \sigma_{\mathrm{noise}}=0.01"}],
                ],
            },
            {
                "id": "denorm", "title": "Denormalisation (inverse)", "phase": "Inverse",
                "note": "The same per-channel statistics invert normalisation at loss and inference time; the log1p inverse clamps the expm1 argument at 80 to prevent float32 overflow.",
                "inputs": ["xhat", "stats"], "outputs": ["xrec"],
                "lines": [
                    [{"id": "xrec", "tex": r"\tilde{\mathbf{x}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\exp\!\big(\min(\hat{x}_c s_c + \mu_c,\ 80)\big) - 1\ \ \text{(log1p)},\ \ \text{else } \hat{x}_c s_c + \mu_c"}],
                ],
            },
        ]
        return {
            "key": "dataset", "name": "Dataset (Loaders)",
            "blurb": "Processed artifacts become PyTorch tensors: split the scene, select secondaries, stack and tile into patches, convert complex passes to real channels, assemble and augment, then apply train-fitted per-slot normalisation.",
            "nodes": nodes, "steps": steps,
        }

    def _training(self) -> dict:
        nodes = [
            {"id": "xhat",  "tex": r"\hat{\mathbf{x}}",            "role": "measured",     "kind": "tensor", "shape": "B x C_in x P x P", "desc": "normalised input batch from the loader",        "sample": [["0.41", "-0.20"], ["-0.33", "0.27"]]},
            {"id": "thn",   "tex": r"\hat{\theta}_{\mathrm{n}}",   "role": "calculated",   "kind": "tensor", "shape": "B x 3K x P x P",   "desc": "raw network output in normalised space",        "sample": ["0.78", "0.55", "0.31", "..."]},
            {"id": "thp",   "tex": r"\hat{\theta}",                "role": "calculated",   "kind": "tensor", "shape": "B x 3K x P x P",   "desc": "denormalised, clamped physical parameters",     "sample": ["12.4", "8.1", "1.9", "..."]},
            {"id": "thrn",  "tex": r"\tilde{\theta}_{\mathrm{n}}", "role": "intermediate", "kind": "tensor", "shape": "B x 3K x P x P",   "desc": "clamped params renormalised for param terms",   "sample": ["0.74", "0.55", "0.33", "..."]},
            {"id": "gtp",   "tex": r"\theta^{\mathrm{GT}}",        "role": "measured",     "kind": "tensor", "shape": "B x 3K x P x P",   "desc": "ground-truth params, physical units",           "sample": ["12.8", "8.0", "1.8", "..."]},
            {"id": "yhat",  "tex": r"\hat{y}",                     "role": "intermediate", "kind": "tensor", "shape": "B x N x P x P",    "desc": "reconstructed predicted elevation curve",       "sample": [["0.05", "0.71"], ["0.33", "0.88"]]},
            {"id": "y",     "tex": r"y",                           "role": "measured",     "kind": "tensor", "shape": "B x N x P x P",    "desc": "ground-truth reconstructed curve",              "sample": [["0.06", "0.70"], ["0.35", "0.86"]]},
            {"id": "err",   "tex": r"e",                           "role": "intermediate", "kind": "tensor", "shape": "B x N x P x P",    "desc": "curve-space residual y-hat minus y",            "sample": ["-0.01", "0.01", "-0.02", "..."]},
            {"id": "Astr",  "tex": r"\mathbf{A}",                  "role": "intermediate", "kind": "matrix", "shape": "N_s x N",         "desc": "tomographic steering matrix exp(j kz xi)",      "sample": [["1+0j", "0.7+0.7j"], ["1+0j", "-0.7+0.7j"]]},
            {"id": "Rcov",  "tex": r"\mathbf{R}[P]",               "role": "intermediate", "kind": "matrix", "shape": "N_s x N_s",       "desc": "synthesised covariance A diag(P) A^H dxi",      "sample": [["3.1", "1.2+0.4j"], ["1.2-0.4j", "2.8"]]},
            {"id": "lj",    "tex": r"\ell_j",                      "role": "calculated",   "kind": "scalar", "shape": "1",               "desc": "raw value of one enabled loss term",            "sample": "0.0317"},
            {"id": "loss",  "tex": r"\mathcal{L}",                 "role": "calculated",   "kind": "scalar", "shape": "1",               "desc": "normalised weighted composite loss",            "sample": "4.1e-2"},
            {"id": "gnorm", "tex": r"\lVert\mathbf{g}\rVert_2",    "role": "intermediate", "kind": "scalar", "shape": "1",               "desc": "global gradient L2 norm",                       "sample": "2.7"},
            {"id": "grad",  "tex": r"\mathbf{g}",                  "role": "intermediate", "kind": "vector", "shape": "|theta|",         "desc": "clipped parameter gradient",                    "sample": ["1.2e-2", "-4e-3", "..."]},
            {"id": "eta",   "tex": r"\eta_{\mathrm{eff}}",         "role": "intermediate", "kind": "scalar", "shape": "1",               "desc": "effective LR (base x schedule x warmup)",       "sample": "7.3e-4"},
            {"id": "w",     "tex": r"\theta_t",                    "role": "intermediate", "kind": "vector", "shape": "|theta|",         "desc": "model weights at optimiser step t",             "sample": ["0.31", "-0.08", "..."]},
            {"id": "wbest", "tex": r"\theta^{\star}",              "role": "final",        "kind": "vector", "shape": "|theta|",         "desc": "best-epoch checkpointed weights",               "sample": ["0.30", "-0.09", "..."]},
        ]
        steps = [
            {
                "id": "forward", "title": "Forward pass", "phase": "Reconstruction",
                "note": "The network maps the normalised input patch to per-pixel Gaussian parameters (interleaved a, mu, sigma per component) in one autocast forward pass.",
                "inputs": ["xhat"], "outputs": ["thn"],
                "lines": [
                    [{"id": "thn", "tex": r"\hat{\theta}_{\mathrm{n}}", "role": "calculated"}, {"tex": "="}, {"tex": r"f_{\theta}\!\big("}, {"id": "xhat", "tex": r"\hat{\mathbf{x}}", "role": "measured"}, {"tex": r"\big)"}],
                ],
            },
            {
                "id": "tdenorm", "title": "Denormalise predictions", "phase": "Reconstruction",
                "note": "Each log1p-encoded channel (amplitude, sigma) is inverted with expm1; the exponent argument is capped at 80 so a pathological early prediction cannot produce a NaN in the backward pass.",
                "inputs": ["thn"], "outputs": ["thp"],
                "lines": [
                    [{"id": "thp", "tex": r"\hat{\theta}", "role": "calculated"}, {"tex": "="}, {"tex": r"\exp\!\mathrm{m1}\big(\min(\hat{\theta}_{\mathrm{n}}\,s + \ell,\ 80)\big)\ \ \text{(log1p)},\ \ \text{else } \hat{\theta}_{\mathrm{n}}\,s + \ell"}],
                ],
            },
            {
                "id": "clamp", "title": "Physical parameter bounds", "phase": "Reconstruction",
                "note": "Predictions are clamped to grid-relative physical bounds with a leaky straight-through slope of 0.01, so the amplitude and sigma heads keep a small gradient through saturation.",
                "inputs": ["thp"], "outputs": ["thp"],
                "lines": [
                    [{"id": "thp", "tex": r"\hat{a}_k \in [0,a_{\max}],\ \ \hat{\mu}_k \in [x_{\min},x_{\max}],\ \ \hat{\sigma}_k \in \big[\tfrac{\Delta x}{2},\tfrac{x_{\max}-x_{\min}}{2}\big]", "role": "calculated"}],
                    [{"tex": r"\mathrm{clamp}_{\mathrm{leaky}}(x) = \mathrm{clip}(x) + 0.01\,\big(x - \mathrm{clip}(x)\big).\mathrm{detach}()"}],
                ],
            },
            {
                "id": "renorm", "title": "Renormalise for parameter terms", "phase": "Reconstruction",
                "note": "The clamped physical predictions are mapped back to training space so the parameter-space loss terms operate in the same normalised units as the labels.",
                "inputs": ["thp"], "outputs": ["thrn"],
                "lines": [
                    [{"id": "thrn", "tex": r"\tilde{\theta}_{\mathrm{n}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\big(\log\!1\mathrm{p}("}, {"id": "thp", "tex": r"\hat{\theta}", "role": "calculated"}, {"tex": r") - \ell\big)\,/\,s"}],
                ],
            },
            {
                "id": "reconstruct", "title": "Gaussian curve reconstruction", "phase": "Reconstruction",
                "note": "Both predicted and ground-truth parameters are evaluated on the elevation axis as an additive Gaussian mixture; the GT curve is built once under no_grad.",
                "inputs": ["thp", "gtp"], "outputs": ["yhat", "y"],
                "lines": [
                    [{"id": "yhat", "tex": r"\hat{y}(x_n)", "role": "intermediate"}, {"tex": "="}, {"tex": r"\sum_{k}"}, {"id": "thp", "tex": r"\hat{a}_k", "role": "calculated"}, {"tex": r"\exp\!\Big(-\tfrac{(x_n-\hat{\mu}_k)^2}{2\hat{\sigma}_k^2}\Big),\qquad K = C/3"}],
                ],
            },
            {
                "id": "residual", "title": "Curve residual", "phase": "Curve loss",
                "note": "The elementwise residual is computed once and shared by the MSE, L1, Huber and Charbonnier terms; shape-sensitive terms take the curves directly.",
                "inputs": ["yhat", "y"], "outputs": ["err"],
                "lines": [
                    [{"id": "err", "tex": r"e", "role": "intermediate"}, {"tex": "="}, {"id": "yhat", "tex": r"\hat{y}", "role": "intermediate"}, {"tex": r"\ -\ "}, {"id": "y", "tex": r"y", "role": "measured"}],
                ],
            },
            {
                "id": "curvepoint", "title": "Pointwise curve terms", "phase": "Curve loss",
                "note": "Four pointwise reductions of the shared residual, each averaged over all elements; Huber transitions at delta and Charbonnier uses an epsilon-smoothed L1.",
                "inputs": ["err"], "outputs": ["lj"],
                "lines": [
                    [{"tex": r"\ell_{\mathrm{MSE}} = \big\langle e^2\big\rangle,\qquad \ell_{L1} = \big\langle |e| \big\rangle"}],
                    [{"id": "lj", "tex": r"\ell_{\mathrm{Hub}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\big\langle \tfrac12 e^2\,[|e|\le\delta] + \delta(|e|-\tfrac{\delta}{2})\,[|e|>\delta]\big\rangle"}],
                    [{"tex": r"\ell_{\mathrm{Charb}} = \big\langle \sqrt{e^2 + \varepsilon^2}\big\rangle"}],
                ],
            },
            {
                "id": "curveshape", "title": "Shape-sensitive curve terms", "phase": "Curve loss",
                "note": "Three terms comparing curve shape rather than magnitude: cosine over valid pixels, a windowed spectral coherence, and per-slice SSIM on jointly normalised images.",
                "inputs": ["yhat", "y"], "outputs": ["lj"],
                "lines": [
                    [{"id": "lj", "tex": r"\ell_{\cos}", "role": "calculated"}, {"tex": "="}, {"tex": r"\Big\langle 1 - \tfrac{\langle\hat{y},y\rangle}{\lVert\hat{y}\rVert\,\lVert y\rVert}\Big\rangle_{\lVert y\rVert>10^{-3}}"}],
                    [{"tex": r"\ell_{\mathrm{SSIM}} = 1 - \overline{\mathrm{SSIM}},\quad \mathrm{SSIM} = \tfrac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}"}],
                ],
            },
            {
                "id": "physgeom", "title": "Tomographic forward operator", "phase": "Physics loss",
                "note": "The physics terms share a Fourier forward operator. The vertical wavenumber uses the monostatic 4-pi factor with the master-relative perpendicular baseline projected by the look angle. These terms are disabled by default.",
                "inputs": [], "outputs": ["Astr"],
                "lines": [
                    [{"tex": r"b_\perp = h\cos\theta + v\sin\theta,\qquad k_z^{(i)} = \tfrac{4\pi\, b_i}{\lambda\, r_0}"}],
                    [{"id": "Astr", "tex": r"A_{in}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\exp\!\big(j\,k_z^{(i)}\,\xi_n\big)"}],
                ],
            },
            {
                "id": "physmoments", "title": "Power and moment terms", "phase": "Physics loss",
                "note": "Ratio-based physics terms compare the relative integrated power and the first three profile moments (mass, centroid, spread), reduced over GT-strong pixels.",
                "inputs": ["yhat", "y"], "outputs": ["lj"],
                "lines": [
                    [{"tex": r"m_0 = \textstyle\sum_n P_n\,\Delta\xi,\quad \bar z = \tfrac{\sum_n P_n \xi_n}{\sum_n P_n},\quad \sigma_z = \sqrt{\tfrac{\sum_n P_n \xi_n^2}{\sum_n P_n} - \bar z^2 + 10^{-8}}"}],
                    [{"id": "lj", "tex": r"\ell_{\mathrm{mom}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\Big\langle \tfrac{w_0\frac{|\Delta m_0|}{m_0^T} + w_1\frac{|\Delta\bar z|}{\xi_{\max}-\xi_{\min}} + w_2\frac{|\Delta\sigma_z|}{\xi_{\max}-\xi_{\min}}}{w_0+w_1+w_2}\Big\rangle"}],
                ],
            },
            {
                "id": "physcov", "title": "Coherence and covariance matching", "phase": "Physics loss",
                "note": "Coherence re-synthesis compares the normalised characteristic functions of the two profiles; covariance matching exploits the linearity of R to transform only the difference, both insensitive to absolute power.",
                "inputs": ["yhat", "y", "Astr"], "outputs": ["Rcov", "lj"],
                "lines": [
                    [{"tex": r"\gamma_P(k_z^{(i)}) = \tfrac{\sum_n P_n e^{j k_z^{(i)}\xi_n}}{\sum_n P_n},\qquad \ell_{\mathrm{coh}} = \big\langle \tfrac{1}{N_s}\textstyle\sum_i |\gamma_P^{(i)} - \gamma_T^{(i)}|^2\big\rangle"}],
                    [{"id": "Rcov", "tex": r"\mathbf{R}[P]", "role": "intermediate"}, {"tex": "="}, {"id": "Astr", "tex": r"\mathbf{A}", "role": "intermediate"}, {"tex": r"\,\mathrm{diag}(P)\,\mathbf{A}^H\Delta\xi,\quad \ell_{\mathrm{cov}} = \big\langle \tfrac{\lVert\mathbf{R}[P]-\mathbf{R}[T]\rVert_F^2}{\lVert\mathbf{R}[T]\rVert_F^2}\big\rangle"}],
                ],
            },
            {
                "id": "physcapon", "title": "Capon cycle-consistency", "phase": "Physics loss",
                "note": "The most expensive physics term synthesises the covariance from the predicted profile, applies signal-adaptive diagonal loading, then forms the Capon spectrum via one solve per pixel and compares mass-normalised spectra.",
                "inputs": ["Rcov", "Astr", "y"], "outputs": ["lj"],
                "lines": [
                    [{"tex": r"\hat{T}_P(\xi_n) = \dfrac{1}{\mathbf{a}^H(\xi_n)\big("}, {"id": "Rcov", "tex": r"\mathbf{R}[P]", "role": "intermediate"}, {"tex": r" + \epsilon\bar\sigma\mathbf{I}\big)^{-1}\mathbf{a}(\xi_n)},\quad \bar\sigma = \tfrac{1}{N_s}\mathrm{tr}\,\mathbf{R}[P]"}],
                    [{"id": "lj", "tex": r"\ell_{\mathrm{cyc}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\Big\langle \tfrac{1}{N}\textstyle\sum_n\big(\tfrac{\hat{T}_P(\xi_n)}{m_0^{\hat T}} - \tfrac{T(\xi_n)}{m_0^{T}}\big)^2\Big\rangle"}],
                ],
            },
            {
                "id": "paramterms", "title": "Parameter-space terms", "phase": "Parameter loss",
                "note": "GT components are sorted by mean; inactive GT slots mask their mu and sigma to zero weight so only amplitude contributes for empty slots. Param-L1/Huber act in normalised space, TV penalises spatial irregularity.",
                "inputs": ["thrn", "gtp"], "outputs": ["lj"],
                "lines": [
                    [{"id": "lj", "tex": r"\ell_{\mathrm{p\text{-}L1}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\big\langle w_p\,m\,|"}, {"id": "thrn", "tex": r"\tilde{\theta}_{\mathrm{n}}", "role": "intermediate"}, {"tex": r" - \theta|\big\rangle,\quad m = \mathbb{1}[a^{\mathrm{GT}}\ge 10^{-3}]\ (\mu,\sigma)"}],
                    [{"tex": r"\ell_{\mathrm{TV}} = \overline{|\tilde\theta_{h}-\tilde\theta_{h-1}|} + \overline{|\tilde\theta_{w}-\tilde\theta_{w-1}|}"}],
                ],
            },
            {
                "id": "composite", "title": "Composite weighted loss", "phase": "Composite",
                "note": "The total loss is the effective-weight-normalised sum over enabled terms; each effective weight is the user weight times a fixed empirical normaliser that brings heterogeneous terms to roughly unit magnitude.",
                "inputs": ["lj"], "outputs": ["loss"],
                "lines": [
                    [{"tex": r"\mathrm{eff}_j = \alpha_j\,\nu_j,\qquad"}, {"id": "loss", "tex": r"\mathcal{L}", "role": "calculated"}, {"tex": "="}, {"tex": r"\dfrac{\sum_j \mathrm{eff}_j\,"}, {"id": "lj", "tex": r"\ell_j", "role": "calculated"}, {"tex": r"}{\sum_j \mathrm{eff}_j}"}],
                ],
            },
            {
                "id": "gradclip", "title": "Backprop and gradient clipping", "phase": "Optimiser step",
                "note": "After a finiteness guard, all gradients are rescaled by a common factor so the global norm never exceeds the threshold, which is fixed or an adaptive percentile over recent norms.",
                "inputs": ["loss"], "outputs": ["gnorm", "grad"],
                "lines": [
                    [{"id": "gnorm", "tex": r"\lVert\mathbf{g}\rVert_2", "role": "intermediate"}, {"tex": "="}, {"tex": r"\Big(\textstyle\sum_i \lVert\nabla_{\theta^{(i)}}\mathcal{L}\rVert_2^2\Big)^{1/2}"}],
                    [{"id": "grad", "tex": r"\mathbf{g}", "role": "intermediate"}, {"tex": r"\leftarrow \mathbf{g}\cdot\min\!\Big(1,\ \tfrac{\tau}{\lVert\mathbf{g}\rVert_2 + \varepsilon}\Big),\quad \tau\in\{\tau_{\mathrm{fix}},\ P_{95},\ \bar g + k\sigma_g\}"}],
                ],
            },
            {
                "id": "adamw", "title": "AdamW update", "phase": "Optimiser step",
                "note": "Bias-corrected adaptive moments with decoupled weight decay and per-group learning rates; the epoch loop drives the training loss down over many steps.",
                "inputs": ["grad", "eta"], "outputs": ["w"],
                "iterative": {"var": "loss", "steps": 100, "unit": "epoch", "symbol": "L",
                              "trace": ["4.1e-2", "2.7e-2", "1.9e-2", "1.4e-2", "1.1e-2", "9.6e-3"]},
                "lines": [
                    [{"tex": r"\hat m_t = \tfrac{m_t}{1-\beta_1^t},\quad \hat v_t = \tfrac{v_t}{1-\beta_2^t}"}],
                    [{"id": "w", "tex": r"\theta_{t+1}", "role": "intermediate"}, {"tex": "="}, {"id": "w", "tex": r"\theta_t", "role": "intermediate"}, {"tex": r"\ -\ \eta_{\mathrm{eff}}\Big(\tfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_t\Big)"}],
                ],
            },
            {
                "id": "schedule", "title": "Warmup, cosine schedule, curriculum", "phase": "Schedule & curriculum",
                "note": "Each group's effective LR is its base rate times the per-epoch cosine factor times the per-step warmup factor; at the swap epoch the loss curriculum moves from the warmup objective to the complete objective.",
                "inputs": [], "outputs": ["eta"],
                "lines": [
                    [{"tex": r"F(t) = \tfrac{\eta_{\min}}{\eta_0} + \tfrac12\big(1 - \tfrac{\eta_{\min}}{\eta_0}\big)\big(1 + \cos\tfrac{\pi\min(t,T)}{T}\big)"}],
                    [{"id": "eta", "tex": r"\eta_{\mathrm{eff}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\eta_0\cdot F(t)\cdot f_{\mathrm{warmup}}(s),\qquad f_{\mathrm{warmup}}(s) = \alpha_0 + (1-\alpha_0)\tfrac{s}{S}"}],
                ],
            },
            {
                "id": "checkpoint", "title": "Validation and checkpoint", "phase": "Eval & checkpoint",
                "note": "Evaluation runs every few epochs; the best epoch is checkpointed on strict improvement and early stopping reverts to it after patience evaluations without a new minimum. The best-loss baseline resets across a curriculum swap.",
                "inputs": ["w"], "outputs": ["wbest"],
                "lines": [
                    [{"id": "wbest", "tex": r"\theta^{\star}", "role": "final"}, {"tex": "="}, {"tex": r"\operatorname*{arg\,min}_{t}\ \mathcal{L}_{\mathrm{val}}("}, {"id": "w", "tex": r"\theta_t", "role": "intermediate"}, {"tex": r")"}],
                ],
            },
        ]
        return {
            "key": "training", "name": "Training (Supervised)",
            "blurb": "One forward pass predicts all Gaussian parameters; predictions are denormalised, clamped and renormalised, then a composite loss over curve, parameter and optional physics space is backpropagated through AdamW with warmup, cosine scheduling, a loss curriculum, and early stopping.",
            "nodes": nodes, "steps": steps,
        }

    def _inference(self) -> dict:
        nodes = [
            {"id": "ckpt",   "tex": r"\theta^{\star}",      "role": "measured",     "kind": "vector", "shape": "|theta|",      "desc": "best-epoch checkpoint weights",             "sample": ["0.31", "-0.08", "..."]},
            {"id": "xaxis",  "tex": r"\mathbf{x}",          "role": "measured",     "kind": "vector", "shape": "N",            "desc": "elevation axis (m) restored from checkpoint", "sample": ["-20", "-19.6", "...", "20"]},
            {"id": "xhat",   "tex": r"\hat{\mathbf{x}}",    "role": "measured",     "kind": "tensor", "shape": "B x C_in x P x P", "desc": "normalised input patch batch",         "sample": [["0.41", "-0.20"], ["-0.33", "0.27"]]},
            {"id": "znorm",  "tex": r"\hat{\mathbf{z}}",    "role": "intermediate", "kind": "tensor", "shape": "B x 3K x P x P", "desc": "raw normalised network output",          "sample": ["0.22", "-1.1", "0.7", "..."]},
            {"id": "th",     "tex": r"\hat{\theta}",        "role": "calculated",   "kind": "tensor", "shape": "B x 3K x P x P", "desc": "denormalised, hard-clamped params",      "sample": ["0.78", "12.1", "1.9", "..."]},
            {"id": "thgt",   "tex": r"\theta^{\mathrm{GT}}","role": "measured",     "kind": "tensor", "shape": "B x 3K x P x P", "desc": "ground-truth params, denormalised",      "sample": ["0.80", "12.0", "1.8", "..."]},
            {"id": "thgts",  "tex": r"\theta^{\mathrm{GT}}_{\pi}", "role": "intermediate", "kind": "tensor", "shape": "B x 3K x P x P", "desc": "mu-sorted GT params (prediction unsorted)", "sample": ["0.80", "11.9", "1.8", "..."]},
            {"id": "p",      "tex": r"\mathbf{p}",          "role": "intermediate", "kind": "tensor", "shape": "N x P x P",    "desc": "reconstructed predicted patch spectrum",   "sample": [["0.05", "0.71"], ["0.33", "0.88"]]},
            {"id": "winv",   "tex": r"w_v",                 "role": "intermediate", "kind": "vector", "shape": "P",            "desc": "1D Hann taper, floored at 1e-3",           "sample": ["0.10", "0.45", "0.85", "1.00"]},
            {"id": "win",    "tex": r"w",                   "role": "intermediate", "kind": "matrix", "shape": "P x P",        "desc": "separable 2D overlap-add window",          "sample": [["0.10", "0.45"], ["0.45", "1.00"]]},
            {"id": "acc",    "tex": r"A",                   "role": "intermediate", "kind": "tensor", "shape": "C x H x Rg",   "desc": "value accumulator",                        "sample": [["1.2", "3.4"], ["0.8", "2.1"]]},
            {"id": "wacc",   "tex": r"W",                   "role": "intermediate", "kind": "matrix", "shape": "H x Rg",       "desc": "weight accumulator",                       "sample": [["0.9", "1.8"], ["1.8", "2.7"]]},
            {"id": "cube",   "tex": r"\hat{C}",             "role": "final",        "kind": "tensor", "shape": "N x Az x Rg",  "desc": "stitched prediction cube",                 "sample": [["0.2", "0.9"], ["0.1", "0.7"]]},
            {"id": "cubegt", "tex": r"C",                   "role": "measured",     "kind": "tensor", "shape": "N x Az x Rg",  "desc": "stitched ground-truth cube",               "sample": [["0.2", "0.9"], ["0.1", "0.7"]]},
            {"id": "pr2",    "tex": r"R^2_{a,r}",           "role": "calculated",   "kind": "matrix", "shape": "Az x Rg",      "desc": "per-pixel R-squared map",                  "sample": [["0.95", "0.88"], ["0.91", "0.97"]]},
            {"id": "r2",     "tex": r"R^2",                 "role": "final",        "kind": "scalar", "shape": "1",            "desc": "overall reconstruction R-squared",         "sample": "0.94"},
            {"id": "psnr",   "tex": r"\mathrm{PSNR}",       "role": "calculated",   "kind": "scalar", "shape": "1",            "desc": "peak signal-to-noise ratio (dB)",          "sample": "28.4"},
            {"id": "elevr2", "tex": r"R^2_{\mathrm{elev}}", "role": "calculated",   "kind": "vector", "shape": "N",            "desc": "per-elevation-bin R-squared",              "sample": ["0.91", "0.93", "..."]},
            {"id": "gmu",    "tex": r"\mathrm{MAE}_{\mu}",  "role": "calculated",   "kind": "scalar", "shape": "1",            "desc": "pooled per-Gaussian mu MAE (active pixels)", "sample": "0.42"},
            {"id": "pcons",  "tex": r"f_{\mathrm{dom}}",    "role": "calculated",   "kind": "scalar", "shape": "1",            "desc": "permutation-consensus dominant fraction",  "sample": "0.81"},
            {"id": "redc",   "tex": r"\mathbf{r}",          "role": "measured",     "kind": "tensor", "shape": "N x Az x Rg",  "desc": "reduced-subset Capon tomogram (re-synthesised)", "sample": [["0.3", "0.8"], ["0.2", "0.6"]]},
            {"id": "dimp",   "tex": r"\Delta_{a,r}",        "role": "final",        "kind": "matrix", "shape": "Az x Rg",      "desc": "improvement map: reduced minus prediction MSE", "sample": [["0.01", "-0.00"], ["0.02", "0.01"]]},
        ]
        steps = [
            {
                "id": "load", "title": "Strict run reconstruction", "phase": "Load run",
                "note": "The trained architecture is rebuilt verbatim from the saved config and the best-epoch weights, x-axis, and norm stats are restored; a single contiguous region is required or stitching fails loudly.",
                "inputs": ["ckpt"], "outputs": ["xaxis"],
                "lines": [
                    [{"id": "ckpt", "tex": r"\theta^{\star}", "role": "measured"}, {"tex": "="}, {"tex": r"\mathrm{ckpt[params]}\quad(\text{no EMA}),\qquad"}, {"id": "xaxis", "tex": r"\mathbf{x}", "role": "measured"}, {"tex": r"= \{x_n\}_{n=1}^{N}"}],
                ],
            },
            {
                "id": "predict", "title": "Windowed prediction", "phase": "Windowed predict",
                "note": "For every patch on the sliding-window grid the model emits raw normalised parameters; patches arrive in deterministic grid order so the cube has no holes.",
                "inputs": ["xhat"], "outputs": ["znorm"],
                "lines": [
                    [{"id": "znorm", "tex": r"\hat{\mathbf{z}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"f_{\theta^{\star}}\!\big("}, {"id": "xhat", "tex": r"\hat{\mathbf{x}}", "role": "measured"}, {"tex": r"\big)"}],
                ],
            },
            {
                "id": "idenorm", "title": "Denormalise and hard clamp", "phase": "Windowed predict",
                "note": "Predictions are denormalised then hard-clamped (no leaky slope) to physical bounds: amplitude to [0, a_max], mean to the elevation axis, spread to half a bin up to half the span.",
                "inputs": ["znorm"], "outputs": ["th"],
                "lines": [
                    [{"id": "th", "tex": r"\hat{a}_k", "role": "calculated"}, {"tex": r"= \mathrm{clip}(\tilde a_k,0,a_{\max}),\quad \hat{\mu}_k = \mathrm{clip}(\tilde\mu_k,x_{\min},x_{\max})"}],
                    [{"id": "th", "tex": r"\hat{\sigma}_k", "role": "calculated"}, {"tex": r"= \mathrm{clip}\!\big(\tilde\sigma_k,\ \tfrac12 x_{\mathrm{step}},\ \tfrac12 x_{\mathrm{range}}\big)"}],
                ],
            },
            {
                "id": "align", "title": "GT slot mu-sort alignment", "phase": "Windowed predict",
                "note": "Per pixel the GT slots are stably sorted by mean with inactive slots pushed to the end; the prediction keeps its raw slot order so the metrics measure the network's ordering.",
                "inputs": ["thgt"], "outputs": ["thgts"],
                "lines": [
                    [{"id": "thgts", "tex": r"\theta^{\mathrm{GT}}_{\pi}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\mathrm{take}\big("}, {"id": "thgt", "tex": r"\theta^{\mathrm{GT}}", "role": "measured"}, {"tex": r",\ \pi^{\star}\big),\quad \pi^{\star} = \operatorname{argsort}_k\,\kappa_k"}],
                    [{"tex": r"\kappa_k = +\infty\ \text{if}\ a^{\mathrm{GT}}_k<10^{-3},\ \ \text{else}\ \mu^{\mathrm{GT}}_k"}],
                ],
            },
            {
                "id": "recon", "title": "Curve reconstruction", "phase": "Windowed predict",
                "note": "Each patch's parameters are evaluated on the elevation axis into a spectrum; amplitudes are rectified at zero and the kernel uses a 2 sigma^2 plus 1e-8 denominator with no sigma floor.",
                "inputs": ["th", "xaxis"], "outputs": ["p"],
                "lines": [
                    [{"id": "p", "tex": r"\mathbf{p}_n", "role": "intermediate"}, {"tex": "="}, {"tex": r"\sum_{k} \max("}, {"id": "th", "tex": r"\hat{a}_k", "role": "calculated"}, {"tex": r",0)\,\exp\!\Big(-\tfrac{(x_n-\hat{\mu}_k)^2}{2\hat{\sigma}_k^2 + 10^{-8}}\Big)"}],
                ],
            },
            {
                "id": "window", "title": "Separable Hann window", "phase": "Cube stitch",
                "note": "A separable Hann window de-emphasises patch borders; each axis factor is floored at 1e-3 before the outer product so every covered position has a strictly positive weight.",
                "inputs": [], "outputs": ["win"],
                "lines": [
                    [{"id": "winv", "tex": r"w_v[i]", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max\!\Big(0.5 - 0.5\cos\tfrac{2\pi(i+0.5)}{P},\ 10^{-3}\Big)"}],
                    [{"id": "win", "tex": r"w", "role": "intermediate"}, {"tex": "="}, {"id": "winv", "tex": r"w_v", "role": "intermediate"}, {"tex": r"\otimes\ w_h"}],
                ],
            },
            {
                "id": "ola", "title": "Weighted overlap-add", "phase": "Cube stitch",
                "note": "Each windowed patch is scattered additively into a value accumulator at its grid origin, alongside the window itself in a parallel weight buffer; the sum is order-independent.",
                "inputs": ["p", "win"], "outputs": ["acc", "wacc"],
                "lines": [
                    [{"id": "acc", "tex": r"A", "role": "intermediate"}, {"tex": r"\mathrel{+}=\ "}, {"id": "p", "tex": r"\mathbf{p}", "role": "intermediate"}, {"tex": r"\cdot"}, {"id": "win", "tex": r"w", "role": "intermediate"}, {"tex": r",\qquad"}, {"id": "wacc", "tex": r"W", "role": "intermediate"}, {"tex": r"\mathrel{+}=\ w"}],
                ],
            },
            {
                "id": "finalise", "title": "Cube finalisation", "phase": "Cube stitch",
                "note": "Accumulated values are divided by accumulated weights (uncovered positions divide by one and yield zero), then the grid padding is trimmed to the scene size.",
                "inputs": ["acc", "wacc"], "outputs": ["cube"],
                "lines": [
                    [{"id": "cube", "tex": r"\hat{C}", "role": "final"}, {"tex": "="}, {"tex": r"\dfrac{"}, {"id": "acc", "tex": r"A", "role": "intermediate"}, {"tex": r"}{\max("}, {"id": "wacc", "tex": r"W", "role": "intermediate"}, {"tex": r",\,1)}\ \Big|_{\mathrm{trim}}"}],
                ],
            },
            {
                "id": "pixelmaps", "title": "Per-pixel metric maps", "phase": "Pixel metrics",
                "note": "Five per-pixel maps are computed over the stitched cubes, reducing over the N elevation bins: MSE, MAE, R-squared, cosine similarity, and peak-bin index error.",
                "inputs": ["cube", "cubegt"], "outputs": ["pr2"],
                "lines": [
                    [{"id": "pr2", "tex": r"R^2_{a,r}", "role": "calculated"}, {"tex": "="}, {"tex": r"1 - \dfrac{\sum_n(\hat C_{n,a,r}-C_{n,a,r})^2}{\sum_n(C_{n,a,r}-\bar C_{a,r})^2 + 10^{-12}}"}],
                    [{"tex": r"\Delta n_{a,r} = \big|\arg\max_n \hat C_{n,a,r} - \arg\max_n C_{n,a,r}\big|"}],
                ],
            },
            {
                "id": "globalcurve", "title": "Global curve metrics", "phase": "Curve metrics",
                "note": "Cube-wide scalars at physical scale: MSE, RMSE, overall R-squared, and PSNR whose peak signal is the GT-only dynamic range.",
                "inputs": ["cube", "cubegt"], "outputs": ["r2", "psnr"],
                "lines": [
                    [{"id": "r2", "tex": r"R^2", "role": "final"}, {"tex": "="}, {"tex": r"1 - \dfrac{\sum_{n,a,r}(\hat C - C)^2}{\sum_{n,a,r}(C - \bar C)^2 + 10^{-12}}"}],
                    [{"id": "psnr", "tex": r"\mathrm{PSNR}", "role": "calculated"}, {"tex": "="}, {"tex": r"10\log_{10}\dfrac{(C_{\max}-C_{\min})^2}{\mathrm{MSE}_{\mathrm{curve}}}\ \ \text{(dB)}"}],
                ],
            },
            {
                "id": "elevssim", "title": "Per-elevation metrics and SSIM", "phase": "Curve metrics",
                "note": "Per elevation bin (pixels as samples): MAE, RMSE, R-squared, and a cross-entropy between column-normalised distributions, plus mean SSIM over slices at GT-only data range.",
                "inputs": ["cube", "cubegt"], "outputs": ["elevr2"],
                "lines": [
                    [{"id": "elevr2", "tex": r"R^2_{\mathrm{elev}}(n)", "role": "calculated"}, {"tex": "="}, {"tex": r"1 - \dfrac{\sum_{a,r}(\hat C_{n,a,r}-C_{n,a,r})^2}{\sum_{a,r}(C_{n,a,r}-\bar C_n)^2 + 10^{-12}}"}],
                    [{"tex": r"\mathrm{CE}(n) = -\tfrac{1}{A_z R_g}\textstyle\sum_{a,r} \bar p^{\mathrm{GT}}_{n}\log \bar p^{\mathrm{pred}}_{n},\quad \bar p_n = \tfrac{C_n}{\sum_m C_m}"}],
                ],
            },
            {
                "id": "paramslot", "title": "Parameter and slot metrics", "phase": "Param metrics",
                "note": "On active pixels: per-Gaussian mu and sigma MAE/RMSE, placeholder detection F1, predicted mu-ordering rate, and a permutation consensus from per-pixel mu-distance assignment.",
                "inputs": ["th", "thgts"], "outputs": ["gmu", "pcons"],
                "lines": [
                    [{"id": "gmu", "tex": r"\mathrm{MAE}_{\mu,k}", "role": "calculated"}, {"tex": "="}, {"tex": r"\tfrac{1}{|\mathcal A_k|}\textstyle\sum_{\mathcal A_k}|\hat\mu_{k}-\mu^{\mathrm{GT}}_{k}|,\quad \mathcal A_k=\{a^{\mathrm{GT}}_k\ge10^{-3}\}"}],
                    [{"id": "pcons", "tex": r"f_{\mathrm{dom}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\dfrac{\max_\pi \mathrm{count}(\pi)}{\sum_\pi \mathrm{count}(\pi)},\quad \pi^{\star} = \operatorname*{arg\,min}_{\pi}\textstyle\sum_k |\hat\mu_k - \mu^{\mathrm{GT}}_{\pi(k)}|"}],
                ],
            },
            {
                "id": "reduced", "title": "Reduced Capon re-synthesis", "phase": "Reduced baseline",
                "note": "When the run used a strict secondary subset, a reduced Capon tomogram is re-synthesised; after an advisory orientation check, GT, prediction and reduced cubes are unit-area normalised and the per-pixel MSE improvement of the network over the reduced baseline is reported.",
                "inputs": ["redc", "cube", "cubegt"], "outputs": ["dimp"],
                "lines": [
                    [{"tex": r"\bar y_{n,a,r} = \dfrac{y_{n,a,r}}{\max(\sum_n y_{n,a,r},\,10^{-12})}\quad (y\in\{C,\hat C,\mathbf{r}\})"}],
                    [{"id": "dimp", "tex": r"\Delta_{a,r}", "role": "final"}, {"tex": "="}, {"tex": r"\mathrm{MSE}^{\mathrm{red}}_{a,r} - \mathrm{MSE}^{\mathrm{pred}}_{a,r}"}],
                ],
            },
        ]
        return {
            "key": "inference", "name": "Inference (Stitching)",
            "blurb": "Sliding-window patch predictions are denormalised and clamped, reconstructed to spectra, blended into dense cubes by weighted overlap-add, then scored by the full per-pixel, curve, per-elevation, parameter and reduced-baseline metric suite.",
            "nodes": nodes, "steps": steps,
        }

    def _tuning(self) -> dict:
        nodes = [
            {"id": "space",     "tex": r"\Theta",            "role": "measured",     "kind": "set",    "shape": "-",  "desc": "joint learning, regularisation and architecture search space", "sample": ["lr", "wd", "features", "..."]},
            {"id": "theta_lr",  "tex": r"\theta_{\mathrm{lr}}", "role": "intermediate", "kind": "vector", "shape": "9", "desc": "4 group LRs, 4 group WDs, dropout",                          "sample": ["2.6e-4", "8e-5", "...", "0.12"]},
            {"id": "theta_ar",  "tex": r"\theta_{\mathrm{arch}}", "role": "intermediate", "kind": "vector", "shape": "5", "desc": "features, bottleneck, activation, norm, upsample",          "sample": ["[64,128,256,512]", "2", "gelu", "group", "bilinear"]},
            {"id": "theta",     "tex": r"\theta",            "role": "intermediate", "kind": "vector", "shape": "d",  "desc": "sampled joint hyperparameter vector",                          "sample": ["3e-4", "1e-4", "64", "..."]},
            {"id": "ell",       "tex": r"\ell(\theta)",      "role": "calculated",   "kind": "scalar", "shape": "1",  "desc": "TPE good-trial KDE density at theta",                          "sample": "4.7"},
            {"id": "g",         "tex": r"g(\theta)",         "role": "calculated",   "kind": "scalar", "shape": "1",  "desc": "TPE bad-trial KDE density at theta",                           "sample": "0.9"},
            {"id": "acq",       "tex": r"a(\theta)",         "role": "calculated",   "kind": "scalar", "shape": "1",  "desc": "TPE acquisition = density ratio",                              "sample": "5.2"},
            {"id": "liar",      "tex": r"\tilde{y}",         "role": "intermediate", "kind": "scalar", "shape": "1",  "desc": "constant-liar phantom objective for pending trials",           "sample": "2.9e-2"},
            {"id": "fobj",      "tex": r"f(\theta)",         "role": "calculated",   "kind": "scalar", "shape": "1",  "desc": "trial objective: best validation loss",                        "sample": "2.3e-2"},
            {"id": "med",       "tex": r"m^{(t)}",           "role": "calculated",   "kind": "scalar", "shape": "1",  "desc": "running median of completed-trial losses at step t",           "sample": "3.4e-2"},
            {"id": "rem",       "tex": r"n_{\mathrm{rem}}",  "role": "intermediate", "kind": "scalar", "shape": "1",  "desc": "remaining trials to dispatch this chunk",                      "sample": "36"},
            {"id": "thetastar", "tex": r"\theta^{*}",        "role": "final",        "kind": "vector", "shape": "d",  "desc": "best joint configuration, decoded and exported",               "sample": ["2.6e-4", "8e-5", "96", "..."]},
        ]
        steps = [
            {
                "id": "spacelr", "title": "Learning and regularisation space", "phase": "Search space",
                "note": "The lr block declares four per-group learning rates and four weight decays as log-uniform floats, plus dropout as a linear-uniform float.",
                "inputs": ["space"], "outputs": ["theta_lr"],
                "lines": [
                    [{"id": "theta_lr", "tex": r"\eta_{\{\mathrm{enc,bot,dec,head}\}}", "role": "intermediate"}, {"tex": r"\sim \log\mathcal{U}(10^{-5},10^{-2}),\quad \lambda_{\{\cdot\}} \sim \log\mathcal{U}(10^{-6},10^{-1})"}],
                    [{"tex": r"p_{\mathrm{drop}} \sim \mathcal{U}(0,\,0.5)"}],
                ],
            },
            {
                "id": "spacearch", "title": "Architecture space", "phase": "Search space",
                "note": "Five categorical hyperparameters; the list-valued features channel is stored as an integer index and decoded back to its list on export.",
                "inputs": ["space"], "outputs": ["theta_ar"],
                "lines": [
                    [{"id": "theta_ar", "tex": r"\mathbf{w}", "role": "intermediate"}, {"tex": r"= C[\,k\,],\quad k\sim\mathcal{U}\{0,\dots,3\}"}],
                    [{"tex": r"b\in\{1,2,4\},\ \ \sigma\in\{\mathrm{relu,lrelu,gelu,silu}\},\ \ \mathrm{norm}\in\{\mathrm{batch,inst,group}\},\ \ \mathrm{up}\in\{\mathrm{tconv,bilin}\}"}],
                ],
            },
            {
                "id": "merge", "title": "Joint space", "phase": "Search space",
                "note": "The two blocks are merged into one joint space; TPE with multivariate sampling models the cross-dependencies directly.",
                "inputs": ["theta_lr", "theta_ar"], "outputs": ["space"],
                "lines": [
                    [{"id": "space", "tex": r"\Theta", "role": "measured"}, {"tex": "="}, {"tex": r"\Theta_{\mathrm{lr}} \times \Theta_{\mathrm{arch}},\qquad d = |\Theta| = 14"}],
                ],
            },
            {
                "id": "tpesplit", "title": "TPE density split", "phase": "Joint search",
                "note": "Once the startup trials exist, TPE splits observed trials by the gamma-quantile of their objective into a good set (low loss) and a bad set, fitting a KDE to each.",
                "inputs": ["fobj", "space"], "outputs": ["ell", "g"],
                "lines": [
                    [{"id": "ell", "tex": r"\ell(\theta)", "role": "calculated"}, {"tex": r"= p(\theta \mid f(\theta) \le y_\gamma),\qquad"}, {"id": "g", "tex": r"g(\theta)", "role": "calculated"}, {"tex": r"= p(\theta \mid f(\theta) > y_\gamma)"}],
                    [{"tex": r"y_\gamma = \inf\{\,y : P(f(\theta) \le y) \ge \gamma\,\}"}],
                ],
            },
            {
                "id": "tpeacq", "title": "Density-ratio proposal", "phase": "Joint search",
                "note": "TPE proposes the candidate maximising the good-over-bad density ratio (equivalent to expected improvement); for the first startup trials it samples uniformly instead.",
                "inputs": ["ell", "g", "space"], "outputs": ["acq", "theta"],
                "lines": [
                    [{"id": "acq", "tex": r"a(\theta)", "role": "calculated"}, {"tex": "="}, {"tex": r"\dfrac{"}, {"id": "ell", "tex": r"\ell(\theta)", "role": "calculated"}, {"tex": r"}{"}, {"id": "g", "tex": r"g(\theta)", "role": "calculated"}, {"tex": r"}"}],
                    [{"id": "theta", "tex": r"\theta", "role": "intermediate"}, {"tex": "="}, {"tex": r"\operatorname*{arg\,max}_{\theta\in\Theta}\ a(\theta)\quad(\text{after } n_0 = 8 \text{ startup trials})"}],
                ],
            },
            {
                "id": "liar", "title": "Constant-liar parallelism", "phase": "Joint search",
                "note": "With many GPU workers sharing one SQLite study, each pending trial is temporarily assigned the worst completed objective so concurrent workers do not all propose the same point.",
                "inputs": ["fobj"], "outputs": ["liar"],
                "lines": [
                    [{"id": "liar", "tex": r"\tilde{y}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max_{j\in\mathcal{C}} f(\theta_j)\quad \forall\,\theta_i\in\mathcal{R}\ (\text{pending})"}],
                    [{"tex": r"\mathrm{seed}_{\mathrm{sampler}} = 42 + \mathrm{gpu\_id}"}],
                ],
            },
            {
                "id": "trialsetup", "title": "Trial isolation and overrides", "phase": "Trial & pruning",
                "note": "Each objective deep-copies the base configs and overrides the epoch budget, scheduler horizon, early-stop patience, and trial seed, keyed on the global trial number.",
                "inputs": ["theta"], "outputs": ["fobj"],
                "lines": [
                    [{"tex": r"\mathrm{setattr}(\mathrm{cfg}, k, v)\quad \forall (k,v)\in"}, {"id": "theta", "tex": r"\theta", "role": "intermediate"}],
                    [{"tex": r"E_{\mathrm{trial}} = 30,\quad \mathrm{patience} = 8,\quad \mathrm{seed} = 42 + \mathrm{trial.number}"}],
                ],
            },
            {
                "id": "trial", "title": "Trial objective", "phase": "Trial & pruning",
                "note": "Each trial trains a full model on the fixed canonical split and returns the minimum validation loss over the epoch budget, capped earlier by early stopping or pruning.",
                "inputs": ["theta"], "outputs": ["fobj"],
                "iterative": {"var": "fobj", "steps": 30, "unit": "epoch", "symbol": "f",
                              "trace": ["6.0e-2", "4.1e-2", "3.0e-2", "2.5e-2", "2.3e-2"]},
                "lines": [
                    [{"id": "fobj", "tex": r"f(\theta)", "role": "calculated"}, {"tex": "="}, {"tex": r"\min_{e \in \{1,\dots,E\}}\ \mathcal{L}^{(e)}_{\mathrm{val}}\!\big("}, {"id": "theta", "tex": r"\theta", "role": "intermediate"}, {"tex": r"\big),\quad E = 30"}],
                ],
            },
            {
                "id": "prune", "title": "Median pruning with gates", "phase": "Trial & pruning",
                "note": "A trial is pruned at step t once its reported loss exceeds the median of completed-trial losses, but only after the startup trials and once t clears the warmup steps. Pruned trials count toward the budget; true failures become FAIL.",
                "inputs": ["fobj", "med"], "outputs": ["fobj"],
                "lines": [
                    [{"id": "med", "tex": r"m^{(t)}", "role": "calculated"}, {"tex": "="}, {"tex": r"\operatorname{median}\big\{\,f^{(t)}_j : j \in \mathrm{COMPLETE}\,\big\}"}],
                    [{"tex": r"\mathrm{prune} \iff \mathcal{L}^{(t)}_{\mathrm{val}} > "}, {"id": "med", "tex": r"m^{(t)}", "role": "calculated"}, {"tex": r"\ \wedge\ n_{\mathrm{done}} \ge 8\ \wedge\ t \ge 8"}],
                ],
            },
            {
                "id": "best", "title": "Chunked top-up and argmin export", "phase": "Best-config export",
                "note": "The study counts done trials and dispatches only the remaining ones across GPUs in chunks; the best config is rewritten after every completed trial, decoding the features index back to its list.",
                "inputs": ["fobj"], "outputs": ["rem", "thetastar"],
                "lines": [
                    [{"id": "rem", "tex": r"n_{\mathrm{rem}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max\!\big(0,\ N - n_{\mathrm{done}}\big),\qquad N = 100"}],
                    [{"id": "thetastar", "tex": r"\theta^{*}", "role": "final"}, {"tex": "="}, {"tex": r"\operatorname*{arg\,min}_{\theta \in \Theta}\ "}, {"id": "fobj", "tex": r"f(\theta)", "role": "calculated"}],
                ],
            },
        ]
        return {
            "key": "tuning", "name": "Tuning (Optuna)",
            "blurb": "A single joint Optuna study wraps the training pipeline: an explicit log-uniform and categorical search space, TPE density-ratio proposals with constant-liar parallelism, gated median pruning, and the best joint configuration exported, resumable in chunks.",
            "nodes": nodes, "steps": steps,
        }

    def collect(self) -> list:
        return [
            self._processing(),
            self._param_extraction(),
            self._dataset(),
            self._training(),
            self._inference(),
            self._tuning(),
        ]
