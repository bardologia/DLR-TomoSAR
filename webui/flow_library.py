class FlowLibrary:

    def _processing(self) -> dict:
        nodes = [
            {"id": "passes", "tex": r"\mathbf{s}",             "role": "measured",     "kind": "tensor", "shape": "N_p x A_z x R_g", "desc": "co-registered SLC pass stack loaded internally by PyRat FuSARtomo", "sample": [["0.8+0.2j", "0.6-0.4j"], ["0.5+0.5j", "0.9-0.1j"]]},
            {"id": "s0",     "tex": r"s_0",                    "role": "measured",     "kind": "matrix", "shape": "A_z x R_g",       "desc": "master (primary) SLC, complex (PyRat RGI-SLC)",                     "sample": [["0.8+0.2j", "0.6-0.4j", "1.1+0.0j"], ["0.5+0.5j", "0.9-0.1j", "0.7+0.3j"], ["1.0+0.2j", "0.4-0.6j", "0.8+0.1j"]]},
            {"id": "si",     "tex": r"s_i",                    "role": "measured",     "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "co-registered secondary SLC stack (PyRat INF-SLC)",                 "sample": [["0.7+0.3j", "0.5-0.3j"], ["0.6+0.4j", "0.8-0.2j"]]},
            {"id": "phidem", "tex": r"\phi_{\mathrm{DEM},i}",  "role": "measured",     "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "DEM-predicted phase per pass (rad)",                                "sample": [["0.12", "0.31"], ["0.27", "0.44"]]},
            {"id": "trk",    "tex": r"\mathbf{t}_i",           "role": "measured",     "kind": "tensor", "shape": "N_p x R x A_z",   "desc": "per-pass INF-TRACK position file; row 2 horizontal, row 3 vertical", "sample": [["...h...", "...v..."]]},
            {"id": "par",    "tex": r"\rho_i",                 "role": "measured",     "kind": "set",    "shape": "N_p",             "desc": "per-pass acquisition params: r, lambda, h0, terrain, antdir (pp_*.xml)", "sample": "lambda, r, h0, terrain, antdir"},
            {"id": "M",      "tex": r"M",                      "role": "calculated",   "kind": "scalar", "shape": "1",               "desc": "number of azimuth subsections",                                     "sample": "15"},
            {"id": "PT",     "tex": r"(P,T)",                  "role": "calculated",   "kind": "scalar", "shape": "1",               "desc": "resolved (workers, threads-per-worker) plan",                       "sample": "(8, 2)"},
            {"id": "Rhat",   "tex": r"\hat{\mathbf{R}}",       "role": "intermediate", "kind": "matrix", "shape": "N_p x N_p",       "desc": "Boxcar-windowed per-pixel sample covariance (PyRat-internal)",      "sample": [["1.0", "0.7"], ["0.7", "1.0"]]},
            {"id": "Tm",     "tex": r"T_m",                    "role": "calculated",   "kind": "tensor", "shape": "H x W_m x R_g",   "desc": "Capon beamformed tomogram of subsection m",                         "sample": [["0.2", "0.9", "0.3"], ["0.1", "0.7", "0.8"], ["0.0", "0.4", "1.0"]]},
            {"id": "DEMm",   "tex": r"D_m",                    "role": "calculated",   "kind": "matrix", "shape": "W_m x R_g",       "desc": "DEM of subsection m (PyRat byproduct)",                             "sample": [["31.2", "30.8"], ["32.1", "31.5"]]},
            {"id": "Tcomb",  "tex": r"T_{\mathrm{comb}}",      "role": "final",        "kind": "tensor", "shape": "H x A_z x R_g",   "desc": "combined beamformed tomogram, model reconstruction reference",      "sample": [["0.2", "0.9", "0.3"], ["0.1", "0.7", "0.8"], ["0.0", "0.4", "1.0"]]},
            {"id": "DEM",    "tex": r"D",                      "role": "final",        "kind": "matrix", "shape": "A_z x R_g",       "desc": "full-stack concatenated DEM (m)",                                   "sample": [["31.2", "30.8"], ["32.1", "31.5"]]},
            {"id": "stilde", "tex": r"\tilde{s}_i",            "role": "intermediate", "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "DEM-deramped secondary SLC",                                        "sample": [["0.6+0.1j", "0.5-0.1j"], ["0.7+0.2j", "0.6-0.0j"]]},
            {"id": "cross",  "tex": r"c_i",                    "role": "intermediate", "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "raw master-secondary cross-product",                                "sample": [["0.45+0.30j", "0.41-0.12j"], ["0.55+0.18j", "0.48-0.05j"]]},
            {"id": "pi",     "tex": r"p_i",                    "role": "intermediate", "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "unit-magnitude interferometric phasor",                             "sample": [["0.83+0.55j", "0.96-0.28j"], ["0.95+0.31j", "0.99-0.10j"]]},
            {"id": "Ai",     "tex": r"A_i",                    "role": "intermediate", "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "clipped secondary amplitude weight",                                "sample": [["0.71", "0.93"], ["1.25", "0.58"]]},
            {"id": "phii",   "tex": r"\tilde{\phi}_i",         "role": "final",        "kind": "tensor", "shape": "N_s x A_z x R_g", "desc": "amplitude-weighted complex interferogram (network input)",          "sample": [["0.51+0.20j", "0.44-0.12j"], ["0.63+0.18j", "0.55-0.05j"]]},
            {"id": "bv",     "tex": r"b^{\mathrm{v}}_i",       "role": "calculated",   "kind": "vector", "shape": "N_p",             "desc": "vertical baseline relative to reference pass (m)",                  "sample": ["0.00", "12.4", "-8.1", "20.7"]},
            {"id": "bh",     "tex": r"b^{\mathrm{h}}_i",       "role": "calculated",   "kind": "vector", "shape": "N_p",             "desc": "horizontal baseline relative to reference pass (m)",                "sample": ["0.00", "5.2", "-3.4", "9.8"]},
            {"id": "prof",   "tex": r"\mathbf{\tau}_i",        "role": "calculated",   "kind": "tensor", "shape": "N_p x A_z",       "desc": "per-azimuth horizontal/vertical track position profiles (m)",       "sample": [["112.3", "112.5"], ["101.7", "101.9"]]},
            {"id": "look",   "tex": r"\theta",                 "role": "calculated",   "kind": "vector", "shape": "R_g",             "desc": "per-range look angle theta (rad)",                                  "sample": ["0.79", "0.80", "...", "1.02"]},
            {"id": "brel",   "tex": r"\mathbf{\beta}_i",       "role": "calculated",   "kind": "tensor", "shape": "N_p x A_z",       "desc": "reference-relative baseline profiles, horizontal & vertical (m)",   "sample": [["0.00", "0.00"], ["5.2", "4.9"]]},
            {"id": "geom",   "tex": r"k_z",                    "role": "final",        "kind": "tensor", "shape": "N_p x A_z x R_g", "desc": "per-pixel interferometric wavenumber k_z (rad/m) for the physics loss", "sample": [["0.00", "0.00"], ["0.11", "0.12"]]},
        ]
        steps = [
            {
                "id": "subdivide", "title": "Azimuth subdivision and worker plan", "phase": "A - Tomogram (Capon)",
                "note": "When the crop azimuth exceeds max_crop_azimuth_width (1000 lines) it is split into M contiguous, non-overlapping subsections; the (workers, threads) plan minimises the number of dispatch waves within a core budget B = floor(0.8 x cores) at the default 'high' effort, with threads per worker capped at 16.",
                "inputs": [], "outputs": ["M", "PT"],
                "lines": [
                    [{"id": "M", "tex": r"M", "role": "calculated"}, {"tex": "="}, {"tex": r"\left\lceil W_{\mathrm{az}} / W_{\max} \right\rceil,\qquad W_{\max} = 1000"}],
                    [{"id": "PT", "tex": r"(P,T)", "role": "calculated"}, {"tex": "="}, {"tex": r"\arg\min_{P \le B}\left\lceil M/P \right\rceil,\quad T = \min(16,\ \lfloor B/P \rfloor),\quad B = \lfloor 0.8\,C \rfloor"}],
                ],
            },
            {
                "id": "covariance", "title": "Boxcar sample covariance", "phase": "A - Tomogram (Capon)",
                "note": "Each subsection is beamformed by an independent PyRat FuSARtomo worker; the Boxcar filter with win = [20, 10] averages the co-registered SLC pass stack over a 20x10-pixel window to estimate the per-pixel sample covariance. The averaging is internal to PyRat, this stage only sets the filter and its window.",
                "inputs": ["passes"], "outputs": ["Rhat"],
                "lines": [
                    [{"id": "Rhat", "tex": r"\hat{\mathbf{R}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\big\langle\,"}, {"id": "passes", "tex": r"\mathbf{s}\,\mathbf{s}^{H}", "role": "measured"}, {"tex": r"\,\big\rangle_{W_{\mathrm{box}}},\qquad W_{\mathrm{box}} = 20 \times 10"}],
                ],
            },
            {
                "id": "capon", "title": "Capon beamforming over elevation", "phase": "A - Tomogram (Capon)",
                "note": "FuSARtomo runs the Capon (MVDR) minimum-variance estimator over the configured height range [-20, 80] m and writes each subsection's tomogram and DEM to HDF5; the steering vector a(xi) is PyRat's internal geometry model, not computed by this stage.",
                "inputs": ["Rhat"], "outputs": ["DEMm", "Tm"],
                "lines": [
                    [{"id": "Tm", "tex": r"T_m(\xi)", "role": "calculated"}, {"tex": r"\;\propto\;"}, {"tex": r"\dfrac{1}{\mathbf{a}^{H}(\xi)\,"}, {"id": "Rhat", "tex": r"\hat{\mathbf{R}}^{-1}", "role": "intermediate"}, {"tex": r"\,\mathbf{a}(\xi)},\qquad \xi \in [-20,\ 80]\,\mathrm{m}"}],
                ],
            },
            {
                "id": "concat", "title": "Subsection concatenation", "phase": "A - Tomogram (Capon)",
                "note": "The worker HDF5 files are reassembled along azimuth into the full-stack products: the 2-D DEM concatenates on axis 0 and the 3-D tomogram on axis 1, then both are saved as .npy. The combined tomogram is the model's reconstruction reference.",
                "inputs": ["Tm", "DEMm"], "outputs": ["DEM", "Tcomb"],
                "lines": [
                    [{"id": "DEM", "tex": r"D", "role": "final"}, {"tex": "="}, {"tex": r"\mathrm{concat}\!\left[\,"}, {"id": "DEMm", "tex": r"D_0, \dots, D_{M-1}", "role": "calculated"}, {"tex": r"\,\right]_{\mathrm{axis}=0}"}],
                    [{"id": "Tcomb", "tex": r"T_{\mathrm{comb}}", "role": "final"}, {"tex": "="}, {"tex": r"\mathrm{concat}\!\left[\,"}, {"id": "Tm", "tex": r"T_0, \dots, T_{M-1}", "role": "calculated"}, {"tex": r"\,\right]_{\mathrm{axis}=1}"}],
                ],
            },
            {
                "id": "slc_load", "title": "SLC loading and co-registration", "phase": "B - Interferograms",
                "note": "The master is read as the geocoded range-Doppler SLC (RGI-SLC); each secondary is the interferometric SLC already co-registered to the master (INF-SLC), loaded together with its DEM-predicted phase (phadem).",
                "inputs": [], "outputs": ["s0", "si", "phidem"],
                "lines": [
                    [{"id": "s0", "tex": r"s_0", "role": "measured"}, {"tex": "="}, {"tex": r"\mathrm{load}\big(\mathrm{master};\ \mathtt{RGI\text{-}SLC}\big)"}],
                    [{"id": "si", "tex": r"s_i", "role": "measured"}, {"tex": "="}, {"tex": r"\mathrm{load}\big(\mathrm{sec}_i;\ \mathtt{INF\text{-}SLC}\big),\quad"}, {"id": "phidem", "tex": r"\phi_{\mathrm{DEM},i}", "role": "measured"}, {"tex": r"= \mathrm{phadem}_i"}],
                ],
            },
            {
                "id": "baselines", "title": "Track baseline extraction", "phase": "B - Interferograms",
                "note": "Per pass, the horizontal (row 2) and vertical (row 3) track positions are read from the INF-TRACK file over the crop azimuth window; the scalar baselines are their azimuth means referenced to the reference pass, and the full per-azimuth profiles are kept to build the geometry field.",
                "inputs": ["trk"], "outputs": ["bv", "bh", "prof"],
                "lines": [
                    [{"id": "bv", "tex": r"b^{\mathrm{v}}_i", "role": "calculated"}, {"tex": "="}, {"tex": r"\overline{\tau^{\mathrm{v}}_i} - \overline{\tau^{\mathrm{v}}_0},\qquad"}, {"id": "bh", "tex": r"b^{\mathrm{h}}_i", "role": "calculated"}, {"tex": "="}, {"tex": r"\overline{\tau^{\mathrm{h}}_i} - \overline{\tau^{\mathrm{h}}_0}"}],
                    [{"id": "prof", "tex": r"\mathbf{\tau}_i", "role": "calculated"}, {"tex": "="}, {"id": "trk", "tex": r"\mathbf{t}_i", "role": "measured"}, {"tex": r"[\,2{:}4,\ \mathtt{az}_0{:}\mathtt{az}_1\,]\quad (\text{h, v rows})"}],
                ],
            },
            {
                "id": "deramp", "title": "DEM-phase deramping", "phase": "B - Interferograms",
                "note": "Each secondary is multiplied by the DEM phasor; after the later conjugation this subtracts the DEM-predicted phase, removing terrain topography and leaving sub-resolution elevation structure.",
                "inputs": ["si", "phidem"], "outputs": ["stilde"],
                "lines": [
                    [{"id": "stilde", "tex": r"\tilde{s}_i", "role": "intermediate"}, {"tex": "="}, {"id": "si", "tex": r"s_i", "role": "measured"}, {"tex": r"\cdot"}, {"tex": r"\exp\!\left(j\,"}, {"id": "phidem", "tex": r"\phi_{\mathrm{DEM},i}", "role": "measured"}, {"tex": r"\right)"}],
                ],
            },
            {
                "id": "crossprod", "title": "Master-secondary cross-product", "phase": "B - Interferograms",
                "note": "Conjugating the deramped secondary against the master leaves the phase difference minus the DEM phase; the conjugation flips the DEM sign, so it is effectively subtracted.",
                "inputs": ["s0", "stilde"], "outputs": ["cross"],
                "lines": [
                    [{"id": "cross", "tex": r"c_i", "role": "intermediate"}, {"tex": "="}, {"id": "s0", "tex": r"s_0", "role": "measured"}, {"tex": r"\cdot"}, {"id": "stilde", "tex": r"\overline{\tilde{s}_i}", "role": "intermediate"}],
                    [{"tex": r"\arg(c_i)"}, {"tex": "="}, {"tex": r"\psi_0 - \psi_i - \phi_{\mathrm{DEM},i}"}],
                ],
            },
            {
                "id": "phasor", "title": "Unit-phasor normalisation", "phase": "B - Interferograms",
                "note": "Dividing by the cross-product magnitude (floored at 1e-30) removes inter-pass amplitude differences while preserving coherence; null pixels collapse to zero instead of NaN.",
                "inputs": ["cross"], "outputs": ["pi"],
                "lines": [
                    [{"id": "pi", "tex": r"p_i", "role": "intermediate"}, {"tex": "="}, {"tex": r"\dfrac{"}, {"id": "cross", "tex": r"c_i", "role": "intermediate"}, {"tex": r"}{\left|c_i\right| + \epsilon},\qquad \epsilon = 10^{-30}"}],
                ],
            },
            {
                "id": "clip", "title": "Amplitude clipping", "phase": "B - Interferograms",
                "note": "The secondary amplitude is clipped at max_amplitude_clip = 1.25 so bright corner reflectors or artefacts cannot dominate the per-pass weight.",
                "inputs": ["si"], "outputs": ["Ai"],
                "lines": [
                    [{"id": "Ai", "tex": r"A_i", "role": "intermediate"}, {"tex": "="}, {"tex": r"\mathrm{clip}\!\big(\big|"}, {"id": "si", "tex": r"s_i", "role": "measured"}, {"tex": r"\big|,\ 0,\ c_{\max}\big),\qquad c_{\max} = 1.25"}],
                ],
            },
            {
                "id": "interf", "title": "Amplitude-weighted interferogram", "phase": "B - Interferograms",
                "note": "Re-attaching the clipped amplitude as the phasor's modulus gives an interferogram whose phase is the residual (sub-DEM) elevation phase and whose magnitude is a bounded signal-strength proxy; this stack is the network input.",
                "inputs": ["Ai", "pi"], "outputs": ["phii"],
                "lines": [
                    [{"id": "phii", "tex": r"\tilde{\phi}_i", "role": "final"}, {"tex": "="}, {"id": "Ai", "tex": r"A_i", "role": "intermediate"}, {"tex": r"\cdot"}, {"id": "pi", "tex": r"p_i", "role": "intermediate"}, {"tex": "="}, {"id": "Ai", "tex": r"A_i", "role": "intermediate"}, {"tex": r"\dfrac{s_0\,\overline{\tilde{s}_i}}{\left|s_0\,\overline{\tilde{s}_i}\right| + \epsilon}"}],
                ],
            },
            {
                "id": "trackgeo", "title": "Look angle and relative baselines", "phase": "C - Geometry field",
                "note": "The look angle is recovered per range bin from the reference-pass geometry, arccos of sensor-height-above-terrain over slant range; the build aborts when that height is non-positive or reaches the nearest slant range, because clamping the ratio into [-1, 1] would otherwise silently yield a zero look angle and an infinite kz. The per-azimuth position profiles become baselines relative to the reference pass. A left-looking stack (antdir <= 0) aborts, since the kz formula assumes a right-looking geometry.",
                "inputs": ["par", "prof"], "outputs": ["look", "brel"],
                "lines": [
                    [{"id": "look", "tex": r"\theta(r)", "role": "calculated"}, {"tex": "="}, {"tex": r"\arccos\!\Big(\mathrm{clip}\big(\tfrac{h_0 - \mathrm{terrain}}{r},\,-1,\,1\big)\Big),\quad r = "}, {"id": "par", "tex": r"\rho_0.r", "role": "measured"}, {"tex": r"[\mathtt{rg}_0{:}\mathtt{rg}_1]"}],
                    [{"id": "brel", "tex": r"\mathbf{\beta}_i", "role": "calculated"}, {"tex": "="}, {"id": "prof", "tex": r"\mathbf{\tau}_i", "role": "calculated"}, {"tex": r"-\ "}, {"id": "prof", "tex": r"\mathbf{\tau}_0", "role": "calculated"}, {"tex": r"\qquad (\text{rel. to reference pass})"}],
                ],
            },
            {
                "id": "geomfield", "title": "Per-pixel wavenumber field", "phase": "C - Geometry field",
                "note": "The perpendicular baseline projects the horizontal and vertical baselines onto the look direction, and the interferometric wavenumber (default height convention, dividing by r sin theta) is stored per (pass, azimuth, range) as the geometry field consumed by the physics loss; the reference pass has zero baseline, so its kz vanishes.",
                "inputs": ["look", "brel", "par"], "outputs": ["geom"],
                "lines": [
                    [{"tex": r"b^{\perp}_i"}, {"tex": "="}, {"id": "brel", "tex": r"b^{\mathrm{h}}_i", "role": "calculated"}, {"tex": r"\cos"}, {"id": "look", "tex": r"\theta", "role": "calculated"}, {"tex": r"+\ "}, {"id": "brel", "tex": r"b^{\mathrm{v}}_i", "role": "calculated"}, {"tex": r"\sin"}, {"id": "look", "tex": r"\theta", "role": "calculated"}],
                    [{"id": "geom", "tex": r"k_{z,i}", "role": "final"}, {"tex": "="}, {"tex": r"\dfrac{4\pi}{\lambda}\,\dfrac{b^{\perp}_i}{r\,\sin\theta}\qquad (\text{height convention})"}],
                ],
            },
        ]
        return {
            "key"   : "processing",
            "name"  : "Processing (Tomogram + Interferograms)",
            "blurb" : "From the F-SAR SLC passes to the model's reference tomogram and input interferograms. Each azimuth subsection is Capon-beamformed by a PyRat FuSARtomo worker and reassembled; the co-registered secondaries are DEM-deramped and amplitude-weighted into the interferometric stack; and the track geometry becomes a per-pixel wavenumber field for the physics loss.",
            "nodes" : nodes,
            "steps" : steps,
        }

    def _param_extraction(self) -> dict:
        nodes = [
            {"id": "T",       "tex": r"T",                        "role": "measured",     "kind": "tensor", "shape": "H x Az x R", "desc": "beamformed tomogram from processing, magnitude taken per profile", "sample": [["0.2", "0.9", "0.3"], ["0.1", "0.7", "0.8"], ["0.0", "0.4", "1.0"]]},
            {"id": "P",       "tex": r"P_h",                      "role": "intermediate", "kind": "vector", "shape": "H",          "desc": "thresholded and truncated magnitude profile over elevation",        "sample": ["0.00", "0.62", "0.31", "0.88", "0.00"]},
            {"id": "active",  "tex": r"\mathbb{1}_{\mathrm{act}}","role": "intermediate", "kind": "scalar", "shape": "1",          "desc": "active-profile gate (profile max above tau_a)",                     "sample": "1"},
            {"id": "scale",   "tex": r"s",                        "role": "intermediate", "kind": "scalar", "shape": "1",          "desc": "per-profile scale = profile max (1 if inactive)",                   "sample": "0.88"},
            {"id": "gtilde",  "tex": r"\tilde{\gamma}_h",         "role": "intermediate", "kind": "vector", "shape": "H",          "desc": "peak-normalised profile (tallest bin equals one)",                  "sample": ["0.00", "0.70", "0.35", "1.00", "0.00"]},
            {"id": "peaks",   "tex": r"\mathcal{P}",              "role": "intermediate", "kind": "set",    "shape": "P",          "desc": "prominence-and-distance gated peak indices, prominence-sorted",      "sample": ["97", "34", "151"]},
            {"id": "sigbase", "tex": r"\sigma_{\mathrm{base}}",   "role": "intermediate", "kind": "scalar", "shape": "1",          "desc": "span-derived width scale, sets d_min and sig0, m",                  "sample": "2.50"},
            {"id": "sig0",    "tex": r"\sigma^{(0)}",             "role": "intermediate", "kind": "vector", "shape": "K",          "desc": "shared initial width guess, m",                                     "sample": ["0.62", "0.62", "0.62"]},
            {"id": "idxs",    "tex": r"\mathcal{I}",              "role": "intermediate", "kind": "set",    "shape": "K",          "desc": "K seed indices (peaks then residual-argmax fill)",                  "sample": ["97", "34", "151", "12", "180"]},
            {"id": "mu0",     "tex": r"\mu",                      "role": "intermediate", "kind": "vector", "shape": "K",          "desc": "seed component means at peak elevations, m",                        "sample": ["12.4", "31.8", "47.0"]},
            {"id": "a0",      "tex": r"a",                        "role": "intermediate", "kind": "vector", "shape": "K",          "desc": "seed component amplitudes at peaks, raw units",                     "sample": ["0.88", "0.62", "0.34"]},
            {"id": "loss",    "tex": r"\mathcal{L}",              "role": "intermediate", "kind": "scalar", "shape": "1",          "desc": "per-profile MSE on the normalised profile",                         "sample": "3.4e-2"},
            {"id": "grad",    "tex": r"g_t",                      "role": "intermediate", "kind": "vector", "shape": "K",          "desc": "loss gradient at step t, masked to free parameters",                "sample": ["-0.04", "0.01", "-0.02"]},
            {"id": "sigstar", "tex": r"\sigma^{*}",               "role": "calculated",   "kind": "vector", "shape": "K",          "desc": "fitted, clamped component widths, m",                               "sample": ["1.84", "2.55", "3.10"]},
            {"id": "mseK",    "tex": r"\mathrm{MSE}_K",           "role": "intermediate", "kind": "vector", "shape": "K_max",      "desc": "per-order residual MSE on the normalised profile",                  "sample": ["0.18", "0.04", "0.05", "0.07", "0.06"]},
            {"id": "penK",    "tex": r"\mathcal{L}_K",            "role": "intermediate", "kind": "vector", "shape": "K_max",      "desc": "penalised score per model order K",                                 "sample": ["0.19", "0.06", "0.08", "0.11", "0.11"]},
            {"id": "Kstar",   "tex": r"K^{*}",                    "role": "calculated",   "kind": "scalar", "shape": "1",          "desc": "selected number of active components",                              "sample": "2"},
            {"id": "aout",    "tex": r"a^{\mathrm{out}}",         "role": "calculated",   "kind": "vector", "shape": "K",          "desc": "winner amplitudes rescaled to raw units",                           "sample": ["0.81", "0.55"]},
            {"id": "theta",   "tex": r"\theta",                   "role": "final",        "kind": "vector", "shape": "3K",         "desc": "mean-ordered interleaved (a, mu, sigma) supervised target",          "sample": ["a_1", "mu_1", "sig_1", "a_2", "..."]},
            {"id": "r2",      "tex": r"R^2",                      "role": "final",        "kind": "matrix", "shape": "Az x R",     "desc": "per-pixel fit-quality map",                                         "sample": [["0.97", "0.91"], ["0.88", "0.99"]]},
            {"id": "mrel",    "tex": r"m_{\mathrm{rel}}",         "role": "final",        "kind": "matrix", "shape": "Az x R",     "desc": "relative K-selection margin diagnostic",                            "sample": [["0.31", "0.04"], ["0.12", "0.55"]]},
        ]
        steps = [
            {
                "id": "threshold", "title": "Profile floor and truncation", "phase": "0 - Conditioning",
                "note": "The elevation profile is the tomogram magnitude; ProfilePreprocessor zeroes every sample not exceeding t_f = 0.25 of the per-profile max, then zeroes all bins from index H_tr = 170 onward. This is the exact conditioning the fit, R2 and contrast all consume.",
                "inputs": ["T"], "outputs": ["P"],
                "lines": [
                    [{"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": "="}, {"id": "T", "tex": r"\left|T_h\right|", "role": "measured"}, {"tex": r"\cdot\,\mathbb{1}\!\left[\left|T_h\right| > t_f \max_h \left|T_h\right|\right],\quad t_f = 0.25"}],
                    [{"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": r"\leftarrow 0\quad \text{for } h \ge H_{\mathrm{tr}} = 170"}],
                ],
            },
            {
                "id": "activity", "title": "Active-profile gate", "phase": "0 - Conditioning",
                "note": "A profile is fitted only if its maximum clears the activity threshold; inactive profiles are skipped (all parameters stay zero) and take scale 1 so normalisation is a no-op.",
                "inputs": ["P"], "outputs": ["active", "scale"],
                "lines": [
                    [{"id": "active", "tex": r"\mathbb{1}_{\mathrm{act}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\mathbb{1}\!\left[\max_h "}, {"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": r" > \tau_a\right],\qquad \tau_a = 10^{-3}"}],
                    [{"id": "scale", "tex": r"s", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max_h "}, {"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": r"\ \ (\text{active}),\qquad 1\ \ (\text{else})"}],
                ],
            },
            {
                "id": "pnorm", "title": "Peak normalisation", "phase": "0 - Conditioning",
                "note": "Dividing by the per-profile maximum sets the tallest peak to one, decoupling the fit loss from absolute backscatter so the MSE and the penalty are comparable across pixels.",
                "inputs": ["P", "scale"], "outputs": ["gtilde"],
                "lines": [
                    [{"id": "gtilde", "tex": r"\tilde{\gamma}_h", "role": "intermediate"}, {"tex": "="}, {"tex": r"\dfrac{1}{"}, {"id": "scale", "tex": r"s", "role": "intermediate"}, {"tex": r"}\;"}, {"id": "P", "tex": r"P_h", "role": "intermediate"}],
                ],
            },
            {
                "id": "peakfind", "title": "Prominence-gated peak detection", "phase": "1 - Init (CPU)",
                "note": "scipy find_peaks runs on the raw thresholded profile with no smoothing, keeping a peak only when its topographic prominence reaches p_frac = 0.05 of the max and it sits at least d_min bins from every rival; survivors are sorted by descending prominence.",
                "inputs": ["P"], "outputs": ["peaks"],
                "lines": [
                    [{"id": "peaks", "tex": r"\mathcal{P}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\left\{\,p : \mathrm{prom}(p) \ge p_{\mathrm{frac}}\,\max_h"}, {"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": r",\ \ |p_i-p_j|\ge d_{\min}\right\},\ \ p_{\mathrm{frac}} = 0.05"}],
                ],
            },
            {
                "id": "geometry", "title": "Width scales and clamp bounds", "phase": "1 - Init (CPU)",
                "note": "The span-derived sigma_base sets the peak-separation distance d_min and, divided by D_sigma = 4, the shared initial width; the Adam width clamp runs from one elevation bin to half the height span.",
                "inputs": [], "outputs": ["sigbase", "sig0"],
                "lines": [
                    [{"id": "sigbase", "tex": r"\sigma_{\mathrm{base}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max\!\left(2\Delta\xi,\ \tfrac{x_{\max}-x_{\min}}{8K}\right),\quad d_{\min} = \max\!\big(1,\ \lfloor \sigma_{\mathrm{base}}/\Delta\xi\rfloor\big)"}],
                    [{"id": "sig0", "tex": r"\sigma^{(0)}", "role": "intermediate"}, {"tex": "="}, {"id": "sigbase", "tex": r"\sigma_{\mathrm{base}}", "role": "intermediate"}, {"tex": r"/D_\sigma,\quad [\sigma_{\mathrm{lo}},\sigma_{\mathrm{hi}}] = \big[\Delta\xi,\ \tfrac{x_{\max}-x_{\min}}{2}\big],\ \ D_\sigma = 4"}],
                ],
            },
            {
                "id": "residfill", "title": "Residual fill of remaining slots", "phase": "1 - Init (CPU)",
                "note": "When fewer than K peaks are found, a window of half-width d_min is zeroed around each detected peak and the remaining slots are filled by repeated argmax of the residual, guaranteeing every seed is d_min apart; a flat profile (max below 1e-10) falls back to K evenly spaced indices.",
                "inputs": ["peaks", "P"], "outputs": ["idxs"],
                "lines": [
                    [{"id": "idxs", "tex": r"\mathcal{I}", "role": "intermediate"}, {"tex": "="}, {"id": "peaks", "tex": r"\mathcal{P}", "role": "intermediate"}, {"tex": r"\ \cup\ \big\{\arg\max_h\,\rho_h\big\}^{\times (K-|\mathcal{P}|)}"}],
                    [{"tex": r"\rho_h"}, {"tex": "="}, {"id": "P", "tex": r"P_h", "role": "intermediate"}, {"tex": r"\cdot\,\mathbb{1}\!\left[\min_{p\in\mathcal{P}}|h-p| > d_{\min}\right]"}],
                ],
            },
            {
                "id": "seed", "title": "Seed amplitudes and means", "phase": "1 - Init (CPU)",
                "note": "Amplitude and mean are read off the raw-profile seed indices (amplitude floored at 1e-10) as the fit's starting point; they stay frozen in the default sigma-only mode and are optimised only when the fit mode frees them. Widths all start at the shared sig0.",
                "inputs": ["idxs", "P"], "outputs": ["mu0", "a0"],
                "lines": [
                    [{"id": "mu0", "tex": r"\mu_k", "role": "intermediate"}, {"tex": "="}, {"tex": r"x_{\,\mathcal{I}_k},\qquad"}, {"id": "a0", "tex": r"a_k", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max\!\big(P_{\mathcal{I}_k},\,10^{-10}\big)"}],
                ],
            },
            {
                "id": "objective", "title": "Mixture-fit objective", "phase": "2 - Fit (GPU)",
                "note": "The loss is the per-profile MSE between the K-Gaussian sum and the normalised profile, with amplitude entering normalised as a/s so the seed peak maps near unity. sigma is floored at 1e-6 and the exponent clipped to [-100, 0] before summing; vmap over pixels, value_and_grad over (a, mu, sigma).",
                "inputs": ["gtilde", "mu0", "a0", "scale", "sig0"], "outputs": ["loss"],
                "lines": [
                    [{"id": "loss", "tex": r"\mathcal{L}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\dfrac{1}{H}\sum_{h}\Big(\sum_{k}\dfrac{"}, {"id": "a0", "tex": r"a_k", "role": "intermediate"}, {"tex": r"}{"}, {"id": "scale", "tex": r"s", "role": "intermediate"}, {"tex": r"}\,e^{-\frac{(x_h-"}, {"id": "mu0", "tex": r"\mu_k", "role": "intermediate"}, {"tex": r")^2}{2\max(\sigma_k,\,10^{-6})^2}} -"}, {"id": "gtilde", "tex": r"\tilde{\gamma}_h", "role": "intermediate"}, {"tex": r"\Big)^{2}"}],
                ],
            },
            {
                "id": "adam", "title": "Bias-corrected Adam width update", "phase": "2 - Fit (GPU)",
                "note": "Hand-written bias-corrected Adam (eta = 0.2, beta_1 = 0.95, beta_2 = 0.999, eps = 1e-8) as one lax.scan over T = 3000 steps compiled into a single XLA program; sigma is always free while amplitude and mean gradients are zeroed by mask unless the mode frees them. After every step a is floored at 0, mu clipped to the height range, sigma clamped to [sigma_lo, sigma_hi].",
                "inputs": ["loss", "sig0"], "outputs": ["sigstar"],
                "iterative": {"unit": "t", "steps": 3000, "symbol": "σ",
                              "trace": ["0.62", "0.81", "1.10", "1.45", "1.70", "1.82", "1.84"]},
                "lines": [
                    [{"id": "grad", "tex": r"g_t", "role": "intermediate"}, {"tex": "="}, {"tex": r"\nabla"}, {"id": "loss", "tex": r"\mathcal{L}", "role": "intermediate"}, {"tex": r"\odot\,\mathbf{m}_{\mathrm{free}},\quad \hat m_t = \tfrac{m_t}{1-\beta_1^{t}},\ \ \hat v_t = \tfrac{v_t}{1-\beta_2^{t}}"}],
                    [{"id": "sigstar", "tex": r"\sigma^{*}_t", "role": "calculated"}, {"tex": "="}, {"tex": r"\mathrm{clip}\!\Big("}, {"id": "sigstar", "tex": r"\sigma^{*}_{t-1}", "role": "calculated"}, {"tex": r"-\eta\,\tfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},\ \sigma_{\mathrm{lo}},\ \sigma_{\mathrm{hi}}\Big),\ \ \eta = 0.2"}],
                ],
            },
            {
                "id": "scoreK", "title": "Per-order penalised score", "phase": "3 - Best-K",
                "note": "Every order K in 1..K_max = 5 is re-scored on the normalised profile with a flat complexity penalty lambda_K = 1e-2 per component, so an extra Gaussian is kept only when it lowers normalised MSE by at least lambda_K.",
                "inputs": ["gtilde", "sigstar", "a0"], "outputs": ["mseK", "penK"],
                "lines": [
                    [{"id": "mseK", "tex": r"\mathrm{MSE}_K", "role": "intermediate"}, {"tex": "="}, {"tex": r"\tfrac{1}{H}\sum_h\big(\hat\gamma_K(x_h) -"}, {"id": "gtilde", "tex": r"\tilde{\gamma}_h", "role": "intermediate"}, {"tex": r"\big)^2"}],
                    [{"id": "penK", "tex": r"\mathcal{L}_K", "role": "intermediate"}, {"tex": "="}, {"id": "mseK", "tex": r"\mathrm{MSE}_K", "role": "intermediate"}, {"tex": r"+\ \lambda_K\,K,\quad \lambda_K = 10^{-2}"}],
                ],
            },
            {
                "id": "selectK", "title": "Best-K argmin selection", "phase": "3 - Best-K",
                "note": "The penalised score is minimised over model order via argmin; on exact ties the smaller K wins, reinforcing parsimony. The selection is never overridden by the post-hoc ambiguity diagnostic.",
                "inputs": ["penK"], "outputs": ["Kstar"],
                "lines": [
                    [{"id": "Kstar", "tex": r"K^{*}", "role": "calculated"}, {"tex": "="}, {"tex": r"\operatorname*{arg\,min}_{K\in\{1,\dots,K_{\max}\}}"}, {"id": "penK", "tex": r"\mathcal{L}_K", "role": "intermediate"}, {"tex": r",\quad K_{\max} = 5"}],
                ],
            },
            {
                "id": "rescale", "title": "Rescale winner to raw amplitude", "phase": "3 - Best-K",
                "note": "Scoring lives on the normalised scale, but the saved amplitudes of the winning order return to raw backscatter units; means and widths are written unchanged.",
                "inputs": ["Kstar", "a0", "scale"], "outputs": ["aout"],
                "lines": [
                    [{"id": "aout", "tex": r"a^{\mathrm{out}}_k", "role": "calculated"}, {"tex": "="}, {"id": "a0", "tex": r"a_k", "role": "intermediate"}, {"tex": r"\cdot"}, {"id": "scale", "tex": r"s", "role": "intermediate"}, {"tex": r",\qquad k \le"}, {"id": "Kstar", "tex": r"K^{*}", "role": "calculated"}],
                ],
            },
            {
                "id": "assemble", "title": "Order and pack the target", "phase": "3 - Best-K",
                "note": "The K_max slots are sorted by ascending mean elevation, with inactive slots (amplitude at or below tau_a) keyed to infinity and pushed last, then written into the interleaved 3K target; slots beyond K* are exact zeros.",
                "inputs": ["Kstar", "aout", "mu0", "sigstar"], "outputs": ["theta"],
                "lines": [
                    [{"id": "theta", "tex": r"\theta", "role": "final"}, {"tex": "="}, {"tex": r"\big[\,"}, {"id": "aout", "tex": r"a^{\mathrm{out}}", "role": "calculated"}, {"tex": ","}, {"id": "mu0", "tex": r"\mu", "role": "intermediate"}, {"tex": ","}, {"id": "sigstar", "tex": r"\sigma^{*}", "role": "calculated"}, {"tex": r"\,\big]_{\pi}^{\,k \le"}, {"id": "Kstar", "tex": r"K^{*}", "role": "calculated"}, {"tex": r"},\ \ \pi=\operatorname{argsort}_k\big(\mu_k\ \text{if}\ a_k>\tau_a\ \text{else}\ \infty\big)"}],
                ],
            },
            {
                "id": "quality", "title": "Fit-quality R-squared map", "phase": "4 - Diagnostics",
                "note": "In the inference/metrics stage, the per-pixel coefficient of determination is computed over elevation against the same thresholded-and-truncated profile the fit saw, in float64, with a 1e-12 stabiliser on the total sum of squares.",
                "inputs": ["theta", "T"], "outputs": ["r2"],
                "lines": [
                    [{"id": "r2", "tex": r"R^2", "role": "final"}, {"tex": "="}, {"tex": r"1 - \dfrac{\sum_h \big(\hat{\gamma}_h("}, {"id": "theta", "tex": r"\theta", "role": "final"}, {"tex": r") - \gamma_h\big)^2}{\sum_h (\gamma_h - \bar{\gamma})^2 + \delta},\quad \delta = 10^{-12}"}],
                ],
            },
            {
                "id": "diagnostics", "title": "K-margin and peak contrast", "phase": "4 - Diagnostics",
                "note": "Post-hoc only and never altering selection: the relative selection margin flags ambiguous pixels (below 0.05), and the uncalibrated peak-to-floor contrast uses the mean of the lowest-amplitude quartile, round(0.25 H) bins, as the noise floor.",
                "inputs": ["penK", "T"], "outputs": ["mrel"],
                "lines": [
                    [{"id": "mrel", "tex": r"m_{\mathrm{rel}}", "role": "final"}, {"tex": "="}, {"tex": r"\dfrac{\mathcal{L}_{2\mathrm{nd}} - \mathcal{L}_{K^{*}}}{\max\!\big(|\mathcal{L}_{K^{*}}|,\,10^{-12}\big)}"}],
                    [{"tex": r"C_{\mathrm{dB}}"}, {"tex": "="}, {"tex": r"10\log_{10}\dfrac{\max_h |T_h|}{\frac{1}{|\mathcal{N}|}\sum_{h\in\mathcal{N}} |T_h|},\quad |\mathcal{N}|=\mathrm{round}(0.25\,H)"}],
                ],
            },
        ]
        return {
            "key"   : "param",
            "name"  : "Parameter Extraction (K-Gaussian fit)",
            "blurb" : "Per-pixel K-Gaussian fit. Threshold and peak-normalise each elevation profile, seed a mixture from prominence peaks on the CPU, fit component widths (and, per mode, amplitudes and means) with clamped bias-corrected Adam on the GPU, select the penalised best order, sort by mean elevation into the interleaved target, then score R2 and K-margin diagnostics.",
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
            {"id": "grid",   "tex": r"\mathcal{G}",       "role": "calculated",   "kind": "set",    "shape": "n_v x n_h",          "desc": "patch grid counts and padding",                              "sample": "n_v=89, n_h=93"},
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
            {"id": "xrec",   "tex": r"\tilde{\mathbf{x}}","role": "calculated",   "kind": "tensor", "shape": "C_in x P x P",       "desc": "denormalised tensor (inverse; clip then expm1)",             "sample": [["0.90", "0.61"], ["0.55", "0.88"]]},
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
                "note": "The deficit between the grid's covered extent and the region is split symmetrically, with the extra pixel going to the bottom or right on odd deficits; border patches are symmetric-padded to fill the overhang.",
                "inputs": ["grid"], "outputs": ["grid"],
                "lines": [
                    [{"tex": r"p_v = P_H + (n_v - 1)\,s - A_z,\qquad p_h = P_W + (n_h - 1)\,s - R_g"}],
                    [{"tex": r"p_{\mathrm{top}} = \lfloor p_v/2\rfloor,\quad p_{\mathrm{bot}} = p_v - p_{\mathrm{top}}"}],
                ],
            },
            {
                "id": "extract", "title": "Patch extraction (contiguous copy)", "phase": "Patch extraction",
                "note": "The clipped read window is copied, never aliased, then symmetric-padded (edge-mirrored) in one pass; the same routine serves the complex stack, the parameters, and the DEM.",
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
                    [{"id": "rep", "tex": r"\mathbf{r}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\Big(\left|"}, {"id": "cpatch", "tex": r"\mathbf{p}", "role": "intermediate"}, {"tex": r"\right|,\ \tfrac{\Re\mathbf{p}}{m},\ \tfrac{\Im\mathbf{p}}{m},\ \angle\mathbf{p}\Big),\quad m=\left|\mathbf{p}\right|\ \big(\!\to 1\ \text{if}\ \left|\mathbf{p}\right|=0\big)"}],
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
                "note": "Statistics are fitted on the train split only, per slot, in float64 over up to 1e6 sampled values. Heavy-tailed magnitude channels (SLC/interferogram magnitude, output amplitude and sigma) use the median and IQR of their log1p values (robust-IQR-log1p); Gaussian-like channels (normalised re/im, SLC phase, output mu, DEM elevation) use the mean and standard deviation (z-score); interferogram phase is mapped by a fixed division by pi. Every fitted scale is floored at 1e-8, and output roles are fitted over active pixels only (amplitude > 1e-3).",
                "inputs": ["xgeo", "gkeys"], "outputs": ["stats"],
                "lines": [
                    [{"tex": r"f(x_c) = \log\!\big(1 + \max(x_c,0)\big)\ \ \text{(log1p slots)},\ \ \text{else } x_c"}],
                    [{"id": "stats", "tex": r"(\mu_c, s_c)", "role": "calculated"}, {"tex": "="}, {"tex": r"\big(P_{50}f,\ \max(P_{75}f{-}P_{25}f,\,10^{-8})\big)\,\text{rob},\ \ \big(\operatorname{mean} f,\ \max(\operatorname{std} f,\,10^{-8})\big)\,\text{z},\ \ (0,\ \pi)\,\text{fix}"}],
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
                "note": "The same per-channel statistics invert normalisation at loss and inference time; for log1p slots the log-domain argument is clipped to [log(1+floor), log(1+ceil)] with floor=0 and ceil=1000 before expm1, bounding the recovered physical value to [0, 1000].",
                "inputs": ["xhat", "stats"], "outputs": ["xrec"],
                "lines": [
                    [{"id": "xrec", "tex": r"\tilde{\mathbf{x}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\operatorname{expm1}\!\big(\operatorname{clip}(\hat{x}_c s_c + \mu_c,\ \log(1{+}f),\ \log(1{+}c))\big)\ \ \text{(log1p)},\ \ \text{else } \hat{x}_c s_c + \mu_c,\quad (f,c)=(0,1000)"}],
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
            {"id": "xhat",  "tex": r"\hat{\mathbf{x}}",            "role": "measured",     "kind": "tensor", "shape": "B x C_in x P x P", "desc": "normalised input batch from the loader",         "sample": [["0.41", "-0.20"], ["-0.33", "0.27"]]},
            {"id": "thn",   "tex": r"\hat{\theta}_{\mathrm{n}}",   "role": "calculated",   "kind": "tensor", "shape": "B x 3K x P x P",   "desc": "raw network output in normalised space",         "sample": ["0.78", "0.55", "0.31", "..."]},
            {"id": "thp",   "tex": r"\hat{\theta}",                "role": "calculated",   "kind": "tensor", "shape": "B x 3K x P x P",   "desc": "denormalised, physically clamped parameters",     "sample": ["12.4", "8.1", "1.9", "..."]},
            {"id": "thrn",  "tex": r"\tilde{\theta}_{\mathrm{n}}", "role": "intermediate", "kind": "tensor", "shape": "B x 3K x P x P",   "desc": "clamped params renormalised for parameter terms", "sample": ["0.74", "0.55", "0.33", "..."]},
            {"id": "gtp",   "tex": r"\theta^{\mathrm{GT}}",        "role": "measured",     "kind": "tensor", "shape": "B x 3K x P x P",   "desc": "ground-truth Gaussian parameters, loader-normalised", "sample": ["0.55", "0.47", "0.34", "..."]},
            {"id": "yhat",  "tex": r"\hat{y}",                     "role": "intermediate", "kind": "tensor", "shape": "B x N x P x P",    "desc": "reconstructed predicted elevation curve",         "sample": [["0.05", "0.71"], ["0.33", "0.88"]]},
            {"id": "y",     "tex": r"y",                           "role": "measured",     "kind": "tensor", "shape": "B x N x P x P",    "desc": "GT elevation curve (denormalised params, no_grad)", "sample": [["0.06", "0.70"], ["0.35", "0.86"]]},
            {"id": "err",   "tex": r"e",                           "role": "intermediate", "kind": "tensor", "shape": "B x N x P x P",    "desc": "curve-space residual y-hat minus y",              "sample": ["-0.01", "0.01", "-0.02", "..."]},
            {"id": "Astr",  "tex": r"\mathbf{A}",                  "role": "intermediate", "kind": "matrix", "shape": "N_s x N",         "desc": "tomographic steering matrix exp(j kz xi)",        "sample": [["1+0j", "0.7+0.7j"], ["1+0j", "-0.7+0.7j"]]},
            {"id": "Rcov",  "tex": r"\mathbf{R}[P]",               "role": "intermediate", "kind": "matrix", "shape": "N_s x N_s",       "desc": "synthesised covariance A diag(P) A^H dxi",        "sample": [["3.1", "1.2+0.4j"], ["1.2-0.4j", "2.8"]]},
            {"id": "lj",    "tex": r"\ell_j",                      "role": "calculated",   "kind": "scalar", "shape": "1",               "desc": "raw value of one enabled loss term",              "sample": "0.0317"},
            {"id": "loss",  "tex": r"\mathcal{L}",                 "role": "calculated",   "kind": "scalar", "shape": "1",               "desc": "weight-normalised composite loss",                "sample": "4.1e-2"},
            {"id": "gnorm", "tex": r"\lVert\mathbf{g}\rVert_2",    "role": "intermediate", "kind": "scalar", "shape": "1",               "desc": "global gradient L2 norm",                         "sample": "2.7"},
            {"id": "grad",  "tex": r"\mathbf{g}",                  "role": "intermediate", "kind": "vector", "shape": "|theta|",         "desc": "clipped parameter gradient",                      "sample": ["1.2e-2", "-4e-3", "..."]},
            {"id": "eta",   "tex": r"\eta_{\mathrm{eff}}",         "role": "intermediate", "kind": "scalar", "shape": "1",               "desc": "effective LR (base x cosine x warmup)",           "sample": "7.3e-4"},
            {"id": "w",     "tex": r"\theta_t",                    "role": "intermediate", "kind": "vector", "shape": "|theta|",         "desc": "model weights at optimiser step t",               "sample": ["0.31", "-0.08", "..."]},
            {"id": "wbest", "tex": r"\theta^{\star}",              "role": "final",        "kind": "vector", "shape": "|theta|",         "desc": "best-epoch checkpointed weights",                 "sample": ["0.30", "-0.09", "..."]},
        ]
        steps = [
            {
                "id": "forward", "title": "Forward pass", "phase": "Reconstruction",
                "note": "The backbone maps the normalised input patch to per-pixel Gaussian parameters (interleaved a, mu, sigma per component) in one forward pass, optionally under bfloat16 autocast.",
                "inputs": ["xhat"], "outputs": ["thn"],
                "lines": [
                    [{"id": "thn", "tex": r"\hat{\theta}_{\mathrm{n}}", "role": "calculated"}, {"tex": "="}, {"tex": r"f_{\theta}\!\big("}, {"id": "xhat", "tex": r"\hat{\mathbf{x}}", "role": "measured"}, {"tex": r"\big),\qquad 3K = 3\,n_g"}],
                ],
            },
            {
                "id": "tdenorm", "title": "Denormalise predictions", "phase": "Reconstruction",
                "note": "Each log1p-encoded output channel (out/amp, out/sigma) is inverted with expm1 after its pre-exponent value is clamped into [log(1+floor), log(1+ceil)] with a leaky slope of 0.1, holding the physical value inside [0, 1000] while keeping a gradient outside; the z-scored out/mu channel is inverted linearly.",
                "inputs": ["thn"], "outputs": ["thp"],
                "lines": [
                    [{"id": "thp", "tex": r"\hat{\theta}", "role": "calculated"}, {"tex": "="}, {"tex": r"\mathrm{expm1}\!\big(\mathrm{clamp}_{0.1}(\hat{\theta}_{\mathrm{n}}\,s + \ell,\ [\,0,\ \log(1{+}c_{\max})\,])\big)\ \ \text{(log1p)},\ \ \text{else } \hat{\theta}_{\mathrm{n}}\,s + \ell"}],
                ],
            },
            {
                "id": "clamp", "title": "Physical parameter bounds", "phase": "Reconstruction",
                "note": "Predictions are clamped to grid-relative physical bounds with a straight-through leaky slope of 0.1, so the amplitude, mu and sigma heads keep a small gradient through saturation.",
                "inputs": ["thp"], "outputs": ["thp"],
                "lines": [
                    [{"id": "thp", "tex": r"\hat{a}_k \in [0,a_{\max}],\ \ \hat{\mu}_k \in [x_{\min},x_{\max}],\ \ \hat{\sigma}_k \in \big[\tfrac{\Delta x}{2},\tfrac{x_{\max}-x_{\min}}{2}\big]", "role": "calculated"}],
                    [{"tex": r"\mathrm{clamp}_{\mathrm{leaky}}(x) = \mathrm{clip}(x) + 0.1\,\big(x - \mathrm{clip}(x)\big),\qquad a_{\max}=1000"}],
                ],
            },
            {
                "id": "renorm", "title": "Renormalise for parameter terms", "phase": "Reconstruction",
                "note": "The clamped physical predictions are mapped back to training space so the parameter-space loss terms operate in the same normalised units as the labels.",
                "inputs": ["thp"], "outputs": ["thrn"],
                "lines": [
                    [{"id": "thrn", "tex": r"\tilde{\theta}_{\mathrm{n}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\big(\log\!1\mathrm{p}("}, {"id": "thp", "tex": r"\hat{\theta}", "role": "calculated"}, {"tex": r") - \ell\big)\,/\,s\ \ \text{(log1p)},\ \ \text{else } (\hat{\theta} - \ell)/s"}],
                ],
            },
            {
                "id": "reconstruct", "title": "Gaussian curve reconstruction", "phase": "Reconstruction",
                "note": "Ground truth is denormalised under no_grad, then both the predicted physical parameters and the GT are evaluated on the elevation axis as an additive Gaussian mixture; the exponent is floored at -100 and sigma at 1e-6.",
                "inputs": ["thp", "gtp"], "outputs": ["yhat", "y"],
                "lines": [
                    [{"id": "yhat", "tex": r"\hat{y}(x_n)", "role": "intermediate"}, {"tex": "="}, {"tex": r"\sum_{k}"}, {"id": "thp", "tex": r"\hat{a}_k", "role": "calculated"}, {"tex": r"\exp\!\Big(-\tfrac{(x_n-\hat{\mu}_k)^2}{2\hat{\sigma}_k^2}\Big),\qquad K = C/3"}],
                ],
            },
            {
                "id": "residual", "title": "Curve residual", "phase": "Curve loss",
                "note": "The elementwise residual is computed once when any pointwise curve term is enabled, and shared by the MSE, L1, Huber and Charbonnier terms; shape-sensitive terms take the curves directly.",
                "inputs": ["yhat", "y"], "outputs": ["err"],
                "lines": [
                    [{"id": "err", "tex": r"e", "role": "intermediate"}, {"tex": "="}, {"id": "yhat", "tex": r"\hat{y}", "role": "intermediate"}, {"tex": r"\ -\ "}, {"id": "y", "tex": r"y", "role": "measured"}],
                ],
            },
            {
                "id": "curvepoint", "title": "Pointwise curve terms", "phase": "Curve loss",
                "note": "Four optional pointwise reductions of the shared residual, each averaged over all elements; Huber transitions at delta=1.0 and Charbonnier uses an epsilon=1e-3 smoothed L1.",
                "inputs": ["err"], "outputs": ["lj"],
                "lines": [
                    [{"tex": r"\ell_{\mathrm{MSE}} = \big\langle e^2\big\rangle,\qquad \ell_{L1} = \big\langle |e| \big\rangle"}],
                    [{"id": "lj", "tex": r"\ell_{\mathrm{Hub}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\big\langle \tfrac12 e^2\,[|e|\le\delta] + \delta(|e|-\tfrac{\delta}{2})\,[|e|>\delta]\big\rangle"}],
                    [{"tex": r"\ell_{\mathrm{Charb}} = \big\langle \sqrt{e^2 + \varepsilon^2}\big\rangle"}],
                ],
            },
            {
                "id": "curveshape", "title": "Shape-sensitive curve term", "phase": "Curve loss",
                "note": "The default curve objective: cosine distance over pixels whose GT curve norm exceeds 1e-3, penalising profile shape rather than magnitude (weight 0.05 in the default curriculum).",
                "inputs": ["yhat", "y"], "outputs": ["lj"],
                "lines": [
                    [{"id": "lj", "tex": r"\ell_{\cos}", "role": "calculated"}, {"tex": "="}, {"tex": r"\Big\langle 1 - \tfrac{\langle\hat{y},y\rangle}{\lVert\hat{y}\rVert\,\lVert y\rVert}\Big\rangle_{\lVert y\rVert>10^{-3}}"}],
                ],
            },
            {
                "id": "physgeom", "title": "Tomographic forward operator", "phase": "Physics loss",
                "note": "The physics terms share a Fourier forward operator. Under the default height convention the wavenumber uses the round-trip 4-pi/(lambda r0) factor divided by the look-angle sine, with the master-relative perpendicular baseline; when any physics term is active a per-pixel kz field replaces this single shared steering matrix.",
                "inputs": [], "outputs": ["Astr"],
                "lines": [
                    [{"tex": r"b_\perp = h\cos\theta + v\sin\theta,\qquad k_z^{(i)} = \dfrac{4\pi\, b_\perp^{(i)}}{\lambda\, r_0\, \sin\theta}"}],
                    [{"id": "Astr", "tex": r"A_{in}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\exp\!\big(j\,k_z^{(i)}\,\xi_n\big)"}],
                ],
            },
            {
                "id": "physmoments", "title": "Power and moment terms", "phase": "Physics loss",
                "note": "Optional ratio-based terms compare the relative integrated power and the first three profile moments (mass, centroid, spread) with unit moment weights, reduced over pixels whose GT power exceeds the physics floor 1e-3.",
                "inputs": ["yhat", "y"], "outputs": ["lj"],
                "lines": [
                    [{"tex": r"m_0 = \textstyle\sum_n P_n\,\Delta\xi,\quad \bar z = \tfrac{\sum_n P_n \xi_n}{\sum_n P_n},\quad \sigma_z = \sqrt{\tfrac{\sum_n P_n \xi_n^2}{\sum_n P_n} - \bar z^2 + 10^{-8}}"}],
                    [{"id": "lj", "tex": r"\ell_{\mathrm{mom}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\Big\langle \tfrac{w_0\frac{|\Delta m_0|}{m_0^T} + w_1\frac{|\Delta\bar z|}{\xi_{\max}-\xi_{\min}} + w_2\frac{|\Delta\sigma_z|}{\xi_{\max}-\xi_{\min}}}{w_0+w_1+w_2}\Big\rangle"}],
                ],
            },
            {
                "id": "physcov", "title": "Coherence and covariance matching", "phase": "Physics loss",
                "note": "The two default physics terms (weight 0.05 each in the complete curriculum): coherence re-synthesis compares the mass-normalised characteristic functions of the two profiles; covariance matching exploits R's linearity to transform only the difference, both insensitive to absolute power.",
                "inputs": ["yhat", "y", "Astr"], "outputs": ["Rcov", "lj"],
                "lines": [
                    [{"tex": r"\gamma_P(k_z^{(i)}) = \tfrac{\sum_n P_n e^{j k_z^{(i)}\xi_n}}{\sum_n P_n},\qquad \ell_{\mathrm{coh}} = \big\langle \tfrac{1}{N_s}\textstyle\sum_i |\gamma_P^{(i)} - \gamma_T^{(i)}|^2\big\rangle"}],
                    [{"id": "Rcov", "tex": r"\mathbf{R}[P]", "role": "intermediate"}, {"tex": "="}, {"id": "Astr", "tex": r"\mathbf{A}", "role": "intermediate"}, {"tex": r"\,\mathrm{diag}(P)\,\mathbf{A}^H\Delta\xi,\quad \ell_{\mathrm{cov}} = \big\langle \tfrac{\lVert\mathbf{R}[P]-\mathbf{R}[T]\rVert_F^2}{\lVert\mathbf{R}[T]\rVert_F^2}\big\rangle"}],
                ],
            },
            {
                "id": "physcapon", "title": "Capon cycle-consistency", "phase": "Physics loss",
                "note": "The most expensive, opt-in physics term synthesises the covariance from the predicted profile, adds signal-adaptive diagonal loading (epsilon=1e-2 times mean trace), Hermitianises, then forms the Capon spectrum by solving the loaded system and compares mass-normalised spectra.",
                "inputs": ["Rcov", "Astr", "y"], "outputs": ["lj"],
                "lines": [
                    [{"tex": r"\hat{T}_P(\xi_n) = \dfrac{1}{\mathbf{a}^H(\xi_n)\big("}, {"id": "Rcov", "tex": r"\mathbf{R}[P]", "role": "intermediate"}, {"tex": r" + \epsilon\bar\sigma\mathbf{I}\big)^{-1}\mathbf{a}(\xi_n)},\quad \bar\sigma = \tfrac{1}{N_s}\mathrm{tr}\,\mathbf{R}[P]"}],
                    [{"id": "lj", "tex": r"\ell_{\mathrm{cyc}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\Big\langle \tfrac{1}{N}\textstyle\sum_n\big(\tfrac{\hat{T}_P(\xi_n)}{m_0^{\hat T}} - \tfrac{T(\xi_n)}{m_0^{T}}\big)^2\Big\rangle"}],
                ],
            },
            {
                "id": "paramterms", "title": "Parameter-space terms", "phase": "Parameter loss",
                "note": "The default sorted-GT matching orders GT slots by mu among active components (inactive amplitudes < 1e-3 pushed to the end); inactive slots mask their mu and sigma to zero so empty slots contribute amplitude only. Param-L1 (the default supervised term) is active-normalised in normalised space; TV is an optional roughness penalty.",
                "inputs": ["thrn", "gtp"], "outputs": ["lj"],
                "lines": [
                    [{"id": "lj", "tex": r"\ell_{\mathrm{p\text{-}L1}}", "role": "calculated"}, {"tex": "="}, {"tex": r"\dfrac{\sum w_p\,m\,|\,\tilde{\theta}_{\mathrm{n}} - \theta^{\mathrm{GT}}\,|}{\sum w_p\,m},\quad m = \mathbb{1}[a^{\mathrm{GT}} > 10^{-3}]\ (\mu,\sigma)"}],
                    [{"tex": r"\ell_{\mathrm{TV}} = \overline{|\tilde\theta_{h}-\tilde\theta_{h-1}|} + \overline{|\tilde\theta_{w}-\tilde\theta_{w-1}|}"}],
                ],
            },
            {
                "id": "composite", "title": "Composite weighted loss", "phase": "Composite",
                "note": "The total loss is the weight-normalised sum over enabled terms; each weight is the user-selected weight_* value directly (no automatic normaliser). The loss-scale probe can suggest factors to fold into those weights, but the user brings heterogeneous terms to comparable magnitude.",
                "inputs": ["lj"], "outputs": ["loss"],
                "lines": [
                    [{"id": "loss", "tex": r"\mathcal{L}", "role": "calculated"}, {"tex": "="}, {"tex": r"\dfrac{\sum_j \alpha_j\,"}, {"id": "lj", "tex": r"\ell_j", "role": "calculated"}, {"tex": r"}{\sum_j \alpha_j},\qquad \alpha_j = \mathtt{weight\_}j"}],
                ],
            },
            {
                "id": "gradclip", "title": "Backprop and gradient clipping", "phase": "Optimiser step",
                "note": "Non-finite batches are skipped (or abort training); surviving gradients are rescaled by a common factor so the global L2 norm never exceeds the threshold, which is fixed at 1.0 or an adaptive percentile / mean-plus-k-sigma over a 200-step window.",
                "inputs": ["loss"], "outputs": ["gnorm", "grad"],
                "lines": [
                    [{"id": "gnorm", "tex": r"\lVert\mathbf{g}\rVert_2", "role": "intermediate"}, {"tex": "="}, {"tex": r"\Big(\textstyle\sum_i \lVert\nabla_{\theta^{(i)}}\mathcal{L}\rVert_2^2\Big)^{1/2}"}],
                    [{"id": "grad", "tex": r"\mathbf{g}", "role": "intermediate"}, {"tex": r"\leftarrow \mathbf{g}\cdot\min\!\Big(1,\ \tfrac{\tau}{\lVert\mathbf{g}\rVert_2 + \varepsilon}\Big),\quad \tau\in\{1.0,\ P_{95},\ \bar g + k\sigma_g\}"}],
                ],
            },
            {
                "id": "adamw", "title": "AdamW update", "phase": "Optimiser step",
                "note": "Bias-corrected adaptive moments (betas 0.9/0.999, eps 1e-8) with decoupled weight decay 0.1 and per-group learning rates; the epoch loop drives the training loss down over many steps, with optional EMA shadow weights.",
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
                "note": "Each group's effective LR is its base rate times the per-epoch cosine factor (T=100, eta_min=1e-6) times the per-step warmup factor (200 steps from start factor 0.1); at swap epoch 15 the loss curriculum moves from the warmup objective to the complete objective, and with the curriculum disabled the complete objective runs from the first epoch.",
                "inputs": [], "outputs": ["eta"],
                "lines": [
                    [{"tex": r"F(t) = \tfrac{\eta_{\min}}{\eta_0} + \tfrac12\big(1 - \tfrac{\eta_{\min}}{\eta_0}\big)\big(1 + \cos\tfrac{\pi\min(t,T)}{T}\big)"}],
                    [{"id": "eta", "tex": r"\eta_{\mathrm{eff}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\eta_0\cdot F(t)\cdot f_{\mathrm{warmup}}(s),\qquad f_{\mathrm{warmup}}(s) = \alpha_0 + (1-\alpha_0)\tfrac{s}{S}"}],
                ],
            },
            {
                "id": "checkpoint", "title": "Validation and checkpoint", "phase": "Eval & checkpoint",
                "note": "Evaluation runs every 5 epochs (EMA weights when enabled); the best epoch is checkpointed on strict improvement and early stopping reverts to it after 15 evaluations without a new minimum. The best-loss baseline resets across a curriculum swap.",
                "inputs": ["w"], "outputs": ["wbest"],
                "lines": [
                    [{"id": "wbest", "tex": r"\theta^{\star}", "role": "final"}, {"tex": "="}, {"tex": r"\operatorname*{arg\,min}_{t}\ \mathcal{L}_{\mathrm{val}}("}, {"id": "w", "tex": r"\theta_t", "role": "intermediate"}, {"tex": r")"}],
                ],
            },
        ]
        return {
            "key": "training", "name": "Training (Supervised backbone)",
            "blurb": "One forward pass predicts all Gaussian parameters; predictions are denormalised, physically clamped and renormalised, then a weight-normalised composite over curve, parameter and optional physics terms is backpropagated through AdamW with linear warmup, cosine annealing, a two-phase loss curriculum, gradient clipping and early stopping.",
            "nodes": nodes, "steps": steps,
        }

    def _inference(self) -> dict:
        nodes = [
            {"id": "ckpt",     "tex": r"\theta^{\star}",           "role": "measured",     "kind": "vector", "shape": "|theta|",        "desc": "best-epoch checkpoint weights (params) with bundled x-axis; per-slot norm stats load from the run meta", "sample": ["0.31", "-0.08", "..."]},
            {"id": "xaxis",    "tex": r"\mathbf{x}",               "role": "measured",     "kind": "vector", "shape": "N",              "desc": "elevation axis (m) restored from the checkpoint",       "sample": ["-20", "-19.6", "...", "20"]},
            {"id": "xhat",     "tex": r"\hat{\mathbf{x}}",         "role": "measured",     "kind": "tensor", "shape": "B x C_in x P x P","desc": "normalised input patch batch from the loader",          "sample": [["0.41", "-0.20"], ["-0.33", "0.27"]]},
            {"id": "znorm",    "tex": r"\hat{\mathbf{z}}",         "role": "intermediate", "kind": "tensor", "shape": "B x 3K x P x P", "desc": "raw normalised network output",                         "sample": ["0.22", "-1.1", "0.7", "..."]},
            {"id": "th",       "tex": r"\hat{\theta}",             "role": "calculated",   "kind": "tensor", "shape": "B x 3K x P x P", "desc": "denormalised, hard-clamped predicted params (a,mu,sig per slot)", "sample": ["0.78", "12.1", "1.9", "..."]},
            {"id": "thgt",     "tex": r"\theta^{\mathrm{GT}}",     "role": "measured",     "kind": "tensor", "shape": "B x 3K x P x P", "desc": "ground-truth params, denormalised (no clamp)",          "sample": ["0.80", "12.0", "1.8", "..."]},
            {"id": "thgts",    "tex": r"\theta^{\mathrm{GT}}_{\pi}","role": "intermediate","kind": "tensor", "shape": "B x 3K x P x P", "desc": "mu-sorted GT params (active first, inactive at the tail)","sample": ["0.80", "11.9", "1.8", "..."]},
            {"id": "p",        "tex": r"\mathbf{p}",               "role": "intermediate", "kind": "tensor", "shape": "N x P x P",      "desc": "per-patch reconstructed elevation spectrum",            "sample": [["0.05", "0.71"], ["0.33", "0.88"]]},
            {"id": "winv",     "tex": r"w_v",                      "role": "intermediate", "kind": "vector", "shape": "P",              "desc": "1D Hann taper, floored at 1e-3",                        "sample": ["0.10", "0.45", "0.85", "1.00"]},
            {"id": "win",      "tex": r"w",                        "role": "intermediate", "kind": "matrix", "shape": "P x P",          "desc": "separable 2D Hann window, also the patch centrality",    "sample": [["0.10", "0.45"], ["0.45", "1.00"]]},
            {"id": "acc",      "tex": r"A",                        "role": "intermediate", "kind": "tensor", "shape": "N x H_pad x W_pad","desc": "curve value accumulator",                              "sample": [["1.2", "3.4"], ["0.8", "2.1"]]},
            {"id": "wacc",     "tex": r"W",                        "role": "intermediate", "kind": "matrix", "shape": "H_pad x W_pad",  "desc": "curve weight accumulator",                              "sample": [["0.9", "1.8"], ["1.8", "2.7"]]},
            {"id": "cube",     "tex": r"\hat{C}",                  "role": "final",        "kind": "tensor", "shape": "N x Az x Rg",    "desc": "stitched predicted curve cube (denorm)",                "sample": [["0.2", "0.9"], ["0.1", "0.7"]]},
            {"id": "cubegt",   "tex": r"C",                        "role": "measured",     "kind": "tensor", "shape": "N x Az x Rg",    "desc": "stitched ground-truth curve cube (denorm)",             "sample": [["0.2", "0.9"], ["0.1", "0.7"]]},
            {"id": "params",   "tex": r"\hat{\Theta}",             "role": "final",        "kind": "tensor", "shape": "3K x Az x Rg",   "desc": "centrality-selected predicted parameter cube",          "sample": ["0.78", "12.1", "1.9", "..."]},
            {"id": "paramsgt", "tex": r"\Theta^{\mathrm{GT}}",     "role": "measured",     "kind": "tensor", "shape": "3K x Az x Rg",   "desc": "GT parameter cube; inactive-slot mu,sigma set to NaN",   "sample": ["0.80", "11.9", "NaN", "..."]},
            {"id": "pr2",      "tex": r"R^2_{a,r}",                "role": "calculated",   "kind": "matrix", "shape": "Az x Rg",        "desc": "per-pixel R-squared map",                               "sample": [["0.95", "0.88"], ["0.91", "0.97"]]},
            {"id": "r2",       "tex": r"R^2",                      "role": "final",        "kind": "scalar", "shape": "1",              "desc": "overall reconstruction R-squared",                      "sample": "0.94"},
            {"id": "psnr",     "tex": r"\mathrm{PSNR}",            "role": "calculated",   "kind": "scalar", "shape": "1",              "desc": "peak signal-to-noise ratio (dB), GT dynamic range",     "sample": "28.4"},
            {"id": "elevr2",   "tex": r"R^2_{\mathrm{elev}}",      "role": "calculated",   "kind": "vector", "shape": "N",              "desc": "per-elevation-bin R-squared",                           "sample": ["0.91", "0.93", "..."]},
            {"id": "gmu",      "tex": r"\mathrm{MAE}_{\mu}",       "role": "calculated",   "kind": "scalar", "shape": "1",              "desc": "matched Gaussian mu MAE over matched active pairs",      "sample": "0.42"},
            {"id": "mf1",      "tex": r"F_1",                      "role": "calculated",   "kind": "scalar", "shape": "1",              "desc": "matched detection F1 within mu tolerance",              "sample": "0.86"},
            {"id": "redc",     "tex": r"\mathbf{r}",               "role": "calculated",   "kind": "tensor", "shape": "N x Az x Rg",    "desc": "reduced-subset Capon tomogram (re-synthesised)",        "sample": [["0.3", "0.8"], ["0.2", "0.6"]]},
            {"id": "dimp",     "tex": r"\Delta_{a,r}",             "role": "final",        "kind": "matrix", "shape": "Az x Rg",        "desc": "improvement map: reduced minus prediction MSE (unit-area)","sample": [["0.01", "-0.00"], ["0.02", "0.01"]]},
            {"id": "kz",       "tex": r"k_z",                      "role": "measured",     "kind": "tensor", "shape": "T x Az x Rg",    "desc": "per-pixel interferometric wavenumber field (rad/m), T=1+N_s","sample": [["0.05", "0.06"], ["0.04", "0.05"]]},
            {"id": "meas",     "tex": r"\tilde{\gamma}",           "role": "intermediate", "kind": "tensor", "shape": "N_s x Az x Rg",  "desc": "multilooked measured interferogram unit phasors",       "sample": [["1+0j", "0.7+0.7j"], ["0+1j", "0.9-0.1j"]]},
            {"id": "coh",      "tex": r"E_{\gamma}",               "role": "final",        "kind": "matrix", "shape": "Az x Rg",        "desc": "per-pixel coherence-resynthesis error map (pred vs GT)", "sample": [["0.02", "0.05"], ["0.01", "0.03"]]},
            {"id": "pha",      "tex": r"\rho_{\varphi}",           "role": "calculated",   "kind": "scalar", "shape": "1",              "desc": "mean measured-vs-synthesised phase agreement",          "sample": "0.78"},
        ]
        steps = [
            {
                "id": "load", "title": "Strict run reconstruction", "phase": "Load run",
                "note": "The trained architecture is rebuilt verbatim from run_summary.json and the saved model config, the best-epoch weights (params), x-axis and per-slot norm stats are restored, and the predictor is wrapped with the norm-stats amplitude ceiling (amp_max); inference requires a single contiguous split region or stitching aborts loudly.",
                "inputs": ["ckpt"], "outputs": ["xaxis"],
                "lines": [
                    [{"id": "ckpt", "tex": r"\theta^{\star}", "role": "measured"}, {"tex": "="}, {"tex": r"\mathrm{ckpt[params]}\ @\ \mathrm{best\ epoch},\qquad"}, {"id": "xaxis", "tex": r"\mathbf{x}", "role": "measured"}, {"tex": r"= \{x_n\}_{n=1}^{N}"}],
                    [{"tex": r"|\Omega_{\mathrm{split}}| = 1\quad(\text{one contiguous crop, else abort})"}],
                ],
            },
            {
                "id": "predict", "title": "Windowed prediction", "phase": "Windowed predict",
                "note": "Under no-grad the wrapped model maps every normalised patch on the deterministic sliding-window grid to raw normalised parameters; patches are consumed in grid order so the stitched cube has no holes.",
                "inputs": ["xhat"], "outputs": ["znorm"],
                "lines": [
                    [{"id": "znorm", "tex": r"\hat{\mathbf{z}}", "role": "intermediate"}, {"tex": "="}, {"tex": r"f_{\theta^{\star}}\!\big("}, {"id": "xhat", "tex": r"\hat{\mathbf{x}}", "role": "measured"}, {"tex": r"\big)"}],
                ],
            },
            {
                "id": "idenorm", "title": "Denormalise and hard clamp", "phase": "Windowed predict",
                "note": "Inside the model wrapper the raw output is denormalised by the per-slot output stats then hard-clamped with leaky_slope=0 to physical bounds: amplitude to [0, a_max], mean to the elevation axis, spread to half a bin up to half the span.",
                "inputs": ["znorm"], "outputs": ["th"],
                "lines": [
                    [{"id": "th", "tex": r"\hat{a}_k", "role": "calculated"}, {"tex": r"= \mathrm{clip}(\tilde a_k,\,0,\,a_{\max}),\quad \hat{\mu}_k = \mathrm{clip}(\tilde\mu_k,\,x_{\min},\,x_{\max})"}],
                    [{"id": "th", "tex": r"\hat{\sigma}_k", "role": "calculated"}, {"tex": r"= \mathrm{clip}\!\big(\tilde\sigma_k,\ \tfrac12 x_{\mathrm{step}},\ \tfrac12 x_{\mathrm{range}}\big),\quad x_{\mathrm{step}} = \tfrac{x_{\max}-x_{\min}}{N-1}"}],
                ],
            },
            {
                "id": "align", "title": "GT denormalise and mu-sort", "phase": "Windowed predict",
                "note": "GT params are denormalised by the dataset output stats (no clamp), then per pixel the slots are argsorted by mean with inactive slots (amplitude at or below 1e-3) pushed to the tail, giving GT a canonical mu-ordered storage order for downstream matching.",
                "inputs": ["thgt"], "outputs": ["thgts"],
                "lines": [
                    [{"id": "thgts", "tex": r"\theta^{\mathrm{GT}}_{\pi}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\mathrm{take}\big("}, {"id": "thgt", "tex": r"\theta^{\mathrm{GT}}", "role": "measured"}, {"tex": r",\ \pi^{\star}\big),\quad \pi^{\star} = \operatorname{argsort}_k\,\kappa_k"}],
                    [{"tex": r"\kappa_k = \mu^{\mathrm{GT}}_k\ \ \text{if}\ a^{\mathrm{GT}}_k > 10^{-3},\ \ \text{else}\ +\infty"}],
                ],
            },
            {
                "id": "recon", "title": "Curve reconstruction", "phase": "Windowed predict",
                "note": "Each patch's parameters, the clamped prediction and the mu-sorted GT alike, are evaluated on the elevation axis into a spectrum; amplitudes are rectified at zero and the kernel denominator is 2 sigma^2 + 1e-8 with no sigma floor.",
                "inputs": ["th", "xaxis"], "outputs": ["p"],
                "lines": [
                    [{"id": "p", "tex": r"\mathbf{p}_n", "role": "intermediate"}, {"tex": "="}, {"tex": r"\sum_{k} \max("}, {"id": "th", "tex": r"\hat{a}_k", "role": "calculated"}, {"tex": r",0)\,\exp\!\Big(-\tfrac{(x_n-\hat{\mu}_k)^2}{2\hat{\sigma}_k^2 + 10^{-8}}\Big)"}],
                ],
            },
            {
                "id": "window", "title": "Separable Hann window", "phase": "Cube stitch",
                "note": "A separable Hann window de-emphasises patch borders; each axis factor is floored at 1e-3 before the outer product, so every covered position keeps a strictly positive overlap weight.",
                "inputs": [], "outputs": ["win"],
                "lines": [
                    [{"id": "winv", "tex": r"w_v[i]", "role": "intermediate"}, {"tex": "="}, {"tex": r"\max\!\Big(0.5 - 0.5\cos\tfrac{2\pi(i+0.5)}{P},\ 10^{-3}\Big)"}],
                    [{"id": "win", "tex": r"w", "role": "intermediate"}, {"tex": "="}, {"id": "winv", "tex": r"w_v", "role": "intermediate"}, {"tex": r"\otimes\ w_h"}],
                ],
            },
            {
                "id": "ola", "title": "Weighted overlap-add", "phase": "Cube stitch",
                "note": "The pred and GT curve patches are stitched by weighted overlap-add: each windowed patch is scattered additively into the value accumulator at its grid origin while the window itself accumulates in a parallel weight buffer, so the blend is order-independent.",
                "inputs": ["p", "win"], "outputs": ["acc", "wacc"],
                "lines": [
                    [{"id": "acc", "tex": r"A", "role": "intermediate"}, {"tex": r"\mathrel{+}=\ "}, {"id": "p", "tex": r"\mathbf{p}", "role": "intermediate"}, {"tex": r"\cdot"}, {"id": "win", "tex": r"w", "role": "intermediate"}, {"tex": r",\qquad"}, {"id": "wacc", "tex": r"W", "role": "intermediate"}, {"tex": r"\mathrel{+}=\ w"}],
                ],
            },
            {
                "id": "finalise", "title": "Curve cube finalisation", "phase": "Cube stitch",
                "note": "The pred and GT curve cubes are formed by dividing the value accumulator by the weight accumulator and trimming the grid padding to the scene; a coverage guard raises if any scene pixel received zero weight.",
                "inputs": ["acc", "wacc"], "outputs": ["cube"],
                "lines": [
                    [{"id": "cube", "tex": r"\hat{C}", "role": "final"}, {"tex": "="}, {"tex": r"\dfrac{"}, {"id": "acc", "tex": r"A", "role": "intermediate"}, {"tex": r"}{"}, {"id": "wacc", "tex": r"W", "role": "intermediate"}, {"tex": r"}\ \Big|_{\mathrm{trim}},\qquad W > 0\ \ \forall\,(a,r)"}],
                ],
            },
            {
                "id": "paramstitch", "title": "Parameter select-stitch", "phase": "Cube stitch",
                "note": "Parameter cubes are not blended: at each pixel the patch with the largest Hann centrality wins, overwriting the running value wherever its centrality exceeds the incumbent; inactive GT slots then have their mu and sigma set to NaN before scoring.",
                "inputs": ["th", "win"], "outputs": ["params"],
                "lines": [
                    [{"id": "params", "tex": r"\hat{\Theta}", "role": "final"}, {"tex": r"[:,m] = "}, {"id": "th", "tex": r"\hat{\theta}", "role": "calculated"}, {"tex": r"[:,m],\quad m = \big\{"}, {"id": "win", "tex": r"w", "role": "intermediate"}, {"tex": r" > w^{\star}\big\},\ \ w^{\star}\!\leftarrow\! w[m]"}],
                    [{"id": "paramsgt", "tex": r"\Theta^{\mathrm{GT}}", "role": "measured"}, {"tex": r"[\mu_k,\sigma_k] = \mathrm{NaN}\ \ \text{where}\ a^{\mathrm{GT}}_k \le 10^{-3}"}],
                ],
            },
            {
                "id": "pixelmaps", "title": "Per-pixel metric maps", "phase": "Pixel metrics",
                "note": "Five per-pixel maps are reduced over the N elevation bins of the stitched curve cubes: MSE, MAE, R-squared (denominator floored at 1e-12), cosine similarity (norms floored at 1e-8), and absolute peak-bin index error.",
                "inputs": ["cube", "cubegt"], "outputs": ["pr2"],
                "lines": [
                    [{"id": "pr2", "tex": r"R^2_{a,r}", "role": "calculated"}, {"tex": "="}, {"tex": r"1 - \dfrac{\sum_n(\hat C_{n,a,r}-C_{n,a,r})^2}{\sum_n(C_{n,a,r}-\bar C_{a,r})^2 + 10^{-12}}"}],
                    [{"tex": r"\Delta n_{a,r} = \big|\arg\max_n \hat C_{n,a,r} - \arg\max_n C_{n,a,r}\big|"}],
                ],
            },
            {
                "id": "globalcurve", "title": "Global curve metrics", "phase": "Curve metrics",
                "note": "Cube-wide scalars at physical (denorm) scale: MSE, RMSE, overall R-squared (eps 1e-12), and PSNR whose peak signal is the GT-only dynamic range C_max minus C_min.",
                "inputs": ["cube", "cubegt"], "outputs": ["r2", "psnr"],
                "lines": [
                    [{"id": "r2", "tex": r"R^2", "role": "final"}, {"tex": "="}, {"tex": r"1 - \dfrac{\sum_{n,a,r}(\hat C - C)^2}{\sum_{n,a,r}(C - \bar C)^2 + 10^{-12}}"}],
                    [{"id": "psnr", "tex": r"\mathrm{PSNR}", "role": "calculated"}, {"tex": "="}, {"tex": r"10\log_{10}\dfrac{(C_{\max}-C_{\min})^2}{\mathrm{MSE}_{\mathrm{curve}}}\ \ \text{(dB)}"}],
                ],
            },
            {
                "id": "elevssim", "title": "Per-elevation metrics and SSIM", "phase": "Curve metrics",
                "note": "Per elevation bin (all pixels as samples): MAE, RMSE, R-squared and a cross-entropy between column-normalised distributions; plus mean SSIM over all slices along each axis, on both denorm and unit-area cubes at each slice's GT data range.",
                "inputs": ["cube", "cubegt"], "outputs": ["elevr2"],
                "lines": [
                    [{"id": "elevr2", "tex": r"R^2_{\mathrm{elev}}(n)", "role": "calculated"}, {"tex": "="}, {"tex": r"1 - \dfrac{\sum_{a,r}(\hat C_{n,a,r}-C_{n,a,r})^2}{\sum_{a,r}(C_{n,a,r}-\bar C_n)^2 + 10^{-12}}"}],
                    [{"tex": r"\mathrm{CE}(n) = -\tfrac{1}{A_z R_g}\textstyle\sum_{a,r} \bar p^{\mathrm{GT}}_{n}\log \bar p^{\mathrm{pred}}_{n},\quad \bar p_n = \tfrac{C_n}{\sum_m C_m}"}],
                ],
            },
            {
                "id": "paramslot", "title": "Matched Gaussian metrics", "phase": "Param metrics",
                "note": "On active pixels each predicted pixel's Gaussians are matched to its GT Gaussians by exact brute-force optimal assignment over all K! permutations on |dmu| (inactive pairs cost 1e7, no Hungarian fallback): matched mu and sigma MAE/RMSE over matched pairs, plus detection recall, precision and F1 counting a hit when |dmu| is at most 5 elevation units.",
                "inputs": ["params", "paramsgt"], "outputs": ["gmu", "mf1"],
                "lines": [
                    [{"id": "gmu", "tex": r"\mathrm{MAE}_{\mu}", "role": "calculated"}, {"tex": "="}, {"tex": r"\tfrac{1}{|\mathcal M|}\textstyle\sum_{(i,j)\in\mathcal M}|\hat\mu_{i}-\mu^{\mathrm{GT}}_{j}|,\quad \pi^{\star}=\operatorname*{argmin}_{\pi}\textstyle\sum_i|\hat\mu_i-\mu^{\mathrm{GT}}_{\pi(i)}|"}],
                    [{"id": "mf1", "tex": r"F_1", "role": "calculated"}, {"tex": "="}, {"tex": r"\dfrac{2\,\mathrm{Prec}\,\mathrm{Rec}}{\mathrm{Prec}+\mathrm{Rec}},\quad \text{hit}:\ |\Delta\mu|\le 5"}],
                ],
            },
            {
                "id": "reduced", "title": "Reduced Capon re-synthesis", "phase": "Reduced baseline",
                "note": "Enabled by default and run only when the run trained on a strict secondary subset: a classical Capon tomogram is re-synthesised from the primary plus that subset (effort high, stetools env); after an advisory orientation check the GT, prediction and reduced cubes are unit-area normalised (eps 1e-12) and the per-pixel MSE improvement of the network over the reduced baseline is mapped.",
                "inputs": ["redc", "cube", "cubegt"], "outputs": ["dimp"],
                "lines": [
                    [{"tex": r"\bar y_{n,a,r} = \dfrac{y_{n,a,r}}{\max(\sum_n y_{n,a,r},\,10^{-12})}\quad (y\in\{C,\hat C,\mathbf{r}\})"}],
                    [{"id": "dimp", "tex": r"\Delta_{a,r}", "role": "final"}, {"tex": "="}, {"tex": r"\mathrm{MSE}^{\mathrm{red}}_{a,r} - \mathrm{MSE}^{\mathrm{pred}}_{a,r}"}],
                ],
            },
            {
                "id": "consistency", "title": "Interferometric data consistency", "phase": "Data consistency",
                "note": "Enabled by default: predicted and GT profiles are reprojected to per-track unit coherences through the per-pixel kz field over pixels whose GT power exceeds physics_floor=1e-3, giving per-pixel coherence-resynthesis and covariance-matching error maps; the phase_multilook=9 multilooked measured interferograms are then correlated against the synthesised coherence, with a conjugate-steering flipped variant as a kz sign check.",
                "inputs": ["cube", "cubegt", "kz", "meas"], "outputs": ["coh", "pha"],
                "lines": [
                    [{"id": "coh", "tex": r"E_{\gamma}", "role": "final"}, {"tex": "="}, {"tex": r"\tfrac{1}{T}\textstyle\sum_{i=0}^{T-1}\Big|\tfrac{\hat\gamma_i}{\hat p_0} - \tfrac{\gamma_i}{p_0}\Big|^2,\ \ \gamma_i = \textstyle\sum_n e^{\,j k_{z,i,n} x_n} C_n\,dx"}],
                    [{"id": "pha", "tex": r"\rho_{\varphi}", "role": "calculated"}, {"tex": "="}, {"tex": r"\Big|\tfrac{1}{|V|}\textstyle\sum_{V} \tilde\gamma\,\overline{(\hat\gamma_i/|\hat\gamma_i|)}\Big|,\quad p_0 = \max(\textstyle\sum_n C_n\,dx,\ 10^{-3})"}],
                ],
            },
        ]
        return {
            "key": "inference", "name": "Inference (Stitching)",
            "blurb": "The trained run is rebuilt strictly from its saved config, then every sliding-window patch is predicted, denormalised and hard-clamped, reconstructed to an elevation spectrum, blended into dense pred and GT cubes by weighted overlap-add for curves and highest-centrality selection for parameters, and scored by the full per-pixel, curve, per-elevation, matched-parameter, reduced-Capon-baseline and interferometric-consistency suite.",
            "nodes": nodes, "steps": steps,
        }

    def _tuning(self) -> dict:
        nodes = [
            {"id": "space",     "tex": r"\Theta",                 "role": "measured",     "kind": "set",    "shape": "-", "desc": "joint learning, regularisation and architecture search space", "sample": ["lr", "wd", "features", "..."]},
            {"id": "theta_lr",  "tex": r"\theta_{\mathrm{lr}}",   "role": "intermediate", "kind": "vector", "shape": "9", "desc": "4 group LRs, 4 group weight decays, dropout",                  "sample": ["2.6e-4", "8e-5", "...", "0.12"]},
            {"id": "theta_ar",  "tex": r"\theta_{\mathrm{arch}}", "role": "intermediate", "kind": "vector", "shape": "5", "desc": "features, bottleneck factor, activation, norm, upsample",      "sample": ["[64,128,256,512]", "2", "gelu", "group", "bilinear"]},
            {"id": "theta",     "tex": r"\theta",                 "role": "intermediate", "kind": "vector", "shape": "d", "desc": "sampled joint hyperparameter vector",                          "sample": ["3e-4", "1e-4", "64", "..."]},
            {"id": "ell",       "tex": r"\ell(\theta)",           "role": "calculated",   "kind": "scalar", "shape": "1", "desc": "TPE good-trial KDE density at theta",                          "sample": "4.7"},
            {"id": "g",         "tex": r"g(\theta)",              "role": "calculated",   "kind": "scalar", "shape": "1", "desc": "TPE bad-trial KDE density at theta",                           "sample": "0.9"},
            {"id": "acq",       "tex": r"a(\theta)",              "role": "calculated",   "kind": "scalar", "shape": "1", "desc": "TPE acquisition = good-over-bad density ratio",                "sample": "5.2"},
            {"id": "liar",      "tex": r"\tilde{y}",              "role": "intermediate", "kind": "scalar", "shape": "1", "desc": "constant-liar phantom objective for pending trials",           "sample": "2.9e-2"},
            {"id": "cfg",       "tex": r"\mathcal{C}_t",          "role": "intermediate", "kind": "set",    "shape": "-", "desc": "per-trial config with budget, patience and seed overrides",    "sample": ["E=30", "pat=8", "seed=42+t"]},
            {"id": "fobj",      "tex": r"f(\theta)",              "role": "calculated",   "kind": "scalar", "shape": "1", "desc": "trial objective: minimum validation loss over the budget",     "sample": "2.3e-2"},
            {"id": "med",       "tex": r"m^{(t)}",                "role": "calculated",   "kind": "scalar", "shape": "1", "desc": "running median of completed-trial losses at step t",           "sample": "3.4e-2"},
            {"id": "rem",       "tex": r"n_{\mathrm{rem}}",       "role": "intermediate", "kind": "scalar", "shape": "1", "desc": "remaining trials to dispatch across GPUs",                     "sample": "36"},
            {"id": "thetastar", "tex": r"\theta^{*}",             "role": "final",        "kind": "vector", "shape": "d", "desc": "best joint configuration, decoded and exported",               "sample": ["2.6e-4", "8e-5", "96", "..."]},
        ]
        steps = [
            {
                "id": "spacelr", "title": "Learning and regularisation space", "phase": "Search space",
                "note": "The lr block declares four per-group learning rates and four weight decays as log-uniform floats, plus dropout as a linear-uniform float.",
                "inputs": [], "outputs": ["theta_lr"],
                "lines": [
                    [{"id": "theta_lr", "tex": r"\eta_{\{\mathrm{enc,bot,dec,head}\}}", "role": "intermediate"}, {"tex": r"\sim \log\mathcal{U}(10^{-5},10^{-2}),\quad \lambda_{\{\cdot\}} \sim \log\mathcal{U}(10^{-6},10^{-1})"}],
                    [{"tex": r"p_{\mathrm{drop}} \sim \mathcal{U}(0,\,0.5)"}],
                ],
            },
            {
                "id": "spacearch", "title": "Architecture space", "phase": "Search space",
                "note": "Five categorical hyperparameters; the list-valued features channel is stored as an integer index and decoded back to its list on export.",
                "inputs": [], "outputs": ["theta_ar"],
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
                "note": "Each objective deep-copies the base trainer and dataset configs and overrides the epoch budget, scheduler horizon, early-stop patience, per-trial log dir and seed, keyed on the global trial number.",
                "inputs": ["theta"], "outputs": ["cfg"],
                "lines": [
                    [{"tex": r"\mathrm{setattr}("}, {"id": "cfg", "tex": r"\mathcal{C}_t", "role": "intermediate"}, {"tex": r", k, v)\quad \forall (k,v)\in"}, {"id": "theta", "tex": r"\theta", "role": "intermediate"}],
                    [{"tex": r"E_{\mathrm{trial}} = 30,\quad \mathrm{patience} = 8,\quad \mathrm{seed} = 42 + \mathrm{trial.number}"}],
                ],
            },
            {
                "id": "trial", "title": "Trial objective", "phase": "Trial & pruning",
                "note": "Each trial trains a full model on the fixed canonical split and returns the minimum validation loss over the epoch budget, capped earlier by early stopping or pruning.",
                "inputs": ["cfg"], "outputs": ["fobj"],
                "iterative": {"var": "fobj", "steps": 30, "unit": "epoch", "symbol": "f",
                              "trace": ["6.0e-2", "4.1e-2", "3.0e-2", "2.5e-2", "2.3e-2"]},
                "lines": [
                    [{"id": "fobj", "tex": r"f(\theta)", "role": "calculated"}, {"tex": "="}, {"tex": r"\min_{e \in \{1,\dots,E\}}\ \mathcal{L}^{(e)}_{\mathrm{val}}\!\big("}, {"id": "cfg", "tex": r"\mathcal{C}_t", "role": "intermediate"}, {"tex": r"\big),\quad E = 30"}],
                ],
            },
            {
                "id": "prune", "title": "Median pruning with gates", "phase": "Trial & pruning",
                "note": "A trial is pruned at step t once its best reported loss exceeds the running median of completed-trial losses, but only after 8 startup trials and once t clears the 8-step warmup. Pruned trials count toward the budget; stale running trials are marked FAIL on resume.",
                "inputs": ["fobj"], "outputs": ["med"],
                "lines": [
                    [{"id": "med", "tex": r"m^{(t)}", "role": "calculated"}, {"tex": "="}, {"tex": r"\operatorname{median}\big\{\,"}, {"id": "fobj", "tex": r"f^{(t)}_j", "role": "calculated"}, {"tex": r": j \in \mathrm{COMPLETE}\,\big\}"}],
                    [{"tex": r"\mathrm{prune} \iff \mathcal{L}^{(t)}_{\mathrm{val}} > "}, {"id": "med", "tex": r"m^{(t)}", "role": "calculated"}, {"tex": r"\ \wedge\ n_{\mathrm{complete}} \ge 8\ \wedge\ t \ge 8"}],
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
