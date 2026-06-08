class FlowLibrary:

    def collect(self) -> list:
        return [
            self._processing(),
            self._param_extraction(),
            self._dataset(),
            self._training(),
            self._inference(),
            self._tuning(),
        ]

    def _processing(self) -> dict:
        nodes = [
            {"id": "s0",      "tex": r"s_0",                   "role": "measured",     "kind": "matrix", "shape": "H x W",       "desc": "master (primary) SLC, complex",            "sample": [["0.8+0.2j", "0.6-0.4j", "1.1+0.0j"], ["0.5+0.5j", "0.9-0.1j", "0.7+0.3j"], ["1.0+0.2j", "0.4-0.6j", "0.8+0.1j"]]},
            {"id": "si",      "tex": r"s_i",                   "role": "measured",     "kind": "tensor", "shape": "N_s x H x W", "desc": "co-registered secondary SLC stack",        "sample": [["0.7+0.3j", "0.5-0.3j"], ["0.6+0.4j", "0.8-0.2j"]]},
            {"id": "phidem",  "tex": r"\phi_{\mathrm{DEM},i}", "role": "measured",     "kind": "tensor", "shape": "N_s x H x W", "desc": "DEM-predicted phase per pass (rad)",       "sample": [["0.12", "0.31"], ["0.27", "0.44"]]},
            {"id": "stilde",  "tex": r"\tilde{s}_i",           "role": "intermediate", "kind": "tensor", "shape": "N_s x H x W", "desc": "DEM-deramped secondary SLC",               "sample": [["0.6+0.1j", "0.5-0.1j"], ["0.7+0.2j", "0.6-0.0j"]]},
            {"id": "Ai",      "tex": r"A_i",                   "role": "intermediate", "kind": "vector", "shape": "N_s",         "desc": "clipped secondary amplitude weight",       "sample": ["0.71", "0.93", "1.25", "0.58"]},
            {"id": "phii",    "tex": r"\phi_i",                "role": "calculated",   "kind": "tensor", "shape": "N_s x H x W", "desc": "amplitude-weighted complex interferogram", "sample": [["0.51+0.20j", "0.44-0.12j"], ["0.63+0.18j", "0.55-0.05j"]]},
            {"id": "Tm",      "tex": r"T_m",                   "role": "calculated",   "kind": "tensor", "shape": "H x W_m x R", "desc": "beamformed tomogram of subsection m",      "sample": [["0.2", "0.9", "0.3"], ["0.1", "0.7", "0.8"], ["0.0", "0.4", "1.0"]]},
            {"id": "Tcomb",   "tex": r"T_{\mathrm{comb}}",     "role": "final",        "kind": "tensor", "shape": "H x W x R",   "desc": "combined tomogram, model input",           "sample": [["0.2", "0.9", "0.3"], ["0.1", "0.7", "0.8"], ["0.0", "0.4", "1.0"]]},
        ]
        steps = [
            {
                "id"      : "deramp",
                "title"   : "DEM-phase deramping",
                "note"    : "Removing the DEM-predicted phase decorrelates the interferogram from terrain topography, leaving sub-resolution elevation structure.",
                "inputs"  : ["si", "phidem"],
                "outputs" : ["stilde"],
                "lines"   : [
                    [
                        {"id": "stilde", "tex": r"\tilde{s}_i",                          "role": "intermediate"},
                        {"tex": "="},
                        {"id": "si",     "tex": r"s_i",                                  "role": "measured"},
                        {"tex": r"\cdot"},
                        {"tex": r"\exp\!\left(j\,\phi_{\mathrm{DEM},i}\right)"},
                    ],
                ],
            },
            {
                "id"      : "interf",
                "title"   : "Amplitude-weighted interferogram",
                "note"    : "Unit-phasor normalisation removes inter-pass amplitude variation while preserving coherence; the clipped secondary amplitude A_i = min(|s_i|, 1.25) is reintroduced as a signal-to-noise proxy.",
                "inputs"  : ["s0", "stilde", "si"],
                "outputs" : ["Ai", "phii"],
                "lines"   : [
                    [
                        {"id": "phii", "tex": r"\phi_i",  "role": "calculated"},
                        {"tex": "="},
                        {"id": "Ai",   "tex": r"A_i",     "role": "intermediate"},
                        {"tex": r"\cdot"},
                        {"id": "s0",   "tex": r"s_0",     "role": "measured"},
                        {"tex": r"\;\dfrac{\overline{\tilde{s}_i}}{\left|s_0\,\overline{\tilde{s}_i}\right|}"},
                    ],
                ],
            },
            {
                "id"      : "beamform",
                "title"   : "Per-worker beamforming (PyRat)",
                "note"    : "Each azimuth subsection is beamformed in an isolated PyRat subprocess; the minimum-variance (Capon) estimator turns the interferometric stack into an elevation power cube.",
                "inputs"  : ["phii"],
                "outputs" : ["Tm"],
                "lines"   : [
                    [
                        {"id": "Tm", "tex": r"T_m(\xi)", "role": "calculated"},
                        {"tex": r"\;\propto\;"},
                        {"tex": r"\dfrac{1}{\mathbf{a}^{H}(\xi)\,\hat{\mathbf{R}}^{-1}\,\mathbf{a}(\xi)}"},
                    ],
                ],
            },
            {
                "id"      : "concat",
                "title"   : "Subsection concatenation",
                "note"    : "Per-worker HDF5 outputs are reassembled along azimuth into the full beamformed tomogram that downstream stages consume.",
                "inputs"  : ["Tm"],
                "outputs" : ["Tcomb"],
                "lines"   : [
                    [
                        {"id": "Tcomb", "tex": r"T_{\mathrm{comb}}", "role": "final"},
                        {"tex": "="},
                        {"tex": r"\mathrm{concat}\!\left[\,"},
                        {"id": "Tm", "tex": r"T_0, \dots, T_{M-1}", "role": "calculated"},
                        {"tex": r"\,\right]_{\mathrm{axis}=1}"},
                    ],
                ],
            },
        ]
        return {
            "key"   : "processing",
            "name"  : "Processing (Pre-process)",
            "blurb" : "From F-SAR SLC passes to the beamformed tomogram. Deramp against the DEM, form the amplitude-weighted interferogram, beamform per azimuth subsection, then reassemble.",
            "nodes" : nodes,
            "steps" : steps,
        }

    def _param_extraction(self) -> dict:
        nodes = [
            {"id": "T",       "tex": r"T",                "role": "measured",     "kind": "tensor", "shape": "H x A x R", "desc": "beamformed tomogram from processing",      "sample": [["0.2", "0.9", "0.3"], ["0.1", "0.7", "0.8"], ["0.0", "0.4", "1.0"]]},
            {"id": "P",       "tex": r"P_h",              "role": "intermediate", "kind": "vector", "shape": "H",         "desc": "per-pixel elevation profile (magnitude)",  "sample": ["0.05", "0.62", "0.31", "0.88", "0.12", "..."]},
            {"id": "scale",   "tex": r"s",                "role": "intermediate", "kind": "scalar", "shape": "1",         "desc": "per-pixel scale (profile maximum)",        "sample": "0.88"},
            {"id": "gtilde",  "tex": r"\tilde{\gamma}_h", "role": "intermediate", "kind": "vector", "shape": "H",         "desc": "normalised, truncated profile",            "sample": ["0.06", "0.70", "0.35", "1.00", "0.14", "..."]},
            {"id": "Psmooth", "tex": r"\tilde{P}_h",      "role": "intermediate", "kind": "vector", "shape": "H",         "desc": "width-5 smoothed profile",                 "sample": ["0.11", "0.58", "0.40", "0.81", "0.19", "..."]},
            {"id": "peaks",   "tex": r"\mathcal{P}",      "role": "intermediate", "kind": "set",    "shape": "P",         "desc": "prominence-gated peak positions (bins)",   "sample": ["34", "97", "151"]},
            {"id": "mu0",     "tex": r"\mu",              "role": "intermediate", "kind": "vector", "shape": "K",         "desc": "component means / elevations (m)",         "sample": ["12.4", "31.8", "47.0"]},
            {"id": "a0",      "tex": r"a",                "role": "intermediate", "kind": "vector", "shape": "K",         "desc": "component amplitudes (normalised)",        "sample": ["0.81", "0.55", "0.30"]},
            {"id": "sig0",    "tex": r"\sigma_0",         "role": "intermediate", "kind": "scalar", "shape": "1",         "desc": "shared initial width guess (m)",           "sample": "2.10"},
            {"id": "loss",    "tex": r"\mathcal{L}",      "role": "intermediate", "kind": "scalar", "shape": "1",         "desc": "per-pixel mean-squared fit error",         "sample": "3.4e-2"},
            {"id": "sigstar", "tex": r"\sigma",           "role": "calculated",   "kind": "vector", "shape": "K",         "desc": "fitted component widths (m)",              "sample": ["1.84", "2.55", "3.10"]},
            {"id": "penK",    "tex": r"\mathrm{pen}_K",   "role": "intermediate", "kind": "vector", "shape": "K_max",     "desc": "penalised score per model order K",        "sample": ["0.21", "0.09", "0.10", "0.14", "0.20"]},
            {"id": "Kstar",   "tex": r"K^{*}",            "role": "calculated",   "kind": "scalar", "shape": "1",         "desc": "selected number of active components",     "sample": "2"},
            {"id": "theta",   "tex": r"\theta",           "role": "final",        "kind": "vector", "shape": "3K",        "desc": "ordered (a, mu, sigma) supervised target", "sample": ["a_1", "mu_1", "sig_1", "a_2", "..."]},
            {"id": "r2",      "tex": r"R^2",              "role": "final",        "kind": "matrix", "shape": "A x R",     "desc": "per-pixel fit-quality map",                "sample": [["0.97", "0.91"], ["0.88", "0.99"]]},
        ]
        steps = [
            {
                "id"      : "condition",
                "title"   : "Profile conditioning",
                "phase"   : "0 - Conditioning",
                "note"    : "Each elevation profile is the tomogram magnitude; samples below a relative floor are zeroed, the upper axis is truncated, and the profile is normalised by its own maximum so the loss is independent of absolute backscatter.",
                "inputs"  : ["T"],
                "outputs" : ["P", "scale", "gtilde"],
                "lines"   : [
                    [
                        {"id": "P", "tex": r"P_h", "role": "intermediate"},
                        {"tex": "="},
                        {"id": "T", "tex": r"\left|T_h\right|", "role": "measured"},
                    ],
                    [
                        {"id": "scale", "tex": r"s", "role": "intermediate"},
                        {"tex": "="},
                        {"tex": r"\max_h"},
                        {"id": "P", "tex": r"P_h", "role": "intermediate"},
                    ],
                    [
                        {"id": "gtilde", "tex": r"\tilde{\gamma}_h", "role": "intermediate"},
                        {"tex": "="},
                        {"tex": r"\dfrac{1}{"},
                        {"id": "scale", "tex": r"s", "role": "intermediate"},
                        {"tex": r"}\;"},
                        {"id": "P", "tex": r"P_h", "role": "intermediate"},
                        {"tex": r"\cdot\mathbf{1}\!\left[P_h > t_f \max_h P_h\right]"},
                    ],
                ],
            },
            {
                "id"      : "smooth",
                "title"   : "Profile smoothing",
                "phase"   : "1 - Init (CPU)",
                "note"    : "A width-5 uniform filter smooths the raw profile before peak detection.",
                "inputs"  : ["P"],
                "outputs" : ["Psmooth"],
                "lines"   : [
                    [
                        {"id": "Psmooth", "tex": r"\tilde{P}_h", "role": "intermediate"},
                        {"tex": "="},
                        {"tex": r"\dfrac{1}{5}\sum_{j=-2}^{2}"},
                        {"id": "P", "tex": r"P_{h+j}", "role": "intermediate"},
                    ],
                ],
            },
            {
                "id"      : "peakfind",
                "title"   : "Prominence-gated peak detection",
                "phase"   : "1 - Init (CPU)",
                "note"    : "find_peaks keeps a peak only if its topographic prominence exceeds a fraction of the profile maximum and it is far enough from every other accepted peak.",
                "inputs"  : ["Psmooth"],
                "outputs" : ["peaks"],
                "lines"   : [
                    [
                        {"id": "peaks", "tex": r"\mathcal{P}", "role": "intermediate"},
                        {"tex": "="},
                        {"tex": r"\left\{\,p : \mathrm{prom}(p) \geq p_{\mathrm{frac}}\,\max_h"},
                        {"id": "Psmooth", "tex": r"\tilde{P}_h", "role": "intermediate"},
                        {"tex": r"\,\right\}"},
                    ],
                ],
            },
            {
                "id"      : "select",
                "title"   : "Top-K selection and width guess",
                "phase"   : "1 - Init (CPU)",
                "note"    : "The K most prominent peaks seed the means and amplitudes; residual maxima fill any remaining slots. Every slot starts from the same width guess.",
                "inputs"  : ["peaks", "Psmooth", "scale"],
                "outputs" : ["mu0", "a0", "sig0"],
                "lines"   : [
                    [
                        {"id": "mu0", "tex": r"\mu", "role": "intermediate"},
                        {"tex": r"\leftarrow"},
                        {"tex": r"\xi\!\left[\operatorname{top}K_{\mathrm{prom}}"},
                        {"id": "peaks", "tex": r"\mathcal{P}", "role": "intermediate"},
                        {"tex": r"\right]"},
                    ],
                    [
                        {"id": "sig0", "tex": r"\sigma_0", "role": "intermediate"},
                        {"tex": "="},
                        {"tex": r"\max\!\left(2\Delta\xi,\ \tfrac{x_{\max}-x_{\min}}{8K}\right)"},
                    ],
                ],
            },
            {
                "id"      : "objective",
                "title"   : "Sigma-fit objective",
                "phase"   : "2 - Sigma fit (GPU)",
                "note"    : "Amplitudes and means are frozen from Phase 1; only the K widths are optimised. Vectorised over pixels with jax.vmap, differentiated with jax.value_and_grad.",
                "inputs"  : ["gtilde", "mu0", "a0", "sig0"],
                "outputs" : ["loss"],
                "lines"   : [
                    [
                        {"id": "loss", "tex": r"\mathcal{L}(\sigma)", "role": "intermediate"},
                        {"tex": "="},
                        {"tex": r"\dfrac{1}{H}\sum_{h}\left(\sum_{k}"},
                        {"id": "a0",  "tex": r"a_k", "role": "intermediate"},
                        {"tex": r"e^{-\frac{(x_h-\mu_k)^2}{2\sigma_k^2}} -"},
                        {"id": "gtilde", "tex": r"\tilde{\gamma}_h", "role": "intermediate"},
                        {"tex": r"\right)^{2}"},
                    ],
                ],
            },
            {
                "id"      : "adam",
                "title"   : "Adam width optimisation",
                "phase"   : "2 - Sigma fit (GPU)",
                "note"    : "Bias-corrected Adam runs as a jax.lax.scan over T = 3000 steps compiled into one XLA computation; the sigmas are clamped to one elevation bin to half the elevation span after every update.",
                "inputs"  : ["loss", "sig0"],
                "outputs" : ["sigstar"],
                "iterative": {"var": "sigstar", "steps": 3000, "unit": "t", "symbol": "σ",
                              "trace": ["2.10", "2.04", "1.97", "1.91", "1.87", "1.85", "1.84"]},
                "lines"   : [
                    [
                        {"id": "sigstar", "tex": r"\sigma_{t}", "role": "calculated"},
                        {"tex": "="},
                        {"id": "sigstar", "tex": r"\sigma_{t-1}", "role": "calculated"},
                        {"tex": r"\;-\;\eta\,\dfrac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}"},
                    ],
                ],
            },
            {
                "id"      : "bestk",
                "title"   : "Penalised best-K selection",
                "phase"   : "3 - Best-K",
                "note"    : "Each model order 1..K_max is scored with a complexity penalty so the full component budget is spent only when the profile is genuinely multi-layered.",
                "inputs"  : ["gtilde", "sigstar", "a0"],
                "outputs" : ["penK", "Kstar"],
                "lines"   : [
                    [
                        {"id": "penK", "tex": r"\mathrm{pen}_K", "role": "intermediate"},
                        {"tex": "="},
                        {"tex": r"\mathrm{MSE}_K + \lambda_K\,K\,\bar{a}_K"},
                    ],
                    [
                        {"id": "Kstar", "tex": r"K^{*}", "role": "calculated"},
                        {"tex": "="},
                        {"tex": r"\operatorname*{arg\,min}_{K}"},
                        {"id": "penK", "tex": r"\mathrm{pen}_K", "role": "intermediate"},
                    ],
                ],
            },
            {
                "id"      : "assemble",
                "title"   : "Rescale, order, emit target",
                "phase"   : "3 - Best-K",
                "note"    : "The winning K's parameters return to the raw amplitude scale, are sorted by ascending mean elevation with inactive slots pushed last, and written into the interleaved 3K supervised target.",
                "inputs"  : ["Kstar", "sigstar", "mu0", "a0", "scale"],
                "outputs" : ["theta"],
                "lines"   : [
                    [
                        {"id": "theta", "tex": r"\theta", "role": "final"},
                        {"tex": "="},
                        {"tex": r"\Big[\,"},
                        {"id": "a0",  "tex": r"a", "role": "intermediate"},
                        {"tex": ","},
                        {"id": "mu0", "tex": r"\mu", "role": "intermediate"},
                        {"tex": ","},
                        {"id": "sigstar", "tex": r"\sigma", "role": "calculated"},
                        {"tex": r"\,\Big]_{\pi},\quad \pi=\operatorname{argsort}_k \mu_k"},
                    ],
                ],
            },
            {
                "id"      : "quality",
                "title"   : "Fit-quality map",
                "phase"   : "3 - Best-K",
                "note"    : "Per-pixel coefficient of determination over the elevation axis on the raw amplitude scale, written as a spatial map alongside the parameter array.",
                "inputs"  : ["theta", "T"],
                "outputs" : ["r2"],
                "lines"   : [
                    [
                        {"id": "r2", "tex": r"R^2", "role": "final"},
                        {"tex": "="},
                        {"tex": r"1 - \dfrac{\sum_h (\hat{\gamma}_h - \gamma_h)^2}{\sum_h (\gamma_h - \bar{\gamma})^2}"},
                    ],
                ],
            },
        ]
        return {
            "key"   : "param",
            "name"  : "Parameter Extraction (Fitting)",
            "blurb" : "Per-pixel three-phase fit. Condition each elevation profile, initialise a K-Gaussian mixture from prominence peaks on the CPU, fit the widths with Adam on the GPU, then select the penalised best model order.",
            "nodes" : nodes,
            "steps" : steps,
        }

    def _dataset(self) -> dict:
        nodes = [
            {"id": "X",      "tex": r"\mathbf{X}",        "role": "measured",     "kind": "tensor", "shape": "(1+2N_s) x Az x Rg", "desc": "stacked complex input: primary, secondaries, interferograms", "sample": [["0.8+0.2j", "0.6-0.4j"], ["0.5+0.5j", "0.9-0.1j"]]},
            {"id": "patch",  "tex": r"\mathbf{p}",        "role": "intermediate", "kind": "tensor", "shape": "C x P x P",          "desc": "one sliding-window patch",                                   "sample": [["0.8+0.2j", "0.6-0.4j", "1.1+0.0j"], ["0.5+0.5j", "0.9-0.1j", "0.7+0.3j"], ["1.0+0.2j", "0.4-0.6j", "0.8+0.1j"]]},
            {"id": "rep",    "tex": r"\mathbf{r}",        "role": "intermediate", "kind": "vector", "shape": "c",                  "desc": "complex-to-real channel set of a pass",                      "sample": ["|s|", "ang", "re", "im"]},
            {"id": "x",      "tex": r"\mathbf{x}",        "role": "calculated",   "kind": "tensor", "shape": "C_in x P x P",       "desc": "assembled real-valued input tensor",                         "sample": [["0.91", "0.62", "1.10"], ["0.55", "0.88", "0.71"]]},
            {"id": "xaug",   "tex": r"\mathbf{x}'",       "role": "intermediate", "kind": "tensor", "shape": "C_in x P x P",       "desc": "augmented patch (flip, rotation, noise)",                    "sample": [["0.93", "0.60", "1.08"], ["0.57", "0.86", "0.70"]]},
            {"id": "stats",  "tex": r"(\mu_c, s_c)",      "role": "intermediate", "kind": "vector", "shape": "C_in",               "desc": "per-channel location and scale, fitted on the train split",  "sample": ["0.12", "0.94", "..."]},
            {"id": "xhat",   "tex": r"\hat{\mathbf{x}}",  "role": "final",        "kind": "tensor", "shape": "C_in x P x P",       "desc": "normalised network input",                                   "sample": [["0.41", "-0.20", "0.88"], ["-0.33", "0.27", "0.05"]]},
            {"id": "y",      "tex": r"\mathbf{y}",        "role": "final",        "kind": "vector", "shape": "C_out",              "desc": "selected Gaussian-parameter target",                         "sample": ["a_1", "mu_1", "sig_1", "..."]},
        ]
        steps = [
            {
                "id": "stack", "title": "Stacked complex input", "phase": "Crop & split",
                "note": "Primary, secondaries and interferograms are written into one pre-allocated complex buffer indexed by pass.",
                "inputs": [], "outputs": ["X"],
                "lines": [[{"id": "X", "tex": r"\mathbf{X}", "role": "measured"}, {"tex": "="}, {"tex": r"\big[\,\mathbf{s}_0 \;;\; S \;;\; I\,\big]"}]],
            },
            {
                "id": "patchgrid", "title": "Sliding-window tiling", "phase": "Patch extraction",
                "note": "The split region is tiled by a strided sliding window; each patch is a slice of the memory-mapped crop, padded by reflection at the borders.",
                "inputs": ["X"], "outputs": ["patch"],
                "lines": [
                    [{"tex": r"N_p = \left\lceil\tfrac{H-P_H}{s}\right\rceil\!+\!1 \;\cdot\; \left\lceil\tfrac{W-P_W}{s}\right\rceil\!+\!1"}],
                    [{"id": "patch", "tex": r"\mathbf{p}", "role": "intermediate"}, {"tex": "="}, {"id": "X", "tex": r"\mathbf{X}", "role": "measured"}, {"tex": r"\big[:,\ v_0{:}v_0{+}P_H,\ h_0{:}h_0{+}P_W\big]"}],
                ],
            },
            {
                "id": "represent", "title": "Complex to real", "phase": "Patch extraction",
                "note": "Each complex pass is converted to real channels; the default keeps magnitude for SLCs and phase for interferograms. Zero magnitude is replaced by one for the normalised components.",
                "inputs": ["patch"], "outputs": ["rep"],
                "lines": [[{"id": "rep", "tex": r"\mathbf{r}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\Big(\,|"}, {"id": "patch", "tex": r"\mathbf{p}", "role": "intermediate"}, {"tex": r"|,\ \angle \mathbf{p},\ \tfrac{p_r}{|\mathbf{p}|},\ \tfrac{p_i}{|\mathbf{p}|}\,\Big)"}]],
            },
            {
                "id": "assemble_in", "title": "Input tensor assembly", "phase": "Patch extraction",
                "note": "Per patch, every enabled source is converted by its representation and concatenated along the channel axis, with the optional DEM channel last.",
                "inputs": ["rep"], "outputs": ["x"],
                "lines": [
                    [{"id": "x", "tex": r"\mathbf{x}", "role": "calculated"}, {"tex": "="}, {"tex": r"\big[\;"}, {"id": "rep", "tex": r"\mathbf{r}_0", "role": "intermediate"}, {"tex": r"\mid \mathbf{r}_S \mid \mathbf{r}_I \mid \mathbf{d}\;\big]"}],
                    [{"tex": r"C_{\mathrm{in}} = c_{\mathrm{prim}} + N_s\,(c_{\mathrm{sec}}+c_{\mathrm{ifg}}) + c_{\mathrm{dem}}"}],
                ],
            },
            {
                "id": "augment", "title": "Augmentation", "phase": "Augmentation",
                "note": "Flips and optional rotations are applied jointly to input and target; additive Gaussian noise perturbs the input only, so the regression target stays clean.",
                "inputs": ["x"], "outputs": ["xaug"],
                "lines": [[{"id": "xaug", "tex": r"\mathbf{x}'", "role": "intermediate"}, {"tex": "="}, {"id": "x", "tex": r"\mathbf{x}", "role": "calculated"}, {"tex": r"+\ \varepsilon,\quad \varepsilon \sim \mathcal{N}\!\left(0,\ \sigma_{\mathrm{noise}}^2 \mathbf{I}\right)"}]],
            },
            {
                "id": "fitstats", "title": "Fit normalisation statistics", "phase": "Normalization",
                "note": "Per-channel statistics are fitted on the training split only, using percentile min-max with an optional log1p compression; the scale is floored at 1e-8.",
                "inputs": ["xaug"], "outputs": ["stats"],
                "lines": [[{"id": "stats", "tex": r"(\mu_c, s_c)", "role": "intermediate"}, {"tex": "="}, {"tex": r"\Big(P_{0.1}\,f(\mathbf{x}'),\ \ P_{99.9}\,f("}, {"id": "xaug", "tex": r"\mathbf{x}'", "role": "intermediate"}, {"tex": r") - P_{0.1}\Big)"}]],
            },
            {
                "id": "normalise", "title": "Forward normalisation", "phase": "Normalization",
                "note": "The fitted statistics are applied identically to every split, yielding the dimensionless tensor the network consumes.",
                "inputs": ["xaug", "stats"], "outputs": ["xhat"],
                "lines": [[{"id": "xhat", "tex": r"\hat{\mathbf{x}}", "role": "final"}, {"tex": "="}, {"tex": r"\dfrac{f(\mathbf{x}') - "}, {"id": "stats", "tex": r"\mu_c", "role": "intermediate"}, {"tex": r"}{"}, {"id": "stats", "tex": r"s_c", "role": "intermediate"}, {"tex": r"}"}]],
            },
            {
                "id": "target", "title": "Target selection", "phase": "Normalization",
                "note": "The configured subset of Gaussian parameters is selected from the interleaved ground-truth layout and paired with the input as the supervised target.",
                "inputs": [], "outputs": ["y"],
                "lines": [[{"id": "y", "tex": r"\mathbf{y}", "role": "final"}, {"tex": "="}, {"tex": r"\big[\,\theta_{c_1}, \dots, \theta_{c_{C_{\mathrm{out}}}}\big],\quad C_{\mathrm{out}} = n_g\,p_g"}]],
            },
        ]
        return {
            "key": "dataset", "name": "Dataset (Loaders)",
            "blurb": "Processed artifacts become PyTorch tensors: tile the crop into patches, convert complex passes to real channels, assemble and augment the input, then apply train-fitted per-channel normalisation.",
            "nodes": nodes, "steps": steps,
        }

    def _training(self) -> dict:
        nodes = [
            {"id": "xhat",   "tex": r"\hat{\mathbf{x}}",        "role": "measured",     "kind": "tensor", "shape": "B x C_in x P x P", "desc": "normalised input batch from the loader",     "sample": [["0.41", "-0.20"], ["-0.33", "0.27"]]},
            {"id": "th",     "tex": r"\hat{\theta}",            "role": "calculated",   "kind": "vector", "shape": "B x 3K x P x P",   "desc": "predicted Gaussian parameters",              "sample": ["0.78", "12.1", "1.9", "..."]},
            {"id": "yhat",   "tex": r"\hat{y}",                 "role": "intermediate", "kind": "vector", "shape": "B x N x P x P",    "desc": "reconstructed elevation curve",              "sample": ["0.05", "0.71", "0.33", "..."]},
            {"id": "y",      "tex": r"y",                       "role": "measured",     "kind": "vector", "shape": "B x N x P x P",    "desc": "ground-truth (experimental) curve",          "sample": ["0.06", "0.70", "0.35", "..."]},
            {"id": "loss",   "tex": r"\mathcal{L}",             "role": "calculated",   "kind": "scalar", "shape": "1",                "desc": "normalised weighted total loss",             "sample": "4.1e-2"},
            {"id": "grad",   "tex": r"\mathbf{g}",              "role": "intermediate", "kind": "vector", "shape": "|theta|",          "desc": "clipped parameter gradient",                 "sample": ["1.2e-2", "-4e-3", "..."]},
            {"id": "w",      "tex": r"\theta",                  "role": "final",        "kind": "vector", "shape": "|theta|",          "desc": "trained model weights (best epoch checkpointed)", "sample": ["0.31", "-0.08", "..."]},
        ]
        steps = [
            {
                "id": "forward", "title": "Forward pass", "phase": "Reconstruction",
                "note": "The network maps the normalised input patch to per-pixel Gaussian parameters in one pass.",
                "inputs": ["xhat"], "outputs": ["th"],
                "lines": [[{"id": "th", "tex": r"\hat{\theta}", "role": "calculated"}, {"tex": "="}, {"tex": r"f_{\theta}\!\big("}, {"id": "xhat", "tex": r"\hat{\mathbf{x}}", "role": "measured"}, {"tex": r"\big)"}]],
            },
            {
                "id": "clamp", "title": "Physical parameter bounds", "phase": "Reconstruction",
                "note": "Denormalised predictions are clamped to physical bounds with a straight-through leaky clamp (slope 0.01) so gradients survive saturation, then renormalised.",
                "inputs": ["th"], "outputs": ["th"],
                "lines": [[{"id": "th", "tex": r"\hat{\theta}", "role": "calculated"}, {"tex": r"\in\ \big[\,0,a_{\max}\big]\times\big[x_{\min},x_{\max}\big]\times\big[\tfrac{\Delta x}{2},\tfrac{x_{\max}-x_{\min}}{2}\big]"}]],
            },
            {
                "id": "reconstruct", "title": "Curve reconstruction", "phase": "Loss",
                "note": "Predicted parameters are evaluated on the elevation axis; the residual against the ground-truth curve drives the curve-space loss terms.",
                "inputs": ["th"], "outputs": ["yhat"],
                "lines": [[{"id": "yhat", "tex": r"\hat{y}(x_n)", "role": "intermediate"}, {"tex": "="}, {"tex": r"\sum_{k}"}, {"id": "th", "tex": r"\hat{a}_k", "role": "calculated"}, {"tex": r"\exp\!\Big(-\tfrac{(x_n-\hat{\mu}_k)^2}{2\hat{\sigma}_k^2}\Big)"}]],
            },
            {
                "id": "loss", "title": "Composite loss", "phase": "Loss",
                "note": "A weighted sum of enabled curve-space and parameter-space terms, each scaled by a fixed empirical normaliser, divided by the total effective weight.",
                "inputs": ["yhat", "y"], "outputs": ["loss"],
                "lines": [
                    [{"tex": r"e = "}, {"id": "yhat", "tex": r"\hat{y}", "role": "intermediate"}, {"tex": r"\ -\ "}, {"id": "y", "tex": r"y", "role": "measured"}],
                    [{"id": "loss", "tex": r"\mathcal{L}", "role": "calculated"}, {"tex": "="}, {"tex": r"\dfrac{\sum_j \mathrm{eff}_j\,\ell_j}{\sum_j \mathrm{eff}_j}"}],
                ],
            },
            {
                "id": "gradclip", "title": "Gradient and clipping", "phase": "Optimiser step",
                "note": "After unscaling, all gradients are scaled by a common factor so the global norm never exceeds the threshold (fixed or adaptive).",
                "inputs": ["loss"], "outputs": ["grad"],
                "lines": [[{"id": "grad", "tex": r"\mathbf{g}", "role": "intermediate"}, {"tex": r"\leftarrow \mathbf{g}\cdot\min\!\Big(1,\ \tfrac{\tau}{\lVert \mathbf{g}\rVert_2}\Big),\quad \mathbf{g}=\nabla_{\theta}"}, {"id": "loss", "tex": r"\mathcal{L}", "role": "calculated"}]],
            },
            {
                "id": "adamw", "title": "AdamW update", "phase": "Optimiser step",
                "note": "Bias-corrected adaptive moments with decoupled weight decay; the epoch loop drives the training loss down over many steps.",
                "inputs": ["grad"], "outputs": ["w"],
                "iterative": {"var": "loss", "steps": 100, "unit": "epoch", "symbol": "L",
                              "trace": ["4.1e-2", "2.7e-2", "1.9e-2", "1.4e-2", "1.1e-2", "9.6e-3"]},
                "lines": [[{"id": "w", "tex": r"\theta_{t+1}", "role": "intermediate"}, {"tex": "="}, {"id": "w", "tex": r"\theta_t", "role": "intermediate"}, {"tex": r"\ -\ \eta\Big(\tfrac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_t\Big)"}]],
            },
            {
                "id": "checkpoint", "title": "Validation and checkpoint", "phase": "Eval and checkpoint",
                "note": "The model is evaluated on the validation split; the best epoch is checkpointed and early stopping reverts to it when validation stagnates.",
                "inputs": ["w"], "outputs": ["w"],
                "lines": [[{"id": "w", "tex": r"\theta^{\star}", "role": "final"}, {"tex": "="}, {"tex": r"\arg\min_{t}\ \mathcal{L}_{\mathrm{val}}(\theta_t)"}]],
            },
        ]
        return {
            "key": "training", "name": "Training (Supervised)",
            "blurb": "One forward pass predicts all Gaussian parameters; the composite loss over curve and parameter space is backpropagated through AdamW with warmup, scheduling, and early stopping.",
            "nodes": nodes, "steps": steps,
        }

    def _inference(self) -> dict:
        nodes = [
            {"id": "xhat",  "tex": r"\hat{\mathbf{x}}",   "role": "measured",     "kind": "tensor", "shape": "C_in x P x P", "desc": "normalised patch over the sliding window",  "sample": [["0.41", "-0.20"], ["-0.33", "0.27"]]},
            {"id": "th",    "tex": r"\hat{\theta}",       "role": "calculated",   "kind": "vector", "shape": "3K x P x P",   "desc": "predicted parameters per patch",            "sample": ["0.78", "12.1", "1.9", "..."]},
            {"id": "p",     "tex": r"\mathbf{p}",         "role": "intermediate", "kind": "tensor", "shape": "N x P x P",    "desc": "reconstructed patch spectrum",              "sample": [["0.05", "0.71"], ["0.33", "0.88"]]},
            {"id": "win",   "tex": r"w",                  "role": "intermediate", "kind": "matrix", "shape": "P x P",        "desc": "2D overlap-add weighting window (Hann)",    "sample": [["0.10", "0.45"], ["0.45", "1.00"]]},
            {"id": "acc",   "tex": r"(A, W)",             "role": "intermediate", "kind": "tensor", "shape": "C x H x Rg",   "desc": "value and weight accumulators",             "sample": [["1.2", "3.4"], ["0.8", "2.1"]]},
            {"id": "cube",  "tex": r"\hat{C}",            "role": "final",        "kind": "tensor", "shape": "N x Az x Rg",  "desc": "stitched prediction cube",                  "sample": [["0.2", "0.9", "0.3"], ["0.1", "0.7", "0.8"]]},
            {"id": "r2",    "tex": r"R^2",                "role": "final",        "kind": "scalar", "shape": "1",            "desc": "overall reconstruction quality",            "sample": "0.94"},
        ]
        steps = [
            {
                "id": "predict", "title": "Windowed prediction", "phase": "Windowed predict",
                "note": "The trained model predicts parameters for every patch of the sliding-window grid over the full scene.",
                "inputs": ["xhat"], "outputs": ["th"],
                "lines": [[{"id": "th", "tex": r"\hat{\theta}", "role": "calculated"}, {"tex": "="}, {"tex": r"f_{\theta}\!\big("}, {"id": "xhat", "tex": r"\hat{\mathbf{x}}", "role": "measured"}, {"tex": r"\big)"}]],
            },
            {
                "id": "recon", "title": "Curve reconstruction", "phase": "Windowed predict",
                "note": "Each patch's parameters are evaluated on the elevation axis to a patch spectrum; amplitudes are rectified at zero.",
                "inputs": ["th"], "outputs": ["p"],
                "lines": [[{"id": "p", "tex": r"\mathbf{p}", "role": "intermediate"}, {"tex": "="}, {"tex": r"\sum_{k}"}, {"id": "th", "tex": r"\hat{a}_k", "role": "calculated"}, {"tex": r"\exp\!\Big(-\tfrac{(x_n-\hat{\mu}_k)^2}{2\hat{\sigma}_k^2}\Big)"}]],
            },
            {
                "id": "window", "title": "Overlap window", "phase": "Cube stitch",
                "note": "A separable Hann window de-emphasises patch borders so overlapping predictions blend without seams.",
                "inputs": [], "outputs": ["win"],
                "lines": [
                    [{"tex": r"w_v[i] = 0.5 - 0.5\cos\!\Big(\tfrac{2\pi(i+0.5)}{P}\Big)"}],
                    [{"id": "win", "tex": r"w", "role": "intermediate"}, {"tex": r"=\ w_v \otimes w_h"}],
                ],
            },
            {
                "id": "ola", "title": "Overlap-add accumulation", "phase": "Cube stitch",
                "note": "Every windowed patch output is scattered into a value accumulator at its grid position, alongside a matching weight accumulator.",
                "inputs": ["p", "win"], "outputs": ["acc"],
                "lines": [[{"id": "acc", "tex": r"(A, W)", "role": "intermediate"}, {"tex": r"\mathrel{+}=\ \big("}, {"id": "p", "tex": r"\mathbf{p}", "role": "intermediate"}, {"tex": r"\cdot"}, {"id": "win", "tex": r"w", "role": "intermediate"}, {"tex": r",\ \ w\,\big)"}]],
            },
            {
                "id": "finalise", "title": "Cube finalisation", "phase": "Cube stitch",
                "note": "Accumulated values are divided by accumulated weights and trimmed of grid padding; uncovered positions divide by one and yield zero.",
                "inputs": ["acc"], "outputs": ["cube"],
                "lines": [[{"id": "cube", "tex": r"\hat{C}", "role": "final"}, {"tex": "="}, {"tex": r"A / W"}, {"tex": r"\ \big|_{\mathrm{trim}}\,,\quad ("}, {"id": "acc", "tex": r"A, W", "role": "intermediate"}, {"tex": r")"}]],
            },
            {
                "id": "metrics", "title": "Overall metrics", "phase": "Metrics",
                "note": "The stitched cubes are scored at physical scale; the overall coefficient of determination summarises reconstruction quality alongside the per-pixel and per-elevation maps.",
                "inputs": ["cube"], "outputs": ["r2"],
                "lines": [[{"id": "r2", "tex": r"R^2", "role": "final"}, {"tex": "="}, {"tex": r"1 - \dfrac{\sum (\hat{C} - C)^2}{\sum (C - \bar{C})^2}"}]],
            },
        ]
        return {
            "key": "inference", "name": "Inference (Stitching)",
            "blurb": "Sliding-window patch predictions are reconstructed to spectra, blended into dense cubes by weighted overlap-add, then scored by the full curve, parameter, and slot metric suite.",
            "nodes": nodes, "steps": steps,
        }

    def _tuning(self) -> dict:
        nodes = [
            {"id": "space",  "tex": r"\Theta",        "role": "measured",     "kind": "set",    "shape": "-",   "desc": "joint learning, regularisation and architecture search space", "sample": ["lr", "wd", "width", "..."]},
            {"id": "theta",  "tex": r"\theta",        "role": "intermediate", "kind": "vector", "shape": "d",   "desc": "sampled hyperparameter vector",                                "sample": ["3e-4", "1e-4", "64", "..."]},
            {"id": "fobj",   "tex": r"f(\theta)",     "role": "calculated",   "kind": "scalar", "shape": "1",   "desc": "trial objective: best validation loss",                        "sample": "2.3e-2"},
            {"id": "thetastar", "tex": r"\theta^{*}", "role": "final",        "kind": "vector", "shape": "d",   "desc": "best joint configuration found so far",                        "sample": ["2.6e-4", "8e-5", "96", "..."]},
        ]
        steps = [
            {
                "id": "sample", "title": "TPE proposal", "phase": "Joint search",
                "note": "A multivariate TPE sampler proposes a joint hyperparameter vector, using constant-liar parallelism across GPU workers.",
                "inputs": ["space"], "outputs": ["theta"],
                "lines": [[{"id": "theta", "tex": r"\theta", "role": "intermediate"}, {"tex": r"\sim\ \mathrm{TPE}\!\big("}, {"id": "space", "tex": r"\Theta", "role": "measured"}, {"tex": r"\big)"}]],
            },
            {
                "id": "trial", "title": "Trial objective", "phase": "Joint search",
                "note": "Each trial trains a full model for the epoch budget and returns its best validation loss.",
                "inputs": ["theta"], "outputs": ["fobj"],
                "iterative": {"var": "fobj", "steps": 30, "unit": "epoch", "symbol": "f",
                              "trace": ["6.0e-2", "4.1e-2", "3.0e-2", "2.5e-2", "2.3e-2"]},
                "lines": [[{"id": "fobj", "tex": r"f(\theta)", "role": "calculated"}, {"tex": "="}, {"tex": r"\min_{e \in \{1,\dots,E\}}\ \mathcal{L}^{(e)}_{\mathrm{val}}\!\big("}, {"id": "theta", "tex": r"\theta", "role": "intermediate"}, {"tex": r"\big)"}]],
            },
            {
                "id": "prune", "title": "Median pruning", "phase": "Joint search",
                "note": "Active after the startup trials and a warmup, a trial is pruned once its reported loss exceeds the running median of completed trials; failures convert to pruned, never crashed.",
                "inputs": ["fobj"], "outputs": ["fobj"],
                "lines": [[{"tex": r"\text{prune}\ \iff\ \mathcal{L}^{(t)}_{\mathrm{val}} > \operatorname{median}\big\{\mathcal{L}^{(t)}_{\mathrm{val}}\big\}"}]],
            },
            {
                "id": "best", "title": "Best-config export", "phase": "Best-config export",
                "note": "The SQLite-backed study is topped up in chunks until the trial target is reached; the minimiser is rewritten after every completed trial.",
                "inputs": ["fobj"], "outputs": ["thetastar"],
                "lines": [[{"id": "thetastar", "tex": r"\theta^{*}", "role": "final"}, {"tex": "="}, {"tex": r"\operatorname*{arg\,min}_{\theta \in \Theta}\ "}, {"id": "fobj", "tex": r"f(\theta)", "role": "calculated"}]],
            },
        ]
        return {
            "key": "tuning", "name": "Tuning (Optuna)",
            "blurb": "A single joint Optuna study wraps the training pipeline: TPE sampling proposes configurations, median pruning kills weak trials early, and the best joint configuration is exported, resumable in chunks.",
            "nodes": nodes, "steps": steps,
        }
