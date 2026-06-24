from __future__ import annotations


class PhysicsLossLibrary:

    def _intro(self) -> dict:
        return {
            "kicker" : "training objective · signal-model supervision",
            "title"  : "The physics loss terms",
            "lead"   : "Five optional loss terms that do not compare curves bin by bin. Each one pushes the predicted Gaussian-mixture tomogram back through the TomoSAR forward model, extracts a physical quantity, and matches it against the same quantity computed from the ground-truth profile. They supervise what the radar actually measures, not the shape of a fitted spectrum.",
            "points" : [
                {
                    "label" : "Shared forward model",
                    "text"  : "Every term reuses one Fourier operator built from the acquisition geometry: the perpendicular baselines set the vertical wavenumbers, those set the steering matrix, and the steering matrix turns a profile into interferometric observables.",
                },
                {
                    "label" : "Supervised against the label",
                    "text"  : "Both the prediction and the reference are curve reconstructions of their Gaussian parameters, in denormalised physical units. The terms match the predicted physical quantity to the GT label, never to a re-estimated Capon profile.",
                },
                {
                    "label" : "Masked to real signal",
                    "text"  : "Each term reduces only over pixels whose ground-truth integrated power exceeds the physics floor; empty sky is never penalised. All five terms default off and must be scale-calibrated before they mix with the curve and parameter losses.",
                },
            ],
        }

    def _operator(self) -> dict:
        return {
            "title"  : "The shared forward operator",
            "blurb"  : "The vertical wavenumber of each pass, the steering matrix sampled on the elevation grid, and the per-bin outer product the covariance terms reuse. Built once per run by TomoGeometry from the resolved geometry; the same matrices feed all five terms.",
            "items"  : [
                {
                    "title" : "Vertical wavenumber",
                    "tex"   : r"k_z^{(i)} = \frac{4\pi\,b_i}{\lambda\,r_0}",
                    "note"  : "Each pass sees elevation through a phase ramp whose rate is set by its perpendicular baseline. The wider the baseline spread, the finer the elevation resolution the stack can resolve.",
                    "vars"  : [
                        {"sym": r"k_z^{(i)}",     "desc": "vertical wavenumber of pass i (rad/m)"},
                        {"sym": r"b_i",           "desc": "perpendicular baseline of pass i (m)"},
                        {"sym": r"\lambda, r_0",  "desc": "radar wavelength and slant range (m)"},
                    ],
                },
                {
                    "title" : "Steering matrix",
                    "tex"   : r"A_{i,n} = \exp\!\left(j\,k_z^{(i)}\,\xi_n\right)",
                    "note"  : "The complex response of pass i to unit reflectivity at elevation bin n. Mapping a profile through A re-synthesises the multibaseline signal a real acquisition would have measured.",
                    "vars"  : [
                        {"sym": r"A_{i,n}", "desc": "steering matrix, pass i at elevation bin n"},
                        {"sym": r"\xi_n",   "desc": "elevation grid sample (m)"},
                        {"sym": r"j",       "desc": "imaginary unit"},
                    ],
                },
                {
                    "title" : "Steering outer product",
                    "tex"   : r"O_{i,j,n} = A_{i,n}\,\overline{A_{j,n}}",
                    "note"  : "Precomputed per elevation bin so the covariance and Capon-cycle terms synthesise a covariance matrix with a single contraction against the profile, instead of rebuilding the steering products at every step.",
                    "vars"  : [
                        {"sym": r"O_{i,j,n}",         "desc": "outer product of the steering columns at bin n"},
                        {"sym": r"\overline{A_{j,n}}", "desc": "complex conjugate of the steering entry"},
                    ],
                },
            ],
        }

    def _terms(self) -> list[dict]:
        return [
            {
                "key"        : "total_power",
                "index"      : 1,
                "name"       : "Total power",
                "code"       : "PhysicalLoss.total_power",
                "tagline"    : "Does the profile carry the right total backscatter?",
                "role"       : "radiometric",
                "role_label" : "kept, but biased",
                "quantity"   : "Integrated reflectivity (mass m₀)",
                "invariant"  : "Absolute power",
                "cost"       : 1,
                "cost_label" : "cheapest",
                "tex"        : r"\ell_{\mathrm{pow}} = \left\langle \frac{\left|m_0^{P} - m_0^{T}\right|}{m_0^{T}} \right\rangle_{\text{valid}}, \qquad m_0 = \sum_n P(\xi_n)\,\Delta\xi",
                "story"      : "The simplest physical check: integrate the predicted spectrum and the reference spectrum over elevation and compare the total. It is the only one of the five that looks at absolute radiometric power rather than a normalised shape.",
                "caveat"     : "Because it is an absolute-power term it inherits the radiometric bias of the Capon estimator that produced the data, roughly 2 dB on-grid and worse off-grid. That bias is why its default weight is zero and why the scale-insensitive terms are preferred as the actual objective.",
                "vars"       : [
                    {"sym": r"\ell_{\mathrm{pow}}", "desc": "total-power relative-error term"},
                    {"sym": r"m_0^{P}, m_0^{T}",    "desc": "integrated power of predicted and GT profiles"},
                    {"sym": r"\Delta\xi",           "desc": "elevation bin spacing (m)"},
                ],
            },
            {
                "key"        : "moments",
                "index"      : 2,
                "name"       : "Profile moments",
                "code"       : "PhysicalLoss.moments",
                "tagline"    : "Is the scatterer at the right height and the right thickness?",
                "role"       : "metric",
                "role_label" : "validation-grade",
                "quantity"   : "Mass, centroid height, vertical spread",
                "invariant"  : "Absolute power (centroid and spread are ratios)",
                "cost"       : 1,
                "cost_label" : "cheap",
                "tex"        : r"\bar{z} = \frac{\sum_n P_n\,\xi_n}{\sum_n P_n}, \quad \sigma_z = \sqrt{\frac{\sum_n P_n\,\xi_n^2}{\sum_n P_n} - \bar{z}^2}, \qquad \ell_{\mathrm{mom}} = \left\langle \frac{w_0\frac{|\Delta m_0|}{m_0^{T}} + w_1\frac{|\Delta\bar{z}|}{\Delta\xi_R} + w_2\frac{|\Delta\sigma_z|}{\Delta\xi_R}}{w_0+w_1+w_2} \right\rangle",
                "story"      : "Summarises each profile by its first three descriptors: total mass, the centroid that locates the scattering phase centre in elevation, and the spread that measures the vertical extent of the scattering. These are the quantities a forest-height or layover analysis actually reads off a tomogram.",
                "caveat"     : "Centroid and spread are mass-normalised ratios, so they are robust to the Capon power bias even though the mass sub-term is not. Used mainly as a validation-grade physics metric rather than the primary loss; the sub-weights default to (1, 1, 1).",
                "vars"       : [
                    {"sym": r"\ell_{\mathrm{mom}}",  "desc": "moments term value"},
                    {"sym": r"\bar{z}, \sigma_z",    "desc": "profile centroid and spread (m)"},
                    {"sym": r"w_0, w_1, w_2",        "desc": "mass / centroid / spread weights, default (1, 1, 1)"},
                    {"sym": r"\Delta\xi_R",          "desc": "elevation axis span x_max - x_min (m)"},
                ],
            },
            {
                "key"        : "coherence_resyn",
                "index"      : 3,
                "name"       : "Coherence re-synthesis",
                "code"       : "PhysicalLoss.coherence_resynthesis",
                "tagline"    : "Would the predicted profile produce the same interferograms?",
                "role"       : "front-runner",
                "role_label" : "scale-insensitive front-runner",
                "quantity"   : "Multibaseline interferometric coherences",
                "invariant"  : "Absolute power (unit-disc ratio)",
                "cost"       : 2,
                "cost_label" : "light",
                "tex"        : r"\gamma_P\!\left(k_z^{(i)}\right) = \frac{\sum_n P_n\,e^{j\,k_z^{(i)}\xi_n}}{\sum_n P_n}, \qquad \ell_{\mathrm{coh\text{-}r}} = \left\langle \frac{1}{N_s}\sum_i \left|\gamma_P^{(i)} - \gamma_T^{(i)}\right|^2 \right\rangle",
                "story"      : "Re-synthesises the complex coherence each baseline pair would observe: the normalised Fourier transform of the profile sampled at the track wavenumbers. This is the characteristic function of the normalised elevation distribution, exactly the interferometric observable the stack measures. For a Gaussian mixture it is a closed-form sum of complex Gaussians, so the term is exactly differentiable in the predicted amplitudes, means, and spreads.",
                "caveat"     : "Each coherence lives on the unit disc by construction, so the term is insensitive to absolute power and sidesteps the Capon radiometric bias entirely. One of the two recommended physics objectives.",
                "vars"       : [
                    {"sym": r"\ell_{\mathrm{coh\text{-}r}}", "desc": "coherence re-synthesis term value"},
                    {"sym": r"\gamma_P, \gamma_T",           "desc": "normalised coherences of predicted and GT profiles"},
                    {"sym": r"k_z^{(i)}",                    "desc": "vertical wavenumber of pass i"},
                    {"sym": r"N_s",                          "desc": "number of passes in the stack"},
                ],
            },
            {
                "key"        : "covariance_match",
                "index"      : 4,
                "name"       : "Covariance matching",
                "code"       : "PhysicalLoss.covariance_matching",
                "tagline"    : "Does the predicted profile rebuild the right sample covariance?",
                "role"       : "front-runner",
                "role_label" : "scale-insensitive front-runner",
                "quantity"   : "Synthesised multibaseline covariance matrix",
                "invariant"  : "Absolute scale (relative Frobenius)",
                "cost"       : 3,
                "cost_label" : "moderate",
                "tex"        : r"\mathbf{R}[P] = \mathbf{A}\,\mathrm{diag}(P)\,\mathbf{A}^{H}\,\Delta\xi, \qquad \ell_{\mathrm{cov}} = \left\langle \frac{\left\|\mathbf{R}[P] - \mathbf{R}[T]\right\|_F^2}{\left\|\mathbf{R}[T]\right\|_F^2} \right\rangle",
                "story"      : "Synthesises the full covariance matrix the profile would generate across the stack and matches it to the reference covariance in relative Frobenius norm. Where coherence re-synthesis checks the normalised diagonal of the interferometric structure, this term constrains the entire matrix at once. Because the covariance is linear in the profile, the implementation transforms only the prediction-minus-GT difference rather than two full matrices.",
                "caveat"     : "Follows the COMET covariance-matching lineage but uses an unweighted relative form with the reference synthesised from the GT profile, so it is insensitive to absolute scale. The second of the two recommended physics objectives.",
                "vars"       : [
                    {"sym": r"\ell_{\mathrm{cov}}",          "desc": "covariance matching term value"},
                    {"sym": r"\mathbf{R}[P], \mathbf{R}[T]", "desc": "covariances synthesised from predicted and GT profiles"},
                    {"sym": r"\mathbf{A}",                   "desc": "steering matrix exp(j kz xi)"},
                    {"sym": r"\|\cdot\|_F",                  "desc": "Frobenius norm over the pass-pair axes"},
                ],
            },
            {
                "key"        : "capon_cycle",
                "index"      : 5,
                "name"       : "Capon cycle-consistency",
                "code"       : "PhysicalLoss.capon_cycle",
                "tagline"    : "Run the prediction through the full estimator and back.",
                "role"       : "faithful",
                "role_label" : "most faithful, most expensive",
                "quantity"   : "Re-estimated Capon spectrum",
                "invariant"  : "Absolute power (mass-normalised spectra)",
                "cost"       : 5,
                "cost_label" : "most expensive",
                "tex"        : r"\hat{T}_P(\xi_n) = \frac{1}{\mathbf{a}^{H}(\xi_n)\big(\mathbf{R}[P] + \epsilon\,\bar{\sigma}\,\mathbf{I}\big)^{-1}\mathbf{a}(\xi_n)}, \qquad \ell_{\mathrm{cyc}} = \left\langle \frac{1}{N}\sum_n \left(\frac{\hat{T}_P(\xi_n)}{m_0^{\hat{T}}} - \frac{T(\xi_n)}{m_0^{T}}\right)^2 \right\rangle",
                "story"      : "The full cycle-consistency loss of inverse imaging: synthesise the covariance from the prediction, apply signal-adaptive diagonal loading, then run the same Capon beamformer that produced the data and compare the re-estimated spectrum with the reference, both mass-normalised. By construction it absorbs the Capon bias and point-spread function rather than fighting them.",
                "caveat"     : "Each pixel needs one N_s x N_s linear solve, making this the most expensive of the five. The loading term guarantees the covariance is invertible and the solve differentiable; the spectrum is formed by solving the system, not by an explicit inverse.",
                "vars"       : [
                    {"sym": r"\ell_{\mathrm{cyc}}",    "desc": "Capon cycle-consistency term value"},
                    {"sym": r"\hat{T}_P(\xi_n)",       "desc": "Capon spectrum re-estimated from the prediction"},
                    {"sym": r"\bar{\sigma}",           "desc": "mean covariance diagonal for adaptive loading"},
                    {"sym": r"\epsilon",               "desc": "capon_loading = 1e-2"},
                    {"sym": r"m_0^{\hat{T}}, m_0^{T}", "desc": "integrated-power normalisers"},
                ],
            },
        ]

    def _comparison(self) -> dict:
        return {
            "title"   : "How the five terms compare",
            "blurb"   : "All default off. The two scale-insensitive front-runners are the recommended objective; total power is kept but disqualified by radiometric bias; moments serve as a validation metric; the Capon cycle is the most faithful but the most expensive.",
            "columns" : ["Term", "Physical quantity", "Scale-invariant", "Compute cost", "Role"],
            "rows"    : [
                {"term": "Total power",            "quantity": "Integrated power",            "invariant": "No",  "cost": "Cheapest",  "role": "Kept, biased"},
                {"term": "Profile moments",        "quantity": "Mass, centroid, spread",      "invariant": "Mostly", "cost": "Cheap",  "role": "Validation metric"},
                {"term": "Coherence re-synthesis", "quantity": "Multibaseline coherences",    "invariant": "Yes", "cost": "Light",     "role": "Front-runner"},
                {"term": "Covariance matching",    "quantity": "Synthesised covariance",      "invariant": "Yes", "cost": "Moderate",  "role": "Front-runner"},
                {"term": "Capon cycle-consistency","quantity": "Re-estimated Capon spectrum", "invariant": "Yes", "cost": "Highest",   "role": "Most faithful"},
            ],
        }

    def _config(self) -> dict:
        return {
            "title"   : "Configuration reference",
            "blurb"   : "Per-term switches and weights live in LossConfig; the shared geometry lives in GeometryConfig. Every term is off by default; weights must be calibrated with the loss scale probe before physics terms are mixed with the curve and parameter losses.",
            "groups"  : [
                {
                    "name"   : "LossConfig",
                    "fields" : [
                        {"field": "use_total_power / weight_total_power",       "default": "False / 0.0",    "meaning": "enable and weight the total-power term"},
                        {"field": "use_moments / weight_moments",               "default": "False / 0.0",    "meaning": "enable and weight the moments term"},
                        {"field": "moments_weights",                            "default": "(1, 1, 1)",      "meaning": "mass / centroid / spread sub-weights"},
                        {"field": "use_coherence_resyn / weight_coherence_resyn","default": "False / 0.0",   "meaning": "enable and weight coherence re-synthesis"},
                        {"field": "use_covariance_match / weight_covariance_match","default": "False / 0.0", "meaning": "enable and weight covariance matching"},
                        {"field": "use_capon_cycle / weight_capon_cycle",       "default": "False / 0.0",    "meaning": "enable and weight the Capon cycle"},
                        {"field": "capon_loading",                              "default": "1e-2",           "meaning": "diagonal loading fraction for the Capon cycle"},
                        {"field": "physics_floor",                              "default": "1e-3",           "meaning": "reference mass below which a pixel is masked out"},
                    ],
                },
                {
                    "name"   : "GeometryConfig",
                    "fields" : [
                        {"field": "wavelength",          "default": "0.23",          "meaning": "radar wavelength (m)"},
                        {"field": "slant_range",         "default": "5000.0",        "meaning": "reference slant range (m)"},
                        {"field": "look_angle_deg",      "default": "45.0",          "meaning": "projects baselines to the perpendicular component"},
                        {"field": "baselines",          "default": "9 values, 0-90", "meaning": "synthetic baselines (m); replaced per dataset"},
                        {"field": "baseline_component",  "default": "perpendicular", "meaning": "baseline component the dataset table resolves to"},
                        {"field": "baselines_source",    "default": "auto",          "meaning": "auto / dataset / manual table resolution"},
                        {"field": "kz_values",           "default": "()",            "meaning": "explicit kz override; bypasses the baselines"},
                    ],
                },
            ],
            "cli"     : [
                "python main/train.py --curriculum.complete.use_coherence_resyn true --curriculum.complete.weight_coherence_resyn 1.0",
                "python main/train.py --geometry.kz_values \"[0.0, 0.12, 0.25, 0.37, 0.49]\"",
            ],
        }

    def collect(self) -> dict:
        return {
            "intro"      : self._intro(),
            "operator"   : self._operator(),
            "terms"      : self._terms(),
            "comparison" : self._comparison(),
            "config"     : self._config(),
        }
