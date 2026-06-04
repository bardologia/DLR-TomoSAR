from __future__ import annotations


class EquationLibrary:

    def collect(self) -> list[dict]:
        return [
            {
                "group" : "Signal Model",
                "blurb" : "How a stack of co-registered SAR passes becomes an elevation power spectrum.",
                "items" : [
                    {
                        "title" : "Tomographic observation model",
                        "tex"   : r"\mathbf{y} = \int_{\xi} \gamma(\xi)\,\mathbf{a}(\xi)\,\mathrm{d}\xi + \mathbf{n}",
                        "note"  : "The complex interferometric vector is the steering-weighted integral of reflectivity over elevation, plus noise.",
                        "vars"  : [
                            {"sym": r"\mathbf{y}",      "desc": "complex observation vector over N_s passes"},
                            {"sym": r"\gamma(\xi)",     "desc": "normalised reflectivity along elevation"},
                            {"sym": r"\mathbf{a}(\xi)", "desc": "steering vector, phase set by the perpendicular baselines"},
                        ],
                    },
                    {
                        "title" : "Capon beamformer",
                        "tex"   : r"\hat{\gamma}_{\text{Capon}}(\xi) = \frac{1}{\mathbf{a}^{H}(\xi)\,\hat{\mathbf{R}}^{-1}\,\mathbf{a}(\xi)}",
                        "note"  : "Minimum-variance distortionless response estimate of the elevation spectrum from the sample covariance.",
                        "vars"  : [
                            {"sym": r"\hat{\mathbf{R}}", "desc": "sample covariance estimated over a spatial window"},
                            {"sym": r"\mathbf{a}^{H}",   "desc": "Hermitian transpose of the steering vector"},
                        ],
                    },
                    {
                        "title" : "Elevation axis",
                        "tex"   : r"x_h = x_{\min} + h\cdot\frac{x_{\max}-x_{\min}}{H-1},\quad h = 0,\dots,H-1",
                        "note"  : "Uniform grid of H elevation bins spanning the configured height range.",
                        "vars"  : [
                            {"sym": r"x_{\min}, x_{\max}", "desc": "height range bounds in metres"},
                            {"sym": r"H",                  "desc": "number of elevation bins"},
                        ],
                    },
                ],
            },
            {
                "group" : "Gaussian Mixture Target",
                "blurb" : "The K-component spectrum the network learns to predict in a single forward pass.",
                "items" : [
                    {
                        "title" : "Gaussian mixture approximation",
                        "tex"   : r"\hat{\gamma}(\xi) = \sum_{k=1}^{K} a_k\,\exp\!\left(-\frac{(\xi-\mu_k)^2}{2\sigma_k^2}\right)",
                        "note"  : "Each per-pixel elevation spectrum is approximated by a sum of K Gaussians.",
                        "vars"  : [
                            {"sym": r"a_k",     "desc": "amplitude (peak reflectivity) of component k"},
                            {"sym": r"\mu_k",   "desc": "mean elevation of component k"},
                            {"sym": r"\sigma_k","desc": "elevation spread of component k"},
                        ],
                    },
                    {
                        "title" : "Per-pixel parameter output",
                        "tex"   : r"\hat{\mathbf{p}} = [\,a_1,\mu_1,\sigma_1,\;\dots,\;a_K,\mu_K,\sigma_K\,]",
                        "note"  : "The model emits 3K channels per pixel: the full parameter set of the mixture.",
                        "vars"  : [
                            {"sym": r"3K", "desc": "output channels, three parameters per Gaussian slot"},
                        ],
                    },
                    {
                        "title" : "Total input channels",
                        "tex"   : r"C_{\text{in}} = c_p + N_s\,(c_s + c_i) + \mathbb{1}[\texttt{use\_dem}]",
                        "note"  : "Input width is set by which sources are enabled and the representation chosen for each.",
                        "vars"  : [
                            {"sym": r"c_p, c_s, c_i", "desc": "channels per pass for primary, secondaries, interferograms"},
                            {"sym": r"N_s",           "desc": "number of secondary passes"},
                        ],
                    },
                ],
            },
            {
                "group" : "Training Objective",
                "blurb" : "How predictions are scored against the experimental curves.",
                "items" : [
                    {
                        "title" : "Curve reconstruction MSE",
                        "tex"   : r"\mathcal{L}_{\text{MSE}} = \frac{1}{B N H W}\sum_{b,n,h,w}\left(\hat{y}_{b,n,h,w}-y_{b,n,h,w}\right)^2",
                        "note"  : "Mean squared error between reconstructed and experimental spectra over the elevation axis.",
                        "vars"  : [
                            {"sym": r"\hat{y}", "desc": "reconstructed curve from predicted parameters"},
                            {"sym": r"y",       "desc": "experimental (target) curve"},
                        ],
                    },
                    {
                        "title" : "Heteroscedastic NLL",
                        "tex"   : r"\mathcal{L}_{\text{NLL}} = \frac{1}{B N H W}\sum_{b,n,h,w}\left[\frac{(\hat{y}-y)^2}{2(\sigma_{\text{noise}}^2+\epsilon)} + \log\sigma_{\text{noise}}\right]",
                        "note"  : "When a noise head is active, the model also predicts a per-pixel uncertainty.",
                        "vars"  : [
                            {"sym": r"\sigma_{\text{noise}}", "desc": "clamped per-pixel noise standard deviation"},
                        ],
                    },
                    {
                        "title" : "Coefficient of determination",
                        "tex"   : r"R^2 = 1 - \frac{\sum (\hat{y}-y)^2}{\sum (y-\bar{y})^2 + \epsilon}",
                        "note"  : "Reported per-pixel and globally to quantify reconstruction quality.",
                        "vars"  : [
                            {"sym": r"\bar{y}", "desc": "mean of the experimental curve values"},
                        ],
                    },
                ],
            },
            {
                "group" : "Optimisation",
                "blurb" : "The update rules and schedules driving the training loop.",
                "items" : [
                    {
                        "title" : "AdamW update",
                        "tex"   : r"\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_t\right)",
                        "note"  : "Decoupled weight decay with bias-corrected first and second moments.",
                        "vars"  : [
                            {"sym": r"\eta",     "desc": "learning rate per parameter group"},
                            {"sym": r"\lambda",  "desc": "weight decay coefficient"},
                        ],
                    },
                    {
                        "title" : "Cosine annealing",
                        "tex"   : r"\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{t\pi}{T_{\max}}\right)",
                        "note"  : "Learning rate decays along a cosine after the warmup phase completes.",
                        "vars"  : [
                            {"sym": r"T_{\max}", "desc": "total scheduled epochs"},
                            {"sym": r"\eta_{\min}", "desc": "minimum learning rate floor"},
                        ],
                    },
                    {
                        "title" : "EMA shadow update",
                        "tex"   : r"\tilde{\theta}_t = \gamma\,\tilde{\theta}_{t-1} + (1-\gamma)\,\theta_t",
                        "note"  : "Shadow weights track the model and replace it at evaluation time.",
                        "vars"  : [
                            {"sym": r"\gamma", "desc": "EMA decay coefficient"},
                        ],
                    },
                ],
            },
        ]
