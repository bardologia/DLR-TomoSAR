"use strict";

class SurveyView {
  static FLIP_LABELS = { azimuth: "azimuth flip", range: "range flip", both: "both flips" };

  constructor(root) {
    this.root         = root;
    this.built        = false;
    this.pollToken    = 0;
    this.runs         = [];
    this.selectedId   = null;
    this.runStrip     = null;
    this.renderedPath = null;
  }

  enter() {
    if (!this.built) {
      this._build();
      this.built = true;
    }
    this._refreshRuns();
    this._pollStatus();
  }

  _build() {
    this.root.innerHTML = `
      <div class="cube-pick">
        <div class="run-strip" id="survey-strip" aria-label="Trained backbone and dual runs"></div>
        <p class="cube-hint" id="survey-hint">Scanning the runs directory for loadable backbone and dual runs&hellip;</p>
        <div class="cube-progress sv-progress" id="survey-progress" hidden>
          <div class="cube-progress__track"><i class="cube-progress__fill" id="survey-progress-fill"></i></div>
          <p class="cube-progress__label" id="survey-progress-label">starting&hellip;</p>
          <button type="button" class="btn btn--mini" id="survey-cancel">Cancel</button>
        </div>
      </div>

      <div class="ms-main">
        <div class="ms-cards" id="survey-cards" hidden>
          <section class="ms-card ms-card--wide">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Surveyed model</h3>
              <span class="ms-badge" id="survey-badge"></span>
            </div>
            <div class="ms-meta" id="survey-meta"></div>
          </section>

          <section class="ms-card">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Prediction fit</h3>
              <span class="ms-card__note">per-pixel curve errors over the whole surveyed region</span>
            </div>
            <table class="ms-slots" id="survey-fit"></table>
            <p class="ms-wifacts" id="survey-detection"></p>
            <p class="ms-card__note ms-witablenote" id="survey-match-note" hidden>hungarian-matched per-parameter mean absolute error over every active ground-truth scatterer</p>
            <table class="ms-slots" id="survey-match"></table>
          </section>

          <section class="ms-card">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Input attribution</h3>
              <span class="ms-card__note">share of |&part;output/&part;input| per input channel, gradients of the summed outputs over every pixel</span>
            </div>
            <div class="ms-bars" id="survey-bars"></div>
            <div class="ms-legend" id="survey-bars-legend"></div>
            <div class="ms-subhead">
              <h4 class="cube-panel-title">Effective receptive field</h4>
              <span class="ms-card__note" id="survey-erf-note"></span>
            </div>
            <canvas id="survey-erf" width="640" height="180" hidden></canvas>
            <p class="ms-wifacts" id="survey-erf-facts"></p>
          </section>

          <section class="ms-card">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Channel ablation</h3>
              <span class="ms-card__note">zero one input channel at a time and re-run &mdash; mean causal damage over every pixel</span>
            </div>
            <div class="ms-abl" id="survey-ablation"></div>
          </section>

          <section class="ms-card">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Symmetry</h3>
              <span class="ms-card__note">mean disagreement between the model and its flipped self over every pixel</span>
            </div>
            <table class="ms-slots" id="survey-flips"></table>
          </section>

          <section class="ms-card">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Noise robustness</h3>
              <span class="ms-card__note">Gaussian noise added to every channel, one fresh draw per tile &mdash; mean damage vs &sigma;</span>
            </div>
            <canvas id="survey-noise" width="640" height="200"></canvas>
            <p class="ms-wifacts" id="survey-noise-facts"></p>
          </section>

          <section class="ms-card">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Occlusion distance profile</h3>
              <span class="ms-card__note">mean damage to a pixel when a zero-patch lands at a given distance from it</span>
            </div>
            <canvas id="survey-occl" width="640" height="200"></canvas>
            <p class="ms-wifacts" id="survey-occl-facts"></p>
          </section>

          <section class="ms-card ms-card--wide">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Layer vitals</h3>
              <span class="ms-card__note">activation statistics accumulated over every surveyed tile</span>
            </div>
            <p class="ms-wifacts" id="survey-vitals-summary"></p>
            <div class="ms-vitals"><table class="ms-slots ms-vitals__table" id="survey-vitals"></table></div>
          </section>
        </div>
      </div>`;

    this.refs = {
      strip      : this.root.querySelector("#survey-strip"),
      hint       : this.root.querySelector("#survey-hint"),
      progress   : this.root.querySelector("#survey-progress"),
      fill       : this.root.querySelector("#survey-progress-fill"),
      label      : this.root.querySelector("#survey-progress-label"),
      cancelBtn  : this.root.querySelector("#survey-cancel"),
      cards      : this.root.querySelector("#survey-cards"),
      badge      : this.root.querySelector("#survey-badge"),
      meta       : this.root.querySelector("#survey-meta"),
      fit        : this.root.querySelector("#survey-fit"),
      detection  : this.root.querySelector("#survey-detection"),
      matchNote  : this.root.querySelector("#survey-match-note"),
      match      : this.root.querySelector("#survey-match"),
      bars       : this.root.querySelector("#survey-bars"),
      barsLegend : this.root.querySelector("#survey-bars-legend"),
      erfNote    : this.root.querySelector("#survey-erf-note"),
      erf        : this.root.querySelector("#survey-erf"),
      erfFacts   : this.root.querySelector("#survey-erf-facts"),
      ablation   : this.root.querySelector("#survey-ablation"),
      flips      : this.root.querySelector("#survey-flips"),
      noise      : this.root.querySelector("#survey-noise"),
      noiseFacts : this.root.querySelector("#survey-noise-facts"),
      occl       : this.root.querySelector("#survey-occl"),
      occlFacts  : this.root.querySelector("#survey-occl-facts"),
      vitalsSum  : this.root.querySelector("#survey-vitals-summary"),
      vitals     : this.root.querySelector("#survey-vitals"),
    };

    this.refs.cancelBtn.addEventListener("click", async () => {
      await window.apiPost("/api/survey/cancel", {});
    });
  }

  _runsBase() {
    try {
      const raw = JSON.parse(localStorage.getItem("results-sources") || "{}");
      return raw.logs || ResultsView.DEFAULT_RUNS;
    } catch (e) {
      return ResultsView.DEFAULT_RUNS;
    }
  }

  async _refreshRuns() {
    const data = await window.apiGet(`/api/survey/runs?base=${encodeURIComponent(this._runsBase())}`);

    if (!data || !data.ok) {
      this.refs.hint.textContent = (data && data.error) || "Backend unreachable.";
      return;
    }

    this.runs = data.runs || [];
    this._renderStrip();

    if (!this.runs.length) {
      this.refs.hint.textContent = `No loadable runs under ${data.root}. Train a backbone or dual run, or point the runs directory in the Results tab at the right place.`;
      return;
    }

    if (this.renderedPath === null) this.refs.hint.textContent = "Pick a run to survey the entire test region. This re-runs the model many times per tile and takes a few minutes.";
  }

  _renderStrip() {
    if (!this.runStrip) {
      this.runStrip = new RunStrip(this.refs.strip, {
        stateFor : (run) => run.id === this.selectedId,
        onPick   : (run) => this._start(run.id),
      });
    }
    this.runStrip.render(this.runs);
  }

  async _start(path) {
    const out = await window.apiPost("/api/survey/start", { path });
    if (!out.ok) {
      this.refs.hint.textContent = out.error || "survey failed to start";
      return;
    }

    this.selectedId   = path;
    this.renderedPath = null;
    this.refs.cards.hidden = true;
    this._renderStrip();
    this._pollStatus();
  }

  async _pollStatus() {
    const token = ++this.pollToken;
    const tick  = async () => {
      if (token !== this.pollToken) return;

      const res = await fetch("/api/survey/status");
      const st  = await res.json();

      if (st.state === "running") {
        this.refs.progress.hidden   = false;
        this.refs.fill.style.width  = `${Math.round((st.progress || 0) * 100)}%`;
        this.refs.label.textContent = `${st.stage || "running"} — ${Math.round((st.progress || 0) * 100)}%`;
        if (st.path && st.path !== this.selectedId) {
          this.selectedId = st.path;
          this._renderStrip();
        }
        setTimeout(tick, 800);
        return;
      }

      this.refs.progress.hidden = true;

      if (st.state === "error") {
        this.refs.hint.textContent = st.error;
        return;
      }
      if (st.state === "cancelled") {
        this.refs.hint.textContent = "Survey cancelled.";
        return;
      }
      if (st.state === "done") {
        if (st.path && st.path !== this.selectedId) {
          this.selectedId = st.path;
          this._renderStrip();
        }
        if (this.renderedPath !== st.path) await this._loadResult(st.path);
      }
    };
    tick();
  }

  async _loadResult(path) {
    const out = await window.apiGet("/api/survey/result");
    if (!out || !out.ok) {
      this.refs.hint.textContent = (out && out.error) || "Backend unreachable.";
      return;
    }

    this.renderedPath = path;
    this._renderResult(out.result);
  }

  _seconds(value) {
    const minutes = Math.floor(value / 60);
    return minutes ? `${minutes} min ${Math.round(value - minutes * 60)} s` : `${value.toFixed(1)} s`;
  }

  _renderResult(r) {
    this.refs.hint.textContent = `Surveyed ${r.backbone} over the ${r.split} split.`;
    this.refs.cards.hidden     = false;

    this.refs.badge.textContent = r.backbone;
    this.refs.meta.innerHTML = [
      `<span class="ms-chip"><b>split</b> ${window.msEsc(r.split)}</span>`,
      `<span class="ms-chip"><b>region</b> ${r.region[0]}&times;${r.region[1]} px</span>`,
      `<span class="ms-chip"><b>surveyed</b> ${r.coverage.tiles} of ${r.coverage.total_tiles} tiles</span>`,
      `<span class="ms-chip"><b>pixels</b> ${r.coverage.pixels.toLocaleString("en-US")}</span>`,
      `<span class="ms-chip"><b>window</b> ${r.patch[0]}&times;${r.patch[1]} px</span>`,
      `<span class="ms-chip"><b>K</b> ${r.n_gaussians}</span>`,
      `<span class="ms-chip"><b>in</b> ${r.channels.length} ch</span>`,
      `<span class="ms-chip"><b>took</b> ${this._seconds(r.seconds)}</span>`,
    ].join("");

    this._renderFit(r);
    this._renderAttribution(r);
    this._renderAblation(r);
    this._renderFlips(r);
    this._renderNoise(r);
    this._renderOcclusion(r);

    window.renderVitalsTable(this.refs.vitalsSum, this.refs.vitals, r.vitals, {});
  }

  _statRow(label, stat) {
    return `
      <tr>
        <td class="ms-slots__key">${label}</td>
        <td>${stat.mean.toExponential(2)}</td>
        <td>${stat.median.toExponential(2)}</td>
        <td>${stat.p90.toExponential(2)}</td>
      </tr>`;
  }

  _renderFit(r) {
    const head = `<thead><tr><th class="ms-slots__key">CURVE MSE</th><th>MEAN</th><th>MEDIAN</th><th>P90</th></tr></thead>`;
    const rows = [];
    if (r.fit.curve_mse_gt) rows.push(this._statRow("vs ground truth", r.fit.curve_mse_gt));
    rows.push(this._statRow("vs raw tomogram", r.fit.curve_mse_raw));
    this.refs.fit.innerHTML = `${head}<tbody>${rows.join("")}</tbody>`;

    if (!r.detection) {
      this.refs.detection.innerHTML  = "no ground truth on this split &mdash; detection and matched errors need GT parameters";
      this.refs.matchNote.hidden     = true;
      this.refs.match.innerHTML      = "";
      return;
    }

    this.refs.detection.innerHTML = [
      `mean active slots <b>${r.detection.mean_pred_active.toFixed(2)}</b> predicted vs <b>${r.detection.mean_gt_active.toFixed(2)}</b> in the ground truth`,
      `exact slot-count match at <b>${(r.detection.count_match_frac * 100).toFixed(1)}%</b> of pixels`,
    ].join(" &middot; ");

    this.refs.matchNote.hidden = false;
    this.refs.match.innerHTML = `
      <thead><tr><th>AMP MAE</th><th>&mu; MAE [m]</th><th>&sigma; MAE [m]</th><th>MATCHED PX&middot;SLOTS</th></tr></thead>
      <tbody><tr>
        <td>${r.matched.amp.toFixed(4)}</td>
        <td>${r.matched.mu.toFixed(3)}</td>
        <td>${r.matched.sigma.toFixed(3)}</td>
        <td>${r.matched.n_matched.toLocaleString("en-US")}</td>
      </tr></tbody>`;
  }

  _renderAttribution(r) {
    window.renderFamilyBars(this.refs.bars, this.refs.barsLegend, r.channels, r.attribution.families, "no dependence anywhere in the region");

    const live = r.erf.families.filter((f) => !f.dead);
    this.refs.erfNote.innerHTML = `cumulative gradient energy vs radius, averaged over ${r.erf.samples} sampled pixels`;

    if (!live.length) {
      this.refs.erf.hidden = true;
      this.refs.erfFacts.innerHTML = "no gradient energy at any sampled pixel";
      return;
    }

    this.refs.erf.hidden = false;
    window.drawLineChart(
      this.refs.erf,
      live[0].radii,
      live.map((f) => ({ values: f.cumulative, color: MicroscopeView.FAMILY_COLORS[f.family], width: 1.5, label: f.family })),
      { xUnit: "px" },
    );

    this.refs.erfFacts.innerHTML = r.erf.families.map((f) => f.dead
      ? `${f.family} &mdash; no dependence`
      : `${f.family} r<sub>50</sub> <b>${f.r50} px</b> / r<sub>90</sub> <b>${f.r90} px</b>`
    ).join(" &middot; ");
  }

  _renderAblation(r) {
    window.renderAblationRows(this.refs.ablation, r.ablation.channels, r.fit.base_power, (c) =>
      c.flips_frac > 0 ? ` &middot; slots flip at ${(c.flips_frac * 100).toFixed(1)}% of px` : ""
    );
  }

  _renderFlips(r) {
    const power = r.fit.base_power;
    const head  = `<thead><tr><th class="ms-slots__key">FLIP</th><th>MEAN &Delta; CURVE MSE</th><th>% OF POWER</th></tr></thead>`;

    const rows = r.flips.filter((e) => e.flip !== "none").map((e) => `
      <tr>
        <td class="ms-slots__key">${SurveyView.FLIP_LABELS[e.flip]}</td>
        <td>${e.delta_mse.toExponential(2)}</td>
        <td>${power > 0 ? ((e.delta_mse / power) * 100).toFixed(2) : "&ndash;"}</td>
      </tr>`);

    this.refs.flips.innerHTML = `${head}<tbody>${rows.join("")}</tbody>`;
  }

  _renderNoise(r) {
    window.drawLineChart(
      this.refs.noise,
      r.noise.map((p) => p.value),
      [{ values: r.noise.map((p) => p.delta_mse), color: MicroscopeView.SERIES_COLORS.perturbed, width: 1.6, label: "Δ curve MSE" }],
      { xUnit: "" },
    );

    const last = r.noise[r.noise.length - 1];
    const rel  = r.fit.base_power > 0 ? ` (${((last.delta_mse / r.fit.base_power) * 100).toFixed(1)}% of power)` : "";
    this.refs.noiseFacts.innerHTML = `mean &Delta; curve MSE at &sigma; ${last.value.toFixed(1)}: <b>${last.delta_mse.toExponential(2)}</b>${rel}`;
  }

  _renderOcclusion(r) {
    window.drawLineChart(
      this.refs.occl,
      r.occlusion.distance,
      [{ values: r.occlusion.delta, color: MicroscopeView.SERIES_COLORS.pred, width: 1.6, label: "Δ curve MSE" }],
      { xUnit: "px" },
    );

    const first = r.occlusion.delta[0];
    const last  = r.occlusion.delta[r.occlusion.delta.length - 1];
    const ratio = last > 0 ? (first / last).toFixed(0) : "&infin;";
    this.refs.occlFacts.innerHTML = `occluder ${r.occlusion.occluder[0]}&times;${r.occlusion.occluder[1]} px &middot; damage right under the occluder is ${ratio}&times; the damage at ${r.occlusion.distance[r.occlusion.distance.length - 1].toFixed(0)} px`;
  }
}

window.SurveyView = SurveyView;
