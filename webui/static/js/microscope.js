"use strict";

class MicroscopeView {
  static FAMILIES      = ["amp", "mu", "sigma"];
  static FAMILY_COLORS = { amp: "#1d4fd8", mu: "#0f766e", sigma: "#a16207" };
  static FAMILY_LABELS = { amp: "amplitude", mu: "mean height", sigma: "width" };
  static SERIES_COLORS = { pred: "#1d4fd8", gt: "#16191b", raw: "#9ca3af", perturbed: "#b91c1c" };

  constructor(root) {
    this.root        = root;
    this.info        = null;
    this.pixel       = null;
    this.layer       = "";
    this.block       = "";
    this.blocks      = [];
    this.layerTypes  = {};
    this.built       = false;
    this.pollToken   = 0;
    this.attribToken = 0;
    this.ablToken    = 0;
    this.occlToken   = 0;
    this.flipToken   = 0;
    this.wiToken     = 0;
    this.runs        = [];
    this.selectedId  = null;
    this.runStrip    = null;
    this.lastPredict = null;
    this.attrib      = null;
    this.attribSlot  = -1;
    this.gridChannel = -1;
    this.fieldsData  = null;
    this.fieldsSlot  = 0;
    this.fieldsToken = 0;
    this.sweepData   = null;
    this.sweepToken  = 0;
    this.vitalsData  = null;
    this.vitalsToken = 0;
    this.wi          = { kind: "drop_channel", channel: 0, factor: 0.5, sigma: 0.5, seed: 0 };
    this.wiTimer     = null;
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
        <div class="run-strip" id="probe-strip" aria-label="Trained backbone and dual runs"></div>
        <p class="cube-hint" id="probe-hint">Scanning the runs directory for loadable backbone and dual runs&hellip;</p>
        <div class="cube-progress" id="probe-progress" hidden>
          <div class="cube-progress__track"><i class="cube-progress__fill" id="probe-progress-fill"></i></div>
          <p class="cube-progress__label" id="probe-progress-label">loading&hellip;</p>
        </div>
      </div>

      <div class="ms-stage" id="probe-stage" hidden>
        <section class="ms-card ms-scene">
          <div class="ms-scene__map">
            <div class="fl-map__frame">
              <span class="cube-axis cube-axis--y">azimuth &darr;</span>
              <div class="fl-map__wrap" id="probe-map-wrap">
                <img id="probe-map-img" alt="Primary SLC amplitude" />
                <div class="fl-marks" id="probe-marks"></div>
              </div>
              <span class="cube-axis cube-axis--x">range &rarr;</span>
            </div>
          </div>
          <div class="ms-scene__info">
            <div class="ms-card__head">
              <h3 class="cube-panel-title">Scene</h3>
              <span class="ms-badge" id="probe-model-badge"></span>
            </div>
            <div class="ms-meta" id="probe-meta"></div>
            <p class="cube-coords" id="probe-coords">Click the map to probe a pixel.</p>
            <div class="fl-pixrow" role="group" aria-label="Probe a pixel by index">
              <span class="cube-jump__cell">
                <label for="probe-az">az</label>
                <input type="number" id="probe-az" class="cube-jump__input" min="0" step="1" inputmode="numeric" />
              </span>
              <span class="cube-jump__cell">
                <label for="probe-rg">rg</label>
                <input type="number" id="probe-rg" class="cube-jump__input" min="0" step="1" inputmode="numeric" />
              </span>
              <button type="button" class="btn btn--mini" id="probe-go">Probe</button>
            </div>
          </div>
        </section>

        <div class="ms-main">
          <div class="ms-idle" id="probe-idle">Click a pixel on the scene map, or type az / rg indices, to put it under the microscope.</div>

          <div class="ms-cards" id="probe-cards" hidden>
            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Predicted profile</h3>
                <span class="ms-card__note">reflectivity vs height at the probed pixel</span>
              </div>
              <canvas id="probe-profile" width="640" height="240"></canvas>
              <table class="ms-slots" id="probe-slots"></table>
              <p class="ms-wifacts" id="probe-fit"></p>
              <p class="ms-card__note ms-witablenote" id="probe-match-note" hidden>ground-truth scatterers and their hungarian-matched predictions &mdash; red marks a miss beyond 0.1 amp or 0.5 m</p>
              <table class="ms-slots" id="probe-match"></table>
            </section>

            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Local parameter fields</h3>
                <span class="ms-card__note">one forward pass over the input window &mdash; how each parameter behaves in the pixel's neighbourhood</span>
              </div>
              <div class="ms-ctlrow">
                <span class="ms-ctlrow__label">scatterer</span>
                <div class="ms-chiprow" id="probe-fields-slots"></div>
              </div>
              <div class="ms-fieldrow" id="probe-fields-maps"></div>
              <div class="ms-fieldrow" id="probe-fields-overview"></div>
            </section>

            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Input attribution</h3>
                <span class="ms-card__note" id="probe-attrib-note">share of |&part;output/&part;input| per input channel, all three outputs side by side</span>
              </div>
              <div class="ms-ctlrow">
                <span class="ms-ctlrow__label">scatterer</span>
                <div class="ms-chiprow" id="probe-attrib-slots"></div>
              </div>
              <div class="ms-bars" id="probe-bars"></div>
              <div class="ms-legend" id="probe-bars-legend"></div>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Spatial gradients</h4>
                <span class="ms-card__note">|&part;output/&part;input| over the input window &mdash; pick an input channel or view all</span>
              </div>
              <div class="ms-ctlrow">
                <span class="ms-ctlrow__label">channel</span>
                <div class="ms-chiprow" id="probe-grid-channels"></div>
              </div>
              <div class="ms-gridwrap"><div class="ms-grid" id="probe-grid"></div></div>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Channel ablation</h4>
                <span class="ms-card__note">zero one input channel at a time and re-run &mdash; causal damage to the prediction, unlike the local gradients above</span>
              </div>
              <div class="ms-abl" id="probe-ablation"></div>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Occlusion sensitivity</h4>
                <span class="ms-card__note">slide a zero-patch across the window and re-run &mdash; where the model's evidence for this pixel actually sits</span>
              </div>
              <div class="ms-fieldrow" id="probe-occlusion"></div>
              <p class="ms-wifacts" id="probe-occl-facts"></p>
            </section>

            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Symmetry</h3>
                <span class="ms-card__note">re-run the model on flipped windows and flip the outputs back &mdash; a flip-equivariant model repeats itself exactly</span>
              </div>
              <canvas id="probe-flips" width="640" height="240"></canvas>
              <p class="ms-wifacts" id="probe-flips-facts"></p>
            </section>

            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">Feature maps</h3>
                <span class="ms-card__note" id="probe-feat-note"></span>
              </div>
              <div class="ms-arch" id="probe-arch" aria-label="Model blocks in forward order"></div>
              <div class="ms-arch__layers" id="probe-arch-layers"></div>
              <div class="ms-feat"><img id="probe-features" alt="Feature maps" hidden /></div>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Layer vitals</h4>
                <span class="ms-card__note">one hooked forward pass at the probed window &mdash; click a row to open that layer's feature maps</span>
              </div>
              <p class="ms-wifacts" id="probe-vitals-summary"></p>
              <div class="ms-vitals"><table class="ms-slots ms-vitals__table" id="probe-vitals"></table></div>
            </section>

            <section class="ms-card">
              <div class="ms-card__head">
                <h3 class="cube-panel-title">What-if</h3>
                <span class="ms-card__note">perturb the input window, re-run the model, compare the predictions</span>
              </div>
              <div class="ms-ctlrow">
                <span class="ms-ctlrow__label">perturb</span>
                <div class="cube-spaces" id="probe-whatif-kind" role="group" aria-label="Perturbation">
                  <button type="button" class="cube-space is-active" data-kind="drop_channel">zero a channel</button>
                  <button type="button" class="cube-space" data-kind="scale_channel">scale a channel</button>
                  <button type="button" class="cube-space" data-kind="noise">add noise</button>
                </div>
              </div>
              <div class="ms-ctlrow" id="probe-whatif-chrow">
                <span class="ms-ctlrow__label">channel</span>
                <div class="ms-chiprow" id="probe-whatif-channels"></div>
                <span class="ms-ctlrow__na" id="probe-whatif-chna" hidden>noise is added to every input channel at once</span>
              </div>
              <div class="ms-ctlrow" id="probe-whatif-valrow">
                <span class="ms-ctlrow__label" id="probe-whatif-value-label">strength</span>
                <div class="ms-wival" id="probe-whatif-valctl" hidden>
                  <input type="range" id="probe-whatif-range" min="0" max="2" step="0.05" value="0.5" />
                  <input type="number" id="probe-whatif-value" class="cube-jump__input" min="0" max="2" step="0.05" value="0.5" />
                  <button type="button" class="btn btn--mini" id="probe-whatif-reroll" hidden>re-roll noise</button>
                </div>
                <span class="ms-ctlrow__na" id="probe-whatif-valna">zeroing removes the channel outright; there is no strength to tune</span>
              </div>
              <p class="ms-wisummary" id="probe-whatif-desc"></p>
              <canvas id="probe-whatif" width="640" height="240"></canvas>
              <p class="ms-wifacts" id="probe-whatif-delta"></p>
              <p class="ms-card__note ms-witablenote" id="probe-whatif-tablenote" hidden>scatterer parameters at the probed pixel, base &rarr; perturbed &mdash; red marks a moved value</p>
              <table class="ms-slots" id="probe-whatif-slots"></table>
              <div class="ms-subhead">
                <h4 class="cube-panel-title">Response sweep</h4>
                <span class="ms-card__note">sweep this perturbation's strength and re-run at each step &mdash; the dashed marker is the strength dialled in above</span>
              </div>
              <canvas id="probe-sweep" width="640" height="200" hidden></canvas>
              <p class="ms-wifacts" id="probe-sweep-facts"></p>
            </section>
          </div>
        </div>
      </div>`;

    this.refs = {
      strip      : this.root.querySelector("#probe-strip"),
      hint       : this.root.querySelector("#probe-hint"),
      progress   : this.root.querySelector("#probe-progress"),
      fill       : this.root.querySelector("#probe-progress-fill"),
      label      : this.root.querySelector("#probe-progress-label"),
      stage      : this.root.querySelector("#probe-stage"),
      badge      : this.root.querySelector("#probe-model-badge"),
      meta       : this.root.querySelector("#probe-meta"),
      mapImg     : this.root.querySelector("#probe-map-img"),
      marks      : this.root.querySelector("#probe-marks"),
      coords     : this.root.querySelector("#probe-coords"),
      azInput    : this.root.querySelector("#probe-az"),
      rgInput    : this.root.querySelector("#probe-rg"),
      goBtn      : this.root.querySelector("#probe-go"),
      idle       : this.root.querySelector("#probe-idle"),
      cards      : this.root.querySelector("#probe-cards"),
      profile    : this.root.querySelector("#probe-profile"),
      slots      : this.root.querySelector("#probe-slots"),
      fit        : this.root.querySelector("#probe-fit"),
      matchNote  : this.root.querySelector("#probe-match-note"),
      match      : this.root.querySelector("#probe-match"),
      fieldSlots : this.root.querySelector("#probe-fields-slots"),
      fieldMaps  : this.root.querySelector("#probe-fields-maps"),
      fieldOver  : this.root.querySelector("#probe-fields-overview"),
      attribNote : this.root.querySelector("#probe-attrib-note"),
      attribSlots: this.root.querySelector("#probe-attrib-slots"),
      bars       : this.root.querySelector("#probe-bars"),
      barsLegend : this.root.querySelector("#probe-bars-legend"),
      gridChips  : this.root.querySelector("#probe-grid-channels"),
      grid       : this.root.querySelector("#probe-grid"),
      ablation   : this.root.querySelector("#probe-ablation"),
      occlusion  : this.root.querySelector("#probe-occlusion"),
      occlFacts  : this.root.querySelector("#probe-occl-facts"),
      flips      : this.root.querySelector("#probe-flips"),
      flipsFacts : this.root.querySelector("#probe-flips-facts"),
      arch       : this.root.querySelector("#probe-arch"),
      archLayers : this.root.querySelector("#probe-arch-layers"),
      features   : this.root.querySelector("#probe-features"),
      featNote   : this.root.querySelector("#probe-feat-note"),
      vitalsSum  : this.root.querySelector("#probe-vitals-summary"),
      vitals     : this.root.querySelector("#probe-vitals"),
      wKind      : this.root.querySelector("#probe-whatif-kind"),
      wDesc      : this.root.querySelector("#probe-whatif-desc"),
      wChRow     : this.root.querySelector("#probe-whatif-chrow"),
      wChannels  : this.root.querySelector("#probe-whatif-channels"),
      wChNa      : this.root.querySelector("#probe-whatif-chna"),
      wValRow    : this.root.querySelector("#probe-whatif-valrow"),
      wValCtl    : this.root.querySelector("#probe-whatif-valctl"),
      wValNa     : this.root.querySelector("#probe-whatif-valna"),
      wValueLabel: this.root.querySelector("#probe-whatif-value-label"),
      wRange     : this.root.querySelector("#probe-whatif-range"),
      wValue     : this.root.querySelector("#probe-whatif-value"),
      wReroll    : this.root.querySelector("#probe-whatif-reroll"),
      wCanvas    : this.root.querySelector("#probe-whatif"),
      wDelta     : this.root.querySelector("#probe-whatif-delta"),
      wTableNote : this.root.querySelector("#probe-whatif-tablenote"),
      wSlots     : this.root.querySelector("#probe-whatif-slots"),
      sweep      : this.root.querySelector("#probe-sweep"),
      sweepFacts : this.root.querySelector("#probe-sweep-facts"),
    };

    this.refs.mapImg.addEventListener("click", (ev) => this._onMapClick(ev));
    this.refs.goBtn.addEventListener("click", () => this._probeInputs());
    this.refs.azInput.addEventListener("keydown", (ev) => { if (ev.key === "Enter") this._probeInputs(); });
    this.refs.rgInput.addEventListener("keydown", (ev) => { if (ev.key === "Enter") this._probeInputs(); });
    this.refs.features.addEventListener("load", () => { this.refs.features.hidden = false; });
    this.refs.features.addEventListener("error", () => {
      this.refs.features.hidden = true;
      this.refs.featNote.textContent = `${this.layer} captured no activation at this pixel`;
    });

    this.refs.wKind.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.refs.wKind.querySelectorAll(".cube-space").forEach((b) => b.classList.toggle("is-active", b === btn));
        this.wi.kind = btn.dataset.kind;
        this._syncWhatIfControls();
        this._scheduleWhatIf();
      });
    });

    this.refs.wRange.addEventListener("input", () => {
      this.refs.wValue.value = this.refs.wRange.value;
      this._readWhatIfValue();
      this._renderWhatIfSummary();
      this._renderSweepChart();
      this._scheduleWhatIf();
    });
    this.refs.wValue.addEventListener("change", () => {
      this.refs.wRange.value = this.refs.wValue.value;
      this._readWhatIfValue();
      this._renderWhatIfSummary();
      this._renderSweepChart();
      this._scheduleWhatIf();
    });
    this.refs.wReroll.addEventListener("click", () => {
      this.wi.seed += 1;
      this._renderWhatIfSummary();
      this._runWhatIf();
    });

    this._syncWhatIfControls();
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
    const base = this._runsBase();
    const data = await window.apiGet(`/api/probe/runs?base=${encodeURIComponent(base)}`);

    if (!data || !data.ok) {
      this.refs.hint.textContent = (data && data.error) || "Backend unreachable.";
      return;
    }

    this.runs = data.runs || [];
    this._renderStrip();

    if (!this.runs.length) {
      this.refs.hint.textContent = `No loadable runs under ${data.root} (a loadable run holds best_model.pt and meta/model_config.json or meta/dual_model_config.json). Train a backbone or dual run, or point the runs directory in the Results tab at the right place.`;
      return;
    }

    if (!this.info) this.refs.hint.textContent = "Pick a trained backbone or dual run to put under the microscope.";
  }

  _renderStrip() {
    if (!this.runStrip) {
      this.runStrip = new RunStrip(this.refs.strip, {
        stateFor : (run) => run.id === this.selectedId,
        onPick   : (run) => this._load(run.id),
      });
    }
    this.runStrip.render(this.runs);
  }

  async _load(path) {
    const out = await window.apiPost("/api/probe/load", { path });
    if (!out.ok) {
      this.refs.hint.textContent = out.error || "load failed";
      return;
    }

    this.selectedId = path;
    this._renderStrip();

    this.refs.progress.hidden = false;
    this._pollStatus();
  }

  async _pollStatus() {
    const token = ++this.pollToken;
    const tick  = async () => {
      if (token !== this.pollToken) return;

      const res = await fetch("/api/probe/status");
      const st  = await res.json();

      if (st.state === "loading") {
        this.refs.progress.hidden  = false;
        this.refs.fill.style.width = `${Math.round((st.progress || 0) * 100)}%`;
        this.refs.label.textContent = st.stage || "loading";
        setTimeout(tick, 500);
        return;
      }

      this.refs.progress.hidden = true;

      if (st.state === "error") {
        this.refs.hint.textContent = st.error;
        return;
      }
      if (st.state === "ready" && st.info) {
        if (st.path && st.path !== this.selectedId) {
          this.selectedId = st.path;
          this._renderStrip();
        }
        this._onReady(st.info);
      }
    };
    tick();
  }

  async _onReady(info) {
    this.info        = info;
    this.pixel       = null;
    this.lastPredict = null;
    this.attrib      = null;
    this.attribSlot  = -1;
    this.gridChannel = -1;
    this.fieldsData  = null;
    this.fieldsSlot  = 0;
    this.sweepData   = null;
    this.vitalsData  = null;
    this.wi.channel  = 0;
    this.wi.seed     = 0;

    this.refs.stage.hidden = false;
    this.refs.idle.hidden  = false;
    this.refs.cards.hidden = true;
    this.refs.marks.innerHTML = "";

    this.refs.hint.textContent   = `Loaded ${info.backbone} — probing the ${info.split} split.`;
    this.refs.badge.textContent  = info.backbone;
    this.refs.coords.textContent = "Click the map to probe a pixel.";
    this.refs.mapImg.src         = `/api/probe/map?t=${Date.now()}`;

    this.refs.azInput.max = info.azimuth_size - 1;
    this.refs.rgInput.max = info.range_size - 1;

    this.refs.meta.innerHTML = [
      `<span class="ms-chip"><b>K</b> ${info.n_gaussians}</span>`,
      `<span class="ms-chip"><b>in</b> ${info.in_channels} ch</span>`,
      `<span class="ms-chip"><b>region</b> ${info.azimuth_size}&times;${info.range_size} px</span>`,
      `<span class="ms-chip"><b>window</b> ${info.patch[0]}&times;${info.patch[1]} px</span>`,
      `<span class="ms-chip"><b>split</b> ${this._esc(info.split)}</span>`,
    ].join("");

    this._buildWhatIfChannels(info.channels);
    this._syncWhatIfControls();

    const layers = await (await fetch("/api/probe/layers")).json();
    if (layers.ok) this._buildArch(layers.layers);
  }

  _chip(label, active, onPick) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "ms-chipbtn" + (active ? " is-active" : "");
    chip.textContent = label;
    chip.addEventListener("click", onPick);
    return chip;
  }

  _buildWhatIfChannels(channels) {
    this.refs.wChannels.innerHTML = "";
    channels.forEach((label, index) => {
      const chip = this._chip(label, index === this.wi.channel, () => {
        this.wi.channel = index;
        this.refs.wChannels.querySelectorAll(".ms-chipbtn").forEach((b, i) => b.classList.toggle("is-active", i === index));
        this._renderWhatIfSummary();
        this._scheduleWhatIf();
      });
      this.refs.wChannels.appendChild(chip);
    });
  }

  _blockKey(name) {
    const parts = name.split(".");
    return parts.length > 1 && /^\d+$/.test(parts[1]) ? `${parts[0]}.${parts[1]}` : parts[0];
  }

  _layerCategory(type) {
    const t = type.toLowerCase();
    if (t.includes("conv")) return "conv";
    if (t.includes("norm")) return "norm";
    if (/relu|gelu|silu|elu|sigmoid|tanh|softmax|mish|swish|activation/.test(t)) return "act";
    if (/linear|attention|embed|mlp/.test(t)) return "dense";
    return "aux";
  }

  _buildArch(layers) {
    this.blocks     = [];
    this.layerTypes = {};

    const index = new Map();
    layers.forEach((layer) => {
      this.layerTypes[layer.name] = layer.type;
      const key = this._blockKey(layer.name);
      if (!index.has(key)) {
        index.set(key, { key, layers: [] });
        this.blocks.push(index.get(key));
      }
      index.get(key).layers.push(layer);
    });

    this.block = this.blocks.length ? this.blocks[0].key : "";
    this.layer = this.blocks.length ? this.blocks[0].layers[0].name : "";

    this._renderArch();
    this._renderArchLayers();
    this._noteFeatureLayer();
  }

  _renderArch() {
    this.refs.arch.innerHTML = "";
    this.blocks.forEach((block) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ms-arch__block" + (block.key === this.block ? " is-active" : "");
      btn.innerHTML = `<span class="ms-arch__name">${this._esc(block.key)}</span><span class="ms-arch__count">${block.layers.length} layer${block.layers.length === 1 ? "" : "s"}</span>`;
      btn.addEventListener("click", () => {
        this.block = block.key;
        this.layer = block.layers[0].name;
        this._renderArch();
        this._renderArchLayers();
        this._noteFeatureLayer();
        this._loadFeatures();
        this._renderVitals();
      });
      this.refs.arch.appendChild(btn);
    });
  }

  _renderArchLayers() {
    this.refs.archLayers.innerHTML = "";
    const block = this.blocks.find((b) => b.key === this.block);
    if (!block) return;

    block.layers.forEach((layer) => {
      const short = layer.name === block.key ? layer.name : layer.name.slice(block.key.length + 1);
      const chip  = document.createElement("button");
      chip.type = "button";
      chip.className = "ms-arch__layer" + (layer.name === this.layer ? " is-active" : "");
      chip.dataset.cat = this._layerCategory(layer.type);
      chip.title = layer.name;
      chip.innerHTML = `${this._esc(short)} <em>${this._esc(layer.type)}</em>`;
      chip.addEventListener("click", () => {
        this.layer = layer.name;
        this._renderArchLayers();
        this._noteFeatureLayer();
        this._loadFeatures();
        this._renderVitals();
      });
      this.refs.archLayers.appendChild(chip);
    });
  }

  _noteFeatureLayer() {
    if (!this.layer) {
      this.refs.featNote.textContent = "";
      return;
    }
    const position = Object.keys(this.layerTypes).indexOf(this.layer) + 1;
    this.refs.featNote.textContent = `${this.layer} (${this.layerTypes[this.layer]}) — layer ${position}/${Object.keys(this.layerTypes).length}, 16 strongest channels over the probed window`;
  }

  _onMapClick(ev) {
    if (!this.info) return;
    const rect = this.refs.mapImg.getBoundingClientRect();
    const az   = Math.round(((ev.clientY - rect.top) / rect.height) * (this.info.azimuth_size - 1));
    const rg   = Math.round(((ev.clientX - rect.left) / rect.width) * (this.info.range_size - 1));
    this._probe(az, rg);
  }

  _probeInputs() {
    const az = parseInt(this.refs.azInput.value, 10);
    const rg = parseInt(this.refs.rgInput.value, 10);
    if (Number.isFinite(az) && Number.isFinite(rg)) this._probe(az, rg);
  }

  async _probe(az, rg) {
    az = Math.max(0, Math.min(this.info.azimuth_size - 1, az));
    rg = Math.max(0, Math.min(this.info.range_size - 1, rg));

    this.pixel = { az, rg };
    this.refs.azInput.value = az;
    this.refs.rgInput.value = rg;
    this.refs.coords.textContent = `az ${az} · rg ${rg} (absolute ${az + this.info.az_offset} · ${rg + this.info.rg_offset})`;
    this._renderMark(az, rg);

    const out = await window.apiPost("/api/probe/predict", { az, rg });
    if (!out.ok) {
      this.refs.coords.textContent = out.error;
      return;
    }

    this.lastPredict       = out;
    this.refs.idle.hidden  = true;
    this.refs.cards.hidden = false;

    this._renderProfile(out);
    this._renderSlots(out);
    this._renderFit(out);
    this._renderMatches(out);
    this._renderAttribSlotChips();
    this._loadAttribution();
    this._loadAblation();
    this._loadOcclusion();
    this._loadFields();
    this._loadFlips();
    this._loadFeatures();
    this._loadVitals();
    this._runWhatIf();
  }

  _renderMark(az, rg) {
    const x = (rg / (this.info.range_size - 1)) * 100;
    const y = (az / (this.info.azimuth_size - 1)) * 100;
    this.refs.marks.innerHTML = `<i class="fl-mark" style="left:${x}%;top:${y}%;background:#e11d48"></i>`;
  }

  _renderProfile(out) {
    const series = [
      { values: out.raw_curve, color: MicroscopeView.SERIES_COLORS.raw,  width: 1.0, label: "raw" },
      { values: out.gt_curve,  color: MicroscopeView.SERIES_COLORS.gt,   width: 1.4, label: "GT" },
      { values: out.curve,     color: MicroscopeView.SERIES_COLORS.pred, width: 1.8, label: "pred" },
    ].filter((s) => Array.isArray(s.values));

    window.drawLineChart(this.refs.profile, out.x_axis, series);
  }

  _slotRow(slot, keyLabel, rowClass, stateLabel) {
    return `
      <tr class="${rowClass}">
        <td class="ms-slots__key">${this._esc(keyLabel)}</td>
        <td>${slot.amp.toFixed(3)}</td>
        <td>${slot.mu.toFixed(2)}</td>
        <td>${slot.sigma.toFixed(2)}</td>
        <td class="ms-slots__state">${stateLabel}</td>
      </tr>`;
  }

  _renderSlots(out) {
    const head = `<thead><tr><th class="ms-slots__key">SLOT</th><th>AMP</th><th>&mu; [m]</th><th>&sigma; [m]</th><th class="ms-slots__state">STATE</th></tr></thead>`;
    const pred = out.slots.map((s) => this._slotRow(s, `${s.slot}`, s.active ? "" : "is-inactive", s.active ? "active" : "off"));
    const gt   = (out.gt_slots || []).filter((s) => s.active).map((s) => this._slotRow(s, `GT ${s.slot}`, "ms-slots__gt", "truth"));

    this.refs.slots.innerHTML = `${head}<tbody>${pred.join("")}${gt.join("")}</tbody>`;
  }

  _renderFit(out) {
    if (!out.fit) {
      this.refs.fit.innerHTML = "";
      return;
    }

    const parts = [];
    if (out.fit.curve_mse_gt !== null) parts.push(`curve MSE vs GT <b>${out.fit.curve_mse_gt.toExponential(3)}</b>`);
    parts.push(`curve MSE vs raw tomogram <b>${out.fit.curve_mse_raw.toExponential(3)}</b>`);

    const active = out.fit.gt_active === null
      ? `${out.fit.pred_active}/${this.info.n_gaussians} predicted slots active (no ground truth on this split)`
      : `${out.fit.pred_active}/${this.info.n_gaussians} predicted slots active vs ${out.fit.gt_active} in the ground truth`;

    this.refs.fit.innerHTML = `${parts.join(" &middot; ")} &middot; ${active}`;
  }

  _renderMatches(out) {
    const rows = out.matches || [];
    this.refs.matchNote.hidden = !rows.length;

    if (!rows.length) {
      this.refs.match.innerHTML = "";
      return;
    }

    const head = `<thead><tr><th class="ms-slots__key">GT SLOT</th><th>PRED SLOT</th><th>&Delta; AMP</th><th>&Delta; &mu; [m]</th><th>&Delta; &sigma; [m]</th></tr></thead>`;
    const body = rows.map((m) => {
      const cells = [["d_amp", 3, 0.1], ["d_mu", 2, 0.5], ["d_sigma", 2, 0.5]].map(([key, digits, eps]) => {
        const missed = Math.abs(m[key]) > eps;
        const signed = `${m[key] >= 0 ? "+" : ""}${m[key].toFixed(digits)}`;
        return `<td><span class="ms-shift${missed ? " is-moved" : ""}">${signed}</span></td>`;
      }).join("");
      return `<tr><td class="ms-slots__key">GT ${m.gt_slot}</td><td>${m.pred_slot}</td>${cells}</tr>`;
    });

    this.refs.match.innerHTML = `${head}<tbody>${body.join("")}</tbody>`;
  }

  _renderAttribSlotChips() {
    this.refs.attribSlots.innerHTML = "";

    const pick = (slot) => () => {
      if (slot === this.attribSlot) return;
      this.attribSlot = slot;
      this._renderAttribSlotChips();
      this._loadAttribution();
    };

    this.refs.attribSlots.appendChild(this._chip("all scatterers", this.attribSlot === -1, pick(-1)));

    ((this.lastPredict && this.lastPredict.slots) || []).forEach((slot) => {
      const label = `slot ${slot.slot} · μ ${slot.mu.toFixed(1)} m${slot.active ? "" : " · off"}`;
      const chip  = this._chip(label, this.attribSlot === slot.slot, pick(slot.slot));
      if (!slot.active) chip.classList.add("is-off");
      this.refs.attribSlots.appendChild(chip);
    });
  }

  _noteAttribution(out) {
    const scope = out.slot < 0 ? `summed over all ${this.info.n_gaussians} scatterer slots` : `scatterer slot ${out.slot} only`;
    this.refs.attribNote.innerHTML = `share of |&part;output/&part;input| per input channel, ${scope}`;
  }

  async _loadAttribution() {
    const token = ++this.attribToken;

    this.refs.bars.innerHTML       = `<p class="cube-hint is-loading">computing gradients&hellip;</p>`;
    this.refs.barsLegend.innerHTML = "";
    this.refs.gridChips.innerHTML  = "";
    this.refs.grid.innerHTML       = "";

    const out = await window.apiPost("/api/probe/attribution", { az: this.pixel.az, rg: this.pixel.rg, slot: this.attribSlot });
    if (token !== this.attribToken) return;

    if (!out.ok) {
      this.attrib = null;
      this.refs.bars.innerHTML = `<p class="cube-hint">${this._esc(out.error)}</p>`;
      return;
    }

    this.attrib = out;
    this._noteAttribution(out);
    this._renderBars(out);
    this._renderGridChips();
    this._renderGrid();
  }

  _renderBars(out) {
    const peak = Math.max(1e-9, ...out.families.filter((f) => !f.dead).flatMap((f) => f.shares));

    this.refs.bars.innerHTML = out.channels.map((label, index) => {
      const cols = out.families.map((f) => {
        const share  = f.shares[index];
        const height = f.dead ? 0 : Math.max(2, (share / peak) * 100);
        const title  = `${MicroscopeView.FAMILY_LABELS[f.family]} · ${label}: ${(share * 100).toFixed(1)}%`;
        return `<i style="height:${height.toFixed(1)}%;background:${MicroscopeView.FAMILY_COLORS[f.family]}" title="${this._esc(title)}"></i>`;
      }).join("");
      return `<div class="ms-bars__group"><div class="ms-bars__cols">${cols}</div><span class="ms-bars__label">${this._esc(label)}</span></div>`;
    }).join("");

    this.refs.barsLegend.innerHTML = out.families.map((f) =>
      `<span><i style="background:${MicroscopeView.FAMILY_COLORS[f.family]}"></i>${f.family} — ${MicroscopeView.FAMILY_LABELS[f.family]}${f.dead ? " (no dependence at this pixel)" : ""}</span>`
    ).join("");
  }

  _renderGridChips() {
    this.refs.gridChips.innerHTML = "";

    const pick = (index) => () => {
      this.gridChannel = index;
      this._renderGridChips();
      this._renderGrid();
    };

    this.refs.gridChips.appendChild(this._chip("all channels", this.gridChannel === -1, pick(-1)));
    this.attrib.channels.forEach((label, index) => {
      this.refs.gridChips.appendChild(this._chip(label, this.gridChannel === index, pick(index)));
    });
  }

  _dot(center, patch) {
    const left = (((center[1] + 0.5) / patch[1]) * 100).toFixed(1);
    const top  = (((center[0] + 0.5) / patch[0]) * 100).toFixed(1);
    return `<i class="ms-cell__dot" style="left:${left}%;top:${top}%"></i>`;
  }

  _gridCell(family, label, cell, share, head) {
    const cap = head
      ? `<figcaption class="ms-cellhead"><i style="background:${MicroscopeView.FAMILY_COLORS[family]}"></i>${family} &middot; ${(share * 100).toFixed(1)}%</figcaption>`
      : "";
    if (!cell) {
      return `<figure class="ms-cell">${cap}<div class="ms-cell--dead"><span>0</span></div></figure>`;
    }
    const img = `
      <span class="ms-cell__frame">
        <img src="data:image/png;base64,${cell}" alt="${this._esc(`${family} sensitivity to ${label}`)}" />
        ${this._dot(this.attrib.center, this.attrib.patch)}
      </span>`;
    return head
      ? `<figure class="ms-cell">${cap}${img}</figure>`
      : `<figure class="ms-cell">${img}<figcaption>${(share * 100).toFixed(1)}%</figcaption></figure>`;
  }

  _renderGrid() {
    const out = this.attrib;

    if (this.gridChannel >= 0) {
      const index = this.gridChannel;
      const label = out.channels[index];

      this.refs.grid.className = "ms-gridone";
      this.refs.grid.style.gridTemplateColumns = "";
      this.refs.grid.innerHTML = out.families.map((f) => this._gridCell(f.family, label, f.cells[index], f.shares[index], true)).join("");
      return;
    }

    this.refs.grid.className = "ms-grid";
    this.refs.grid.style.gridTemplateColumns = `92px repeat(${out.channels.length}, minmax(58px, 150px))`;

    const cells = [`<span class="ms-grid__corner"></span>`];
    out.channels.forEach((label) => cells.push(`<span class="ms-grid__col">${this._esc(label)}</span>`));

    out.families.forEach((f) => {
      cells.push(`<span class="ms-grid__row"><i style="background:${MicroscopeView.FAMILY_COLORS[f.family]}"></i>${f.family}</span>`);
      out.channels.forEach((label, index) => cells.push(this._gridCell(f.family, label, f.cells[index], f.shares[index], false)));
    });

    this.refs.grid.innerHTML = cells.join("");
  }

  async _loadAblation() {
    const token = ++this.ablToken;

    this.refs.ablation.innerHTML = `<p class="cube-hint is-loading">re-running the model without each channel&hellip;</p>`;

    const out = await window.apiPost("/api/probe/ablation", { az: this.pixel.az, rg: this.pixel.rg });
    if (token !== this.ablToken) return;

    if (!out.ok) {
      this.refs.ablation.innerHTML = `<p class="cube-hint">${this._esc(out.error)}</p>`;
      return;
    }

    const peak = Math.max(1e-12, ...out.channels.map((c) => c.delta_mse));

    this.refs.ablation.innerHTML = out.channels.map((c) => {
      const width = Math.max(1.5, (c.delta_mse / peak) * 100).toFixed(1);
      const rel   = out.base_power > 0 ? ` &middot; ${((c.delta_mse / out.base_power) * 100).toFixed(2)}% of power` : "";
      const flips = c.flips ? ` &middot; ${c.flips} slot${c.flips === 1 ? "" : "s"} flip` : "";
      const mu    = c.max_mu_shift > 0.5 ? ` &middot; &mu; moves ${c.max_mu_shift.toFixed(1)} m` : "";
      return `
        <div class="ms-abl__row">
          <span class="ms-abl__label" title="${this._esc(c.label)}">${this._esc(c.label)}</span>
          <span class="ms-abl__bar"><span class="ms-abl__track"><i style="width:${width}%"></i></span><em>curve MSE ${c.delta_mse.toExponential(2)}${rel}${flips}${mu}</em></span>
        </div>`;
    }).join("");
  }

  _fmtVal(v) {
    if (v === 0) return "0";
    const abs = Math.abs(v);
    return abs >= 1e4 || abs < 1e-2 ? v.toExponential(1) : Number(v.toPrecision(3)).toString();
  }

  async _loadOcclusion() {
    const token = ++this.occlToken;

    this.refs.occlFacts.innerHTML = "";
    this.refs.occlusion.innerHTML = `<p class="cube-hint is-loading">sliding an occluder across the window&hellip;</p>`;

    const out = await window.apiPost("/api/probe/occlusion", { az: this.pixel.az, rg: this.pixel.rg });
    if (token !== this.occlToken) return;

    if (!out.ok) {
      this.refs.occlusion.innerHTML = `<p class="cube-hint">${this._esc(out.error)}</p>`;
      return;
    }

    this.refs.occlusion.innerHTML = `
      <figure class="ms-field ms-field--wide">
        <span class="ms-cell__frame">
          <img src="data:image/png;base64,${out.map.png}" alt="Occlusion sensitivity map" />
          ${this._dot(out.center_cell, out.grid)}
        </span>
        <figcaption>&Delta; curve MSE per occluded cell &middot; ${this._fmtVal(out.map.min)} &rarr; ${this._fmtVal(out.map.max)}</figcaption>
      </figure>`;

    const dAz  = (out.worst.row + 0.5) * out.occluder[0] - out.center[0];
    const dRg  = (out.worst.col + 0.5) * out.occluder[1] - out.center[1];
    const dist = Math.hypot(dAz, dRg);
    const rel  = out.base_power > 0 ? ` (${((out.worst.delta_mse / out.base_power) * 100).toFixed(2)}% of the base curve's power)` : "";

    this.refs.occlFacts.innerHTML = `occluder ${out.occluder[0]}&times;${out.occluder[1]} px &middot; peak damage <b>${out.worst.delta_mse.toExponential(2)}</b>${rel} about ${dist.toFixed(0)} px from the probed pixel`;
  }

  async _loadFields() {
    const token = ++this.fieldsToken;

    this.refs.fieldSlots.innerHTML = "";
    this.refs.fieldOver.innerHTML  = "";
    this.refs.fieldMaps.innerHTML  = `<p class="cube-hint is-loading">computing parameter fields&hellip;</p>`;

    const out = await window.apiPost("/api/probe/fields", { az: this.pixel.az, rg: this.pixel.rg });
    if (token !== this.fieldsToken) return;

    if (!out.ok) {
      this.fieldsData = null;
      this.refs.fieldMaps.innerHTML = `<p class="cube-hint">${this._esc(out.error)}</p>`;
      return;
    }

    this.fieldsData   = out;
    const firstActive = out.slots.find((s) => s.active);
    this.fieldsSlot   = firstActive ? firstActive.slot : 0;

    this._renderFieldsChips();
    this._renderFieldsMaps();
  }

  _renderFieldsChips() {
    this.refs.fieldSlots.innerHTML = "";

    const pick = (slot) => () => {
      if (slot === this.fieldsSlot) return;
      this.fieldsSlot = slot;
      this._renderFieldsChips();
      this._renderFieldsMaps();
    };

    this.fieldsData.slots.forEach((slot) => {
      const predicted = (this.lastPredict && this.lastPredict.slots[slot.slot]) || null;
      const mu        = predicted ? ` · μ ${predicted.mu.toFixed(1)} m` : "";
      const chip      = this._chip(`slot ${slot.slot}${mu}${slot.active ? "" : " · off"}`, this.fieldsSlot === slot.slot, pick(slot.slot));
      if (!slot.active) chip.classList.add("is-off");
      this.refs.fieldSlots.appendChild(chip);
    });
  }

  _fieldFigure(payload, title, rangeText) {
    const range = rangeText || `${this._fmtVal(payload.min)} &rarr; ${this._fmtVal(payload.max)}`;
    return `
      <figure class="ms-field">
        <span class="ms-cell__frame">
          <img src="data:image/png;base64,${payload.png}" alt="${this._esc(title)}" />
          ${this._dot(this.fieldsData.center, this.fieldsData.patch)}
        </span>
        <figcaption>${title} &middot; ${range}</figcaption>
      </figure>`;
  }

  _renderFieldsMaps() {
    const out  = this.fieldsData;
    const slot = out.slots.find((s) => s.slot === this.fieldsSlot) || out.slots[0];

    this.refs.fieldMaps.innerHTML = slot.families.map((f) =>
      this._fieldFigure(f, `${MicroscopeView.FAMILY_LABELS[f.family]}${f.family === "amp" ? "" : " [m]"}`)
    ).join("");

    const over = [this._fieldFigure(out.activity, "active scatterers", `${this._fmtVal(out.activity.min)} &rarr; ${this._fmtVal(out.activity.max)} of ${this.info.n_gaussians}`)];
    if (out.error) over.push(this._fieldFigure(out.error, "curve MSE vs GT"));
    else over.push(`<p class="ms-ctlrow__na">no ground truth on this split &mdash; the local error map needs GT parameters</p>`);

    this.refs.fieldOver.innerHTML = over.join("");
  }

  async _loadFlips() {
    const token = ++this.flipToken;

    this.refs.flipsFacts.innerHTML = `<span class="cube-hint is-loading">re-running the model on flipped windows&hellip;</span>`;

    const out = await window.apiPost("/api/probe/flips", { az: this.pixel.az, rg: this.pixel.rg });
    if (token !== this.flipToken) return;

    if (!out.ok) {
      this.refs.flipsFacts.innerHTML = this._esc(out.error);
      return;
    }

    const styles = {
      none    : { color: MicroscopeView.SERIES_COLORS.pred,      width: 1.8, label: "original" },
      azimuth : { color: "#0f766e",                              width: 1.2, label: "az flip" },
      range   : { color: "#a16207",                              width: 1.2, label: "rg flip" },
      both    : { color: MicroscopeView.SERIES_COLORS.perturbed, width: 1.2, label: "both" },
    };

    window.drawLineChart(this.refs.flips, out.x_axis, out.entries.map((e) => ({ values: e.curve, ...styles[e.flip] })));

    const worst = Math.max(...out.entries.map((e) => e.delta_mse));
    const parts = out.entries.filter((e) => e.flip !== "none").map((e) => `${styles[e.flip].label} &Delta; <b>${e.delta_mse.toExponential(2)}</b>`);
    const rel   = out.base_power > 0 ? ` &mdash; worst is ${((worst / out.base_power) * 100).toFixed(2)}% of the base curve's power` : "";

    this.refs.flipsFacts.innerHTML = `${parts.join(" &middot; ")}${rel}`;
  }

  _loadFeatures() {
    if (!this.pixel || !this.layer) return;
    this.refs.features.src = `/api/probe/features?az=${this.pixel.az}&rg=${this.pixel.rg}&layer=${encodeURIComponent(this.layer)}&t=${Date.now()}`;
  }

  async _loadVitals() {
    const token = ++this.vitalsToken;

    this.refs.vitalsSum.innerHTML = `<span class="cube-hint is-loading">recording activation statistics&hellip;</span>`;
    this.refs.vitals.innerHTML    = "";

    const out = await window.apiPost("/api/probe/vitals", { az: this.pixel.az, rg: this.pixel.rg });
    if (token !== this.vitalsToken) return;

    if (!out.ok) {
      this.refs.vitalsSum.innerHTML = this._esc(out.error);
      return;
    }

    this.vitalsData = out;
    this._renderVitals();
  }

  _renderVitals() {
    const out = this.vitalsData;
    if (!out) return;

    const s       = out.summary;
    const flagged = s.flagged ? `<b>${s.flagged} flagged</b> (nonfinite, mostly zero, or &gt;25% dead channels)` : "none flagged";
    this.refs.vitalsSum.innerHTML = `${s.n_layers} live layers &middot; ${s.total_params.toLocaleString("en-US")} parameters &middot; ${flagged}`;

    const head = `<thead><tr>
      <th>#</th><th class="ms-slots__key">LAYER</th><th class="ms-vitals__type">TYPE</th><th>OUT</th><th>PARAMS</th>
      <th>ZERO %</th><th>DEAD CH</th><th>EFF CH %</th><th>MAX |ACT|</th><th class="ms-vitals__flags">FLAGS</th>
    </tr></thead>`;

    const rows = out.entries.map((e, index) => {
      const dead    = e.dead === null ? "&ndash;" : `${e.dead}/${e.channels}`;
      const eff     = e.eff_frac === null ? "&ndash;" : (e.eff_frac * 100).toFixed(0);
      const classes = [e.flags.length ? "is-flagged" : "", e.name === this.layer ? "is-current" : ""].filter(Boolean).join(" ");
      return `
        <tr class="${classes}" data-layer="${this._esc(e.name)}">
          <td>${index + 1}</td>
          <td class="ms-slots__key" title="${this._esc(e.name)}">${this._esc(e.name)}</td>
          <td class="ms-vitals__type">${this._esc(e.type)}</td>
          <td>${(e.shape || []).join("&times;")}</td>
          <td>${e.params.toLocaleString("en-US")}</td>
          <td>${(e.zero_frac * 100).toFixed(1)}</td>
          <td>${dead}</td>
          <td>${eff}</td>
          <td>${this._fmtVal(e.max_abs)}</td>
          <td class="ms-vitals__flags">${this._esc(e.flags.join(", "))}</td>
        </tr>`;
    });

    this.refs.vitals.innerHTML = `${head}<tbody>${rows.join("")}</tbody>`;

    this.refs.vitals.querySelectorAll("tr[data-layer]").forEach((row) => {
      row.addEventListener("click", () => {
        const name = row.dataset.layer;
        if (!(name in this.layerTypes)) return;
        this.block = this._blockKey(name);
        this.layer = name;
        this._renderArch();
        this._renderArchLayers();
        this._noteFeatureLayer();
        this._loadFeatures();
        this._renderVitals();
      });
    });
  }

  _renderWhatIfSummary() {
    if (!this.info) {
      this.refs.wDesc.innerHTML = "";
      return;
    }

    const channel = this._esc(this.info.channels[this.wi.channel] || `ch ${this.wi.channel}`);
    const window  = `${this.info.patch[0]}&times;${this.info.patch[1]}`;

    if (this.wi.kind === "drop_channel") {
      this.refs.wDesc.innerHTML = `Testing: zero out <b>${channel}</b> across the whole ${window} input window, then re-run the model.`;
    } else if (this.wi.kind === "scale_channel") {
      this.refs.wDesc.innerHTML = `Testing: multiply <b>${channel}</b> by <b>${this.wi.factor.toFixed(2)}</b> across the whole ${window} input window (1.00 leaves it untouched), then re-run the model.`;
    } else {
      this.refs.wDesc.innerHTML = `Testing: add Gaussian noise with &sigma; = <b>${this.wi.sigma.toFixed(2)}</b> (normalized units) to every channel of the ${window} input window, then re-run the model &mdash; noise draw #${this.wi.seed + 1}.`;
    }
  }

  _syncWhatIfControls() {
    const kind = this.wi.kind;

    const noChannel = kind === "noise";
    this.refs.wChRow.classList.toggle("is-off", noChannel);
    this.refs.wChannels.hidden = noChannel;
    this.refs.wChNa.hidden     = !noChannel;

    const noValue = kind === "drop_channel";
    this.refs.wValRow.classList.toggle("is-off", noValue);
    this.refs.wValCtl.hidden = noValue;
    this.refs.wValNa.hidden  = !noValue;

    this.refs.wReroll.hidden          = kind !== "noise";
    this.refs.wValueLabel.textContent = kind === "noise" ? "sigma" : (kind === "scale_channel" ? "factor" : "strength");

    const value = kind === "noise" ? this.wi.sigma : this.wi.factor;
    this.refs.wRange.value = value;
    this.refs.wValue.value = value;

    this._renderWhatIfSummary();
  }

  _readWhatIfValue() {
    const value = parseFloat(this.refs.wValue.value);
    if (!Number.isFinite(value)) return;
    if (this.wi.kind === "noise") this.wi.sigma = value;
    else this.wi.factor = value;
  }

  _scheduleWhatIf() {
    if (!this.pixel) return;
    clearTimeout(this.wiTimer);
    this.wiTimer = setTimeout(() => this._runWhatIf(), 250);
  }

  async _runWhatIf() {
    if (!this.pixel) return;

    const kind         = this.wi.kind;
    const perturbation = { kind };
    if (kind !== "noise") perturbation.channel = this.wi.channel;
    if (kind === "scale_channel") perturbation.factor = this.wi.factor;
    if (kind === "noise") {
      perturbation.sigma = this.wi.sigma;
      perturbation.seed  = this.wi.seed;
    }

    const token = ++this.wiToken;
    const out   = await window.apiPost("/api/probe/whatif", { az: this.pixel.az, rg: this.pixel.rg, perturbation });
    if (token !== this.wiToken) return;

    if (!out.ok) {
      this.refs.wDelta.textContent  = out.error;
      this.refs.wSlots.innerHTML    = "";
      this.refs.wTableNote.hidden   = true;
      return;
    }

    const series = [];
    if (this.lastPredict && Array.isArray(this.lastPredict.gt_curve)) {
      series.push({ values: this.lastPredict.gt_curve, color: MicroscopeView.SERIES_COLORS.raw, width: 1.0, label: "GT" });
    }
    series.push({ values: out.base_curve,      color: MicroscopeView.SERIES_COLORS.pred,      width: 1.6, label: "base" });
    series.push({ values: out.perturbed_curve, color: MicroscopeView.SERIES_COLORS.perturbed, width: 1.6, label: "perturbed" });

    window.drawLineChart(this.refs.wCanvas, out.x_axis, series);

    const power = out.base_curve.reduce((acc, v) => acc + v * v, 0) / out.base_curve.length;
    const rel   = power > 0 ? ` &mdash; ${((out.delta_mse / power) * 100).toFixed(2)}% of the base curve's power` : "";
    this.refs.wDelta.innerHTML = `prediction shift: curve MSE <b>${out.delta_mse.toExponential(3)}</b>${rel}`;

    this.refs.wTableNote.hidden = false;
    this._renderWhatIfSlots(out.base_slots, out.perturbed_slots);
    this._ensureSweep();
  }

  async _ensureSweep() {
    if (this.wi.kind === "drop_channel") {
      this.sweepData = null;
      this.refs.sweep.hidden = true;
      this.refs.sweepFacts.innerHTML = `<span class="ms-ctlrow__na">zeroing a channel is all-or-nothing &mdash; pick the scale or noise perturbation to sweep a strength</span>`;
      return;
    }

    const key = [this.pixel.az, this.pixel.rg, this.wi.kind, this.wi.kind === "scale_channel" ? this.wi.channel : "all"].join(":");
    if (this.sweepData && this.sweepData.key === key) {
      this._renderSweepChart();
      return;
    }

    const token = ++this.sweepToken;
    this.refs.sweep.hidden = true;
    this.refs.sweepFacts.innerHTML = `<span class="cube-hint is-loading">sweeping perturbation strengths&hellip;</span>`;

    const body = { az: this.pixel.az, rg: this.pixel.rg, kind: this.wi.kind };
    if (this.wi.kind === "scale_channel") body.channel = this.wi.channel;

    const out = await window.apiPost("/api/probe/sweep", body);
    if (token !== this.sweepToken) return;

    if (!out.ok) {
      this.refs.sweepFacts.innerHTML = this._esc(out.error);
      return;
    }

    out.key                = key;
    this.sweepData         = out;
    this.refs.sweep.hidden = false;
    this._renderSweepChart();
    this._renderSweepFacts();
  }

  _renderSweepChart() {
    if (!this.sweepData || this.refs.sweep.hidden) return;

    const marker = this.wi.kind === "noise" ? this.wi.sigma : this.wi.factor;

    window.drawLineChart(
      this.refs.sweep,
      this.sweepData.points.map((p) => p.value),
      [{ values: this.sweepData.points.map((p) => p.delta_mse), color: MicroscopeView.SERIES_COLORS.perturbed, width: 1.6, label: "Δ curve MSE" }],
      { xUnit: "", marker },
    );
  }

  _renderSweepFacts() {
    const points = this.sweepData.points;
    const peak   = points.reduce((best, p) => (p.delta_mse > best.delta_mse ? p : best), points[0]);
    const flip   = points.find((p) => p.flips > 0);
    const sym    = this.wi.kind === "noise" ? "&sigma;" : "factor";
    const rel    = this.sweepData.base_power > 0 ? ` (${((peak.delta_mse / this.sweepData.base_power) * 100).toFixed(2)}% of power)` : "";
    const flips  = flip ? `slots first flip at ${sym} ${flip.value.toFixed(2)}` : "no slot flips across the sweep";

    this.refs.sweepFacts.innerHTML = `peak &Delta; curve MSE <b>${peak.delta_mse.toExponential(2)}</b>${rel} at ${sym} ${peak.value.toFixed(2)} &middot; ${flips}`;
  }

  _renderWhatIfSlots(base, perturbed) {
    const head = `<thead><tr><th class="ms-slots__key">SLOT</th><th>AMP</th><th>&mu; [m]</th><th>&sigma; [m]</th></tr></thead>`;

    const rows = base.map((b, k) => {
      const p     = perturbed[k];
      const cells = [["amp", 3, 0.005], ["mu", 2, 0.05], ["sigma", 2, 0.05]].map(([param, digits, eps]) => {
        const moved = Math.abs(p[param] - b[param]) > eps;
        return `<td>${b[param].toFixed(digits)} <span class="ms-shift${moved ? " is-moved" : ""}">&rarr; ${p[param].toFixed(digits)}</span></td>`;
      }).join("");
      return `<tr class="${b.active || p.active ? "" : "is-inactive"}"><td class="ms-slots__key">${k}</td>${cells}</tr>`;
    });

    this.refs.wSlots.innerHTML = `${head}<tbody>${rows.join("")}</tbody>`;
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }
}

window.drawLineChart = (canvas, xAxis, series, opts = {}) => {
  if (!canvas.dataset.chartW) {
    canvas.dataset.chartW = canvas.width;
    canvas.dataset.chartH = canvas.height;
  }

  const baseW = parseInt(canvas.dataset.chartW, 10);
  const baseH = parseInt(canvas.dataset.chartH, 10);
  const dpr   = window.devicePixelRatio || 1;

  canvas.style.width  = "100%";
  canvas.style.height = `${baseH}px`;

  const W = canvas.clientWidth || baseW;
  const H = baseH;

  canvas.width  = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  if (!all.length) return;

  const lo = Math.min(0, ...all);
  const hi = Math.max(...all) * 1.05 || 1;
  const x0 = xAxis[0];
  const x1 = xAxis[xAxis.length - 1];

  const pad = { l: 52, r: 12, t: 10, b: 24 };
  const px  = (x) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const py  = (v) => H - pad.b - ((v - lo) / (hi - lo)) * (H - pad.t - pad.b);

  const fmt = (v) => {
    if (v === 0) return "0";
    const abs = Math.abs(v);
    return abs >= 1e4 || abs < 1e-2 ? v.toExponential(1) : Number(v.toPrecision(3)).toString();
  };

  ctx.font = "10px JetBrains Mono, monospace";

  for (let tick = 0; tick <= 4; tick += 1) {
    const v = lo + ((hi - lo) * tick) / 4;
    const y = py(v);

    ctx.strokeStyle = "rgba(22, 25, 27, 0.07)";
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(W - pad.r, y);
    ctx.stroke();

    ctx.fillStyle = "rgba(125, 133, 139, 0.95)";
    ctx.textAlign = "right";
    ctx.fillText(fmt(v), pad.l - 6, y + 3);
  }

  const xUnit = opts.xUnit === undefined ? " m" : (opts.xUnit ? ` ${opts.xUnit}` : "");

  [[x0, "left"], [(x0 + x1) / 2, "center"], [x1, "right"]].forEach(([x, align]) => {
    ctx.fillStyle = "rgba(125, 133, 139, 0.95)";
    ctx.textAlign = align;
    ctx.fillText(`${fmt(x)}${xUnit}`, px(x), H - 8);
  });

  ctx.strokeStyle = "rgba(154, 161, 150, 0.8)";
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, H - pad.b);
  ctx.lineTo(W - pad.r, H - pad.b);
  ctx.stroke();

  series.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth   = s.width;
    ctx.lineJoin    = "round";
    ctx.beginPath();
    let pen = false;
    s.values.forEach((v, i) => {
      if (!Number.isFinite(v)) {
        pen = false;
        return;
      }
      const x = px(xAxis[i]);
      const y = py(v);
      if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      pen = true;
    });
    ctx.stroke();
  });

  if (Number.isFinite(opts.marker) && opts.marker >= Math.min(x0, x1) && opts.marker <= Math.max(x0, x1)) {
    ctx.strokeStyle = "#e11d48";
    ctx.lineWidth   = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(px(opts.marker), pad.t);
    ctx.lineTo(px(opts.marker), H - pad.b);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.textAlign = "left";
  const entryWidths = series.map((s) => 16 + ctx.measureText(s.label).width + 14);
  let legendX = W - pad.r - entryWidths.reduce((acc, w) => acc + w, 0);
  const legendY = pad.t + 8;

  series.forEach((s, index) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth   = 2;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY);
    ctx.lineTo(legendX + 10, legendY);
    ctx.stroke();

    ctx.fillStyle = "rgba(63, 71, 76, 0.95)";
    ctx.fillText(s.label, legendX + 14, legendY + 3);
    legendX += entryWidths[index];
  });
};

window.MicroscopeView = MicroscopeView;
