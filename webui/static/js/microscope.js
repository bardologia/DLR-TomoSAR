"use strict";

class MicroscopeView {
  static DEFAULT_RUN = "/ste/rnd/User/vice_vi/DLR-TomoSAR/runs/backbone";
  static FAMILIES    = ["amp", "mu", "sigma"];

  constructor(root) {
    this.root      = root;
    this.info      = null;
    this.pixel     = null;
    this.family    = "mu";
    this.layer     = "";
    this.built     = false;
    this.pollToken = 0;
  }

  enter() {
    if (!this.built) {
      this._build();
      this.built = true;
    }
    this._pollStatus();
  }

  _build() {
    this.root.innerHTML = `
      <div class="cube-pick">
        <div class="fl-src" role="group" aria-label="Run directory">
          <label for="probe-run">Run dir</label>
          <input id="probe-run" type="text" spellcheck="false" autocomplete="off" placeholder="${MicroscopeView.DEFAULT_RUN}" />
          <button type="button" class="btn btn--mini" id="probe-load">Load</button>
        </div>
        <p class="cube-hint" id="probe-hint">Point at a training run directory holding best_model.pt, then load it.</p>
        <div class="cube-progress" id="probe-progress" hidden>
          <div class="cube-progress__track"><i class="cube-progress__fill" id="probe-progress-fill"></i></div>
          <p class="cube-progress__label" id="probe-progress-label">loading&hellip;</p>
        </div>
      </div>

      <div class="fl-stage" id="probe-stage" hidden>
        <div class="fl-deck">
          <aside class="fl-map">
            <div class="fl-map__head">
              <h3 class="cube-panel-title">Pixel</h3>
              <span class="cube-coords" id="probe-model-badge"></span>
            </div>
            <div class="fl-map__frame">
              <span class="cube-axis cube-axis--y">range &rarr;</span>
              <div class="fl-map__wrap" id="probe-map-wrap">
                <img id="probe-map-img" alt="Primary SLC amplitude" />
                <div class="fl-marks" id="probe-marks"></div>
              </div>
              <span class="cube-axis cube-axis--x">azimuth &rarr;</span>
            </div>
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
          </aside>

          <div class="fl-results">
            <div class="fl-card">
              <h3 class="cube-panel-title">Predicted profile</h3>
              <canvas id="probe-profile" width="640" height="240"></canvas>
              <div id="probe-slots" class="cube-hint"></div>
            </div>

            <div class="fl-card">
              <div class="fl-map__head">
                <h3 class="cube-panel-title">Input attribution</h3>
                <div class="cube-spaces" id="probe-family" role="group" aria-label="Output family">
                  ${MicroscopeView.FAMILIES.map((f) => `<button type="button" class="cube-space${f === "mu" ? " is-active" : ""}" data-family="${f}">${f}</button>`).join("")}
                </div>
              </div>
              <div id="probe-shares"></div>
              <canvas id="probe-saliency" width="320" height="200"></canvas>
              <p class="cube-hint">Spatial |&part;output/&part;input| around the probed pixel, summed over channels.</p>
            </div>

            <div class="fl-card">
              <div class="fl-map__head">
                <h3 class="cube-panel-title">Feature maps</h3>
                <select class="lb-select" id="probe-layer" aria-label="Layer"></select>
              </div>
              <img id="probe-features" alt="Feature maps" style="max-width:100%" hidden />
            </div>

            <div class="fl-card">
              <div class="fl-map__head">
                <h3 class="cube-panel-title">What-if</h3>
                <div class="cube-spaces" id="probe-whatif-kind" role="group" aria-label="Perturbation">
                  <button type="button" class="cube-space is-active" data-kind="drop_channel">drop</button>
                  <button type="button" class="cube-space" data-kind="scale_channel">scale</button>
                  <button type="button" class="cube-space" data-kind="noise">noise</button>
                </div>
              </div>
              <div class="fl-pixrow" role="group" aria-label="Perturbation parameters">
                <span class="cube-jump__cell" id="probe-whatif-channel-cell">
                  <label for="probe-whatif-channel">channel</label>
                  <select class="lb-select" id="probe-whatif-channel"></select>
                </span>
                <span class="cube-jump__cell" id="probe-whatif-value-cell" hidden>
                  <label for="probe-whatif-value" id="probe-whatif-value-label">factor</label>
                  <input type="number" id="probe-whatif-value" class="cube-jump__input" step="0.1" value="0.5" />
                </span>
                <button type="button" class="btn btn--mini" id="probe-whatif-run">Perturb</button>
              </div>
              <canvas id="probe-whatif" width="640" height="240"></canvas>
              <p class="cube-hint" id="probe-whatif-delta"></p>
            </div>
          </div>
        </div>
      </div>`;

    this.refs = {
      runInput   : this.root.querySelector("#probe-run"),
      loadBtn    : this.root.querySelector("#probe-load"),
      hint       : this.root.querySelector("#probe-hint"),
      progress   : this.root.querySelector("#probe-progress"),
      fill       : this.root.querySelector("#probe-progress-fill"),
      label      : this.root.querySelector("#probe-progress-label"),
      stage      : this.root.querySelector("#probe-stage"),
      badge      : this.root.querySelector("#probe-model-badge"),
      mapWrap    : this.root.querySelector("#probe-map-wrap"),
      mapImg     : this.root.querySelector("#probe-map-img"),
      marks      : this.root.querySelector("#probe-marks"),
      coords     : this.root.querySelector("#probe-coords"),
      azInput    : this.root.querySelector("#probe-az"),
      rgInput    : this.root.querySelector("#probe-rg"),
      goBtn      : this.root.querySelector("#probe-go"),
      profile    : this.root.querySelector("#probe-profile"),
      slots      : this.root.querySelector("#probe-slots"),
      familyWrap : this.root.querySelector("#probe-family"),
      shares     : this.root.querySelector("#probe-shares"),
      saliency   : this.root.querySelector("#probe-saliency"),
      layerSel   : this.root.querySelector("#probe-layer"),
      features   : this.root.querySelector("#probe-features"),
      wKind      : this.root.querySelector("#probe-whatif-kind"),
      wChannel   : this.root.querySelector("#probe-whatif-channel"),
      wChannelCell: this.root.querySelector("#probe-whatif-channel-cell"),
      wValue     : this.root.querySelector("#probe-whatif-value"),
      wValueCell : this.root.querySelector("#probe-whatif-value-cell"),
      wValueLabel: this.root.querySelector("#probe-whatif-value-label"),
      wRun       : this.root.querySelector("#probe-whatif-run"),
      wCanvas    : this.root.querySelector("#probe-whatif"),
      wDelta     : this.root.querySelector("#probe-whatif-delta"),
    };

    this.refs.runInput.value = MicroscopeView.DEFAULT_RUN;
    this.refs.loadBtn.addEventListener("click", () => this._load());
    this.refs.runInput.addEventListener("keydown", (ev) => { if (ev.key === "Enter") this._load(); });
    this.refs.mapImg.addEventListener("click", (ev) => this._onMapClick(ev));
    this.refs.goBtn.addEventListener("click", () => this._probeInputs());
    this.refs.layerSel.addEventListener("change", () => this._loadFeatures());
    this.refs.wRun.addEventListener("click", () => this._runWhatIf());

    this.refs.familyWrap.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.refs.familyWrap.querySelectorAll(".cube-space").forEach((b) => b.classList.toggle("is-active", b === btn));
        this.family = btn.dataset.family;
        if (this.pixel) this._loadSaliency();
      });
    });

    this.refs.wKind.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.refs.wKind.querySelectorAll(".cube-space").forEach((b) => b.classList.toggle("is-active", b === btn));
        this._syncWhatIfControls(btn.dataset.kind);
      });
    });
  }

  _syncWhatIfControls(kind) {
    const needsChannel = kind !== "noise";
    this.refs.wChannelCell.hidden = !needsChannel;
    this.refs.wValueCell.hidden   = kind === "drop_channel";
    this.refs.wValueLabel.textContent = kind === "noise" ? "sigma" : "factor";
    if (kind === "noise") this.refs.wValue.value = "0.5";
  }

  async _load() {
    const path = this.refs.runInput.value.trim();
    if (!path) return;

    const res = await fetch("/api/probe/load", {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ path }),
    });
    const out = await res.json();
    if (!out.ok) {
      this.refs.hint.textContent = out.error || "load failed";
      return;
    }

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
        this._onReady(st.info);
      }
    };
    tick();
  }

  async _onReady(info) {
    this.info = info;
    this.refs.stage.hidden = false;
    this.refs.hint.textContent = `${info.backbone} · ${info.in_channels} channels · K=${info.n_gaussians} · ${info.azimuth_size}×${info.range_size} px (${info.split})`;
    this.refs.badge.textContent = info.backbone;
    this.refs.mapImg.src = `/api/probe/map?t=${Date.now()}`;

    this.refs.wChannel.innerHTML = info.channels.map((c, i) => `<option value="${i}">${this._esc(c)}</option>`).join("");

    const layers = await (await fetch("/api/probe/layers")).json();
    if (layers.ok) {
      this.refs.layerSel.innerHTML = layers.layers.map((l) => `<option value="${this._esc(l.name)}">${this._esc(l.name)} (${this._esc(l.type)})</option>`).join("");
      this.layer = layers.layers.length ? layers.layers[0].name : "";
    }
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
    this.pixel = { az, rg };
    this.refs.azInput.value = az;
    this.refs.rgInput.value = rg;
    this.refs.coords.textContent = `probing az ${az}, rg ${rg}`;
    this._renderMark(az, rg);

    const res = await fetch("/api/probe/predict", {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ az, rg }),
    });
    const out = await res.json();
    if (!out.ok) {
      this.refs.coords.textContent = out.error;
      return;
    }

    this._renderProfile(out);
    this._renderSlots(out);
    this._loadSaliency();
    this._loadFeatures();
    this.refs.wDelta.textContent = "";
  }

  _renderMark(az, rg) {
    const x = (rg / (this.info.range_size - 1)) * 100;
    const y = (az / (this.info.azimuth_size - 1)) * 100;
    this.refs.marks.innerHTML = `<i class="fl-mark" style="left:${x}%;top:${y}%;background:#e11d48"></i>`;
  }

  _renderProfile(out) {
    const series = [
      { values: out.raw_curve, color: "#9ca3af", width: 1.0, label: "raw" },
      { values: out.gt_curve,  color: "#111827", width: 1.4, label: "GT" },
      { values: out.curve,     color: "#1d4fd8", width: 1.8, label: "pred" },
    ].filter((s) => Array.isArray(s.values));

    this._lineChart(this.refs.profile, out.x_axis, series);
  }

  _renderSlots(out) {
    const rows = out.slots.map((s) =>
      `slot ${s.slot}: a=${s.amp.toFixed(3)} μ=${s.mu.toFixed(2)} σ=${s.sigma.toFixed(2)}${s.active ? "" : " (inactive)"}`
    );
    const gt = (out.gt_slots || []).filter((s) => s.active).map((s) =>
      `gt: a=${s.amp.toFixed(3)} μ=${s.mu.toFixed(2)} σ=${s.sigma.toFixed(2)}`
    );
    this.refs.slots.innerHTML = rows.concat(gt).map((r) => this._esc(r)).join("<br>");
  }

  async _loadSaliency() {
    if (!this.pixel) return;

    const res = await fetch("/api/probe/saliency", {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ az: this.pixel.az, rg: this.pixel.rg, family: this.family }),
    });
    const out = await res.json();

    if (!out.ok) {
      this.refs.shares.innerHTML = `<p class="cube-hint">${this._esc(out.error)}</p>`;
      const ctx = this.refs.saliency.getContext("2d");
      ctx.clearRect(0, 0, this.refs.saliency.width, this.refs.saliency.height);
      return;
    }

    const top = out.channels
      .map((label, i) => ({ label, share: out.shares[i] }))
      .sort((a, b) => b.share - a.share);

    this.refs.shares.innerHTML = top.map((row) => `
      <div class="lb-bar-row" style="display:flex;align-items:center;gap:8px;margin:2px 0">
        <span style="flex:0 0 130px;font-size:11px;text-align:right">${this._esc(row.label)}</span>
        <span style="flex:1;background:rgba(29,79,216,.12);height:12px;position:relative">
          <i style="position:absolute;inset:0;width:${(row.share * 100).toFixed(1)}%;background:#1d4fd8"></i>
        </span>
        <span style="flex:0 0 52px;font-size:11px">${(row.share * 100).toFixed(1)}%</span>
      </div>`).join("");

    this._heatmap(this.refs.saliency, out.map, out.center);
  }

  _loadFeatures() {
    if (!this.pixel || !this.layer) return;
    this.layer = this.refs.layerSel.value || this.layer;
    this.refs.features.hidden = false;
    this.refs.features.src = `/api/probe/features?az=${this.pixel.az}&rg=${this.pixel.rg}&layer=${encodeURIComponent(this.layer)}&t=${Date.now()}`;
  }

  async _runWhatIf() {
    if (!this.pixel) return;

    const kind         = this.refs.wKind.querySelector(".is-active").dataset.kind;
    const perturbation = { kind };
    if (kind !== "noise") perturbation.channel = parseInt(this.refs.wChannel.value, 10);
    if (kind === "scale_channel") perturbation.factor = parseFloat(this.refs.wValue.value);
    if (kind === "noise") perturbation.sigma = parseFloat(this.refs.wValue.value);

    const res = await fetch("/api/probe/whatif", {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ az: this.pixel.az, rg: this.pixel.rg, perturbation }),
    });
    const out = await res.json();

    if (!out.ok) {
      this.refs.wDelta.textContent = out.error;
      return;
    }

    this._lineChart(this.refs.wCanvas, out.x_axis, [
      { values: out.base_curve,      color: "#1d4fd8", width: 1.6, label: "base" },
      { values: out.perturbed_curve, color: "#b91c1c", width: 1.6, label: "perturbed" },
    ]);
    this.refs.wDelta.textContent = `curve MSE shift: ${out.delta_mse.toExponential(3)}`;
  }

  _lineChart(canvas, xAxis, series) {
    window.drawLineChart(canvas, xAxis, series);
  }

  _heatmap(canvas, map, center) {
    const ctx  = canvas.getContext("2d");
    const rows = map.length;
    const cols = map[0].length;
    const cw   = canvas.width / cols;
    const ch   = canvas.height / rows;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const v = Math.pow(Math.max(0, Math.min(1, map[r][c])), 0.5);
        ctx.fillStyle = `rgba(29, 79, 216, ${v.toFixed(3)})`;
        ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }

    if (center) {
      ctx.strokeStyle = "#e11d48";
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      ctx.arc((center[1] + 0.5) * cw, (center[0] + 0.5) * ch, 5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }
}

window.drawLineChart = (canvas, xAxis, series) => {
  const ctx = canvas.getContext("2d");
  const W   = canvas.width;
  const H   = canvas.height;
  const pad = { l: 42, r: 10, t: 8, b: 22 };

  ctx.clearRect(0, 0, W, H);

  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  if (!all.length) return;

  const lo = Math.min(0, ...all);
  const hi = Math.max(...all) * 1.05 || 1;
  const x0 = xAxis[0];
  const x1 = xAxis[xAxis.length - 1];

  const px = (x) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const py = (v) => H - pad.b - ((v - lo) / (hi - lo)) * (H - pad.t - pad.b);

  ctx.strokeStyle = "rgba(120,130,150,.35)";
  ctx.lineWidth   = 1;
  ctx.strokeRect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b);

  ctx.fillStyle = "rgba(110,120,140,.9)";
  ctx.font      = "10px JetBrains Mono, monospace";
  ctx.fillText(hi.toPrecision(3), 4, pad.t + 10);
  ctx.fillText(lo.toPrecision(3), 4, H - pad.b);
  ctx.fillText(`${x0.toFixed(0)}m`, pad.l, H - 6);
  ctx.fillText(`${x1.toFixed(0)}m`, W - pad.r - 34, H - 6);

  series.forEach((s, index) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth   = s.width;
    ctx.beginPath();
    s.values.forEach((v, i) => {
      const x = px(xAxis[i]);
      const y = py(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = s.color;
    ctx.fillText(s.label, W - pad.r - 60, pad.t + 12 + index * 12);
  });
};

window.MicroscopeView = MicroscopeView;
