"use strict";

class TomogramView {
  static LABELS = { pred: "pred", gt: "gt", reduced: "capon reduced", full: "capon full" };

  constructor(refs) {
    this.strip = refs.strip;
    this.stage = refs.stage;
    this.deck = refs.deck;
    this.topdown = refs.topdown;
    this.cross = refs.cross;
    this.coords = refs.coords;
    this.back = refs.back;
    this.hint = refs.hint;
    this.panels = refs.panels;
    this.slicesEl = refs.slices;
    this.slicesAt = refs.slicesAt;
    this.sourcesEl = refs.sources;
    this.profilesEl = refs.profiles;
    this.profAt = refs.profAt;
    this.profModeBtns = refs.profModeBtns;
    this.profPanels = refs.profPanels;
    this.spaceBtns = refs.spaceBtns || [];
    this.progress = refs.progress;
    this.progressFill = refs.progressFill;
    this.progressLabel = refs.progressLabel;

    this.atLabels = {
      range   : this.slicesEl.querySelector('.cube-cutgroup__at[data-axis="range"]'),
      azimuth : this.slicesEl.querySelector('.cube-cutgroup__at[data-axis="azimuth"]'),
    };

    this.cubes = [];
    this.selectedId = null;
    this.meta = null;
    this.space = "physical";
    this.point = null;
    this.mode = "map";
    this.locked = null;
    this.entered = false;
    this.polling = false;
    this.profMode = "raw";
    this.profData = null;
    this.profQueued = null;
    this.profFetching = false;
    this.colors = {};
    this.visible = new Set();

    this.mapWrap = this.topdown.closest(".cube-map__wrap");

    this.topdown.addEventListener("mousemove", (ev) => this._onMove(ev));
    this.topdown.addEventListener("click", (ev) => this._onClick(ev));
    this.topdown.addEventListener("load", () => this.mapWrap.classList.remove("is-loading"));
    this.topdown.addEventListener("error", () => this.mapWrap.classList.remove("is-loading"));

    this.back.addEventListener("click", () => this._exitSlices());
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && this.mode === "slices") this._exitSlices();
    });

    this.panels.forEach((panel) => {
      panel.canvas.addEventListener("mousemove", (ev) => this._onSliceMove(panel, ev));
    });

    this.spaceBtns.forEach((btn) => {
      btn.addEventListener("click", () => this._setSpace(btn.dataset.space));
    });
    this.profModeBtns.forEach((btn) => {
      btn.addEventListener("click", () => this._setProfMode(btn.dataset.mode));
    });
  }

  async enter() {
    if (this.entered) return;
    this.entered = true;
    await this.refresh();
  }

  async refresh() {
    this.hint.hidden = false;
    this.hint.textContent = "Loading saved cubes…";
    this.hint.classList.add("is-loading");

    let data;
    try {
      data = await window.apiGet(`/api/cubes?base=${encodeURIComponent(this._runsBase())}`);
    } catch (e) {
      this.hint.classList.remove("is-loading");
      this.hint.textContent = "Backend unreachable.";
      return;
    }

    this.hint.classList.remove("is-loading");
    this.cubes = data.cubes || [];
    this._renderStrip();

    if (!this.cubes.length) {
      this.hint.textContent = data.error || "No saved cubes found. Run an inference with save_cubes=True first.";
      this.hint.hidden = false;
      this.stage.hidden = true;
      return;
    }

    this.hint.textContent = "Select a cube directory to load it into memory.";
    this.hint.hidden = false;
  }

  _runsBase() {
    try {
      const raw = JSON.parse(localStorage.getItem("results-sources") || "{}");
      return raw.logs || "";
    } catch (e) {
      return "";
    }
  }

  _renderStrip() {
    this.strip.innerHTML = "";
    this.cubes.forEach((cube) => {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "tb-pill" + (cube.id === this.selectedId ? " is-active" : "");
      pill.title = cube.id;
      pill.innerHTML =
        `<span class="tb-pill__name">${this._esc(cube.group)}/${this._esc(cube.run)}</span>` +
        `<span class="tb-pill__state">${this._esc(cube.stamp)}</span>`;
      pill.addEventListener("click", () => this.select(cube.id));
      this.strip.appendChild(pill);
    });
  }

  async select(cubeId) {
    if (this.polling) {
      window.toast("A cube is still loading.", "warn");
      return;
    }
    if (cubeId === this.selectedId && this.meta) return;

    this.selectedId = cubeId;
    this.meta = null;
    this.point = null;
    this.locked = null;
    this.mode = "map";
    this.panels.forEach((panel) => {
      panel.bitmap = null;
      panel.key = null;
      panel.drawnSpace = null;
      panel.marker = 0;
      panel.queued = null;
      panel.fetching = false;
    });
    this.profQueued = null;
    this.profData = null;
    this.deck.dataset.mode = "map";
    this.back.hidden = true;
    this.cross.hidden = true;
    this._hideRefs();
    this.coords.textContent = "Hover the image to cut every tomogram · click to lock the slices";
    this.slicesAt.textContent = "";
    this.profAt.textContent = "Hover a slice to read the profiles at that position.";
    if (this.atLabels.range)   this.atLabels.range.textContent   = "";
    if (this.atLabels.azimuth) this.atLabels.azimuth.textContent = "";
    this.stage.hidden = true;
    this.slicesEl.hidden = true;
    this.slicesEl.classList.remove("is-in");
    this.hint.hidden = true;
    this._renderStrip();

    const res = await window.apiPost("/api/cubes/load", { id: cubeId });
    if (!res.ok) {
      this.hint.textContent = res.error || "Cube load failed.";
      this.hint.hidden = false;
      return;
    }

    this._setProgress(0, "loading");
    this.progress.hidden = false;
    await this._poll();
  }

  async _poll() {
    this.polling = true;

    while (true) {
      let st;
      try {
        st = await window.apiGet("/api/cubes/status");
      } catch (e) {
        this._failLoad("Backend unreachable.");
        break;
      }

      if (st.id !== this.selectedId) {
        this.progress.hidden = true;
        break;
      }

      if (st.state === "loading") {
        this._setProgress(st.progress || 0, st.stage || "loading");
        await new Promise((r) => setTimeout(r, 400));
        continue;
      }

      if (st.state === "ready" && st.cube) {
        this._setProgress(1, "ready");
        this._display(st.cube);
        break;
      }

      this._failLoad(st.error || "Cube load failed.");
      break;
    }

    this.polling = false;
  }

  _failLoad(message) {
    this.progress.hidden = true;
    this.hint.textContent = message;
    this.hint.hidden = false;
  }

  _setProgress(frac, stage) {
    const pct = Math.max(0, Math.min(100, Math.round(frac * 100)));
    this.progressFill.style.width = `${pct}%`;
    const label = TomogramView.LABELS[stage] || stage;
    this.progressLabel.textContent = `${label} — ${pct}%`;
  }

  _display(meta) {
    this.meta = meta;
    this.progress.hidden = true;

    const css = getComputedStyle(this.stage);
    this.colors = {
      pred    : css.getPropertyValue("--src-pred").trim(),
      gt      : css.getPropertyValue("--src-gt").trim(),
      reduced : css.getPropertyValue("--src-reduced").trim(),
      full    : css.getPropertyValue("--src-full").trim(),
    };

    this._syncSpaceBtns();
    this._syncProfModeBtns();

    const kept = meta.sources.filter((s) => this.visible.has(s));
    this.visible = new Set(kept.length ? kept : meta.sources);
    this._renderSourceToggles();
    this._applyVisibility();

    this.hint.hidden = true;
    this.stage.hidden = false;
    this.mapWrap.classList.add("is-loading");
    this.topdown.src = `/api/cubes/primary?id=${encodeURIComponent(this.selectedId)}`;

    this._follow({ az: Math.floor(meta.n_az / 2), rg: Math.floor(meta.n_rg / 2), fx: 0.5, fy: 0.5 }, true);
  }

  _setSpace(space) {
    if (space === this.space || !["physical", "normalized"].includes(space)) return;
    this.space = space;
    this._syncSpaceBtns();

    if (!this.meta || !this.point) return;
    this._drawSlices(this.point.az, this.point.rg);
  }

  _setProfMode(mode) {
    if (mode === this.profMode || !["raw", "unit"].includes(mode)) return;
    this.profMode = mode;
    this._syncProfModeBtns();
    this._drawProfiles();
  }

  _renderSourceToggles() {
    this.sourcesEl.innerHTML = "";
    this.meta.sources.forEach((source) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "cube-source";
      btn.dataset.source = source;
      btn.innerHTML = `<i class="cube-dot" data-source="${source}"></i>${TomogramView.LABELS[source]}`;
      btn.addEventListener("click", () => this._toggleSource(source));
      this.sourcesEl.appendChild(btn);
    });
  }

  _toggleSource(source) {
    if (this.visible.has(source)) {
      if (this.visible.size === 1) {
        window.toast("At least one source must stay visible.", "warn");
        return;
      }
      this.visible.delete(source);
    } else {
      this.visible.add(source);
    }

    this._applyVisibility();
    if (this.point) this._drawSlices(this.point.az, this.point.rg);
    this._drawProfiles();
  }

  _applyVisibility() {
    this.panels.forEach((panel) => {
      panel.root.hidden = !this.visible.has(panel.source);
    });
    this.profPanels.forEach((panel) => {
      panel.root.hidden = !this.visible.has(panel.source);
    });
    this.sourcesEl.querySelectorAll(".cube-source").forEach((btn) => {
      btn.classList.toggle("is-active", this.visible.has(btn.dataset.source));
    });
    this.slicesEl.style.setProperty("--cube-rows", String(this.visible.size));
  }

  _syncSpaceBtns() {
    this.spaceBtns.forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.space === this.space);
    });
  }

  _syncProfModeBtns() {
    this.profModeBtns.forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.mode === this.profMode);
    });
  }

  _pointFromEvent(ev) {
    if (!this.meta) return null;

    const rect = this.topdown.getBoundingClientRect();
    const fx = (ev.clientX - rect.left) / rect.width;
    const fy = (ev.clientY - rect.top) / rect.height;

    return {
      az: Math.min(this.meta.n_az - 1, Math.max(0, Math.floor(fy * this.meta.n_az))),
      rg: Math.min(this.meta.n_rg - 1, Math.max(0, Math.floor(fx * this.meta.n_rg))),
      fx,
      fy,
    };
  }

  _onMove(ev) {
    if (this.mode !== "map") return;
    const point = this._pointFromEvent(ev);
    if (!point) return;
    this._follow(point);
  }

  _onClick(ev) {
    if (this.mode !== "map") return;
    const point = this._pointFromEvent(ev);
    if (!point) return;
    this._follow(point, true);
    this._enterSlices(point);
  }

  _enterSlices(point) {
    this.mode = "slices";
    this.locked = { az: point.az, rg: point.rg };
    this.deck.dataset.mode = "slices";
    this.back.hidden = false;
    this.slicesAt.textContent = `locked at az = ${point.az} · rg = ${point.rg} · hover a slice for profiles · Esc to go back`;
    this._queueProfiles(point.az, point.rg);
  }

  _exitSlices() {
    if (this.mode !== "slices") return;
    this.mode = "map";
    this.locked = null;
    this.deck.dataset.mode = "map";
    this.back.hidden = true;
    this.slicesAt.textContent = "";
    this._hideRefs();
  }

  _hideRefs() {
    this.panels.forEach((panel) => {
      if (panel.ref) panel.ref.hidden = true;
    });
  }

  _onSliceMove(panel, ev) {
    if (this.mode !== "slices" || !this.meta || !this.locked) return;

    const rect = panel.canvas.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
    const n    = panel.axis === "range" ? this.meta.n_az : this.meta.n_rg;
    const idx  = Math.min(n - 1, Math.floor(frac * n));

    this.panels.forEach((p) => {
      if (!p.ref || p.root.hidden) return;
      if (p.axis === panel.axis) {
        p.ref.hidden = false;
        p.ref.style.left = `${frac * 100}%`;
      } else {
        p.ref.hidden = true;
      }
    });

    const az = panel.axis === "range" ? idx : this.locked.az;
    const rg = panel.axis === "range" ? this.locked.rg : idx;
    this._queueProfiles(az, rg);
  }

  _follow(point, force = false) {
    if (!force && this.point && point.az === this.point.az && point.rg === this.point.rg) {
      this._moveCross(point);
      return;
    }

    this.point = point;
    this._moveCross(point);
    this.coords.textContent = `az = ${point.az} · rg = ${point.rg} · click to lock`;

    this._drawSlices(point.az, point.rg);
  }

  _moveCross(point) {
    this.cross.hidden = false;
    this.cross.style.left = `${point.fx * 100}%`;
    this.cross.style.top = `${point.fy * 100}%`;
  }

  _revealSlices() {
    if (!this.slicesEl.hidden) return;
    this.slicesEl.hidden = false;
    requestAnimationFrame(() => requestAnimationFrame(() => this.slicesEl.classList.add("is-in")));
  }

  _drawSlices(az, rg) {
    if (!this.meta) return;

    this._revealSlices();

    if (this.atLabels.range)   this.atLabels.range.textContent   = `rg = ${rg}`;
    if (this.atLabels.azimuth) this.atLabels.azimuth.textContent = `az = ${az}`;

    this.panels.forEach((panel) => {
      if (panel.root.hidden) return;
      this._updatePanel(panel, az, rg);
    });
  }

  _updatePanel(panel, az, rg) {
    const key = panel.axis === "range" ? rg : az;
    panel.marker = panel.axis === "range" ? az / this.meta.n_az : rg / this.meta.n_rg;

    if (panel.bitmap && panel.key === key && panel.drawnSpace === this.space) {
      this._paintSlice(panel);
      return;
    }

    panel.queued = { az, rg, key };
    this._panelPump(panel);
  }

  async _panelPump(panel) {
    if (panel.fetching) return;
    panel.fetching = true;

    while (panel.queued) {
      const job = panel.queued;
      panel.queued = null;
      await this._fetchSlice(panel, job);
    }

    panel.fetching = false;
  }

  async _fetchSlice(panel, job) {
    const space = this.space;
    const url = `/api/cubes/slice?id=${encodeURIComponent(this.selectedId)}&source=${panel.source}&axis=${panel.axis}&az=${job.az}&rg=${job.rg}&space=${space}`;
    const skeletonTimer = panel.bitmap ? null : setTimeout(() => panel.root.classList.add("is-loading"), 120);

    try {
      const res = await fetch(url);
      if (!res.ok) return;

      panel.bitmap = await createImageBitmap(await res.blob());
      panel.key = job.key;
      panel.drawnSpace = space;
      this._paintSlice(panel);
    } catch (e) {
    } finally {
      if (skeletonTimer) clearTimeout(skeletonTimer);
      panel.root.classList.remove("is-loading");
    }
  }

  _paintSlice(panel) {
    const bitmap = panel.bitmap;
    const canvas = panel.canvas;

    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }

    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(panel.marker * canvas.width, 0);
    ctx.lineTo(panel.marker * canvas.width, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _queueProfiles(az, rg) {
    this.profAt.textContent = `profiles at az = ${az} · rg = ${rg}`;
    this.profQueued = { az, rg };
    this._profPump();
  }

  async _profPump() {
    if (this.profFetching || !this.profQueued) return;
    this.profFetching = true;

    while (this.profQueued) {
      const { az, rg } = this.profQueued;
      this.profQueued = null;
      await this._fetchProfiles(az, rg);
    }

    this.profFetching = false;
  }

  async _fetchProfiles(az, rg) {
    const staleTimer = setTimeout(() => {
      this.profPanels.forEach((panel) => panel.root.classList.add("is-stale"));
    }, 180);

    let data;
    try {
      data = await window.apiGet(`/api/cubes/profiles?id=${encodeURIComponent(this.selectedId)}&az=${az}&rg=${rg}`);
    } catch (e) {
      data = null;
    }

    clearTimeout(staleTimer);
    this.profPanels.forEach((panel) => panel.root.classList.remove("is-stale"));

    if (!data || !data.ok) return;
    this.profData = data;
    this._drawProfiles();
  }

  _drawProfiles() {
    if (!this.profData) return;

    let shared = null;
    if (this.profMode === "unit") {
      shared = 0;
      this.profPanels.forEach((panel) => {
        if (panel.root.hidden) return;
        const series = this.profData.sources[panel.source];
        if (series) shared = Math.max(shared, ...this._scaledValues(series));
      });
      if (!shared) shared = 1;
    }

    this.profPanels.forEach((panel) => {
      if (panel.root.hidden) return;
      const series = this.profData.sources[panel.source];
      if (series) this._drawProfile(panel, series, shared);
    });
  }

  _drawProfile(panel, series, shared) {
    const canvas = panel.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 240;
    const h = canvas.clientHeight || 190;

    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const heights = series.heights;
    const values = this._scaledValues(series);
    const n = values.length;
    if (n < 2) return;

    const padL = 8, padR = 8, padT = 16, padB = 8;
    const innerW = w - padL - padR;
    const innerH = h - padT - padB;
    const hMin = heights[0];
    const hMax = heights[n - 1];
    const span = hMax - hMin || 1;
    const vMax = shared || Math.max(...values) || 1;

    const xAt = (value) => padL + (value / vMax) * innerW * 0.96;
    const yAt = (height) => padT + (1 - (height - hMin) / span) * innerH;

    ctx.strokeStyle = "rgba(20, 25, 30, 0.07)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 3; i++) {
      const x = padL + (innerW * i) / 4;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.moveTo(xAt(0), yAt(heights[0]));
    for (let i = 0; i < n; i++) ctx.lineTo(xAt(values[i]), yAt(heights[i]));
    ctx.lineTo(xAt(0), yAt(heights[n - 1]));
    ctx.closePath();
    ctx.globalAlpha = 0.12;
    ctx.fillStyle = this.colors[panel.source] || "#555";
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      if (i === 0) ctx.moveTo(xAt(values[i]), yAt(heights[i]));
      else ctx.lineTo(xAt(values[i]), yAt(heights[i]));
    }
    ctx.strokeStyle = this.colors[panel.source] || "#555";
    ctx.lineWidth = 1.6;
    ctx.stroke();

    let peak = 0;
    for (let i = 1; i < n; i++) if (values[i] > values[peak]) peak = i;
    ctx.beginPath();
    ctx.arc(xAt(values[peak]), yAt(heights[peak]), 2.6, 0, Math.PI * 2);
    ctx.fillStyle = this.colors[panel.source] || "#555";
    ctx.fill();

    ctx.fillStyle = "rgba(20, 25, 30, 0.55)";
    ctx.font = "9.5px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText(this._fmt(hMax), padL + 2, padT - 5);
    ctx.fillText(this._fmt(hMin), padL + 2, h - padB - 3);
    ctx.textAlign = "right";
    ctx.fillText(`peak ${this._fmt(values[peak])} @ ${this._fmt(heights[peak])}`, w - 4, padT - 5);
  }

  _scaledValues(series) {
    const values = series.values.map((v) => (Number.isFinite(v) ? Math.max(v, 0) : 0));
    if (this.profMode === "raw") return values;

    let area = 0;
    for (let i = 1; i < values.length; i++) {
      const dh = Math.abs(series.heights[i] - series.heights[i - 1]);
      area += 0.5 * (values[i] + values[i - 1]) * dh;
    }

    if (area <= 0) return values.map(() => 0);
    return values.map((v) => v / area);
  }

  _fmt(value) {
    const abs = Math.abs(value);
    if (abs >= 1000) return value.toFixed(0);
    if (abs >= 10)   return value.toFixed(1);
    if (abs >= 0.01 || abs === 0) return value.toFixed(2);
    return value.toExponential(1);
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
}

window.TomogramView = TomogramView;
