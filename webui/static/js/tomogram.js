"use strict";

class TomogramSweep {
  constructor(refs, host) {
    this.host = host;
    this.axis = refs.axis;
    this.grid = refs.grid;
    this.atLabel = refs.at;
    this.fill = refs.fill;
    this.input = refs.input;
    this.rangeLabel = refs.range;
    this.playBtn = refs.play;
    this.panels = refs.panels || [];

    this.idx = 0;
    this.steps = 1;
    this.playing = false;
    this.token = 0;
    this.frameMs = 130;

    if (this.grid)    this.grid.addEventListener("wheel", (ev) => this._onWheel(ev), { passive: false });
    if (this.input)   this.input.addEventListener("change", () => this._onManual());
    if (this.playBtn) this.playBtn.addEventListener("click", () => this.toggle());
  }

  configure() {
    const meta = this.host.meta;
    this.steps = this._axisSteps(meta);
    this.idx   = Math.floor((this.steps - 1) / 2);

    if (this.input) {
      this.input.min = 1;
      this.input.max = this.steps;
      this.input.value = this.idx + 1;
    }
    if (this.rangeLabel) this.rangeLabel.textContent = this._rangeText(meta);

    this.panels.forEach((panel) => {
      panel.root.hidden = !meta.sources.includes(panel.source);
      panel.bitmap = null;
    });

    this.stop();
  }

  applyVisibility() {
    this.panels.forEach((panel) => {
      panel.root.hidden = !this.host.visible.has(panel.source);
    });
    if (this.host.meta && !this.playing && this.host.view === this.axis) this._renderFrame();
  }

  syncSpace() {
    if (this.host.meta && !this.playing) this._renderFrame();
  }

  render() {
    if (this.host.meta) this._renderFrame();
  }

  play() {
    if (!this.host.meta || this.playing) return;
    if (this.steps < 2) { this._renderFrame(); return; }
    this.playing = true;
    this._syncPlayBtn();
    this._loop();
  }

  stop() {
    this.playing = false;
    this._syncPlayBtn();
  }

  toggle() {
    if (this.playing) this.stop();
    else this.play();
  }

  async _loop() {
    while (this.playing && this.host.meta && this.host.selectedId) {
      await this._renderFrame();
      if (!this.playing) break;
      await this._sleep(this.frameMs);
      if (!this.playing) break;
      this.idx = (this.idx + 1) % this.steps;
    }
  }

  _onWheel(ev) {
    if (!this.host.meta) return;
    ev.preventDefault();
    this.stop();

    const step = ev.deltaY > 0 ? 1 : -1;
    const next = Math.min(this.steps - 1, Math.max(0, this.idx + step));
    if (next === this.idx) return;

    this.idx = next;
    this._renderFrame();
  }

  _onManual() {
    if (!this.host.meta || !this.input) return;
    this.stop();
    this.idx = this.host._clampInt(Number(this.input.value) - 1, this.steps);
    this._renderFrame();
  }

  _renderFrame() {
    if (!this.host.meta) return Promise.resolve();

    this.token += 1;
    const token = this.token;
    this._updateLabels();

    const jobs = this.panels.filter((panel) => !panel.root.hidden).map((panel) => this._fetch(panel, this.idx, token));
    return Promise.all(jobs);
  }

  _updateLabels() {
    const frac = this.steps > 1 ? this.idx / (this.steps - 1) : 0;

    if (this.fill) this.fill.style.width = `${frac * 100}%`;
    if (this.input && document.activeElement !== this.input) this.input.value = this.idx + 1;
    if (this.atLabel) this.atLabel.textContent = this._atText(frac);
  }

  async _fetch(panel, idx, token) {
    const url = this._url(panel.source, idx);
    const skeletonTimer = panel.bitmap ? null : setTimeout(() => panel.root.classList.add("is-loading"), 120);

    try {
      const res = await fetch(url);
      if (!res.ok) return;

      const bitmap = await createImageBitmap(await res.blob());
      if (token !== this.token) { if (bitmap.close) bitmap.close(); return; }

      panel.bitmap = bitmap;
      this._paint(panel);
    } catch (e) {
    } finally {
      if (skeletonTimer) clearTimeout(skeletonTimer);
      panel.root.classList.remove("is-loading");
    }
  }

  _paint(panel) {
    const bitmap = panel.bitmap;
    const canvas = panel.canvas;

    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }

    canvas.getContext("2d").drawImage(bitmap, 0, 0);
  }

  _url(source, idx) {
    const id    = encodeURIComponent(this.host.selectedId);
    const space = this.host.space;

    if (this.axis === "elevation") {
      const frac = this.steps > 1 ? idx / (this.steps - 1) : 0;
      return `/api/cubes/plane?id=${id}&source=${source}&frac=${frac}&space=${space}`;
    }
    if (this.axis === "azimuth") {
      return `/api/cubes/slice?id=${id}&source=${source}&axis=azimuth&az=${idx}&rg=0&space=${space}`;
    }
    return `/api/cubes/slice?id=${id}&source=${source}&axis=range&az=0&rg=${idx}&space=${space}`;
  }

  _axisSteps(meta) {
    if (this.axis === "elevation") {
      const primary = meta.sources.includes("pred") ? "pred" : meta.sources[0];
      return Math.max(1, meta.n_elev[primary] || 1);
    }
    if (this.axis === "azimuth") return Math.max(1, meta.n_az);
    return Math.max(1, meta.n_rg);
  }

  _rangeText(meta) {
    if (this.axis === "elevation") return `1–${this.steps} · ${this.host._fmt(meta.x_min)} … ${this.host._fmt(meta.x_max)}`;
    return `1–${this.steps} · index 0–${this.steps - 1}`;
  }

  _atText(frac) {
    if (this.axis === "elevation") {
      const height = this.host.meta.x_min + frac * (this.host.meta.x_max - this.host.meta.x_min);
      return `elevation ≈ ${this.host._fmt(height)} · bin ${this.idx + 1} / ${this.steps} · scroll or play to sweep`;
    }
    return `${this.axis} index ${this.idx} · bin ${this.idx + 1} / ${this.steps} · scroll or play to sweep`;
  }

  _syncPlayBtn() {
    if (!this.playBtn) return;
    this.playBtn.classList.toggle("is-playing", this.playing);
    this.playBtn.textContent = this.playing ? "Pause" : "Play";
  }

  _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

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
    this.profMetricsEl = refs.profMetrics;
    this.profModeBtns = refs.profModeBtns;
    this.profPanels = refs.profPanels;
    this.spaceBtns = refs.spaceBtns || [];
    this.modeBtns = refs.modeBtns || [];
    this.viewEls = refs.views || [];
    this.jumpAz = refs.jumpAz;
    this.jumpRg = refs.jumpRg;
    this.jumpGo = refs.jumpGo;
    this.jumpAzRange = refs.jumpAzRange;
    this.jumpRgRange = refs.jumpRgRange;
    this.progress = refs.progress;
    this.progressFill = refs.progressFill;
    this.progressLabel = refs.progressLabel;

    this.atLabels = {
      range   : this.slicesEl.querySelector('.cube-cutgroup__at[data-axis="range"]'),
      azimuth : this.slicesEl.querySelector('.cube-cutgroup__at[data-axis="azimuth"]'),
    };

    this.cubes = [];
    this.selectedId = null;
    this.openGroups = new Set();
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
    this.ssimQueued = null;
    this.ssimFetching = false;
    this.view = "explorer";
    this.colors = {};
    this.visible = new Set();

    this.sweeps = (refs.sweeps || []).map((sweep) => new TomogramSweep(sweep, this));

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
    this.modeBtns.forEach((btn) => {
      btn.addEventListener("click", () => this._setView(btn.dataset.view));
    });

    if (this.jumpAz) this.jumpAz.addEventListener("change", () => this._setManualCut());
    if (this.jumpRg) this.jumpRg.addEventListener("change", () => this._setManualCut());
    if (this.jumpGo) this.jumpGo.addEventListener("click", () => this._setManualCut());
  }

  leave() {
    this._stopSweeps();
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
      return raw.logs || ResultsView.DEFAULT_RUNS;
    } catch (e) {
      return ResultsView.DEFAULT_RUNS;
    }
  }

  _renderStrip() {
    this.strip.innerHTML = "";

    const groups = new Map();
    this.cubes.forEach((cube) => {
      if (!groups.has(cube.group)) groups.set(cube.group, []);
      groups.get(cube.group).push(cube);
    });

    const selectedGroup = (this.cubes.find((c) => c.id === this.selectedId) || {}).group;

    groups.forEach((cubes, group) => {
      const isOpen = this.openGroups.has(group) || group === selectedGroup;
      const label  = group === "." ? "runs" : group;

      const card = document.createElement("div");
      card.className = "cube-group" + (isOpen ? " is-open" : "");

      const head = document.createElement("button");
      head.type = "button";
      head.className = "cube-group__head";
      head.title = label;
      head.innerHTML =
        `<span class="cube-group__chev" aria-hidden="true"></span>` +
        `<span class="cube-group__name">${this._esc(label)}</span>` +
        `<span class="cube-group__count">${cubes.length}</span>`;
      head.addEventListener("click", () => this._toggleGroup(group));
      card.appendChild(head);

      const body = document.createElement("div");
      body.className = "cube-group__body";
      cubes.forEach((cube) => {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "cube-run" + (cube.id === this.selectedId ? " is-active" : "");
        row.title = cube.id;
        row.innerHTML =
          `<span class="cube-run__name">${this._esc(cube.run)}</span>` +
          `<span class="cube-run__stamp">${this._esc(cube.stamp)}</span>`;
        row.addEventListener("click", () => this.select(cube.id));
        body.appendChild(row);
      });
      card.appendChild(body);

      this.strip.appendChild(card);
    });
  }

  _toggleGroup(group) {
    if (this.openGroups.has(group)) this.openGroups.delete(group);
    else this.openGroups.add(group);
    this._renderStrip();
  }

  async select(cubeId) {
    if (this.polling) {
      window.toast("A cube is still loading.", "warn");
      return;
    }
    if (cubeId === this.selectedId && this.meta) return;

    this._stopSweeps();
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
      if (panel.metric) panel.metric.textContent = "";
    });
    this.profQueued = null;
    this.profData = null;
    this.ssimQueued = null;
    this._clearProfMetrics();
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
      range   : css.getPropertyValue("--cut-range").trim(),
      azimuth : css.getPropertyValue("--cut-azimuth").trim(),
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

    this._initCutBounds();
    this.sweeps.forEach((sweep) => sweep.configure());

    this._follow({ az: Math.floor(meta.n_az / 2), rg: Math.floor(meta.n_rg / 2), fx: 0.5, fy: 0.5 }, true);

    const sweep = this._sweepFor(this.view);
    if (sweep) sweep.play();
  }

  _setSpace(space) {
    if (space === this.space || !["physical", "normalized"].includes(space)) return;
    this.space = space;
    this._syncSpaceBtns();

    if (!this.meta) return;
    const sweep = this._sweepFor(this.view);
    if (sweep) sweep.syncSpace();
    if (this.point) this._drawSlices(this.point.az, this.point.rg);
  }

  _setView(view) {
    if (!["explorer", "elevation", "azimuth", "range"].includes(view) || view === this.view) return;

    this._stopSweeps();
    this.view = view;

    this.modeBtns.forEach((btn) => btn.classList.toggle("is-active", btn.dataset.view === view));
    this.viewEls.forEach((el) => { el.hidden = el.dataset.view !== view; });

    const sweep = this._sweepFor(view);
    if (sweep && this.meta) sweep.play();
  }

  _sweepFor(view) {
    return this.sweeps.find((sweep) => sweep.axis === view) || null;
  }

  _stopSweeps() {
    this.sweeps.forEach((sweep) => sweep.stop());
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
    if (this.profMetricsEl) {
      this.profMetricsEl.querySelectorAll("tr[data-source]").forEach((row) => {
        row.hidden = !this.visible.has(row.dataset.source);
      });
    }
    this.slicesEl.style.setProperty("--cube-rows", String(this.visible.size));

    this.sweeps.forEach((sweep) => sweep.applyVisibility());
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

  _setManualCut() {
    if (!this.meta) return;

    const az = this._clampInt(this.jumpAz ? this.jumpAz.value : 0, this.meta.n_az);
    const rg = this._clampInt(this.jumpRg ? this.jumpRg.value : 0, this.meta.n_rg);
    const point = { az, rg, fx: (rg + 0.5) / this.meta.n_rg, fy: (az + 0.5) / this.meta.n_az };

    this._follow(point, true);
    this._enterSlices(point);
    this._syncCutInputs(az, rg, true);
  }

  _syncCutInputs(az, rg, force = false) {
    if (this.jumpAz && (force || document.activeElement !== this.jumpAz)) this.jumpAz.value = az;
    if (this.jumpRg && (force || document.activeElement !== this.jumpRg)) this.jumpRg.value = rg;
  }

  _initCutBounds() {
    if (this.jumpAz) { this.jumpAz.min = 0; this.jumpAz.max = this.meta.n_az - 1; }
    if (this.jumpRg) { this.jumpRg.min = 0; this.jumpRg.max = this.meta.n_rg - 1; }
    if (this.jumpAzRange) this.jumpAzRange.textContent = `0–${this.meta.n_az - 1}`;
    if (this.jumpRgRange) this.jumpRgRange.textContent = `0–${this.meta.n_rg - 1}`;
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
    this._syncCutInputs(point.az, point.rg);

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

    this._queueSsim(az, rg);
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

    const x = panel.marker * canvas.width;

    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();

    ctx.strokeStyle = this.colors[panel.axis] || "#fff";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _queueSsim(az, rg) {
    this.ssimQueued = { az, rg };
    this._ssimPump();
  }

  async _ssimPump() {
    if (this.ssimFetching || !this.ssimQueued) return;
    this.ssimFetching = true;

    while (this.ssimQueued) {
      const { az, rg } = this.ssimQueued;
      this.ssimQueued = null;
      await this._fetchSsim(az, rg);
    }

    this.ssimFetching = false;
  }

  async _fetchSsim(az, rg) {
    const space = this.space;

    let data;
    try {
      data = await window.apiGet(`/api/cubes/ssim?id=${encodeURIComponent(this.selectedId)}&az=${az}&rg=${rg}&space=${space}`);
    } catch (e) {
      return;
    }

    if (!data || !data.ok) return;
    this._applySsim(data);
  }

  _applySsim(data) {
    this.panels.forEach((panel) => {
      if (!panel.metric) return;

      if (panel.source === "gt") {
        panel.metric.textContent = "reference";
        return;
      }

      const group = data[panel.axis] || {};
      const value = group[panel.source];
      panel.metric.textContent = value === undefined || value === null ? "SSIM –" : `SSIM ${value.toFixed(3)}`;
    });
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

    this._fillProfMetrics();
  }

  _fillProfMetrics() {
    if (!this.profMetricsEl) return;

    const gt = this.profData && this.profData.sources.gt;

    ["pred", "reduced"].forEach((source) => {
      const row = this.profMetricsEl.querySelector(`tr[data-source="${source}"]`);
      if (!row) return;

      const series = this.profData && this.profData.sources[source];
      const scores = gt && series ? this._scoreCurve(series.values, gt.values) : null;

      ["mae", "mse", "r2"].forEach((key) => {
        const cell = row.querySelector(`td[data-k="${key}"]`);
        if (cell) cell.textContent = scores ? this._fmtMetric(scores[key], key) : "–";
      });
    });
  }

  _scoreCurve(values, ref) {
    const n = Math.min(values.length, ref.length);
    if (!n) return null;

    let sumAbs = 0, sumSq = 0, sumRef = 0;
    for (let i = 0; i < n; i++) {
      const d = values[i] - ref[i];
      sumAbs += Math.abs(d);
      sumSq  += d * d;
      sumRef += ref[i];
    }

    const mean = sumRef / n;
    let ssTot = 0;
    for (let i = 0; i < n; i++) ssTot += (ref[i] - mean) ** 2;

    return { mae: sumAbs / n, mse: sumSq / n, r2: 1 - sumSq / (ssTot + 1e-12) };
  }

  _fmtMetric(value, key) {
    if (!Number.isFinite(value)) return "–";
    if (key === "r2") return value.toFixed(3);
    return this._fmt(value);
  }

  _clearProfMetrics() {
    if (!this.profMetricsEl) return;
    this.profMetricsEl.querySelectorAll("td[data-k]").forEach((cell) => { cell.textContent = "–"; });
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

  _clampInt(value, count) {
    const n = Math.floor(Number(value));
    if (!Number.isFinite(n)) return 0;
    return Math.min(count - 1, Math.max(0, n));
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
