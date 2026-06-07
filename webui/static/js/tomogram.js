"use strict";

class TomogramView {
  static LABELS = { pred: "pred", gt: "gt", reduced: "capon reduced", full: "capon full" };

  constructor(refs) {
    this.strip = refs.strip;
    this.stage = refs.stage;
    this.topdown = refs.topdown;
    this.cross = refs.cross;
    this.coords = refs.coords;
    this.sourceBtns = refs.sourceBtns;
    this.hint = refs.hint;
    this.panels = refs.panels;
    this.slicesEl = refs.slices;
    this.progress = refs.progress;
    this.progressFill = refs.progressFill;
    this.progressLabel = refs.progressLabel;

    this.cubes = [];
    this.selectedId = null;
    this.meta = null;
    this.source = "pred";
    this.point = null;
    this.pinned = false;
    this.entered = false;
    this.polling = false;
    this.fetching = false;
    this.queued = null;

    this.topdown.addEventListener("mousemove", (ev) => this._onMove(ev));
    this.topdown.addEventListener("click", (ev) => this._onClick(ev));
    this.sourceBtns.forEach((btn) => {
      btn.addEventListener("click", () => this._setSource(btn.dataset.source));
    });
  }

  async enter() {
    if (this.entered) return;
    this.entered = true;
    await this.refresh();
  }

  async refresh() {
    let data;
    try {
      data = await window.apiGet("/api/cubes");
    } catch (e) {
      this.hint.textContent = "Backend unreachable.";
      return;
    }

    this.cubes = data.cubes || [];
    this._renderStrip();

    if (!this.cubes.length) {
      this.hint.textContent = "No saved cubes found. Run an inference with save_cubes=True first.";
      this.hint.hidden = false;
      this.stage.hidden = true;
      return;
    }

    this.hint.textContent = "Select a cube directory to load it into memory.";
    this.hint.hidden = false;
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
    this.pinned = false;
    this.queued = null;
    this.cross.hidden = true;
    this.coords.textContent = "hover the map to explore, click to pin";
    this.stage.hidden = true;
    this.slicesEl.hidden = true;
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

    if (!meta.sources.includes(this.source)) this.source = meta.sources[0];
    this._syncSourceBtns();

    this.panels.forEach((panel) => {
      panel.root.hidden = !meta.sources.includes(panel.source);
    });

    this.hint.hidden = true;
    this.stage.hidden = false;
    this.topdown.src = `/api/cubes/topdown?id=${encodeURIComponent(this.selectedId)}&source=${this.source}`;
  }

  _setSource(source) {
    if (source === this.source || !this.meta || !this.meta.sources.includes(source)) return;
    this.source = source;
    this._syncSourceBtns();
    this.topdown.src = `/api/cubes/topdown?id=${encodeURIComponent(this.selectedId)}&source=${this.source}`;
  }

  _syncSourceBtns() {
    this.sourceBtns.forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.source === this.source);
      btn.disabled = !this.meta || !this.meta.sources.includes(btn.dataset.source);
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
    if (this.pinned) return;
    const point = this._pointFromEvent(ev);
    if (!point) return;
    this._follow(point);
  }

  _onClick(ev) {
    const point = this._pointFromEvent(ev);
    if (!point) return;

    if (this.pinned) {
      this.pinned = false;
      this._follow(point);
    } else {
      this.pinned = true;
      this._follow(point, true);
    }
  }

  _follow(point, force = false) {
    if (!force && this.point && point.az === this.point.az && point.rg === this.point.rg) {
      this._moveCross(point);
      return;
    }

    this.point = point;
    this._moveCross(point);
    this.coords.textContent = `az = ${point.az}   rg = ${point.rg}${this.pinned ? "   [pinned — click to release]" : ""}`;

    this.queued = { az: point.az, rg: point.rg };
    this._pump();
  }

  _moveCross(point) {
    this.cross.hidden = false;
    this.cross.style.left = `${point.fx * 100}%`;
    this.cross.style.top = `${point.fy * 100}%`;
  }

  async _pump() {
    if (this.fetching || !this.queued) return;
    this.fetching = true;

    while (this.queued) {
      const { az, rg } = this.queued;
      this.queued = null;
      await this._drawSlices(az, rg);
    }

    this.fetching = false;
  }

  async _drawSlices(az, rg) {
    if (!this.meta) return;

    this.slicesEl.hidden = false;

    const jobs = this.panels
      .filter((panel) => !panel.root.hidden)
      .map(async (panel) => {
        const url = `/api/cubes/slice?id=${encodeURIComponent(this.selectedId)}&source=${panel.source}&axis=${panel.axis}&az=${az}&rg=${rg}`;
        try {
          const res = await fetch(url);
          if (!res.ok) return;
          const bitmap = await createImageBitmap(await res.blob());

          const canvas = panel.canvas;
          if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
          }
          const ctx = canvas.getContext("2d");
          ctx.drawImage(bitmap, 0, 0);

          const markerFrac = panel.axis === "range" ? az / this.meta.n_az : rg / this.meta.n_rg;
          ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(markerFrac * canvas.width, 0);
          ctx.lineTo(markerFrac * canvas.width, canvas.height);
          ctx.stroke();
          ctx.setLineDash([]);

          const label = TomogramView.LABELS[panel.source] || panel.source;
          panel.caption.textContent =
            panel.axis === "range"
              ? `${label} — range cut @ rg=${rg}`
              : `${label} — azimuth cut @ az=${az}`;
        } catch (e) {}
      });

    await Promise.all(jobs);
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
}

window.TomogramView = TomogramView;
