"use strict";

class TomogramView {
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

    this.cubes = [];
    this.selectedId = null;
    this.source = "pred";
    this.point = null;
    this.pinned = false;
    this.loaded = false;
    this.fetching = false;
    this.queued = null;

    this.topdown.addEventListener("mousemove", (ev) => this._onMove(ev));
    this.topdown.addEventListener("click", (ev) => this._onClick(ev));
    this.sourceBtns.forEach((btn) => {
      btn.addEventListener("click", () => this._setSource(btn.dataset.source));
    });
  }

  async enter() {
    if (this.loaded) return;
    this.loaded = true;
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

    if (!this.cubes.some((c) => c.id === this.selectedId)) this.select(this.cubes[0].id);
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

  _cube() {
    return this.cubes.find((c) => c.id === this.selectedId) || null;
  }

  select(cubeId) {
    this.selectedId = cubeId;
    this.point = null;
    this.pinned = false;
    this.queued = null;
    this.cross.hidden = true;
    this.coords.textContent = "hover the map to explore, click to pin";
    this.slicesEl.hidden = true;
    this._renderStrip();

    const cube = this._cube();
    if (!cube) return;

    if (!cube.sources.includes(this.source)) this.source = cube.sources[0] || "pred";
    this._syncSourceBtns();

    this.panels.forEach((panel) => {
      panel.root.hidden = panel.source === "gt" && !cube.sources.includes("gt");
    });

    this.hint.hidden = true;
    this.stage.hidden = false;
    this.topdown.src = `/api/cubes/topdown?id=${encodeURIComponent(cubeId)}&source=${this.source}`;
  }

  _setSource(source) {
    const cube = this._cube();
    if (source === this.source || !cube || !cube.sources.includes(source)) return;
    this.source = source;
    this._syncSourceBtns();
    this.topdown.src = `/api/cubes/topdown?id=${encodeURIComponent(this.selectedId)}&source=${this.source}`;
  }

  _syncSourceBtns() {
    const cube = this._cube();
    this.sourceBtns.forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.source === this.source);
      btn.disabled = !cube || !cube.sources.includes(btn.dataset.source);
    });
  }

  _pointFromEvent(ev) {
    const cube = this._cube();
    if (!cube) return null;

    const rect = this.topdown.getBoundingClientRect();
    const fx = (ev.clientX - rect.left) / rect.width;
    const fy = (ev.clientY - rect.top) / rect.height;

    return {
      az: Math.min(cube.n_az - 1, Math.max(0, Math.floor(fy * cube.n_az))),
      rg: Math.min(cube.n_rg - 1, Math.max(0, Math.floor(fx * cube.n_rg))),
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
    const cube = this._cube();
    if (!cube) return;

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

          const markerFrac = panel.axis === "range" ? az / cube.n_az : rg / cube.n_rg;
          ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(markerFrac * canvas.width, 0);
          ctx.lineTo(markerFrac * canvas.width, canvas.height);
          ctx.stroke();
          ctx.setLineDash([]);

          panel.caption.textContent =
            panel.axis === "range"
              ? `${panel.source} — range cut @ rg=${rg}`
              : `${panel.source} — azimuth cut @ az=${az}`;
        } catch (e) {}
      });

    await Promise.all(jobs);
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
}

window.TomogramView = TomogramView;
