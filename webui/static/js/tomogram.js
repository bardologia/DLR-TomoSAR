"use strict";

class TomogramView {
  constructor(refs) {
    this.strip = refs.strip;
    this.stage = refs.stage;
    this.topdown = refs.topdown;
    this.cross = refs.cross;
    this.coords = refs.coords;
    this.sourceBtns = refs.sourceBtns;
    this.rangeImg = refs.rangeImg;
    this.azimuthImg = refs.azimuthImg;
    this.hint = refs.hint;

    this.cubes = [];
    this.selectedId = null;
    this.source = "pred";
    this.point = null;
    this.loaded = false;

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
    this.cross.hidden = true;
    this.coords.textContent = "click a point on the map";
    this.rangeImg.src = "";
    this.azimuthImg.src = "";
    this.rangeImg.parentElement.hidden = true;
    this.azimuthImg.parentElement.hidden = true;
    this._renderStrip();

    const cube = this._cube();
    if (!cube) return;

    const hasSource = cube.sources.includes(this.source);
    if (!hasSource) this.source = cube.sources[0] || "pred";
    this._syncSourceBtns();

    this.hint.hidden = true;
    this.stage.hidden = false;
    this.topdown.src = `/api/cubes/topdown?id=${encodeURIComponent(cubeId)}&source=${this.source}`;
  }

  _setSource(source) {
    if (source === this.source) return;
    const cube = this._cube();
    if (!cube || !cube.sources.includes(source)) return;
    this.source = source;
    this._syncSourceBtns();
    this.topdown.src = `/api/cubes/topdown?id=${encodeURIComponent(this.selectedId)}&source=${this.source}`;
    if (this.point) this._loadSlices();
  }

  _syncSourceBtns() {
    const cube = this._cube();
    this.sourceBtns.forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.source === this.source);
      btn.disabled = !cube || !cube.sources.includes(btn.dataset.source);
    });
  }

  _onClick(ev) {
    const cube = this._cube();
    if (!cube) return;

    const rect = this.topdown.getBoundingClientRect();
    const fx = (ev.clientX - rect.left) / rect.width;
    const fy = (ev.clientY - rect.top) / rect.height;

    const rg = Math.min(cube.n_rg - 1, Math.max(0, Math.floor(fx * cube.n_rg)));
    const az = Math.min(cube.n_az - 1, Math.max(0, Math.floor(fy * cube.n_az)));

    this.point = { az, rg };
    this.coords.textContent = `az = ${az}   rg = ${rg}`;

    this.cross.hidden = false;
    this.cross.style.left = `${fx * 100}%`;
    this.cross.style.top = `${fy * 100}%`;

    this._loadSlices();
  }

  _loadSlices() {
    const { az, rg } = this.point;
    const base = `/api/cubes/slice?id=${encodeURIComponent(this.selectedId)}&source=${this.source}&az=${az}&rg=${rg}`;
    this.rangeImg.parentElement.hidden = false;
    this.azimuthImg.parentElement.hidden = false;
    this.rangeImg.src = `${base}&axis=range`;
    this.azimuthImg.src = `${base}&axis=azimuth`;
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
}

window.TomogramView = TomogramView;
