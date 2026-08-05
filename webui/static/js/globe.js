"use strict";

class TomogramGlobe {
  static CESIUM_VERSION = "1.144.0";
  static IMAGERY_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
  static IMAGERY_CREDIT = "Esri, Maxar, Earthstar Geographics, and the GIS User Community";

  constructor(refs, host) {
    this.host = host;
    this.sourceEl = refs.source;
    this.colorEl = refs.color;
    this.thrEl = refs.thr;
    this.thrLabel = refs.thrLabel;
    this.thrValEl = refs.thrVal;
    this.maxEl = refs.max;
    this.exaggEl = refs.exagg;
    this.liftEl = refs.lift;
    this.clampEl = refs.clamp;
    this.reframeEl = refs.reframe;
    this.atEl = refs.at;
    this.container = refs.container;

    this.source = "pred";
    this.colorBy = "mu";
    this.available = false;
    this.points = null;
    this.total = 0;
    this.muRange = null;
    this.flown = false;
    this.loader = null;
    this.viewer = null;
    this.collection = null;
    this.debounceTimer = null;

    this.sourceEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => this._setSource(btn.dataset.source));
    });
    this.colorEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.addEventListener("click", () => this._setColor(btn.dataset.color));
    });

    this.thrEl.addEventListener("input", () => this._onThreshold());
    this.maxEl.addEventListener("change", () => this._fetch());
    this.exaggEl.addEventListener("change", () => this._redraw());
    this.liftEl.addEventListener("change", () => this._redraw());
    this.clampEl.addEventListener("change", () => this._redraw());
    this.reframeEl.addEventListener("click", () => this._flyToScene(true));
  }

  configure(meta) {
    const sources = this._sources(meta);
    this.available = !!meta.globe && sources.length > 0;
    if (!sources.includes(this.source)) this.source = sources[0] || "pred";

    this.points = null;
    this.muRange = null;
    this.flown = false;
    if (this.collection) {
      this.collection.removeAll();
      this.viewer.scene.requestRender();
    }
    this._syncThresholdLabel();
  }

  _sources(meta = this.host.meta) {
    const out = meta.params ? meta.params.sources.slice() : [];
    if (meta.sources.includes("reduced")) out.push("reduced");
    return out;
  }

  _isParam() {
    return this.source === "pred" || this.source === "gt";
  }

  _ampMin() {
    return TomogramCloud.ampFloor(this.host.meta, this.source, Number(this.thrEl.value) / 100);
  }

  _syncThresholdLabel() {
    const meta = this.host.meta;
    if (!meta || (this._isParam() && !meta.params)) return;
    this.thrLabel.textContent = this._isParam() ? "amp ≥" : "int ≥";
    this.thrValEl.textContent = this.host._fmt(this._ampMin());
  }

  _syncBtns() {
    const sources = this._sources();
    this.sourceEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.source === this.source);
      btn.disabled = !sources.includes(btn.dataset.source);
    });
    this.colorEl.querySelectorAll(".cube-space").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.color === this.colorBy);
    });
  }

  _setSource(source) {
    if (source === this.source) return;
    if (!this._sources().includes(source)) {
      window.toast(TomogramCloud.MISSING[source] || "This source is not available.", "warn");
      return;
    }
    this.source = source;
    this._syncBtns();
    this._syncThresholdLabel();
    this._fetch();
  }

  _setColor(colorBy) {
    if (colorBy === this.colorBy) return;
    this.colorBy = colorBy;
    this._syncBtns();
    this._redraw();
  }

  _onThreshold() {
    this._syncThresholdLabel();
    clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => this._fetch(), 250);
  }

  _ensureCesium() {
    if (this.loader) return this.loader;

    const base = `https://cdn.jsdelivr.net/npm/cesium@${TomogramGlobe.CESIUM_VERSION}/Build/Cesium/`;
    window.CESIUM_BASE_URL = base;

    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = `${base}Widgets/widgets.css`;
    document.head.appendChild(css);

    this.loader = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `${base}Cesium.js`;
      script.onload = () => resolve();
      script.onerror = () => {
        this.loader = null;
        reject(new Error("CesiumJS failed to load"));
      };
      document.head.appendChild(script);
    });
    return this.loader;
  }

  _ensureViewer() {
    if (this.viewer) return;

    const provider = new Cesium.UrlTemplateImageryProvider({
      url: TomogramGlobe.IMAGERY_URL,
      credit: TomogramGlobe.IMAGERY_CREDIT,
      maximumLevel: 19,
    });

    this.viewer = new Cesium.Viewer(this.container, {
      baseLayer: new Cesium.ImageryLayer(provider),
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      requestRenderMode: true,
      maximumRenderTimeChange: Infinity,
    });

    this.viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#10151a");
    this.collection = this.viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
  }

  async _fetch() {
    if (!this.available || !this.viewer) return;

    const url = `/api/cubes/globe_points?id=${encodeURIComponent(this.host.selectedId)}&source=${this.source}` +
      `&amp_min=${this._ampMin()}&max=${this.maxEl.value}`;

    let raw;
    try {
      const res = await fetch(url);
      if (!res.ok) return;
      raw = new Float32Array(await res.arrayBuffer());
    } catch (e) {
      return;
    }

    this.points = raw.subarray(4);
    this.total = raw[1];
    this.muRange = TomogramCloud.sampleRange(this.points, 3, 5);
    this._redraw();
  }

  _flyToScene(animate) {
    if (!this.viewer || !this.host.meta || !this.host.meta.globe) return;

    const globe = this.host.meta.globe;
    const [west, south, east, north] = globe.bbox;
    const midLat = (south + north) / 2;

    const extentEast = (east - west) * 111320.0 * Math.cos(midLat * Math.PI / 180);
    const extentNorth = (north - south) * 110574.0;
    const radius = Math.max(0.5 * Math.max(extentEast, extentNorth), 100.0);

    const anchor = new Cesium.Cartesian3(globe.anchor_ecef[0], globe.anchor_ecef[1], globe.anchor_ecef[2]);
    const up = Cesium.Ellipsoid.WGS84.geodeticSurfaceNormal(anchor, new Cesium.Cartesian3());
    const lift = Number(this.liftEl.value || 0) - (this.clampEl.checked ? globe.base_height : 0);
    const center = new Cesium.Cartesian3(anchor.x + up.x * lift, anchor.y + up.y * lift, anchor.z + up.z * lift);

    this.viewer.camera.flyToBoundingSphere(new Cesium.BoundingSphere(center, radius), {
      offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-32), radius * 3.4),
      duration: animate ? 1.2 : 0,
    });
  }

  _redraw() {
    if (!this.viewer || !this.points || this.host.view !== "globe") return;

    const meta = this.host.meta;
    const globe = meta.globe;
    const rows = this.points;

    const anchor = new Cesium.Cartesian3(globe.anchor_ecef[0], globe.anchor_ecef[1], globe.anchor_ecef[2]);
    const up = Cesium.Ellipsoid.WGS84.geodeticSurfaceNormal(anchor, new Cesium.Cartesian3());
    const lift = Number(this.liftEl.value || 0) - (this.clampEl.checked ? globe.base_height : 0);
    const exagg = Number(this.exaggEl.value || 1);

    const [muLo, muHi] = this.muRange || [meta.x_min, meta.x_max];
    const muSpan = (muHi - muLo) || 1;

    const logAmp = this._isParam();
    let ampLo, ampHi;
    if (logAmp) {
      ampLo = Math.log(Math.max(meta.params.threshold, 1e-6));
      ampHi = Math.log(Math.max(meta.params.ranges.amp[1], meta.params.threshold * 10));
    } else {
      [ampLo, ampHi] = meta.intensity[this.source];
    }

    this.collection.removeAll();

    for (let i = 0; i < rows.length; i += 5) {
      const mu = rows[i + 3];
      const amp = rows[i + 4];
      const t = this.colorBy === "amp"
        ? ((logAmp ? Math.log(Math.max(amp, 1e-6)) : amp) - ampLo) / Math.max(ampHi - ampLo, 1e-6)
        : (mu - muLo) / muSpan;
      const rgb = TomogramCloud.palette(t);

      const upComp = rows[i] * up.x + rows[i + 1] * up.y + rows[i + 2] * up.z;
      const rise = upComp * (exagg - 1) + lift;

      this.collection.add({
        position: new Cesium.Cartesian3(
          anchor.x + rows[i] + up.x * rise,
          anchor.y + rows[i + 1] + up.y * rise,
          anchor.z + rows[i + 2] + up.z * rise,
        ),
        color: Cesium.Color.fromBytes(rgb[0], rgb[1], rgb[2], 255),
        pixelSize: 2.5,
      });
    }

    if (!this.flown) {
      this._flyToScene(false);
      this.flown = true;
    }

    const shown = rows.length / 5;
    const exaggNote = exagg > 1 ? ` · height ${exagg}×` : "";
    this.atEl.textContent = `${shown.toLocaleString()} of ${Math.round(this.total).toLocaleString()} scatterers${exaggNote} · corner fit ±${globe.residual_rms_m.toFixed(1)} m · ctrl+drag tilts · Esri World Imagery`;
    this.viewer.scene.requestRender();
  }

  render() {
    this._syncBtns();
    this._syncThresholdLabel();

    this._ensureCesium()
      .then(() => {
        this._ensureViewer();
        this.viewer.resize();
        if (!this.points) this._fetch();
        else this._redraw();
      })
      .catch(() => {
        this.atEl.textContent = "CesiumJS could not be loaded — the globe needs internet access.";
      });
  }
}
