"use strict";

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

class CanvasBase {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = 0;
    this.h = 0;
    this._resize = this._resize.bind(this);
    window.addEventListener("resize", this._resize);
    this._resize();
  }

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.w = Math.max(1, rect.width);
    this.h = Math.max(1, rect.height);
    this.canvas.width = this.w * this.dpr;
    this.canvas.height = this.h * this.dpr;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.onResize();
  }

  resize() {
    this._resize();
  }

  onResize() {}
}

class RadarScene extends CanvasBase {
  constructor(canvas) {
    super(canvas);
    this.angle = 0;
    this.scatterers = [];
    this._seed();
    if (REDUCED_MOTION) {
      this._draw();
    } else {
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
    }
  }

  onResize() {
    this._seed();
    if (REDUCED_MOTION && this.angle != null) this._draw();
  }

  _seed() {
    const count = Math.round((this.w * this.h) / 26000);
    this.scatterers = [];
    for (let i = 0; i < count; i++) {
      this.scatterers.push({
        a: Math.random() * Math.PI * 2,
        r: 0.12 + Math.random() * 0.88,
        s: 0.6 + Math.random() * 1.8,
        lit: 0,
      });
    }
  }

  _center() {
    return { cx: this.w * 0.74, cy: this.h * 0.52, rad: Math.max(this.w, this.h) * 0.62 };
  }

  _draw() {
    const ctx = this.ctx;
    const { cx, cy, rad } = this._center();
    ctx.clearRect(0, 0, this.w, this.h);

    ctx.save();
    ctx.translate(cx, cy);

    ctx.strokeStyle = "rgba(29, 79, 216, 0.14)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 6; i++) {
      ctx.beginPath();
      ctx.arc(0, 0, (rad / 6) * i, 0, Math.PI * 2);
      ctx.stroke();
    }
    for (let i = 0; i < 12; i++) {
      const a = (Math.PI * 2 * i) / 12;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(Math.cos(a) * rad, Math.sin(a) * rad);
      ctx.strokeStyle = "rgba(29, 79, 216, 0.08)";
      ctx.stroke();
    }

    if (!REDUCED_MOTION) {
      const grad = ctx.createConicGradient(this.angle, 0, 0);
      grad.addColorStop(0, "rgba(29, 79, 216, 0.0)");
      grad.addColorStop(0.04, "rgba(29, 79, 216, 0.08)");
      grad.addColorStop(0.09, "rgba(29, 79, 216, 0.007)");
      grad.addColorStop(1, "rgba(29, 79, 216, 0.0)");
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.arc(0, 0, rad, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(Math.cos(this.angle) * rad, Math.sin(this.angle) * rad);
      ctx.strokeStyle = "rgba(29, 79, 216, 0.18)";
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }

    for (const p of this.scatterers) {
      const x = Math.cos(p.a) * p.r * rad;
      const y = Math.sin(p.a) * p.r * rad;
      const base = 0.18 + p.lit * 0.82;
      ctx.beginPath();
      ctx.arc(x, y, p.s * (0.7 + p.lit * 1.4), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(29, 79, 216, ${base})`;
      ctx.fill();
      if (p.lit > 0.05) {
        ctx.beginPath();
        ctx.arc(x, y, p.s * 4 * p.lit, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(29, 79, 216, ${0.12 * p.lit})`;
        ctx.fill();
      }
    }

    ctx.restore();
  }

  _loop() {
    this.angle += 0.012;
    if (this.angle > Math.PI * 2) this.angle -= Math.PI * 2;

    for (const p of this.scatterers) {
      let diff = Math.abs(((p.a - this.angle) % (Math.PI * 2)));
      if (diff > Math.PI) diff = Math.PI * 2 - diff;
      if (diff < 0.08) p.lit = 1;
      else p.lit *= 0.965;
    }

    this._draw();
    requestAnimationFrame(this._loop);
  }
}

class SpectrumScene extends CanvasBase {
  constructor(canvas, readout) {
    super(canvas);
    this.readout = readout;
    this.t = 0;
    this.hues = ["45, 212, 191", "111, 155, 255", "167, 139, 250"];
    this.components = [
      { a: 0.9, mu: 0.32, sig: 0.05, va: 0.12, vm: 0.04, vs: 0.015, pa: 0, pm: 1.2, ps: 2.1 },
      { a: 0.55, mu: 0.58, sig: 0.07, va: 0.1, vm: 0.05, vs: 0.02, pa: 2, pm: 0.7, ps: 1.1 },
      { a: 0.32, mu: 0.78, sig: 0.045, va: 0.08, vm: 0.03, vs: 0.012, pa: 4, pm: 1.6, ps: 3.0 },
    ];
    if (REDUCED_MOTION) {
      this._draw();
    } else {
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
    }
  }

  onResize() {
    if (REDUCED_MOTION && this.components) this._draw();
  }

  _params() {
    return this.components.map((c) => ({
      a: Math.max(0.05, c.a + Math.sin(this.t * c.pa * 0.0 + this.t * 0.7 + c.pa) * c.va),
      mu: Math.min(0.95, Math.max(0.05, c.mu + Math.sin(this.t * 0.5 + c.pm) * c.vm)),
      sig: Math.max(0.02, c.sig + Math.sin(this.t * 0.6 + c.ps) * c.vs),
    }));
  }

  _curve(params, x) {
    let y = 0;
    for (const p of params) {
      const d = x - p.mu;
      y += p.a * Math.exp(-(d * d) / (2 * p.sig * p.sig));
    }
    return y;
  }

  _draw() {
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    const padL = 30;
    const padB = 22;
    const padT = 10;
    const padR = 8;
    const params = this._params();

    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + ((h - padT - padB) / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
    }
    for (let i = 0; i <= 5; i++) {
      const x = padL + ((w - padL - padR) / 5) * i;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
    }

    const px = (x) => padL + x * (w - padL - padR);
    const maxY = 1.7;
    const py = (y) => padT + (1 - Math.min(1, y / maxY)) * (h - padT - padB);

    const steps = 160;

    params.forEach((p, k) => {
      const hue = this.hues[k % this.hues.length];

      ctx.save();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = `rgba(${hue}, 0.3)`;
      ctx.beginPath();
      ctx.moveTo(px(p.mu), py(0));
      ctx.lineTo(px(p.mu), py(this._curve(params, p.mu)));
      ctx.stroke();
      ctx.restore();

      ctx.beginPath();
      for (let i = 0; i <= steps; i++) {
        const x = i / steps;
        const d = x - p.mu;
        const y = p.a * Math.exp(-(d * d) / (2 * p.sig * p.sig));
        const sx = px(x);
        const sy = py(y);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.strokeStyle = `rgba(${hue}, 0.55)`;
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.lineTo(px(1), py(0));
      ctx.lineTo(px(0), py(0));
      ctx.closePath();
      ctx.fillStyle = `rgba(${hue}, 0.05)`;
      ctx.fill();
    });

    ctx.beginPath();
    for (let i = 0; i <= steps; i++) {
      const x = i / steps;
      const y = this._curve(params, x);
      const sx = px(x);
      const sy = py(y);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }

    const stroke = ctx.getLineDash ? ctx : ctx;
    const lineGrad = ctx.createLinearGradient(padL, 0, w - padR, 0);
    lineGrad.addColorStop(0, "#6f9bff");
    lineGrad.addColorStop(1, "#2dd4bf");
    ctx.strokeStyle = lineGrad;
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(45, 212, 191, 0.6)";
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.lineTo(px(1), py(0));
    ctx.lineTo(px(0), py(0));
    ctx.closePath();
    const fill = ctx.createLinearGradient(0, padT, 0, h - padB);
    fill.addColorStop(0, "rgba(45, 212, 191, 0.28)");
    fill.addColorStop(1, "rgba(45, 212, 191, 0.0)");
    ctx.fillStyle = fill;
    ctx.fill();

    ctx.fillStyle = "#8d979d";
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.fillText("P", 8, padT + 8);
    ctx.fillText("ξ", w - padR - 8, h - 6);

    if (this.readout) {
      let html = `K <b>${params.length}</b>`;
      params.forEach((p, k) => {
        const hue = this.hues[k % this.hues.length];
        const muM = (-20 + p.mu * 100).toFixed(1);
        const sigM = (p.sig * 100).toFixed(1);
        html += `<span class="sp-k"><i style="background:rgb(${hue})"></i>μ <b>${muM}</b> σ <b>${sigM}</b> a <b>${p.a.toFixed(2)}</b></span>`;
      });
      this.readout.innerHTML = html;
    }
  }

  _loop() {
    this.t += 0.016;
    this._draw();
    requestAnimationFrame(this._loop);
  }
}

window.RadarScene = RadarScene;
window.SpectrumScene = SpectrumScene;
window.REDUCED_MOTION = REDUCED_MOTION;
