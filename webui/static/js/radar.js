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
    if (rect.width < 2 || rect.height < 2) return;
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

    requestAnimationFrame(this._loop);
    this._draw();
  }
}


window.RadarScene = RadarScene;
window.REDUCED_MOTION = REDUCED_MOTION;
