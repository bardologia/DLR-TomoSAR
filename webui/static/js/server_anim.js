"use strict";

class ServerScene extends CanvasBase {

  constructor(canvas) {
    super(canvas);
    this.t     = 0;
    this.ink   = "20, 30, 40";
    this.blue  = "29, 79, 216";
    this.teal  = "15, 118, 110";

    this.units = [0, 1, 2, 3].map((i) => ({
      seed     : i * 2.61 + 0.7,
      fan      : i * 1.3,
      fanSpeed : 0.05 + 0.02 * ((i * 7) % 3),
    }));

    this.packets = Array.from({ length: 6 }, (_, i) => ({
      p     : (i * 0.37) % 1,
      lane  : i % 2,
      speed : 0.0035 + 0.0012 * ((i * 5) % 3),
    }));

    if (REDUCED_MOTION) {
      this.t = 3;
      this._draw();
    } else {
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
    }
  }

  onResize() {
    if (REDUCED_MOTION && this.units) this._draw();
  }

  _draw() {
    const ctx = this.ctx;
    const w   = this.w;
    const h   = this.h;
    ctx.clearRect(0, 0, w, h);

    const rackX = w * 0.40;
    const rackY = 10;
    const rackW = w * 0.56;
    const rackH = h - 20;
    const gap   = 6;
    const uh    = (rackH - gap * (this.units.length - 1)) / this.units.length;

    ctx.strokeStyle = `rgba(${this.ink}, 0.45)`;
    ctx.lineWidth   = 1.2;
    this._round(ctx, rackX - 7, rackY - 7, rackW + 14, rackH + 14, 5);
    ctx.stroke();

    this.units.forEach((u, i) => {
      const uy = rackY + i * (uh + gap);
      this._unit(ctx, u, rackX, uy, rackW, uh);
    });

    this._network(ctx, rackX - 7, h);
  }

  _unit(ctx, u, ux, uy, uw, uh) {
    ctx.strokeStyle = `rgba(${this.ink}, 0.35)`;
    ctx.lineWidth   = 1;
    this._round(ctx, ux, uy, uw, uh, 3);
    ctx.stroke();

    const cy   = uy + uh / 2;
    const live = Math.sin(this.t * 6 + u.seed * 9) > -0.4 ? 1 : 0.25;

    ctx.beginPath();
    ctx.arc(ux + 11, cy - 5, 2.4, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${this.blue}, ${0.25 + 0.65 * live})`;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(ux + 11, cy + 5, 2.4, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${this.teal}, 0.8)`;
    ctx.fill();

    const bars = 11;
    const bx0  = ux + 24;
    const bx1  = ux + uw - 34;
    const bw   = (bx1 - bx0) / bars;
    for (let k = 0; k < bars; k++) {
      const v  = 0.2 + 0.8 * Math.abs(Math.sin(this.t * 0.9 + u.seed + k * 0.73) * Math.sin(this.t * 0.41 + k * 1.31 + u.seed * 2));
      const bh = v * (uh - 12);
      ctx.fillStyle = `rgba(${this.blue}, ${0.18 + 0.5 * v})`;
      ctx.fillRect(bx0 + k * bw, uy + uh - 6 - bh, bw - 2.5, bh);
    }

    const fx  = ux + uw - 17;
    const rot = REDUCED_MOTION ? u.fan : this.t * u.fanSpeed * 60;
    ctx.strokeStyle = `rgba(${this.ink}, 0.4)`;
    ctx.beginPath();
    ctx.arc(fx, cy, Math.min(8, uh * 0.32), 0, Math.PI * 2);
    ctx.stroke();
    for (let s = 0; s < 3; s++) {
      const a = rot + (s * Math.PI * 2) / 3;
      ctx.beginPath();
      ctx.moveTo(fx, cy);
      ctx.lineTo(fx + Math.cos(a) * Math.min(7, uh * 0.28), cy + Math.sin(a) * Math.min(7, uh * 0.28));
      ctx.stroke();
    }
  }

  _network(ctx, endX, h) {
    const lanes = [h * 0.40, h * 0.60];

    ctx.save();
    ctx.setLineDash([2, 5]);
    ctx.strokeStyle = `rgba(${this.ink}, 0.22)`;
    ctx.lineWidth   = 1;
    lanes.forEach((y) => {
      ctx.beginPath();
      ctx.moveTo(4, y);
      ctx.lineTo(endX - 4, y);
      ctx.stroke();
    });
    ctx.restore();

    this.packets.forEach((pk) => {
      const span = endX - 12;
      const x    = pk.lane === 0 ? 6 + pk.p * span : endX - 6 - pk.p * span;
      const y    = lanes[pk.lane];
      const tint = pk.lane === 0 ? this.blue : this.teal;
      const edge = Math.min(1, Math.min(pk.p, 1 - pk.p) * 8);

      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${tint}, ${0.85 * edge})`;
      ctx.fill();
    });
  }

  _round(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  _loop() {
    this.t += 0.016;
    this.packets.forEach((pk) => {
      pk.p += pk.speed;
      if (pk.p > 1) pk.p -= 1;
    });
    this._draw();
    requestAnimationFrame(this._loop);
  }
}

window.ServerScene = ServerScene;
