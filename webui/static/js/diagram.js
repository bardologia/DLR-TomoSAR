"use strict";

class ModelDiagram {
  static C = {
    accent: "#35e6d0",
    accent2: "#7cff9b",
    boxStroke: "#35e6d0",
    boxFill: "rgba(53,230,208,0.07)",
    ioStroke: "rgba(124,160,176,0.5)",
    ioFill: "#0d141d",
    flow: "rgba(140,168,182,0.5)",
    skip: "rgba(124,255,155,0.55)",
  };

  static render(model) {
    return this.build(model).network;
  }

  static build(model) {
    const bp = this._blueprint(model);
    const blocks = {};
    bp.blocks.forEach((b) => { blocks[b.id] = b; });
    const network =
      `<figure class="dgm-net"><div class="dgm-frame dgm-frame--net">${this._network(bp.network)}</div>` +
      `<figcaption class="dgm-hint">Click any block to zoom into its operations</figcaption></figure>`;
    return { network, blocks };
  }

  static blockSvg(def) {
    return this._block(def);
  }

  static spec(model) {
    const skip = (model.skip || "").toLowerCase();
    let skipKind = "concat";
    if (skip.includes("attention")) skipKind = "attention";
    else if (skip.includes("nested")) skipKind = "nested";
    else if (skip.includes("additive")) skipKind = "additive";
    else if (skip.includes("residual")) skipKind = "residual";
    else if (skip.includes("token")) skipKind = "tokens";
    const head = (model.head || "").toLowerCase();
    let heads = 1;
    if (head.includes("3 ")) heads = 3;
    else if (head.includes("k ")) heads = "K";
    const isT = (model.family || "").toLowerCase().startsWith("transformer");
    return { skipKind, heads, type: isT ? "transformer" : "unet", key: model.key };
  }

  /* ---------- primitives ---------- */

  static _defs() {
    const C = this.C;
    return (
      `<defs>` +
      `<marker id="tipm" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,2 L8,5 L0,8" fill="none" stroke="${C.flow}" stroke-width="1.4"/></marker>` +
      `<marker id="tip2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,2 L8,5 L0,8" fill="none" stroke="${C.skip}" stroke-width="1.4"/></marker>` +
      `</defs>`
    );
  }

  static _node(x, y, w, h, label, sub, variant, idx, pick, subId) {
    const C = this.C;
    const stroke = variant === "io" ? C.ioStroke : C.boxStroke;
    const fill = variant === "io" ? C.ioFill : C.boxFill;
    const cx = x + w / 2;
    const delay = (idx * 0.03).toFixed(2);
    let text;
    if (sub) {
      text =
        `<text x="${cx}" y="${y + h / 2 - 3}" class="dgm-t">${label}</text>` +
        `<text x="${cx}" y="${y + h / 2 + 11}" class="dgm-s">${sub}</text>`;
    } else {
      text = `<text x="${cx}" y="${y + h / 2 + 4}" class="dgm-t">${label}</text>`;
    }
    let cls = "dgm-block", data = "";
    if (pick) { cls += " dgm-pick"; data = ` data-block="${pick}"`; }
    else if (subId) { cls += " dgm-pick dgm-sub"; data = ` data-subblock="${subId}"`; }
    return (
      `<g class="${cls}"${data} style="animation-delay:${delay}s">` +
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${fill}" stroke="${stroke}" stroke-width="1.4"/>` +
      text +
      `</g>`
    );
  }

  static _gridStack(cx, cy, opts, idx) {
    const C = opts.out ? this.C.accent2 : this.C.accent;
    const S = 46;
    const off = 4;
    const delay = (idx * 0.03).toFixed(2);
    let layers = "";
    for (let i = 3; i >= 1; i--) {
      const lx = cx - S / 2 + i * off;
      const ly = cy - S / 2 - i * off;
      layers += `<rect x="${lx}" y="${ly}" width="${S}" height="${S}" rx="3" fill="${this.C.ioFill}" stroke="${C}" stroke-width="1" opacity="${0.3 + (3 - i) * 0.12}"/>`;
    }
    const fx = cx - S / 2;
    const fy = cy - S / 2;
    let grid = `<rect x="${fx}" y="${fy}" width="${S}" height="${S}" rx="3" fill="${this.C.ioFill}" stroke="${C}" stroke-width="1.5"/>`;
    for (let k = 1; k < 6; k++) {
      const p = (S / 6) * k;
      grid += `<line x1="${fx + p}" y1="${fy}" x2="${fx + p}" y2="${fy + S}" stroke="${C}" stroke-width="0.5" opacity="0.4"/>`;
      grid += `<line x1="${fx}" y1="${fy + p}" x2="${fx + S}" y2="${fy + p}" stroke="${C}" stroke-width="0.5" opacity="0.4"/>`;
    }
    const lab =
      `<text x="${cx}" y="${cy - S / 2 - 30}" class="dgm-t">${opts.label}</text>` +
      `<text x="${cx}" y="${cy - S / 2 - 17}" class="dgm-s">${opts.sub}</text>`;
    const attr = opts.pick ? ` data-block="${opts.pick}" class="dgm-block dgm-pick"` : ` class="dgm-block"`;
    return `<g${attr} style="animation-delay:${delay}s">${layers}${grid}${lab}</g>`;
  }

  static _sum(cx, cy, sym, idx, r) {
    const C = this.C;
    const rad = r || 13;
    const delay = (idx * 0.03).toFixed(2);
    return (
      `<g class="dgm-block" style="animation-delay:${delay}s">` +
      `<circle cx="${cx}" cy="${cy}" r="${rad}" fill="${C.ioFill}" stroke="${C.accent2}" stroke-width="1.4"/>` +
      `<text x="${cx}" y="${cy + 4.5}" class="dgm-sum">${sym}</text></g>`
    );
  }

  static _arrow(x1, y1, x2, y2, second, idx) {
    const C = this.C;
    const base = second ? C.skip : C.flow;
    const lit = second ? C.accent2 : C.accent;
    const delay = (idx * 0.025 + 0.1).toFixed(2);
    return (
      `<line class="dgm-link" style="animation-delay:${delay}s" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${base}" stroke-width="1.4"/>` +
      `<line class="dgm-flow" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${lit}"/>`
    );
  }

  static _path(points, second, idx) {
    const C = this.C;
    const base = second ? C.skip : C.flow;
    const lit = second ? C.accent2 : C.accent;
    const delay = (idx * 0.03 + 0.18).toFixed(2);
    const cls = second ? "dgm-skip" : "dgm-link";
    const d = points.map((p) => `${p[0]},${p[1]}`).join(" ");
    return (
      `<polyline class="${cls}" style="animation-delay:${delay}s" points="${d}" fill="none" stroke="${base}" stroke-width="1.4"/>` +
      `<polyline class="dgm-flow" points="${d}" fill="none" stroke="${lit}"/>`
    );
  }

  static _label(x, y, text, second, anchor) {
    const a = anchor ? ` style="text-anchor:${anchor}"` : "";
    return `<text x="${x}" y="${y}"${a} class="${second ? "dgm-lbl" : "dgm-lbl dgm-lbl--flow"}">${text}</text>`;
  }

  /* ---------- network (orthogonal U) ---------- */

  static _network(g) {
    const colX = { left: 212, center: 400, right: 588 };
    const W = 720;
    const rowY = (r) => (r === 0 ? 66 : 210 + (r - 1) * 76);
    const bw = 156, bh = 52, bbw = 188, gw = 50;

    const box = {};
    g.nodes.forEach((n) => {
      const grid = n.type === "grid";
      const w = grid ? gw : n.col === "center" ? bbw : bw;
      const h = grid ? gw : bh;
      const cx = colX[n.col];
      const cy = n.yPix != null ? n.yPix : rowY(n.row);
      box[n.id] = { x: cx - w / 2, y: cy - h / 2, w, h, cx, cy };
    });

    let edges = "";
    let labels = "";
    g.edges.forEach((e, i) => {
      const a = box[e.from];
      const b = box[e.to];
      const second = e.kind === "skip";
      let pts;
      if (e.route === "V") {
        if (b.cy > a.cy) pts = [[a.cx, a.y + a.h], [b.cx, b.y]];
        else pts = [[a.cx, a.y], [b.cx, b.y + b.h]];
        edges += this._path(pts, second, i);
        if (e.label) labels += this._label(a.cx + 10, (a.y + a.h + b.y) / 2 + 3, e.label, second, "start");
      } else if (e.route === "H") {
        if (b.cx > a.cx) pts = [[a.x + a.w, a.cy], [b.x, b.cy]];
        else pts = [[a.x, a.cy], [b.x + b.w, b.cy]];
        edges += this._path(pts, second, i);
        if (e.label) labels += this._label((a.x + a.w + b.x) / 2, a.cy - 13, e.label, second);
        if (e.deco) edges += this._deco((a.x + a.w + b.x) / 2, a.cy, b.x, e.deco, i);
      } else if (e.route === "VH") {
        pts = [[a.cx, a.y + a.h], [a.cx, b.cy], [b.x, b.cy]];
        edges += this._path(pts, second, i);
      } else if (e.route === "HV") {
        pts = [[a.x + a.w, a.cy], [b.cx, a.cy], [b.cx, b.y + b.h]];
        edges += this._path(pts, second, i);
      }
    });

    let nodes = "";
    g.nodes.forEach((n, i) => {
      const b = box[n.id];
      if (n.type === "grid") nodes += this._gridStack(b.cx, b.cy, { label: n.label, sub: n.sub, out: n.out, pick: n.block }, i);
      else nodes += this._node(b.x, b.y, b.w, b.h, n.label, n.sub, n.variant || "op", i, n.block);
    });

    const H = rowY(g.rows) + bh / 2 + 30;
    return `<svg class="dgm dgm--net" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">${this._defs()}${edges}${labels}${nodes}</svg>`;
  }

  static _deco(mx, cy, endX, kind, idx) {
    const C = this.C;
    const delay = (idx * 0.03 + 0.35).toFixed(2);
    if (kind === "gate") {
      return (
        `<g class="dgm-skip" style="animation-delay:${delay}s">` +
        `<path d="M${mx - 9},${cy} L${mx},${cy - 9} L${mx + 9},${cy} L${mx},${cy + 9} Z" fill="${C.ioFill}" stroke="${C.accent2}" stroke-width="1.2"/>` +
        `<text x="${mx}" y="${cy + 3}" class="dgm-deco">AG</text></g>`
      );
    }
    if (kind === "add") {
      return `<g class="dgm-skip" style="animation-delay:${delay}s"><circle cx="${endX - 18}" cy="${cy}" r="9" fill="${C.ioFill}" stroke="${C.accent2}" stroke-width="1.2"/><text x="${endX - 18}" y="${cy + 4}" class="dgm-sum">+</text></g>`;
    }
    if (kind === "dots") {
      return `<g class="dgm-skip" style="animation-delay:${delay}s"><circle cx="${mx - 13}" cy="${cy}" r="4.5" fill="${C.ioFill}" stroke="${C.accent2}" stroke-width="1.1"/><circle cx="${mx + 13}" cy="${cy}" r="4.5" fill="${C.ioFill}" stroke="${C.accent2}" stroke-width="1.1"/></g>`;
    }
    return "";
  }

  /* ---------- block detail ---------- */

  static _block(block) {
    if (block.fan) return this._fan(block);
    if (block.horizontal) return this._blockH(block);

    const ops = block.ops;
    const up = block.upward;
    const wb = 208;
    const x0 = 24;
    const cx = x0 + wb / 2;
    const gap = 15;
    const topPad = 22;
    let out = "";

    const hs = ops.map((op) => (op.kind === "sum" || op.kind === "mul" ? 26 : op.kind === "io" ? 32 : 34));
    const off = [];
    let c = 0;
    ops.forEach((op, i) => { off[i] = c; c += hs[i] + gap; });
    const totalH = c - gap;

    const ys = ops.map((op, i) => {
      const yTop = up ? topPad + (totalH - off[i] - hs[i]) : topPad + off[i];
      return { y: yTop, h: hs[i] };
    });

    const first = ys[0];
    if (up) out += this._arrow(cx, first.y + first.h + 18, cx, first.y + first.h, false, 0);
    else out += this._arrow(cx, 4, cx, first.y, false, 0);

    for (let i = 0; i < ops.length - 1; i++) {
      const a = ys[i], b = ys[i + 1];
      if (a.y < b.y) out += this._arrow(cx, a.y + a.h, cx, b.y, false, i + 1);
      else out += this._arrow(cx, a.y, cx, b.y + b.h, false, i + 1);
    }

    ops.forEach((op, i) => {
      const slot = ys[i];
      if (op.kind === "sum") out += this._sum(cx, slot.y + slot.h / 2, "+", i);
      else if (op.kind === "mul") out += this._sum(cx, slot.y + slot.h / 2, "x", i);
      else out += this._node(x0, slot.y, wb, slot.h, op.label, op.sub, op.kind === "io" ? "io" : "op", i, null, op.block);
    });

    const rightX = x0 + wb + 40;
    (block.shortcuts || []).forEach((sc, k) => {
      const target = ys[sc.to];
      const ty = target.y + target.h / 2;
      const startY = sc.from === "top" ? (up ? first.y + first.h + 12 : 12) : ys[sc.from].y + ys[sc.from].h / 2;
      const startX = sc.from === "top" ? cx : x0 + wb;
      out += this._path([[startX, startY], [rightX, startY], [rightX, ty], [x0 + wb, ty]], true, k);
      out += this._label(rightX + 8, (startY + ty) / 2 + 3, sc.label, true, "start");
    });

    const W = rightX + 112;
    const H = topPad + totalH + (up ? 28 : 8);
    return `<svg class="dgm" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">${this._defs()}${out}</svg>`;
  }

  static _blockH(block) {
    const ops = block.ops;
    const hb = 58, gap = 26, rowCy = 42, wb = 126, sumW = 32;
    let x = 18;
    const xs = [];
    let out = "";

    ops.forEach((op) => {
      const isNode = op.kind === "sum" || op.kind === "mul";
      if (isNode) {
        xs.push({ cx: x + sumW / 2, w: sumW, node: true });
        x += sumW + gap;
      } else {
        xs.push({ x, cx: x + wb / 2, w: wb });
        x += wb + gap;
      }
    });

    out += this._arrow(3, rowCy, 16, rowCy, false, 0);
    for (let i = 0; i < ops.length - 1; i++) {
      const a = xs[i], b = xs[i + 1];
      const ax = a.node ? a.cx + a.w / 2 : a.x + a.w;
      const bx = b.node ? b.cx - b.w / 2 : b.x;
      out += this._arrow(ax, rowCy, bx, rowCy, false, i + 1);
    }

    ops.forEach((op, i) => {
      const s = xs[i];
      if (op.kind === "sum") out += this._sum(s.cx, rowCy, "+", i, 15);
      else if (op.kind === "mul") out += this._sum(s.cx, rowCy, "x", i, 15);
      else out += this._node(s.x, rowCy - hb / 2, s.w, hb, op.label, op.sub, op.kind === "io" ? "io" : "op", i, null, op.block);
    });

    const belowY = rowCy + hb / 2 + 26;
    const hasSc = block.shortcuts && block.shortcuts.length;
    (block.shortcuts || []).forEach((sc, k) => {
      const startX = sc.from === "top" ? 12 : xs[sc.from].cx;
      const startY = sc.from === "top" ? rowCy : rowCy + hb / 2;
      const sumCx = xs[sc.to].cx;
      out += this._path([[startX, startY], [startX, belowY], [sumCx, belowY], [sumCx, rowCy + 13]], true, k);
      out += this._label((startX + sumCx) / 2, belowY + 12, sc.label, true);
    });

    const W = x - gap + 18;
    const H = hasSc ? belowY + 22 : rowCy + hb / 2 + 14;
    return `<svg class="dgm dgm--h" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">${this._defs()}${out}</svg>`;
  }

  static _fan(block) {
    const labels = block.fan.slice(0, 3);
    const up = block.upward;
    const W = 336, wb = 92, hb = 32, cx = W / 2, inH = 30, outH = 26, vgap = 28, bgap = 14;
    let out = "";

    let y = 14;
    const P = {};
    const order = up ? ["out", "branch", "in"] : ["in", "branch", "out"];
    order.forEach((k) => {
      if (k === "in") { P.in = y; y += inH + vgap; }
      else if (k === "branch") { P.branch = y; y += hb + bgap; }
      else { P.out = y; y += outH + vgap; }
    });
    const H = y;

    out += this._node(cx - 96, P.in, 192, inH, block.inLabel, block.inSub, "io", 0);

    const branchesBelowInput = P.branch > P.in;
    const hubY = branchesBelowInput ? P.in + inH + 12 : P.in - 12;
    out += this._arrow(cx, branchesBelowInput ? P.in + inH : P.in, cx, hubY, false, 1);

    const slotW = wb + 14;
    const startX = cx - ((labels.length - 1) * slotW) / 2;
    labels.forEach((lab, i) => {
      const bCx = startX + i * slotW;
      const bx = bCx - wb / 2;
      out += this._path([[cx, hubY], [bCx, hubY], [bCx, branchesBelowInput ? P.branch : P.branch + hb]], false, 2 + i);
      out += this._node(bx, P.branch, wb, hb, lab.t, lab.sub, "op", 2 + i);
      if (P.out > P.branch) out += this._arrow(bCx, P.branch + hb, bCx, P.out, false, 5 + i);
      else out += this._arrow(bCx, P.branch, bCx, P.out + outH, false, 5 + i);
      out += this._node(bx, P.out, wb, outH, lab.out, null, "io", 5 + i);
    });

    return `<svg class="dgm" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">${this._defs()}${out}</svg>`;
  }

  /* ---------- op helpers ---------- */

  static _o = {
    conv: (a, b) => ({ label: "Conv 3x3", sub: a ? `${a} -> ${b}` : undefined }),
    na: () => ({ label: "Norm -> Act" }),
    pool: () => ({ label: "MaxPool 2x2", sub: "down" }),
    up: () => ({ label: "Up-conv 2x2", sub: "up" }),
    cat: () => ({ label: "concat skip" }),
    sum: () => ({ label: "+", kind: "sum" }),
    mul: () => ({ kind: "mul" }),
    io: (label, sub) => ({ label, sub, kind: "io" }),
    txt: (label, sub) => ({ label, sub }),
  };

  /* ---------- per-model blueprints ---------- */

  static _uGraph(o) {
    const sh = o.shapes;
    const nodes = [
      { id: "in", col: "left", row: 0, type: "grid", label: "input", sub: sh[0] },
      { id: "e1", col: "left", row: 1, label: o.enc, sub: sh[1], block: "enc" },
      { id: "e2", col: "left", row: 2, label: o.enc, sub: sh[2], block: "enc" },
      { id: "e3", col: "left", row: 3, label: o.enc, sub: sh[3], block: "enc" },
      { id: "e4", col: "left", row: 4, label: o.enc, sub: sh[4], block: "enc" },
      { id: "bn", col: "center", row: 5, label: o.bott, sub: sh[5], block: "bridge" },
      { id: "d4", col: "right", row: 4, label: o.dec, sub: sh[6], block: "dec" },
      { id: "d3", col: "right", row: 3, label: o.dec, sub: sh[7], block: "dec" },
      { id: "d2", col: "right", row: 2, label: o.dec, sub: sh[8], block: "dec" },
      { id: "d1", col: "right", row: 1, label: o.dec, sub: sh[9], block: "dec" },
      { id: "hd", col: "right", yPix: 132, label: "Output head", sub: o.headSub, block: "head" },
      { id: "out", col: "right", row: 0, type: "grid", out: true, label: "params", sub: sh[10] },
    ];
    const edges = [
      { from: "in", to: "e1", route: "V" },
      { from: "e1", to: "e2", route: "V" },
      { from: "e2", to: "e3", route: "V" },
      { from: "e3", to: "e4", route: "V" },
      { from: "e4", to: "bn", route: "VH" },
      { from: "bn", to: "d4", route: "HV" },
      { from: "d4", to: "d3", route: "V" },
      { from: "d3", to: "d2", route: "V" },
      { from: "d2", to: "d1", route: "V" },
      { from: "d1", to: "hd", route: "V" },
      { from: "hd", to: "out", route: "V" },
      { from: "e1", to: "d1", route: "H", kind: "skip", label: o.skip, deco: o.deco },
      { from: "e2", to: "d2", route: "H", kind: "skip", label: o.skip, deco: o.deco },
      { from: "e3", to: "d3", route: "H", kind: "skip", label: o.skip, deco: o.deco },
      { from: "e4", to: "d4", route: "H", kind: "skip", label: o.skip, deco: o.deco },
    ];
    return { nodes, edges, rows: 5 };
  }

  static _blocks(s) {
    const o = this._o;
    const head1 = { id: "head", title: "Output head", caption: "1x1 conv to params", ops: [o.txt("Conv 1x1", "F1 -> 3K"), o.io("params", "3K x P x P")] };

    if (s.type === "transformer") {
      const mhsa = s.key === "swin_unet" ? "W-MSA" : "Multi-head attn";
      const txBlock = { id: "tx", title: "Transformer block", caption: "attention then MLP, each residual", ops: [o.txt("LayerNorm"), o.txt(mhsa), o.sum(), o.txt("LayerNorm"), o.txt("MLP", "GELU"), o.sum(), o.io("output")], shortcuts: [{ from: "top", to: 2, label: "res" }, { from: 2, to: 5, label: "res" }] };
      const down = s.key === "swin_unet" ? o.txt("Patch merge", "down") : o.txt("Down-sample", "down");
      const up = s.key === "unetr" ? o.txt("Deconv 2x2", "up") : s.key === "swin_unet" ? o.txt("Patch expand", "up") : o.txt("Upsample", "up");
      const txOp = { label: "Transformer block", sub: "x N", block: "tx" };
      const enc = { id: "enc", title: "Encoder block", caption: "downsample, then transformer block", ops: [down, txOp, o.io("output")] };
      const bridge = { id: "bridge", title: "Bridge", caption: "deepest transformer block", ops: [{ label: "Transformer", sub: "x N", block: "tx" }, o.io("output")] };
      const decRefine = s.key === "transunet" ? o.txt("Conv 3x3") : { label: "Transformer block", block: "tx" };
      const dec = { id: "dec", title: "Decoder block", caption: "upsample, fuse skip, refine", ops: [up, o.cat(), decRefine, o.io("output")] };
      return [enc, bridge, txBlock, dec, head1];
    }

    if (s.skipKind === "residual") {
      const enc = { id: "enc", title: "Encoder block", caption: "residual block, then max-pool", ops: [o.na(), o.conv("Cin", "F"), o.na(), o.conv("F", "F"), o.sum(), o.pool(), o.io("output", "F x H/2")], shortcuts: [{ from: "top", to: 4, label: "Proj" }] };
      const bridge = { id: "bridge", title: "Bridge", caption: "residual bottleneck", ops: [o.na(), o.conv("F", "2F"), o.na(), o.conv("2F", "2F"), o.sum(), o.io("output", "2F x H")], shortcuts: [{ from: "top", to: 4, label: "Proj" }] };
      const dec = { id: "dec", title: "Decoder block", caption: "up-conv, concat, residual block", ops: [o.up(), o.cat(), o.na(), o.conv("2F", "F"), o.na(), o.conv("F", "F"), o.sum(), o.io("output", "F x 2H")], shortcuts: [{ from: 1, to: 6, label: "Proj" }] };
      return [enc, bridge, dec, head1];
    }

    if (s.skipKind === "additive") {
      const enc = { id: "enc", title: "Encoder block", caption: "residual conv, then downsample", ops: [o.conv("Cin", "F"), o.na(), o.conv("F", "F"), o.sum(), o.pool(), o.io("output", "F x H/2")], shortcuts: [{ from: "top", to: 3, label: "skip" }] };
      const bridge = { id: "bridge", title: "Bridge", caption: "bottleneck conv", ops: [o.conv("F", "2F"), o.na(), o.conv("2F", "2F"), o.na(), o.io("output", "2F x H")] };
      const dec = { id: "dec", title: "Decoder block", caption: "reduce, up-conv, add encoder skip", ops: [o.txt("Conv 1x1", "C -> C/4"), o.txt("ConvT 3x3", "up"), o.txt("Conv 1x1", "C/4 -> C"), o.sum(), o.io("output", "F x 2H")], shortcuts: [{ from: "top", to: 3, label: "encoder skip" }] };
      return [enc, bridge, dec, head1];
    }

    const enc = { id: "enc", title: "Encoder block", caption: "double conv, then 2x2 max-pool", ops: [o.conv("Cin", "F"), o.na(), o.conv("F", "F"), o.na(), o.pool(), o.io("output", "F x H/2")] };
    const bridge = { id: "bridge", title: "Bridge", caption: "bottleneck double conv", ops: [o.conv("F", "2F"), o.na(), o.conv("2F", "2F"), o.na(), o.io("output", "2F x H")] };

    if (s.skipKind === "attention") {
      const dec = { id: "dec", title: "Decoder block", caption: "up-conv, attention-gated skip, double conv", ops: [o.up(), { label: "Attention gate", sub: "on skip", block: "attn" }, o.cat(), o.conv("2F", "F"), o.na(), o.conv("F", "F"), o.na(), o.io("output", "F x 2H")] };
      const gate = { id: "attn", title: "Attention gate", caption: "filters the skip feature", ops: [o.io("gate g + skip x"), o.txt("Conv 1x1 -> Add"), o.txt("ReLU"), o.txt("Conv 1x1 -> Sig", "psi in [0,1]"), o.mul(), o.io("gated skip")], shortcuts: [{ from: 0, to: 4, label: "skip x" }] };
      return [enc, bridge, dec, gate, head1];
    }

    if (s.skipKind === "nested") {
      const dec = { id: "dec", title: "Dense node", caption: "fuse dense skips, double conv", ops: [o.io("concat dense inputs", "up + same level"), o.conv("", ""), o.na(), o.conv("", ""), o.io("node X(i,j)")] };
      return [enc, bridge, dec, head1];
    }

    const decPlain = { id: "dec", title: "Decoder block", caption: "up-conv, concat skip, double conv", ops: [o.up(), o.cat(), o.conv("2F", "F"), o.na(), o.conv("F", "F"), o.na(), o.io("output", "F x 2H")] };

    if (s.heads === 3 || s.heads === "K") {
      const labs = s.heads === 3
        ? [{ t: "PixelMLP", sub: "amp", out: "a" }, { t: "PixelMLP", sub: "mean", out: "mu" }, { t: "PixelMLP", sub: "spread", out: "sig" }]
        : [{ t: "PixelMLP", sub: "k=1", out: "a,mu,s" }, { t: "PixelMLP", sub: "k=2", out: "a,mu,s" }, { t: "PixelMLP", sub: "k=K", out: "a,mu,s" }];
      const headFan = { id: "head", title: s.heads === 3 ? "Per-type heads" : "Per-Gaussian heads", fan: labs, inLabel: "decoder features", inSub: "d1 : F1 x P x P" };
      return [enc, bridge, decPlain, headFan];
    }

    return [enc, bridge, decPlain, head1];
  }

  static _blueprint(model) {
    const s = this.spec(model);
    const cnnShapes = ["Cin x P x P", "F1 x P/2 x P/2", "F2 x P/4 x P/4", "F3 x P/8 x P/8", "F4 x P/16 x P/16", "Fb x P/16 x P/16", "F4 x P/8 x P/8", "F3 x P/4 x P/4", "F2 x P/2 x P/2", "F1 x P x P", "3K x P x P"];
    const txShapes = ["Cin x P x P", "D1 x N1", "D2 x N2", "D3 x N3", "D4 x N4", "Db x N4", "D4 x N3", "D3 x N2", "D2 x N1", "D1 x N1", "3K x P x P"];
    const headSub = s.heads === 3 ? "3 x PixelMLP" : s.heads === "K" ? "K x PixelMLP" : "1x1 conv";

    const labels = {
      unet: { enc: "Encoder", dec: "Decoder", bott: "Bridge", skip: "concat" },
      resunet: { enc: "Residual encoder", dec: "Residual decoder", bott: "Residual bridge", skip: "concat" },
      attention_unet: { enc: "Encoder", dec: "Decoder", bott: "Bridge", skip: "concat", deco: "gate" },
      unetplusplus: { enc: "Encoder", dec: "Dense node", bott: "Bridge", skip: "dense", deco: "dots" },
      linknet: { enc: "Residual encoder", dec: "Decoder", bott: "Bridge", skip: "add", deco: "add" },
      swin_unet: { enc: "Swin encoder", dec: "Swin decoder", bott: "Swin bridge", skip: "concat" },
      transunet: { enc: "ViT encoder", dec: "CNN decoder", bott: "ViT bridge", skip: "skip" },
      unetr: { enc: "ViT encoder", dec: "Deconv decoder", bott: "ViT bridge", skip: "skip" },
      unet_multihead: { enc: "Encoder", dec: "Decoder", bott: "Bridge", skip: "concat" },
      unet_pergaussian: { enc: "Encoder", dec: "Decoder", bott: "Bridge", skip: "concat" },
    };

    const o = labels[s.key] || labels.unet;
    o.headSub = headSub;
    o.shapes = s.type === "transformer" ? txShapes : cnnShapes;
    const blocks = this._blocks(s);
    blocks.forEach((b) => {
      if (b.id === "bridge") b.horizontal = true;
      if (b.id === "dec" || b.id === "head") b.upward = true;
    });
    return { network: this._uGraph(o), blocks };
  }
}

window.ModelDiagram = ModelDiagram;
