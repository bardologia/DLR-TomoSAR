"use strict";

class RepoMapView {
  constructor(root) {
    this.root     = root;
    this.folders  = [];
    this.folder   = null;
    this.diagram  = null;
    this.loaded   = false;
    this.selNode  = null;
    this.ro       = null;
    this.rafId    = null;
    this.wireEls  = [];
    this.labelEls = [];
    this.trace    = { on: false, col: -1, timer: null };

    this.ROLE_COLORS = {
      entry        : "#6d28d9",
      orchestrator : "#1d4fd8",
      config       : "#b45309",
      transform    : "#0e7490",
      model        : "#be185d",
      data         : "#15803d",
      io           : "#0f766e",
      metric       : "#c2410c",
      external     : "#64748b",
    };
    this.ROLE_LABELS = {
      entry        : "Entry",
      orchestrator : "Orchestrator",
      config       : "Config",
      transform    : "Transform",
      model        : "Model",
      data         : "Data",
      io           : "I/O",
      metric       : "Metric",
      external     : "External",
    };
    this.ROLE_ORDER = ["entry", "orchestrator", "config", "transform", "model", "data", "io", "metric", "external"];
  }

  async enter() {
    if (!this.loaded) { await this.load(); return; }
    this._redraw();
  }

  async load() {
    const data = await window.apiGet("/api/repomap");
    if (!data || data.error) { this._empty("Backend unreachable."); return; }
    this.folders = data.folders || [];
    this.loaded  = true;
    if (!this.folders.length) { this._empty("No repo map data yet."); return; }
    this._buildShell();
    this._selectFolder(this.folders[0].folder);
  }

  _empty(msg) {
    this.root.innerHTML = `<div class="rm-empty">${this._esc(msg)}</div>`;
  }

  _buildShell() {
    this.root.innerHTML = "";
    this.root.classList.add("rm");

    this.foldersEl = document.createElement("div");
    this.foldersEl.className = "rm__folders";
    this.foldersEl.setAttribute("role", "tablist");
    this.foldersEl.setAttribute("aria-label", "Subsystems");
    this.folders.forEach((f) => {
      const b = document.createElement("button");
      b.className   = "rm-folder";
      b.dataset.key = f.folder;
      b.setAttribute("role", "tab");
      b.innerHTML   = `<span class="rm-folder__name">${this._esc(f.title)}</span><span class="rm-folder__n">${f.diagrams.length}</span>`;
      b.addEventListener("click", () => this._selectFolder(f.folder));
      this.foldersEl.appendChild(b);
    });

    this.panel = document.createElement("div");
    this.panel.className = "rm__panel";
    this.panel.innerHTML = `
      <div class="rm__head">
        <div class="rm__subtabs" role="tablist" aria-label="Stages"></div>
        <div class="rm__tools">
          <button class="rm-tool rm-tool--labels" type="button" title="Show every data-exchange label at once (otherwise hover a class to reveal its labels)">
            <span class="rm-tool__ic">&#8801;</span><span class="rm-tool__lb">Labels</span>
          </button>
          <button class="rm-tool rm-tool--trace" type="button" title="Animate the data flow through the pipeline">
            <span class="rm-tool__ic">&#9654;</span><span class="rm-tool__lb">Trace flow</span>
          </button>
        </div>
      </div>
      <div class="rm__title">
        <h3 class="rm__stage-name"></h3>
        <a class="rm__entry" target="_blank" rel="noopener"></a>
      </div>
      <div class="rm__stage">
        <div class="rm__legend"></div>
        <div class="rm-graph">
          <div class="rm-canvas">
            <svg class="rm-wires" aria-hidden="true"><defs>
              <marker id="rm-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
                <path d="M0,1 L9,5 L0,9" fill="none" stroke="context-stroke" stroke-width="1.5"></path>
              </marker></defs>
            </svg>
            <div class="rm-cols"></div>
            <div class="rm-labels"></div>
          </div>
        </div>
      </div>
      <div class="rm__ledger"></div>`;

    this.subtabsEl = this.panel.querySelector(".rm__subtabs");
    this.labelsBtn = this.panel.querySelector(".rm-tool--labels");
    this.traceBtn  = this.panel.querySelector(".rm-tool--trace");
    this.nameEl    = this.panel.querySelector(".rm__stage-name");
    this.entryEl   = this.panel.querySelector(".rm__entry");
    this.legendEl  = this.panel.querySelector(".rm__legend");
    this.graphEl   = this.panel.querySelector(".rm-graph");
    this.canvasEl  = this.panel.querySelector(".rm-canvas");
    this.wiresEl   = this.panel.querySelector(".rm-wires");
    this.colsEl    = this.panel.querySelector(".rm-cols");
    this.labelsEl  = this.panel.querySelector(".rm-labels");
    this.ledgerEl  = this.panel.querySelector(".rm__ledger");

    this.traceBtn.addEventListener("click", () => this._toggleTrace());
    this.labelsBtn.addEventListener("click", () => {
      const on = this.canvasEl.classList.toggle("rm-showlabels");
      this.labelsBtn.classList.toggle("is-on", on);
    });

    this.root.appendChild(this.foldersEl);
    this.root.appendChild(this.panel);

    if (window.ResizeObserver) {
      this.ro = new ResizeObserver(() => this._scheduleWires());
      this.ro.observe(this.graphEl);
    }
    window.addEventListener("resize", () => this._scheduleWires());
  }

  _selectFolder(key) {
    this.folder = this.folders.find((f) => f.folder === key) || this.folders[0];
    [...this.foldersEl.children].forEach((b) => b.classList.toggle("is-active", b.dataset.key === this.folder.folder));

    this.subtabsEl.innerHTML = "";
    this.folder.diagrams.forEach((d) => {
      const b = document.createElement("button");
      b.className   = "rm-sub";
      b.dataset.key = d.key;
      b.textContent = d.stage;
      b.setAttribute("role", "tab");
      b.addEventListener("click", () => this._selectDiagram(d.key));
      this.subtabsEl.appendChild(b);
    });

    this._selectDiagram(this.folder.diagrams[0].key);
  }

  _selectDiagram(key) {
    this._stopTrace();
    this.diagram = this.folder.diagrams.find((d) => d.key === key) || this.folder.diagrams[0];
    this.selNode = null;
    [...this.subtabsEl.children].forEach((b) => b.classList.toggle("is-active", b.dataset.key === this.diagram.key));

    this.nameEl.textContent  = this.diagram.title;

    if (this.diagram.entry) {
      this.entryEl.style.display = "";
      this.entryEl.textContent   = this.diagram.entry;
      this.entryEl.href          = "#/scripts";
    } else {
      this.entryEl.style.display = "none";
    }

    this._buildLegend();
    this._buildNodes();
    this._buildLedger();
    this._redraw();
  }

  _buildLegend() {
    this.legendEl.innerHTML = "";
    const present = new Set(this.diagram.nodes.map((n) => n.role));
    this.ROLE_ORDER.filter((r) => present.has(r)).forEach((role) => {
      const item = document.createElement("span");
      item.className = "rm-leg";
      item.style.setProperty("--role", this.ROLE_COLORS[role]);
      item.innerHTML = `<i></i>${this._esc(this.ROLE_LABELS[role])}`;
      this.legendEl.appendChild(item);
    });
  }

  _buildNodes() {
    this.colsEl.innerHTML   = "";
    this.labelsEl.innerHTML = "";
    this.nodeById = {};

    const ncol = this.diagram.ncol || (this.diagram.nodes.reduce((m, n) => Math.max(m, n.col || 0), 0) + 1);
    const nrow = this.diagram.nrow || (this.diagram.nodes.reduce((m, n) => Math.max(m, n.row || 0), 0) + 1);
    this.colsEl.style.setProperty("--ncol", ncol);
    this.colsEl.style.setProperty("--nrow", nrow);

    this.diagram.nodes.forEach((n) => {
      const el = document.createElement("div");
      el.className   = "rm-node rm-node--" + n.role;
      el.dataset.id  = n.id;
      el.dataset.col = String(n.col || 0);
      el.style.gridColumn = String((n.col || 0) + 1);
      el.style.gridRow    = String((n.row || 0) + 1);
      el.style.setProperty("--role", this.ROLE_COLORS[n.role] || this.ROLE_COLORS.external);

      const io = [];
      if (n.writes && n.writes.length) io.push(`<span class="rm-io rm-io--w" title="writes ${this._esc(n.writes.join(", "))}">out ${n.writes.length}</span>`);
      if (n.reads && n.reads.length)   io.push(`<span class="rm-io rm-io--r" title="reads ${this._esc(n.reads.join(", "))}">in ${n.reads.length}</span>`);

      el.innerHTML =
        `<div class="rm-node__top"><span class="rm-node__role">${this._esc(this.ROLE_LABELS[n.role] || n.role)}</span><span class="rm-node__io">${io.join("")}</span></div>` +
        `<h4 class="rm-node__name">${this._esc(n.label)}</h4>` +
        `<p class="rm-node__fn" title="${this._esc(n.fn)}">${this._esc(n.fn)}</p>` +
        `<code class="rm-node__mod">${this._esc(n.module)}</code>`;

      el.addEventListener("click", () => this._focusNode(n.id));
      el.addEventListener("mouseenter", () => this._hoverNode(n.id, true));
      el.addEventListener("mouseleave", () => this._hoverNode(n.id, false));
      this.colsEl.appendChild(el);
      this.nodeById[n.id] = el;
    });
  }

  _buildLedger() {
    const arts = this.diagram.artifacts || [];
    if (!arts.length) { this.ledgerEl.innerHTML = ""; return; }

    const rows = arts.map((a) => {
      const cross = a.scope === "cross" ? `<span class="rm-tag rm-tag--cross">cross-pipeline</span>` : "";
      const cons  = (a.consumers || []).map((c) => this._esc(c)).join(", ") || "&mdash;";
      return `<tr data-producer="${this._esc(a.producer)}" data-consumers="${this._esc((a.consumers || []).join("|"))}">
        <td class="rm-led__name"><code>${this._esc(a.name)}</code>${cross}</td>
        <td class="rm-led__fmt">${this._esc(a.fmt || "")}</td>
        <td class="rm-led__by">${this._esc(a.producer)}</td>
        <td class="rm-led__to">${cons}</td>
        <td class="rm-led__desc">${this._esc(a.desc || "")}</td>
      </tr>`;
    }).join("");

    this.ledgerEl.classList.remove("is-open");
    this.ledgerEl.innerHTML = `
      <button class="rm-led__toggle" type="button" aria-expanded="false">
        <span class="rm-led__chev">&#9656;</span>
        <span class="rm-led__lbl">Artifact ledger</span>
        <span class="rm-led__count">${arts.length}</span>
        <span class="rm-led__sub">what each stage saves and who reads it</span>
      </button>
      <div class="rm-led__body" hidden>
        <div class="rm-led__scroll"><table class="rm-led">
          <thead><tr><th>Artifact</th><th>Format</th><th>Written by</th><th>Read by</th><th>Holds</th></tr></thead>
          <tbody>${rows}</tbody>
        </table></div>
      </div>`;

    const toggle = this.ledgerEl.querySelector(".rm-led__toggle");
    const body   = this.ledgerEl.querySelector(".rm-led__body");
    toggle.addEventListener("click", () => {
      const open = body.hasAttribute("hidden");
      if (open) body.removeAttribute("hidden"); else body.setAttribute("hidden", "");
      toggle.setAttribute("aria-expanded", String(open));
      this.ledgerEl.classList.toggle("is-open", open);
    });

    this.ledgerEl.querySelectorAll("tbody tr").forEach((tr) => {
      tr.addEventListener("mouseenter", () => this._highlightArtifact(tr));
      tr.addEventListener("mouseleave", () => this._clearHighlight());
    });
  }

  _focusNode(id) {
    if (this.selNode === id) { this.selNode = null; this._clearHighlight(); return; }
    this.selNode = id;
    const incident = new Set([id]);
    (this.diagram.edges || []).forEach((e) => {
      if (e.from === id || e.to === id) { incident.add(e.from); incident.add(e.to); }
    });
    this.canvasEl.classList.add("is-focused");
    Object.entries(this.nodeById).forEach(([nid, el]) => el.classList.toggle("is-dim", !incident.has(nid)));
    this.wireEls.forEach((w) => {
      const on = w.from === id || w.to === id;
      w.base.classList.toggle("is-lit", on);
      w.base.classList.toggle("is-dim", !on);
      w.flow.classList.toggle("is-dim", !on);
    });
    this.labelEls.forEach((l) => {
      const on = l.from === id || l.to === id;
      l.el.classList.toggle("is-lit", on);
      l.el.classList.toggle("is-dim", !on);
    });
  }

  _hoverNode(id, on) {
    if (this.selNode || this.trace.on) return;
    this.labelEls.forEach((l) => {
      if (l.from === id || l.to === id) l.el.classList.toggle("is-show", on);
    });
    this.wireEls.forEach((w) => {
      if (w.from === id || w.to === id) w.base.classList.toggle("is-lit", on);
    });
  }

  _highlightArtifact(tr) {
    const names = new Set();
    if (tr.dataset.producer) names.add(tr.dataset.producer);
    (tr.dataset.consumers ? tr.dataset.consumers.split("|") : []).forEach((n) => names.add(n));
    this.canvasEl.classList.add("is-focused");
    this.diagram.nodes.forEach((n) => {
      const hit = names.has(n.label) || names.has(n.id);
      const el  = this.nodeById[n.id];
      if (el) el.classList.toggle("is-dim", !hit);
    });
  }

  _clearHighlight() {
    if (this.selNode) return;
    this.canvasEl.classList.remove("is-focused");
    Object.values(this.nodeById).forEach((el) => el.classList.remove("is-dim"));
    this.wireEls.forEach((w) => { w.base.classList.remove("is-lit", "is-dim"); w.flow.classList.remove("is-dim"); });
    this.labelEls.forEach((l) => l.el.classList.remove("is-lit", "is-dim", "is-show"));
  }

  _scheduleWires() {
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = requestAnimationFrame(() => this._drawWires());
  }

  _redraw() {
    requestAnimationFrame(() => requestAnimationFrame(() => this._drawWires()));
  }

  _drawWires() {
    if (!this.diagram || !this.canvasEl) return;
    const w = this.canvasEl.offsetWidth;
    const h = this.canvasEl.offsetHeight;
    if (!w || !h) return;

    this.wiresEl.setAttribute("width", w);
    this.wiresEl.setAttribute("height", h);
    this.wiresEl.setAttribute("viewBox", `0 0 ${w} ${h}`);

    const defs = this.wiresEl.querySelector("defs");
    this.wiresEl.innerHTML = "";
    this.wiresEl.appendChild(defs);
    this.labelsEl.innerHTML = "";
    this.wireEls  = [];
    this.labelEls = [];

    const crect = this.canvasEl.getBoundingClientRect();
    const box   = (id) => {
      const el = this.nodeById[id];
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { x: r.left - crect.left, y: r.top - crect.top, w: r.width, h: r.height, col: Number(el.dataset.col) };
    };
    const roleOf = (id) => { const n = this.diagram.nodes.find((nn) => nn.id === id); return n ? n.role : "external"; };

    // Faces are directional: inputs enter LEFT/TOP, outputs leave RIGHT/BOTTOM. Adjacent cards are
    // joined by a straight line between their closest face centres; diagonals use a single elbow
    // (one horizontal + one vertical face); backward/blocked edges take a side route.
    const AV = 22;
    const nodeBoxes = this.diagram.nodes.map((n) => box(n.id)).filter(Boolean);
    const recs = [];
    (this.diagram.edges || []).forEach((e) => {
      const a = box(e.from), b = box(e.to);
      if (!a || !b) return;
      const acx = a.x + a.w / 2, acy = a.y + a.h / 2;
      const bcx = b.x + b.w / 2, bcy = b.y + b.h / 2;
      const dx  = bcx - acx, dy = bcy - acy;

      // A straight line is only safe if no other card sits on it.
      const between = (vertical) => nodeBoxes.some((o) => {
        const ocx = o.x + o.w / 2, ocy = o.y + o.h / 2;
        return vertical
          ? (Math.abs(ocx - acx) <= o.w / 2 && ocy > Math.min(acy, bcy) + 1 && ocy < Math.max(acy, bcy) - 1)
          : (Math.abs(ocy - acy) <= o.h / 2 && ocx > Math.min(acx, bcx) + 1 && ocx < Math.max(acx, bcx) - 1);
      });
      const sameRow = Math.abs(dy) <= AV, sameCol = Math.abs(dx) <= AV;

      let fa, fb, aligned = false;
      if (sameRow && !between(false)) { aligned = true; if (dx >= 0) { fa = "R"; fb = "L"; } else { fa = "L"; fb = "R"; } }
      else if (sameCol && !between(true)) { aligned = true; if (dy >= 0) { fa = "B"; fb = "T"; } else { fa = "T"; fb = "B"; } }
      else if (dx >= 0 && dy > 0 && !sameCol) { if (dx >= dy) { fa = "R"; fb = "T"; } else { fa = "B"; fb = "L"; } }
      else if (dx >= 0) { fa = "R"; fb = "L"; }
      else { fa = "L"; fb = "R"; }
      recs.push({ e, a, b, fa, fb, aligned, acx, acy, bcx, bcy, bend: 0 });
    });

    // Keep each face one-directional: if a face carries both entering and leaving wires, move the
    // movable (non-straight) minority onto an empty or same-direction face that points its way.
    const NB = { R: [1, 0], L: [-1, 0], B: [0, 1], T: [0, -1] };
    const faceDir = {};                    // "id|face" -> { out: [recs], in: [recs] }
    const reg = (id, f, r, dir) => { const k = id + "|" + f; (faceDir[k] = faceDir[k] || { out: [], in: [] })[dir].push(r); };
    recs.forEach((r) => { reg(r.e.from, r.fa, r, "out"); reg(r.e.to, r.fb, r, "in"); });
    this.diagram.nodes.forEach((n) => {
      const id = n.id;
      ["R", "L", "T", "B"].forEach((f) => {
        const dd = faceDir[id + "|" + f];
        if (!dd || !dd.out.length || !dd.in.length) return;         // absent or not mixed
        const outMovable = dd.out.every((r) => !r.aligned), inMovable = dd.in.every((r) => !r.aligned);
        let dir = (outMovable && inMovable) ? (dd.out.length <= dd.in.length ? "out" : "in")
                : outMovable ? "out" : inMovable ? "in" : null;
        if (!dir) return;
        const opp = dir === "out" ? "in" : "out";
        dd[dir].slice().forEach((r) => {
          const self = dir === "out" ? r.a : r.b, other = dir === "out" ? r.b : r.a;
          const vx = (other.x + other.w / 2) - (self.x + self.w / 2), vy = (other.y + other.h / 2) - (self.y + self.h / 2);
          let best = null, bestDot = 0.01;
          ["R", "L", "T", "B"].forEach((cf) => {
            if (cf === f) return;
            const cd = faceDir[id + "|" + cf];
            if (cd && (!cd[dir].length || cd[opp].length)) return;  // occupied by the opposite direction
            const dot = NB[cf][0] * vx + NB[cf][1] * vy;
            if (dot > bestDot) { bestDot = dot; best = cf; }
          });
          if (!best) return;
          faceDir[id + "|" + f][dir] = faceDir[id + "|" + f][dir].filter((x) => x !== r);
          reg(id, best, r, dir);
          if (dir === "out") r.fa = best; else r.fb = best;
        });
      });
    });

    // Straight wires attach at the exact middle of their facing sides. Every other wire gets a
    // distinct slot along its face, sorted by the opposite endpoint so the fan stays
    // crossing-free; a face that carries a straight wire keeps its middle free for it.
    const faceCenter = (bx, f) =>
      f === "R" ? { x: bx.x + bx.w,     y: bx.y + bx.h / 2 } :
      f === "L" ? { x: bx.x,            y: bx.y + bx.h / 2 } :
      f === "B" ? { x: bx.x + bx.w / 2, y: bx.y + bx.h }     :
                  { x: bx.x + bx.w / 2, y: bx.y };
    const groups = {}, reserved = {};
    const push = (id, face, item) => { (groups[id + "|" + face] = groups[id + "|" + face] || []).push(item); };
    recs.forEach((r) => {
      if (r.aligned) {
        r.pa = faceCenter(r.a, r.fa);
        r.pb = faceCenter(r.b, r.fb);
        reserved[r.e.from + "|" + r.fa] = true;
        reserved[r.e.to   + "|" + r.fb] = true;
        return;
      }
      push(r.e.from, r.fa, { r, end: "a", perp: (r.fa === "L" || r.fa === "R") ? r.bcy : r.bcx });
      push(r.e.to,   r.fb, { r, end: "b", perp: (r.fb === "L" || r.fb === "R") ? r.acy : r.acx });
    });
    Object.entries(groups).forEach(([key, list]) => {
      list.sort((p, q) => p.perp - q.perp);
      const face = key.slice(key.indexOf("|") + 1);
      const horiz = face === "L" || face === "R";
      const n = list.length, keepCenter = !!reserved[key];
      list.forEach((item, i) => {
        const r = item.r, bx = item.end === "a" ? r.a : r.b;
        const span = horiz ? bx.h : bx.w;
        const m    = Math.min(horiz ? 13 : 16, span / 2 - 2);
        const frac = n === 1 ? (keepCenter ? 0.72 : 0.5) : i / (n - 1);
        const along = m + (span - 2 * m) * frac;
        const pt = {};
        if      (face === "R") { pt.x = bx.x + bx.w; pt.y = bx.y + along; }
        else if (face === "L") { pt.x = bx.x;        pt.y = bx.y + along; }
        else if (face === "B") { pt.y = bx.y + bx.h; pt.x = bx.x + along; }
        else                   { pt.y = bx.y;        pt.x = bx.x + along; }
        const zish = ((r.fa === "L" || r.fa === "R") === (r.fb === "L" || r.fb === "R"));
        if (item.end === "a") r.pa = pt;
        else { r.pb = pt; r.bend = (zish && n > 1) ? (i - (n - 1) / 2) * 9 : 0; }
      });
    });

    recs.forEach((r) => {
      const geom  = this._route(r.pa, r.fa, r.pb, r.fb, r.bend);
      const color = this.ROLE_COLORS[roleOf(r.e.from)] || this.ROLE_COLORS.external;

      const base = this._path(geom.d, "rm-wire", color);
      base.setAttribute("marker-end", "url(#rm-arrow)");
      const flow = this._path(geom.d, "rm-flow", color);
      this.wiresEl.appendChild(base);
      this.wiresEl.appendChild(flow);
      this.wireEls.push({ base, flow, from: r.e.from, to: r.e.to, fcol: r.a.col, tcol: r.b.col });

      if (r.e.label) {
        const lab = document.createElement("span");
        lab.className   = "rm-elabel rm-elabel--" + (r.e.kind || "data");
        lab.style.left  = geom.mx + "px";
        lab.style.top   = geom.my + "px";
        lab.textContent = r.e.label;
        this.labelsEl.appendChild(lab);
        this.labelEls.push({ el: lab, from: r.e.from, to: r.e.to, tcol: r.b.col });
      }
    });

    if (this.selNode) { const s = this.selNode; this.selNode = null; this._focusNode(s); }
  }

  _path(d, cls, color) {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    p.setAttribute("class", cls);
    p.setAttribute("stroke", color);
    return p;
  }

  _route(pa, fa, pb, fb, bend) {
    const bnd = bend || 0;
    const hA = fa === "L" || fa === "R", hB = fb === "L" || fb === "R";

    if (hA && hB) {                                              // both side faces: straight or Z
      if (Math.abs(pa.y - pb.y) < 1) return { d: `M ${pa.x} ${pa.y} L ${pb.x} ${pb.y}`, mx: (pa.x + pb.x) / 2, my: pa.y };
      let mx = (pa.x + pb.x) / 2 + bnd;
      const lo = Math.min(pa.x, pb.x) + 8, hi = Math.max(pa.x, pb.x) - 8;
      if (lo <= hi) mx = Math.max(lo, Math.min(hi, mx));
      return { d: `M ${pa.x} ${pa.y} H ${mx} V ${pb.y} H ${pb.x}`, mx, my: (pa.y + pb.y) / 2 };
    }
    if (!hA && !hB) {                                            // both top/bottom faces: straight or Z
      if (Math.abs(pa.x - pb.x) < 1) return { d: `M ${pa.x} ${pa.y} L ${pb.x} ${pb.y}`, mx: pa.x, my: (pa.y + pb.y) / 2 };
      let my = (pa.y + pb.y) / 2 + bnd;
      const lo = Math.min(pa.y, pb.y) + 8, hi = Math.max(pa.y, pb.y) - 8;
      if (lo <= hi) my = Math.max(lo, Math.min(hi, my));
      return { d: `M ${pa.x} ${pa.y} V ${my} H ${pb.x} V ${pb.y}`, mx: (pa.x + pb.x) / 2, my };
    }
    if (hA) return { d: `M ${pa.x} ${pa.y} H ${pb.x} V ${pb.y}`, mx: (pa.x + pb.x) / 2, my: pa.y };  // side -> top/bottom
    return { d: `M ${pa.x} ${pa.y} V ${pb.y} H ${pb.x}`, mx: pa.x, my: (pa.y + pb.y) / 2 };          // top/bottom -> side
  }

  _toggleTrace() {
    if (this.trace.on) { this._stopTrace(); return; }
    if (window.REDUCED_MOTION) { this._revealAll(); return; }
    this.trace.on  = true;
    this.trace.col = -1;
    this.traceBtn.classList.add("is-on");
    this.traceBtn.querySelector(".rm-tool__ic").innerHTML = "&#10073;&#10073;";
    this.traceBtn.querySelector(".rm-tool__lb").textContent = "Stop";
    this.canvasEl.classList.add("is-tracing");
    Object.values(this.nodeById).forEach((el) => el.classList.remove("is-lit", "is-active"));
    this.wireEls.forEach((w) => { w.flow.classList.remove("is-flow"); w.base.classList.remove("is-lit"); });
    this._traceStep();
  }

  _traceStep() {
    const maxCol = this.diagram.nodes.reduce((m, n) => Math.max(m, n.col || 0), 0);
    this.trace.col += 1;
    const c = this.trace.col;

    Object.values(this.nodeById).forEach((el) => el.classList.remove("is-active"));
    this.diagram.nodes.forEach((n) => {
      if ((n.col || 0) <= c) this.nodeById[n.id].classList.add("is-lit");
      if ((n.col || 0) === c) this.nodeById[n.id].classList.add("is-active");
    });
    this.wireEls.forEach((w) => {
      const flowing = w.tcol <= c && w.fcol < c + 1;
      w.flow.classList.toggle("is-flow", flowing);
      w.base.classList.toggle("is-lit", flowing);
    });
    this.labelEls.forEach((l) => l.el.classList.toggle("is-lit", l.tcol <= c));

    if (c >= maxCol) { this.trace.timer = setTimeout(() => this._stopTrace(), 1500); return; }
    this.trace.timer = setTimeout(() => this._traceStep(), 840);
  }

  _revealAll() {
    Object.values(this.nodeById).forEach((el) => el.classList.add("is-lit"));
    this.wireEls.forEach((w) => w.base.classList.add("is-lit"));
  }

  _stopTrace() {
    this.trace.on = false;
    if (this.trace.timer) { clearTimeout(this.trace.timer); this.trace.timer = null; }
    if (this.traceBtn) {
      this.traceBtn.classList.remove("is-on");
      this.traceBtn.querySelector(".rm-tool__ic").innerHTML = "&#9654;";
      this.traceBtn.querySelector(".rm-tool__lb").textContent = "Trace flow";
    }
    if (!this.canvasEl) return;
    this.canvasEl.classList.remove("is-tracing");
    Object.values(this.nodeById || {}).forEach((el) => el.classList.remove("is-lit", "is-active"));
    this.wireEls.forEach((w) => { w.flow.classList.remove("is-flow"); w.base.classList.remove("is-lit"); });
    this.labelEls.forEach((l) => l.el.classList.remove("is-lit"));
  }

  _esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
}

window.RepoMapView = RepoMapView;
