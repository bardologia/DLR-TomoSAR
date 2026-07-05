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
          <button class="rm-tool rm-tool--trace" type="button" title="Animate the data flow through the pipeline">
            <span class="rm-tool__ic">&#9654;</span><span class="rm-tool__lb">Trace flow</span>
          </button>
        </div>
      </div>
      <div class="rm__title">
        <div class="rm__titletext">
          <h3 class="rm__stage-name"></h3>
          <p class="rm__blurb"></p>
        </div>
        <a class="rm__entry" target="_blank" rel="noopener"></a>
      </div>
      <div class="rm__legend"></div>
      <div class="rm__stage">
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
    this.traceBtn  = this.panel.querySelector(".rm-tool--trace");
    this.nameEl    = this.panel.querySelector(".rm__stage-name");
    this.blurbEl   = this.panel.querySelector(".rm__blurb");
    this.entryEl   = this.panel.querySelector(".rm__entry");
    this.legendEl  = this.panel.querySelector(".rm__legend");
    this.graphEl   = this.panel.querySelector(".rm-graph");
    this.canvasEl  = this.panel.querySelector(".rm-canvas");
    this.wiresEl   = this.panel.querySelector(".rm-wires");
    this.colsEl    = this.panel.querySelector(".rm-cols");
    this.labelsEl  = this.panel.querySelector(".rm-labels");
    this.ledgerEl  = this.panel.querySelector(".rm__ledger");

    this.traceBtn.addEventListener("click", () => this._toggleTrace());

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
    this.blurbEl.textContent = this.diagram.blurb || "";

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

    const maxCol = this.diagram.nodes.reduce((m, n) => Math.max(m, n.col || 0), 0);
    const cols   = [];
    for (let c = 0; c <= maxCol; c++) {
      const col = document.createElement("div");
      col.className = "rm-col";
      this.colsEl.appendChild(col);
      cols.push(col);
    }

    this.diagram.nodes.forEach((n) => {
      const el = document.createElement("div");
      el.className   = "rm-node rm-node--" + n.role;
      el.dataset.id  = n.id;
      el.dataset.col = String(n.col || 0);
      el.style.setProperty("--role", this.ROLE_COLORS[n.role] || this.ROLE_COLORS.external);

      const io = [];
      if (n.writes && n.writes.length) io.push(`<span class="rm-io rm-io--w" title="writes ${this._esc(n.writes.join(", "))}">out ${n.writes.length}</span>`);
      if (n.reads && n.reads.length)   io.push(`<span class="rm-io rm-io--r" title="reads ${this._esc(n.reads.join(", "))}">in ${n.reads.length}</span>`);

      el.innerHTML =
        `<div class="rm-node__top"><span class="rm-node__role">${this._esc(this.ROLE_LABELS[n.role] || n.role)}</span><span class="rm-node__io">${io.join("")}</span></div>` +
        `<h4 class="rm-node__name">${this._esc(n.label)}</h4>` +
        `<p class="rm-node__fn">${this._esc(n.fn)}</p>` +
        `<code class="rm-node__mod">${this._esc(n.module)}</code>`;

      el.addEventListener("click", () => this._focusNode(n.id));
      cols[n.col || 0].appendChild(el);
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

    this.ledgerEl.innerHTML = `
      <div class="rm-led__head"><h4>Artifact ledger</h4><span class="rm-led__sub">what each stage saves and who reads it</span></div>
      <div class="rm-led__scroll"><table class="rm-led">
        <thead><tr><th>Artifact</th><th>Format</th><th>Written by</th><th>Read by</th><th>Holds</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;

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
    this.labelEls.forEach((l) => l.el.classList.remove("is-lit", "is-dim"));
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
    const colOf  = (id) => { const n = this.diagram.nodes.find((nn) => nn.id === id); return n ? (n.col || 0) : 0; };
    const roleOf = (id) => { const n = this.diagram.nodes.find((nn) => nn.id === id); return n ? n.role : "external"; };

    const counts = {};
    (this.diagram.edges || []).forEach((e) => {
      const fc = colOf(e.from), tc = colOf(e.to);
      if (tc > fc) { const k = fc + "_" + tc; counts[k] = (counts[k] || 0) + 1; }
    });
    const seen = {};

    (this.diagram.edges || []).forEach((e) => {
      const a = box(e.from), b = box(e.to);
      if (!a || !b) return;

      let offset = 0;
      if (b.col > a.col) {
        const k = a.col + "_" + b.col;
        const idx = (seen[k] = (seen[k] === undefined ? 0 : seen[k] + 1));
        offset = idx - (counts[k] - 1) / 2;
      }

      const geom  = this._orthGeom(a, b, offset);
      const color = this.ROLE_COLORS[roleOf(e.from)] || this.ROLE_COLORS.external;

      const base = this._path(geom.d, "rm-wire", color);
      base.setAttribute("marker-end", "url(#rm-arrow)");
      const flow = this._path(geom.d, "rm-flow", color);
      this.wiresEl.appendChild(base);
      this.wiresEl.appendChild(flow);
      this.wireEls.push({ base, flow, from: e.from, to: e.to, fcol: a.col, tcol: b.col });

      if (e.label) {
        const lab = document.createElement("span");
        lab.className   = "rm-elabel rm-elabel--" + (e.kind || "data");
        lab.style.left  = geom.mx + "px";
        lab.style.top   = geom.my + "px";
        lab.textContent = e.label;
        this.labelsEl.appendChild(lab);
        this.labelEls.push({ el: lab, from: e.from, to: e.to, tcol: b.col });
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

  _orthGeom(a, b, offset) {
    const ax = a.x + a.w, ay = a.y + a.h / 2;
    const bx = b.x,       by = b.y + b.h / 2;

    if (bx >= ax + 24) {
      let mx = (ax + bx) / 2 + offset * 22;
      mx = Math.max(ax + 16, Math.min(bx - 16, mx));
      return { d: `M ${ax} ${ay} H ${mx} V ${by} H ${bx}`, mx, my: (ay + by) / 2 };
    }

    const sx = a.x + a.w / 2, sy = a.y + a.h;
    const tx = b.x + b.w / 2, ty = b.y + b.h;
    const ly = Math.max(sy, ty) + 36;
    return { d: `M ${sx} ${sy} V ${ly} H ${tx} V ${ty}`, mx: (sx + tx) / 2, my: ly };
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
