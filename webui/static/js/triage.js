"use strict";

class TriageView {
  static DEFAULT_BASE = "/ste/rnd/User/vice_vi/DLR-TomoSAR/runs";
  static VERDICTS     = ["label problem", "model problem", "interesting"];

  constructor(root) {
    this.root    = root;
    this.cubes   = [];
    this.cubeId  = null;
    this.cases   = [];
    this.payload = null;
    this.built   = false;
  }

  enter() {
    if (!this.built) {
      this._build();
      this.built = true;
    }
  }

  _build() {
    this.root.innerHTML = `
      <div class="cube-pick">
        <div class="fl-src" role="group" aria-label="Runs directory">
          <label for="triage-base">Runs dir</label>
          <input id="triage-base" type="text" spellcheck="false" autocomplete="off" placeholder="${TriageView.DEFAULT_BASE}" />
          <button type="button" class="btn btn--mini" id="triage-scan">Scan</button>
        </div>
        <div class="cube-grouplist" id="triage-strip" aria-label="Saved inferences"></div>
        <p class="cube-hint" id="triage-hint">Scan a runs directory for saved inference cubes, then pick one to triage.</p>
      </div>
      <div id="triage-cases"></div>`;

    this.refs = {
      base  : this.root.querySelector("#triage-base"),
      scan  : this.root.querySelector("#triage-scan"),
      strip : this.root.querySelector("#triage-strip"),
      hint  : this.root.querySelector("#triage-hint"),
      cases : this.root.querySelector("#triage-cases"),
    };

    this.refs.base.value = TriageView.DEFAULT_BASE;
    this.refs.scan.addEventListener("click", () => this._scan());
    this.refs.base.addEventListener("keydown", (ev) => { if (ev.key === "Enter") this._scan(); });
  }

  async _scan() {
    const base = this.refs.base.value.trim();
    if (!base) return;

    const data = await window.apiGet(`/api/cubes?base=${encodeURIComponent(base)}`);
    if (!data.ok) {
      this.refs.hint.textContent = data.error || "scan failed";
      return;
    }

    this.cubes = data.cubes || [];
    this.refs.hint.textContent = this.cubes.length ? `${this.cubes.length} saved inference(s); pick one.` : "no saved inference cubes under this directory";

    this.refs.strip.innerHTML = "";
    this.cubes.forEach((cube) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "cube-space";
      btn.textContent = `${cube.run} · ${cube.stamp || ""}`;
      btn.title = cube.id;
      btn.addEventListener("click", () => this._pick(cube.id, btn));
      this.refs.strip.appendChild(btn);
    });
  }

  async _pick(cubeId, btn) {
    this.refs.strip.querySelectorAll(".cube-space").forEach((b) => b.classList.toggle("is-active", b === btn));
    this.cubeId = cubeId;

    const data = await window.apiGet(`/api/triage/cases?id=${encodeURIComponent(cubeId)}&n=40`);
    if (!data.ok) {
      this.refs.cases.innerHTML = `<p class="cube-hint">${this._esc(data.error || "failed to build cases")}</p>`;
      return;
    }

    this.cases   = data.cases;
    this.payload = data;
    this._renderCases();
  }

  _renderCases() {
    if (!this.cases.length) {
      this.refs.cases.innerHTML = `<p class="cube-hint">no triage cases; every block is below the error floor</p>`;
      return;
    }

    const cards = this.cases.map((row, index) => this._card(row, index)).join("");
    this.refs.cases.innerHTML = `<div class="script-grid">${cards}</div>`;
    this._wire(this.refs.cases);
  }

  _card(row, index) {
    const data       = this.payload || {};
    const annotation = row.annotation || {};
    const modeBadge  = data.has_modes && row.mode ? `<span class="cube-coords">mode: ${this._esc(row.mode)} (${(row.fail_frac * 100).toFixed(0)}% failing)</span><br>` : "";
    const aux        = (data.aux || [])
      .filter((name) => row[name] !== null && row[name] !== undefined)
      .map((name) => `${this._esc(name)}: ${Number(row[name]).toPrecision(3)}`)
      .join(" · ");

    const verdictBtns = TriageView.VERDICTS.map((verdict) => {
      const active = annotation.verdict === verdict;
      return `<button type="button" class="cube-space${active ? " is-active" : ""}" data-case="${index}" data-verdict="${active ? "" : verdict}">${verdict}</button>`;
    }).join("");

    return `
      <article class="fl-card">
        <h3 class="cube-panel-title">#${index + 1} · az ${row.az0}–${row.az1}, rg ${row.rg0}–${row.rg1}</h3>
        <p class="cube-hint">
          block MSE ${row.mse_mean.toExponential(2)} (peak ${row.mse_max.toExponential(2)} at az ${row.worst_az}, rg ${row.worst_rg})<br>
          ${modeBadge}${aux ? `${aux}<br>` : ""}
        </p>
        <div class="cube-spaces" role="group" aria-label="Verdict">${verdictBtns}</div>
        <div class="fl-pixrow">
          <input type="text" class="cube-jump__input cube-jump__input--wide" placeholder="note" value="${this._esc(annotation.note || "")}" data-note="${index}" />
          <button type="button" class="btn btn--mini" data-open="${index}">Open cuts</button>
        </div>
        ${annotation.updated ? `<p class="cube-hint">last touched ${this._esc(annotation.updated)}</p>` : ""}
      </article>`;
  }

  async _annotate(index, verdict, noteOverride) {
    const row        = this.cases[index];
    const annotation = row.annotation || {};
    const body       = {
      id      : this.cubeId,
      case    : row.case,
      verdict : verdict === null ? (annotation.verdict || "") : verdict,
      note    : noteOverride !== undefined ? noteOverride : (annotation.note || ""),
    };

    const out = await window.apiPost("/api/triage/annotate", body);

    if (out.ok) {
      row.annotation = out.annotation || null;
      this._repaintCard(index);
    }
  }

  _repaintCard(index) {
    const grid = this.refs.cases.querySelector(".script-grid");
    const old  = grid ? grid.children[index] : null;
    if (!old) return;

    const holder = document.createElement("div");
    holder.innerHTML = this._card(this.cases[index], index);
    const card = holder.firstElementChild;

    old.replaceWith(card);
    this._wire(card);
  }

  _wire(scope) {
    scope.querySelectorAll("[data-open]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const row = this.cases[Number(btn.dataset.open)];
        if (window.tomogramView) {
          window.tomogramView.openAt(this.cubeId, row.worst_az, row.worst_rg);
          window.location.hash = "#/cube";
        }
      });
    });

    scope.querySelectorAll("[data-verdict]").forEach((btn) => {
      btn.addEventListener("click", () => this._annotate(Number(btn.dataset.case), btn.dataset.verdict));
    });

    scope.querySelectorAll("[data-note]").forEach((input) => {
      input.addEventListener("change", () => this._annotate(Number(input.dataset.note), null, input.value));
    });
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }
}

window.TriageView = TriageView;
