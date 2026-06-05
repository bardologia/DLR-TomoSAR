"use strict";

class ResultsView {
  constructor(listEl, detailEl) {
    this.listEl = listEl;
    this.detailEl = detailEl;
    this.runs = [];
    this.selectedId = null;
    this.activeStamp = null;
    this.loaded = false;

    this.lightbox = document.getElementById("lightbox");
    this.lightboxImg = document.getElementById("lightbox-img");
    this.lightboxCap = document.getElementById("lightbox-cap");
    this.lightbox.addEventListener("click", () => this._closeLightbox());
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") this._closeLightbox();
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
      data = await window.apiGet("/api/runs");
    } catch (e) {
      this.listEl.innerHTML = `<div class="clist__empty">Backend unreachable.</div>`;
      return;
    }

    this.runs = data.runs || [];
    this._renderList();

    if (!this.runs.length) {
      this.detailEl.innerHTML = `<div class="res-empty">No runs with inference outputs found under <code>logs/</code>. Run an inference first.</div>`;
      return;
    }

    if (!this.runs.some((r) => r.id === this.selectedId)) this.select(this.runs[0].id);
  }

  _renderList() {
    this.listEl.innerHTML = "";

    if (!this.runs.length) {
      this.listEl.innerHTML = `<div class="clist__empty">No inference runs yet.</div>`;
      return;
    }

    const groups = new Map();
    this.runs.forEach((run) => {
      const key = run.group || "logs";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(run);
    });

    groups.forEach((runs, group) => {
      const head = document.createElement("div");
      head.className = "run-group__head";
      head.innerHTML = `<span>${this._esc(group)}</span><span class="run-group__count">${runs.length}</span>`;
      this.listEl.appendChild(head);

      runs.forEach((run) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "run-item" + (run.id === this.selectedId ? " is-active" : "");
        item.innerHTML =
          `<span class="run-item__name">${this._esc(run.name)}</span>` +
          `<span class="run-item__meta">${run.modified.replace("T", " ")} &middot; ${run.outputs} output${run.outputs === 1 ? "" : "s"}</span>`;
        item.addEventListener("click", () => this.select(run.id));
        this.listEl.appendChild(item);
      });
    });
  }

  async select(runId) {
    this.selectedId = runId;
    this.activeStamp = null;
    this._renderList();
    this.detailEl.innerHTML = `<div class="res-empty">Loading&hellip;</div>`;

    let detail;
    try {
      detail = await window.apiGet(`/api/runs?id=${encodeURIComponent(runId)}`);
    } catch (e) {
      detail = null;
    }

    if (!detail || !detail.ok) {
      this.detailEl.innerHTML = `<div class="res-empty">Could not load this run.</div>`;
      return;
    }

    this.detail = detail;
    this.activeStamp = detail.outputs.length ? detail.outputs[0].stamp : null;
    this._renderDetail();
  }

  _renderDetail() {
    const detail = this.detail;
    const output = detail.outputs.find((o) => o.stamp === this.activeStamp) || detail.outputs[0];

    let html =
      `<header class="res-head">` +
      `<div><h3 class="res-title">${this._esc(detail.name)}</h3>` +
      `<p class="res-path">${this._esc(detail.id)}</p></div>`;

    if (detail.outputs.length > 1) {
      html += `<div class="res-stamps">`;
      detail.outputs.forEach((o) => {
        html += `<button type="button" class="res-stamp${o.stamp === output.stamp ? " is-active" : ""}" data-stamp="${this._esc(o.stamp)}">${this._esc(o.stamp)}</button>`;
      });
      html += `</div>`;
    } else if (output) {
      html += `<span class="res-stamp is-active is-single">${this._esc(output.stamp)}</span>`;
    }
    html += `</header>`;

    if (!output) {
      this.detailEl.innerHTML = html + `<div class="res-empty">This run has no inference outputs.</div>`;
      return;
    }

    if (output.metrics) html += this._metricsHtml(output.metrics);

    output.groups.forEach((group) => {
      html += this._sectionHtml(group.title, group.items, "img");
    });

    if (output.gifs.length) html += this._sectionHtml("Animations", output.gifs, "gif");

    if (output.report_url) {
      html += `<p class="res-report"><a href="${output.report_url}" target="_blank" rel="noopener">Open report.md</a></p>`;
    }

    this.detailEl.innerHTML = html;
    this._bind(output);
  }

  _metricsHtml(metrics) {
    const entries = Object.entries(metrics).filter(([, v]) => typeof v === "number" || typeof v === "string");
    if (!entries.length) return "";

    const cells = entries.map(([key, value]) => {
      const shown = typeof value === "number" ? this._fmt(value) : this._esc(String(value));
      return `<div class="res-metric"><dt>${shown}</dt><dd>${this._esc(key)}</dd></div>`;
    }).join("");

    return (
      `<section class="res-section res-section--metrics">` +
      `<h4 class="res-section__cap">Metrics <span>${entries.length}</span></h4>` +
      `<dl class="res-metrics">${cells}</dl>` +
      `</section>`
    );
  }

  _sectionHtml(title, items, kind) {
    const figs = items.map((item) =>
      `<figure class="res-fig${kind === "gif" ? " res-fig--gif" : ""}" data-url="${item.url}" data-name="${this._esc(item.name)}">` +
      `<img src="${item.url}" alt="${this._esc(item.name)}" loading="lazy" />` +
      `<figcaption>${this._esc(item.name)}</figcaption>` +
      `</figure>`
    ).join("");

    return (
      `<section class="res-section">` +
      `<h4 class="res-section__cap">${this._esc(title)} <span>${items.length}</span></h4>` +
      `<div class="res-grid${kind === "gif" ? " res-grid--gif" : ""}">${figs}</div>` +
      `</section>`
    );
  }

  _bind(output) {
    this.detailEl.querySelectorAll(".res-stamp[data-stamp]").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.activeStamp = btn.dataset.stamp;
        this._renderDetail();
      });
    });

    this.detailEl.querySelectorAll(".res-fig").forEach((fig) => {
      fig.addEventListener("click", () => this._openLightbox(fig.dataset.url, fig.dataset.name));
    });
  }

  _openLightbox(url, name) {
    this.lightboxImg.src = url;
    this.lightboxCap.textContent = name;
    this.lightbox.hidden = false;
  }

  _closeLightbox() {
    if (this.lightbox.hidden) return;
    this.lightbox.hidden = true;
    this.lightboxImg.src = "";
  }

  _fmt(value) {
    if (Number.isInteger(value)) return String(value);
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(3);
    return value.toFixed(4);
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
}

window.ResultsView = ResultsView;
