"use strict";

class ScriptPanel {
  constructor(refs) {
    this.gridEl = refs.grid;
    this.filterEl = refs.filters;
    this.scripts = [];
    this.filter = "All";
  }

  async load() {
    const data = await window.apiGet("/api/scripts");
    this.scripts = data.scripts || [];
    this._renderFilters();
    this._renderGrid();
  }

  _categories() {
    return ["All", ...new Set(this.scripts.map((s) => s.category))];
  }

  _renderFilters() {
    this.filterEl.innerHTML = "";
    this._categories().forEach((cat) => {
      const chip = document.createElement("button");
      chip.className = "chip" + (cat === this.filter ? " is-active" : "");
      chip.textContent = cat;
      chip.addEventListener("click", () => {
        this.filter = cat;
        [...this.filterEl.children].forEach((c) => c.classList.toggle("is-active", c.textContent === cat));
        this._renderGrid();
      });
      this.filterEl.appendChild(chip);
    });
  }

  _renderGrid() {
    this.gridEl.innerHTML = "";
    const items = this.scripts.filter((s) => this.filter === "All" || s.category === this.filter);

    items.forEach((s, i) => {
      const card = document.createElement("a");
      card.className = "script-card reveal";
      card.href = `#/launch/${s.key}`;
      card.style.transitionDelay = `${i * 0.04}s`;
      card.innerHTML =
        `<span class="script-card__glow"></span>` +
        `<div class="script-card__top"><span class="script-card__cat">${s.category}</span>` +
        `<span class="script-card__file">${s.file}</span></div>` +
        `<h3 class="script-card__title">${s.title}</h3>` +
        `<p class="script-card__purpose">${s.purpose}</p>` +
        `<div class="script-card__foot"><span>${s.config_class || "no entry config"}</span>` +
        `<span class="arrow">configure &rarr;</span></div>`;
      this.gridEl.appendChild(card);
    });

    window.revealScan();
  }
}

window.ScriptPanel = ScriptPanel;
