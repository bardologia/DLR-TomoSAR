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

    const seenGroups = new Set();
    let i = 0;
    items.forEach((s) => {
      if (s.group) {
        if (seenGroups.has(s.group)) return;
        seenGroups.add(s.group);
      }

      const card = document.createElement("a");
      card.className = "script-card reveal";
      card.style.transitionDelay = `${i * 0.04}s`;
      i += 1;

      if (s.group) {
        const variants = s.variants || [];
        const chips = variants.map((v) => `<span class="script-card__variant">${v.label}</span>`).join("");
        card.href = `#/launch/${variants[0] ? variants[0].key : s.key}`;
        card.innerHTML =
          `<span class="script-card__glow"></span>` +
          `<div class="script-card__top"><span class="script-card__cat">${s.group_category || s.category}</span>` +
          `<span class="script-card__file">${s.file}</span></div>` +
          `<h3 class="script-card__title">${s.group_title || s.title}</h3>` +
          `<p class="script-card__purpose">${s.group_purpose || s.purpose}</p>` +
          `<div class="script-card__variants">${chips}</div>` +
          `<div class="script-card__foot"><span>${variants.length} stages</span>` +
          `<span class="arrow">configure &rarr;</span></div>`;
      } else {
        card.href = `#/launch/${s.key}`;
        card.innerHTML =
          `<span class="script-card__glow"></span>` +
          `<div class="script-card__top"><span class="script-card__cat">${s.category}</span>` +
          `<span class="script-card__file">${s.file}</span></div>` +
          `<h3 class="script-card__title">${s.title}</h3>` +
          `<p class="script-card__purpose">${s.purpose}</p>` +
          `<div class="script-card__foot"><span>${s.config_class || "no entry config"}</span>` +
          `<span class="arrow">configure &rarr;</span></div>`;
      }

      this.gridEl.appendChild(card);
    });

    window.revealScan();
  }
}

window.ScriptPanel = ScriptPanel;
