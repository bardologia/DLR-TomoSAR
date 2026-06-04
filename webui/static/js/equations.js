"use strict";

class EquationView {
  constructor(tabsEl, gridEl) {
    this.tabsEl = tabsEl;
    this.gridEl = gridEl;
    this.groups = [];
    this.active = 0;
  }

  async load() {
    const data = await window.apiGet("/api/equations");
    this.groups = data.groups || [];
    this._renderTabs();
    this._renderGroup(0);
  }

  _renderTabs() {
    this.tabsEl.innerHTML = "";
    this.groups.forEach((group, i) => {
      const tab = document.createElement("button");
      tab.className = "eq-tab" + (i === 0 ? " is-active" : "");
      tab.textContent = group.group;
      tab.addEventListener("click", () => this._select(i));
      this.tabsEl.appendChild(tab);
    });
  }

  _select(index) {
    this.active = index;
    [...this.tabsEl.children].forEach((t, i) => t.classList.toggle("is-active", i === index));
    this._renderGroup(index);
  }

  _renderGroup(index) {
    const group = this.groups[index];
    if (!group) return;
    this.gridEl.innerHTML = "";

    group.items.forEach((item, i) => {
      const card = document.createElement("div");
      card.className = "eq-card reveal";
      card.style.transitionDelay = `${i * 0.05}s`;

      const title = document.createElement("h3");
      title.className = "eq-card__title";
      title.textContent = item.title;

      const tex = document.createElement("div");
      tex.className = "eq-card__tex";
      this._renderTex(tex, item.tex);

      const note = document.createElement("p");
      note.className = "eq-card__note";
      note.textContent = item.note;

      card.appendChild(title);
      card.appendChild(tex);
      card.appendChild(note);

      if (item.vars && item.vars.length) {
        const vars = document.createElement("div");
        vars.className = "eq-card__vars";
        item.vars.forEach((v) => {
          const row = document.createElement("div");
          row.className = "eq-var";
          const sym = document.createElement("span");
          sym.className = "eq-var__sym";
          this._renderTex(sym, v.sym);
          const desc = document.createElement("span");
          desc.className = "eq-var__desc";
          desc.textContent = v.desc;
          row.appendChild(sym);
          row.appendChild(desc);
          vars.appendChild(row);
        });
        card.appendChild(vars);
      }

      this.gridEl.appendChild(card);
    });

    window.revealScan();
  }

  _renderTex(el, tex) {
    if (window.katex) {
      try {
        window.katex.render(tex, el, { throwOnError: false, displayMode: false });
        return;
      } catch (e) {}
    }
    el.textContent = tex;
  }
}

window.EquationView = EquationView;
