"use strict";

class ConfigBrowser {
  constructor(listEl, detailEl, searchEl) {
    this.listEl = listEl;
    this.detailEl = detailEl;
    this.searchEl = searchEl;
    this.groups = [];
    this.flat = [];
    this.activeId = null;
    this.query = "";
    this.searchEl.addEventListener("input", () => {
      this.query = this.searchEl.value.trim().toLowerCase();
      this._renderList();
      this._reselect();
    });
  }

  async load() {
    const data = await window.apiGet("/api/configs");
    this.groups = data.groups || [];
    this.flat = [];
    this.groups.forEach((g) => {
      g.classes.forEach((c) => this.flat.push({ id: `${g.module}::${c.name}`, group: g.title, module: g.module, name: c.name, fields: c.fields }));
    });
    this._renderList();
    if (this.flat[0]) this._select(this.flat[0].id);
  }

  _matchField(f) {
    if (!this.query) return true;
    return (
      f.name.toLowerCase().includes(this.query) ||
      (f.type || "").toLowerCase().includes(this.query) ||
      (f.default || "").toLowerCase().includes(this.query)
    );
  }

  _visibleClasses() {
    if (!this.query) return this.flat;
    return this.flat.filter((c) => c.fields.some((f) => this._matchField(f)));
  }

  _renderList() {
    this.listEl.innerHTML = "";
    const visible = this._visibleClasses();

    if (!visible.length) {
      this.listEl.innerHTML = `<div class="clist__empty">No config matches.</div>`;
      return;
    }

    let currentGroup = null;
    visible.forEach((c) => {
      if (c.group !== currentGroup) {
        currentGroup = c.group;
        const label = document.createElement("div");
        label.className = "clist__label";
        label.textContent = c.group;
        this.listEl.appendChild(label);
      }
      const count = this.query ? c.fields.filter((f) => this._matchField(f)).length : c.fields.length;
      const item = document.createElement("button");
      item.className = "clist__item";
      item.dataset.id = c.id;
      item.innerHTML = `<span class="clist__name">${c.name}</span><span class="clist__count">${count}</span>`;
      item.addEventListener("click", () => this._select(c.id));
      this.listEl.appendChild(item);
    });
  }

  _reselect() {
    const visible = this._visibleClasses();
    if (!visible.length) {
      this.detailEl.innerHTML = `<div class="cdetail__empty">No fields match that search.</div>`;
      this.activeId = null;
      return;
    }
    if (!visible.find((c) => c.id === this.activeId)) {
      this._select(visible[0].id);
    } else {
      this._renderDetail();
    }
  }

  _select(id) {
    this.activeId = id;
    this.listEl.querySelectorAll(".clist__item").forEach((b) => b.classList.toggle("is-active", b.dataset.id === id));
    this._renderDetail();
  }

  _renderDetail() {
    const cls = this.flat.find((c) => c.id === this.activeId);
    if (!cls) return;
    const fields = cls.fields.filter((f) => this._matchField(f));

    let rows = "";
    fields.forEach((f) => {
      rows +=
        `<div class="cfield">` +
        `<div class="cfield__main"><span class="cfield__name">${this._mark(f.name)}</span>` +
        `<span class="cfield__type">${this._esc(f.type)}</span></div>` +
        `<span class="cfield__default">${this._esc(f.default)}</span>` +
        `</div>`;
    });

    this.detailEl.innerHTML =
      `<div class="cdetail__head">` +
      `<div><span class="cdetail__group">${cls.group}</span>` +
      `<h3 class="cdetail__name">${cls.name}</h3></div>` +
      `<span class="cdetail__count">${fields.length} field${fields.length === 1 ? "" : "s"}</span>` +
      `</div>` +
      `<div class="cdetail__path">configuration/${cls.module}.py</div>` +
      `<div class="cfields">${rows}</div>`;

    this.detailEl.classList.remove("is-swap");
    void this.detailEl.offsetWidth;
    this.detailEl.classList.add("is-swap");
  }

  _esc(s) {
    return String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }

  _mark(name) {
    const safe = this._esc(name);
    if (!this.query) return safe;
    const idx = name.toLowerCase().indexOf(this.query);
    if (idx < 0) return safe;
    const end = idx + this.query.length;
    return this._esc(name.slice(0, idx)) + "<mark>" + this._esc(name.slice(idx, end)) + "</mark>" + this._esc(name.slice(end));
  }
}

window.ConfigBrowser = ConfigBrowser;
