"use strict";

class RunStrip {
  static FILTER_MIN = 7;

  constructor(container, opts = {}) {
    this.container  = container;
    this.opts       = opts;
    this.entries    = [];
    this.filter     = "";
    this.openGroups = new Set();
    this.seeded     = false;

    this.container.classList.add("run-strip");
    this.container.innerHTML = `
      <div class="run-strip__bar" hidden>
        <input type="text" class="run-strip__filter" placeholder="filter runs&hellip;" spellcheck="false" autocomplete="off" />
        <span class="run-strip__total"></span>
      </div>
      <div class="run-strip__groups"></div>`;

    this.bar    = this.container.querySelector(".run-strip__bar");
    this.input  = this.container.querySelector(".run-strip__filter");
    this.total  = this.container.querySelector(".run-strip__total");
    this.groups = this.container.querySelector(".run-strip__groups");

    this.input.addEventListener("input", () => {
      this.filter = this.input.value.trim().toLowerCase();
      this._paint();
    });
  }

  static _esc(text) {
    const div = document.createElement("div");
    div.textContent = text == null ? "" : String(text);
    return div.innerHTML;
  }

  static prettyStamp(stamp) {
    const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/.exec(stamp || "");
    if (!m) return stamp || "";
    return `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}`;
  }

  static parseName(name) {
    const m      = /^(.*)_(\d{8}_\d{6})(?:_(.+))?$/.exec(name || "");
    const base   = m ? m[1] : (name || "");
    const when   = m ? RunStrip.prettyStamp(m[2]) : "";
    const suffix = m && m[3] ? m[3] : "";
    return { tokens: base.split("-").filter(Boolean), when, suffix };
  }

  static nameHtml(name) {
    const parts  = RunStrip.parseName(name);
    const tokens = parts.tokens.map((tok, i) => `<span class="run-name__tok${i === 0 ? " run-name__tok--model" : ""}">${RunStrip._esc(tok)}</span>`);
    return `<span class="run-name">${tokens.join(`<span class="run-name__sep">-</span>`)}</span>`;
  }

  static metaHtml(entry) {
    const parts  = RunStrip.parseName(entry.run);
    const pieces = [];

    if (parts.when)   pieces.push(`<span class="cube-run__when">${RunStrip._esc(parts.when)}</span>`);
    if (entry.stamp)  pieces.push(`<span class="cube-run__when">cube ${RunStrip._esc(RunStrip.prettyStamp(entry.stamp))}</span>`);
    if (parts.suffix) pieces.push(`<span class="cube-run__badge">${RunStrip._esc(parts.suffix)}</span>`);

    if (!pieces.length) return "";
    return `<span class="cube-run__meta">${pieces.join("")}</span>`;
  }

  open(group)  { this.openGroups.add(group); }
  close(group) { this.openGroups.delete(group); }

  _stateFor(entry) {
    return this.opts.stateFor ? this.opts.stateFor(entry) : false;
  }

  _matches(entry) {
    if (!this.filter) return true;
    return `${entry.group}/${entry.run} ${entry.stamp || ""}`.toLowerCase().includes(this.filter);
  }

  _row(entry) {
    const state = this._stateFor(entry);
    const tag   = this.opts.onPick ? "button" : "div";
    const row   = document.createElement(tag);

    if (tag === "button") row.type = "button";
    row.className = "cube-run" + (this.opts.rowClass ? ` ${this.opts.rowClass}` : "") + (state ? " is-active" : "");
    row.title     = entry.id;

    const main = document.createElement("span");
    main.className = "cube-run__main";
    main.innerHTML = RunStrip.nameHtml(entry.run) + RunStrip.metaHtml(entry);
    row.appendChild(main);

    if (this.opts.extras) {
      const extras = document.createElement("span");
      extras.className = "cube-run__extras";
      this.opts.extras(entry).forEach((node) => extras.appendChild(node));
      row.appendChild(extras);
    }

    if (this.opts.onPick) row.addEventListener("click", () => this.opts.onPick(entry));

    return row;
  }

  _card(group, entries) {
    const label   = group === "." ? "runs" : group;
    const shown   = entries.filter((entry) => this._matches(entry));
    const current = entries.some((entry) => this._stateFor(entry));
    const isOpen  = this.filter ? shown.length > 0 : this.openGroups.has(group);

    if (this.filter && !shown.length) return null;

    const card = document.createElement("div");
    card.className = "cube-group" + (isOpen ? " is-open" : "") + (current ? " is-current" : "");

    const picked = entries.filter((entry) => this._stateFor(entry)).length;
    const count  = this.opts.multi && picked ? `${picked}/${entries.length}` : `${entries.length}`;

    const head = document.createElement("button");
    head.type      = "button";
    head.className = "cube-group__head";
    head.title     = label;
    head.innerHTML =
      `<span class="cube-group__chev" aria-hidden="true"></span>` +
      `<span class="cube-group__name">${RunStrip._esc(label)}</span>` +
      `<span class="cube-group__count">${RunStrip._esc(count)}</span>`;
    head.addEventListener("click", () => {
      if (this.openGroups.has(group)) this.openGroups.delete(group);
      else this.openGroups.add(group);
      this._paint();
    });
    card.appendChild(head);

    const body = document.createElement("div");
    body.className = "cube-group__body";
    shown.forEach((entry) => body.appendChild(this._row(entry)));
    card.appendChild(body);

    return card;
  }

  _paint() {
    this.groups.innerHTML = "";

    const byGroup = new Map();
    this.entries.forEach((entry) => {
      if (!byGroup.has(entry.group)) byGroup.set(entry.group, []);
      byGroup.get(entry.group).push(entry);
    });

    byGroup.forEach((entries, group) => {
      const card = this._card(group, entries);
      if (card) this.groups.appendChild(card);
    });

    const shown = this.entries.filter((entry) => this._matches(entry)).length;
    this.bar.hidden = this.entries.length < RunStrip.FILTER_MIN && !this.filter;
    this.total.textContent = this.filter ? `${shown} / ${this.entries.length} runs` : `${this.entries.length} runs`;
  }

  repaint() {
    this._paint();
  }

  render(entries) {
    this.entries = entries || [];

    if (!this.seeded && this.entries.length) {
      this.seeded = true;
      const groups = new Set(this.entries.map((entry) => entry.group));
      if (groups.size === 1) this.openGroups.add([...groups][0]);
      this.entries.forEach((entry) => { if (this._stateFor(entry)) this.openGroups.add(entry.group); });
    }

    this._paint();
  }
}
