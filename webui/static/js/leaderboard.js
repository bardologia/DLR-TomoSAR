"use strict";

class LeaderboardView {
  static AXIS_COLS = [
    { key: "model",    label: "model" },
    { key: "head",     label: "head" },
    { key: "matching", label: "matching" },
    { key: "k",        label: "K" },
    { key: "aug",      label: "aug" },
    { key: "presence", label: "presence" },
    { key: "loss",     label: "loss" },
  ];

  constructor(root) {
    this.root = root;

    this.data     = null;
    this.view     = "table";
    this.diffData = null;

    this.sortKey  = "mtime";
    this.sortDir  = -1;
    this.filter   = "";
    this.axisSel  = {};
    this.selected = [];

    this.diffFilter  = "";
    this.diffSeries  = false;
    this.diffChanged = false;

    this.visible = this._loadVisible();
  }

  enter() {
    this.load();
  }

  async load() {
    this.root.innerHTML = `<div class="res-empty">Scanning inference results&hellip;</div>`;

    const base = this._runsBase();
    const data = await window.apiGet(`/api/leaderboard?base=${encodeURIComponent(base)}`);

    if (!data || !data.ok) {
      this.data = null;
      this.root.innerHTML = `<div class="res-empty">${this._esc((data && data.error) || "Could not scan the runs directory.")}</div>`;
      return;
    }

    this.data     = data;
    this.selected = this.selected.filter((id) => data.rows.some((row) => row.id === id));
    if (this.visible === null) this.visible = data.columns.filter((c) => c.default).map((c) => c.key);

    this._render();
  }

  async showDiff() {
    if (this.selected.length !== 2) return;

    const [a, b] = this.selected;
    const data   = await window.apiGet(`/api/leaderboard/diff?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);

    if (!data || !data.ok) {
      window.toast((data && data.error) || "Could not load the comparison.", "warn");
      return;
    }

    this.diffData = data;
    this.view     = "diff";
    this._render();
  }

  _render() {
    if (this.view === "diff" && this.diffData) {
      this.root.innerHTML = this._diffHtml();
      this._bindDiff();
      return;
    }

    this.root.innerHTML = this._tableHtml();
    this._bindTable();
  }

  _rows() {
    let rows = this.data.rows;

    if (this.filter) {
      const needle = this.filter.toLowerCase();
      rows = rows.filter((row) => row.run.toLowerCase().includes(needle) || row.group.toLowerCase().includes(needle));
    }

    Object.entries(this.axisSel).forEach(([axis, value]) => {
      if (value === "") return;
      rows = rows.filter((row) => row.axes && String(row.axes[axis]) === value);
    });

    const dir = this.sortDir;
    const key = this.sortKey;

    return [...rows].sort((ra, rb) => {
      const va = this._sortValue(ra, key);
      const vb = this._sortValue(rb, key);
      if (va === null && vb === null) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      if (typeof va === "string") return va.localeCompare(vb) * dir;
      return (va - vb) * dir;
    });
  }

  _sortValue(row, key) {
    if (key === "run")   return row.run;
    if (key === "mtime") return row.mtime;
    if (LeaderboardView.AXIS_COLS.some((c) => c.key === key)) {
      if (!row.axes || row.axes[key] === undefined) return null;
      return row.axes[key];
    }
    const value = row.metrics[key];
    return value === undefined ? null : value;
  }

  _tableHtml() {
    const rows    = this._rows();
    const columns = this.data.columns.filter((c) => this.visible.includes(c.key));
    const ranges  = this._ranges(rows, columns);

    let html = this._toolbarHtml(rows.length);

    if (this.data.errors && this.data.errors.length) {
      html += `<p class="lb-errors">${this.data.errors.length} unreadable metrics file${this.data.errors.length === 1 ? "" : "s"} skipped.</p>`;
    }

    if (!this.data.rows.length) {
      return html + `<div class="res-empty">No inference results found. Run an inference first, then revisit this tab.</div>`;
    }
    if (!rows.length) {
      return html + `<div class="res-empty">No run matches the current filters.</div>`;
    }

    html += `<div class="lb-scroll"><table class="lb-table"><thead><tr>`;
    html += this._headCell("run", "run");
    LeaderboardView.AXIS_COLS.forEach((col) => { html += this._headCell(col.key, col.label); });
    columns.forEach((col) => { html += this._headCell(col.key, col.label, true); });
    html += this._headCell("mtime", "when");
    html += `</tr></thead><tbody>`;

    rows.forEach((row) => {
      const selected = this.selected.includes(row.id);
      html += `<tr class="lb-row${selected ? " is-selected" : ""}" data-id="${this._esc(row.id)}" title="${this._esc(row.id)}">`;
      html += `<td class="lb-run"><span class="lb-run__name">${this._esc(row.run)}</span><span class="lb-run__stamp">${this._esc(row.group !== "." ? row.group + " / " : "")}${this._esc(row.stamp)}</span></td>`;

      LeaderboardView.AXIS_COLS.forEach((col) => {
        const value = row.axes ? row.axes[col.key] : undefined;
        html += `<td class="lb-axis">${value === undefined || value === "" ? "&ndash;" : this._esc(value)}</td>`;
      });

      columns.forEach((col) => {
        const value = row.metrics[col.key];
        if (value === undefined) {
          html += `<td class="lb-val lb-val--none">&ndash;</td>`;
          return;
        }
        const good = this._goodness(value, ranges[col.key], col.direction);
        const best = good !== null && good > 0.999;
        html += `<td class="lb-val${best ? " is-best" : ""}">`;
        if (good !== null) html += `<i class="lb-bar" style="width:${Math.round(good * 100)}%"></i>`;
        html += `<span>${this._fmt(value)}</span></td>`;
      });

      html += `<td class="lb-when">${this._ago(row.mtime)}</td>`;
      html += `</tr>`;
    });

    html += `</tbody></table></div>`;
    return html;
  }

  _toolbarHtml(shown) {
    const total = this.data.rows.length;
    const count = shown === total ? `${total}` : `${shown} of ${total}`;

    let html = `<div class="lb-bar-top">`;
    html += `<span class="lb-count">${count} inference result${total === 1 ? "" : "s"}</span>`;
    html += `<input type="search" class="res-filter" id="lb-filter" placeholder="Filter by run name" value="${this._esc(this.filter)}" spellcheck="false" />`;

    LeaderboardView.AXIS_COLS.forEach((col) => {
      if (col.key === "loss") return;
      const values = [...new Set(this.data.rows.filter((r) => r.axes).map((r) => String(r.axes[col.key])))].sort();
      if (values.length < 2) return;
      const active = this.axisSel[col.key] || "";
      html += `<select class="lb-select" data-axis="${col.key}" aria-label="Filter by ${col.label}">`;
      html += `<option value="">${col.label}: all</option>`;
      values.forEach((v) => { html += `<option value="${this._esc(v)}"${v === active ? " selected" : ""}>${this._esc(v)}</option>`; });
      html += `</select>`;
    });

    html += `<details class="lb-cols"><summary>Columns</summary><div class="lb-cols__menu">`;
    this.data.columns.forEach((col) => {
      const on = this.visible.includes(col.key);
      html += `<label><input type="checkbox" data-col="${col.key}"${on ? " checked" : ""} /> ${this._esc(col.label)}</label>`;
    });
    html += `</div></details>`;

    html += `<button type="button" class="btn btn--mini" id="lb-compare" ${this.selected.length === 2 ? "" : "disabled"}>Compare ${this.selected.length}/2</button>`;
    html += `</div>`;
    return html;
  }

  _headCell(key, label, isMetric = false) {
    const active = this.sortKey === key;
    const arrow  = active ? (this.sortDir === 1 ? " &#9650;" : " &#9660;") : "";
    return `<th class="lb-th${isMetric ? " lb-th--metric" : ""}${active ? " is-sorted" : ""}" data-sort="${key}">${this._esc(label)}${arrow}</th>`;
  }

  _ranges(rows, columns) {
    const ranges = {};
    columns.forEach((col) => {
      const values = rows.map((row) => row.metrics[col.key]).filter((v) => v !== undefined && isFinite(v));
      if (values.length >= 2) ranges[col.key] = { min: Math.min(...values), max: Math.max(...values) };
    });
    return ranges;
  }

  _goodness(value, range, direction) {
    if (!range || !isFinite(value) || range.max === range.min || !direction) return null;
    const norm = (value - range.min) / (range.max - range.min);
    return direction > 0 ? norm : 1 - norm;
  }

  _bindTable() {
    const filter = this.root.querySelector("#lb-filter");
    if (filter) filter.addEventListener("input", () => {
      this.filter = filter.value.trim();
      this._rerenderKeepFocus(filter);
    });

    this.root.querySelectorAll(".lb-select").forEach((sel) => {
      sel.addEventListener("change", () => {
        this.axisSel[sel.dataset.axis] = sel.value;
        this._render();
      });
    });

    this.root.querySelectorAll(".lb-cols input[type=checkbox]").forEach((box) => {
      box.addEventListener("change", () => {
        if (box.checked) this.visible = [...this.visible, box.dataset.col];
        else             this.visible = this.visible.filter((k) => k !== box.dataset.col);
        localStorage.setItem("leaderboard-cols", JSON.stringify(this.visible));
        this._render();
        const menu = this.root.querySelector(".lb-cols");
        if (menu) menu.open = true;
      });
    });

    this.root.querySelectorAll(".lb-th").forEach((th) => {
      th.addEventListener("click", () => {
        const key = th.dataset.sort;
        if (this.sortKey === key) this.sortDir = -this.sortDir;
        else { this.sortKey = key; this.sortDir = key === "run" ? 1 : -1; }
        this._render();
      });
    });

    this.root.querySelectorAll(".lb-row").forEach((tr) => {
      tr.addEventListener("click", () => this._toggleSelect(tr.dataset.id));
    });

    const compare = this.root.querySelector("#lb-compare");
    if (compare) compare.addEventListener("click", () => this.showDiff());
  }

  _rerenderKeepFocus(input) {
    const at = input.selectionStart;
    this._render();
    const next = this.root.querySelector("#lb-filter");
    if (next) { next.focus(); next.setSelectionRange(at, at); }
  }

  _toggleSelect(id) {
    if (this.selected.includes(id)) this.selected = this.selected.filter((s) => s !== id);
    else this.selected = [...this.selected, id].slice(-2);
    this._render();
  }

  _diffHtml() {
    const { a, b, directions } = this.diffData;

    let html = `<div class="lb-bar-top">`;
    html += `<button type="button" class="btn btn--mini" id="lb-back">&larr; Back to table</button>`;
    html += `<span class="lb-count">Comparing two inference results</span>`;
    html += `</div>`;

    html += `<div class="lb-diff__heads">`;
    [["A", a], ["B", b]].forEach(([tag, side]) => {
      html += `<div class="lb-diff__head"><span class="lb-diff__tag">${tag}</span><span class="lb-run__name">${this._esc(side.run)}</span><span class="lb-run__stamp">${this._esc(side.stamp)}</span></div>`;
    });
    html += `</div>`;

    const all    = [...new Set([...Object.keys(a.metrics), ...Object.keys(b.metrics)])].sort();
    const needle = this.diffFilter.toLowerCase();

    const stems = {};
    all.forEach((key) => {
      const stem = key.replace(/\d+$/, "");
      if (stem !== key) stems[stem] = (stems[stem] || 0) + 1;
    });
    const isSeries = (key) => {
      const stem = key.replace(/\d+$/, "");
      return stem !== key && stems[stem] >= 10;
    };

    const keys = all.filter((key) => {
      if (!this.diffSeries && isSeries(key)) return false;
      if (needle && !key.toLowerCase().includes(needle)) return false;
      if (this.diffChanged && a.metrics[key] === b.metrics[key]) return false;
      return true;
    });

    html += `<section class="lb-diff__section"><h4 class="res-section__cap">Metrics <span>${keys.length} of ${all.length}</span></h4>`;
    html += `<div class="lb-diff__controls">`;
    html += `<input type="search" class="res-filter" id="lb-diff-filter" placeholder="Filter metrics by name" value="${this._esc(this.diffFilter)}" spellcheck="false" />`;
    html += `<label class="lb-check"><input type="checkbox" id="lb-diff-changed"${this.diffChanged ? " checked" : ""} /> differing only</label>`;
    html += `<label class="lb-check"><input type="checkbox" id="lb-diff-series"${this.diffSeries ? " checked" : ""} /> per-index series</label>`;
    html += `</div>`;
    html += `<div class="lb-scroll"><table class="lb-table lb-table--diff"><thead><tr><th class="lb-th">metric</th><th class="lb-th">A</th><th class="lb-th">B</th><th class="lb-th">&Delta; (B&minus;A)</th><th class="lb-th">&Delta;%</th></tr></thead><tbody>`;

    keys.forEach((key) => {
      const va = a.metrics[key];
      const vb = b.metrics[key];
      let cells;

      if (va === undefined || vb === undefined) {
        cells = `<td>${va === undefined ? "&ndash;" : this._fmt(va)}</td><td>${vb === undefined ? "&ndash;" : this._fmt(vb)}</td><td>&ndash;</td><td>&ndash;</td>`;
      } else {
        const delta = vb - va;
        const pct   = va !== 0 ? (delta / Math.abs(va)) * 100 : null;
        const cls   = this._deltaClass(delta, directions[key]);
        cells = `<td>${this._fmt(va)}</td><td>${this._fmt(vb)}</td><td class="${cls}">${delta === 0 ? "=" : this._fmt(delta)}</td><td class="${cls}">${pct === null || delta === 0 ? "&ndash;" : (pct > 0 ? "+" : "") + pct.toFixed(1) + "%"}</td>`;
      }

      html += `<tr><td class="lb-key">${this._esc(key)}</td>${cells}</tr>`;
    });
    html += `</tbody></table></div>`;
    if (!keys.length) html += `<div class="res-empty res-empty--tight">No metric matches the current filters.</div>`;
    html += `</section>`;

    html += this._configDiffHtml(a.config, b.config);
    return html;
  }

  _configDiffHtml(ca, cb) {
    const keys    = [...new Set([...Object.keys(ca), ...Object.keys(cb)])].sort();
    const differs = keys.filter((key) => String(ca[key]) !== String(cb[key]));

    let html = `<section class="lb-diff__section"><h4 class="res-section__cap">Config differences <span>${differs.length} of ${keys.length}</span></h4>`;

    if (!keys.length) {
      return html + `<div class="res-empty res-empty--tight">No resolved config files were found for these runs.</div></section>`;
    }
    if (!differs.length) {
      return html + `<div class="res-empty res-empty--tight">The resolved configs are identical.</div></section>`;
    }

    html += `<div class="lb-scroll"><table class="lb-table lb-table--diff"><thead><tr><th class="lb-th">field</th><th class="lb-th">A</th><th class="lb-th">B</th></tr></thead><tbody>`;
    differs.forEach((key) => {
      const va = ca[key] === undefined ? "&ndash;" : this._esc(String(ca[key]));
      const vb = cb[key] === undefined ? "&ndash;" : this._esc(String(cb[key]));
      html += `<tr><td class="lb-key">${this._esc(key)}</td><td>${va}</td><td>${vb}</td></tr>`;
    });
    html += `</tbody></table></div></section>`;
    return html;
  }

  _deltaClass(delta, direction) {
    if (!direction || delta === 0) return "lb-delta";
    const improved = direction > 0 ? delta > 0 : delta < 0;
    return improved ? "lb-delta lb-delta--good" : "lb-delta lb-delta--bad";
  }

  _bindDiff() {
    const back = this.root.querySelector("#lb-back");
    if (back) back.addEventListener("click", () => {
      this.view = "table";
      this._render();
    });

    const filter = this.root.querySelector("#lb-diff-filter");
    if (filter) filter.addEventListener("input", () => {
      this.diffFilter = filter.value.trim();
      const at = filter.selectionStart;
      this._render();
      const next = this.root.querySelector("#lb-diff-filter");
      if (next) { next.focus(); next.setSelectionRange(at, at); }
    });

    const changed = this.root.querySelector("#lb-diff-changed");
    if (changed) changed.addEventListener("change", () => {
      this.diffChanged = changed.checked;
      this._render();
    });

    const series = this.root.querySelector("#lb-diff-series");
    if (series) series.addEventListener("change", () => {
      this.diffSeries = series.checked;
      this._render();
    });
  }

  _loadVisible() {
    try {
      const raw = localStorage.getItem("leaderboard-cols");
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  _runsBase() {
    try {
      const raw = JSON.parse(localStorage.getItem("results-sources") || "{}");
      return raw.logs || ResultsView.DEFAULT_RUNS;
    } catch (e) {
      return ResultsView.DEFAULT_RUNS;
    }
  }

  _fmt(value) {
    if (value === 0) return "0";
    const magnitude = Math.abs(value);
    if (magnitude >= 10000 || magnitude < 0.001) return value.toExponential(2);
    return String(Number(value.toPrecision(4)));
  }

  _ago(mtime) {
    const seconds = Date.now() / 1000 - mtime;
    if (seconds < 3600)   return `${Math.max(1, Math.round(seconds / 60))}m ago`;
    if (seconds < 86400)  return `${Math.round(seconds / 3600)}h ago`;
    return `${Math.round(seconds / 86400)}d ago`;
  }

  _esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
}

window.LeaderboardView = LeaderboardView;
