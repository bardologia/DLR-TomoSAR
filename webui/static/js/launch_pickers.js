"use strict";

class DatasetPicker {
  static CUSTOM = "__custom__";

  static MULTI_MODES = {
    datasets:     { endpoint: "/api/fs/datasets", key: "datasets", noun: "datasets", hint: "nothing selected = every dataset is processed", badge: (d) => (d.has_params ? null : { text: "no params", tone: "warn" }) },
    runs:         { endpoint: "/api/fs/runs",     key: "runs",     noun: "runs",     hint: "nothing selected = every run is processed",     badge: (d) => (!d.has_checkpoint ? { text: "no checkpoint", tone: "warn" } : d.has_inference ? { text: "inferred", tone: "ok" } : null) },
    runs_compare: { endpoint: "/api/fs/runs",     key: "runs",     noun: "runs",     hint: "select 2 or more runs to compare",            badge: (d) => (d.has_inference ? { text: "inferred", tone: "ok" } : { text: "no inference", tone: "warn" }) },
    param_trials: { endpoint: "/api/fs/param_trials", key: "trials", noun: "trials", hint: "select 2 or more trials to compare across datasets", badge: (d) => (d.dataset ? { text: d.dataset, tone: "ok" } : null) },
  };

  constructor(view, leaf, spec) {
    this.view = view;
    this.leaf = leaf;
    this.spec = spec;
    this.el = null;
    this.select = null;
    this.custom = null;
    this.note = null;
    this.board = null;
    this.count = null;
    this.hint = null;
    this.filter = null;
    this.body = null;
    this.items = [];
    this.options = [];
    this.selected = new Set();
    this.reloadTimer = null;
  }

  build() {
    this.el = document.createElement("div");
    this.el.className = "picker";

    if (this.spec.multi) this._buildMulti();
    else this._buildSingle();

    if (this.spec.datasetFrom) {
      this.view._onDependency(this.spec.datasetFrom, (value) => this._reloadSingle(value));
    }

    return { el: this.el, input: this.select, reset: () => this._reset() };
  }

  _buildSingle() {
    this.select = document.createElement("select");
    this.select.className = "cfg-edit__input picker__select";

    this.custom = document.createElement("input");
    this.custom.className = "cfg-edit__input picker__custom";
    this.custom.spellcheck = false;
    this.custom.hidden = true;
    this.custom.placeholder = "absolute path";

    this.note = document.createElement("span");
    this.note.className = "picker__note";

    this.select.addEventListener("change", () => this._onSelect());
    this.custom.addEventListener("input", () => {
      this.custom.classList.toggle("is-dirty", this.custom.value !== this.leaf.value);
      this.view._setValue(this.leaf, this.custom.value);
      this.view._fireDependents(this.leaf.path, this.custom.value);
    });

    this.el.appendChild(this.select);
    this.el.appendChild(this.custom);
    this.el.appendChild(this.note);

    if (this.spec.baseFrom) {
      this.view._onDependency(this.spec.baseFrom, () => this._loadSingle());
    }

    this._renderSingleOptions([]);
    this._loadSingle();
  }

  _base() {
    if (this.spec.baseFromParent) {
      const current = this.view._effective(this.leaf);
      const trimmed = String(current).replace(/\/+$/, "");
      const cut = trimmed.lastIndexOf("/");
      return cut > 0 ? trimmed.slice(0, cut) : trimmed;
    }
    if (this.spec.baseFrom) {
      const source = this.view._leafByPath(this.spec.baseFrom);
      return source ? this.view._effective(source) : "";
    }
    return "";
  }

  async _loadSingle() {
    if (this.spec.mode === "params") {
      const dataset = this.view._effective(this.view._leafByPath(this.spec.datasetFrom));
      await this._fetchParams(dataset);
      return;
    }
    if (this.spec.mode === "runs") {
      await this._fetchRuns(this._base());
      return;
    }
    await this._fetchDatasets(this._base());
  }

  async _fetchRuns(base) {
    if (!base) {
      this.note.textContent = "set a runs directory first";
      this._renderSingleOptions([]);
      return;
    }
    this.note.textContent = "listing...";
    const res = await window.apiGet(`/api/fs/runs?base=${encodeURIComponent(base)}`);
    if (!res.ok) {
      this.note.textContent = res.error || "could not list runs";
      this._renderSingleOptions([]);
      return;
    }
    let items = res.runs || [];
    if (this.spec.checkpointOnly) items = items.filter((r) => r.has_checkpoint);
    this.note.textContent = items.length ? `${items.length} run${items.length > 1 ? "s" : ""} in ${res.base}` : `no completed runs in ${res.base}`;
    this._renderSingleOptions(items.map((r) => ({ value: r.name, label: r.name + (r.has_checkpoint ? "" : "  (no checkpoint)") })));
  }

  async _fetchDatasets(base) {
    this.note.textContent = "listing...";
    const res = await window.apiGet(`/api/fs/datasets?base=${encodeURIComponent(base || "")}`);
    if (!res.ok) {
      this.note.textContent = res.error || "could not list datasets";
      this._renderSingleOptions([]);
      return;
    }
    const items = (res.datasets || []).filter((d) => (this.spec.validOnly ? d.is_dataset : true));
    this.note.textContent = items.length ? `${items.length} in ${res.base}` : `no datasets in ${res.base}`;
    this._renderSingleOptions(items.map((d) => ({ value: d.path, label: d.name + (d.has_params ? "" : "  (no params)") })));
  }

  async _fetchParams(dataset) {
    if (!dataset) {
      this.note.textContent = "select a dataset first";
      this._renderSingleOptions([]);
      return;
    }
    this.note.textContent = "listing...";
    const res = await window.apiGet(`/api/fs/params?dataset=${encodeURIComponent(dataset)}`);
    if (!res.ok) {
      this.note.textContent = res.error || "could not list params";
      this._renderSingleOptions([]);
      return;
    }
    const items = res.files || [];
    this.note.textContent = items.length ? `${items.length} under ${res.params_root}` : "no extracted params yet";
    this._renderSingleOptions(items.map((f) => ({ value: f.path, label: f.name })));
  }

  _renderSingleOptions(items) {
    const current = this.view._effective(this.leaf);
    const known = new Set(items.map((it) => it.value));

    this.options = items.slice();
    this.select.innerHTML = "";

    if (current && !known.has(current)) {
      this.select.appendChild(this._option(current, `${this._tail(current)} (current)`));
    }
    items.forEach((it) => this.select.appendChild(this._option(it.value, it.label)));
    this.select.appendChild(this._option(DatasetPicker.CUSTOM, "Custom path..."));

    const isCustom = !this.custom.hidden && this.custom.value && !known.has(this.custom.value);
    this.select.value = isCustom ? DatasetPicker.CUSTOM : current;
    if (this.select.value !== DatasetPicker.CUSTOM) this.custom.hidden = true;

    this.select.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
  }

  _onSelect() {
    if (this.select.value === DatasetPicker.CUSTOM) {
      this.custom.hidden = false;
      if (!this.custom.value) this.custom.value = this.view._effective(this.leaf);
      this.custom.focus();
      this.view._setValue(this.leaf, this.custom.value);
      this.view._fireDependents(this.leaf.path, this.custom.value);
      return;
    }
    this.custom.hidden = true;
    this.select.classList.toggle("is-dirty", this.select.value !== this.leaf.value);
    this.view._setValue(this.leaf, this.select.value);
    this.view._fireDependents(this.leaf.path, this.select.value);
  }

  async _reloadSingle(datasetPath) {
    if (this.spec.mode !== "params") return;
    await this._fetchParams(datasetPath);
  }

  _buildMulti() {
    const mode = DatasetPicker.MULTI_MODES[this.spec.mode];

    this.board = document.createElement("div");
    this.board.className = "picker__board";

    const head = document.createElement("div");
    head.className = "picker__board-head";

    this.count = document.createElement("span");
    this.count.className = "picker__count";

    this.hint = document.createElement("span");
    this.hint.className = "picker__hint";

    this.filter = document.createElement("input");
    this.filter.className = "picker__filter";
    this.filter.spellcheck = false;
    this.filter.placeholder = `filter ${mode.noun}...`;
    this.filter.addEventListener("input", () => this._renderItems());

    head.appendChild(this.count);
    head.appendChild(this.hint);
    head.appendChild(this.filter);
    head.appendChild(window.LaunchWidgetDom.mini("All", () => this._setAll(true)));
    head.appendChild(window.LaunchWidgetDom.mini("None", () => this._setAll(false)));
    head.appendChild(window.LaunchWidgetDom.mini("Reload", () => this._loadMulti()));

    this.body = document.createElement("div");
    this.body.className = "picker__items";

    this.board.appendChild(head);
    this.board.appendChild(this.body);
    this.el.appendChild(this.board);

    if (this.spec.baseFrom) {
      this.view._onDependency(this.spec.baseFrom, () => this._scheduleReload());
    }

    this._loadMulti();
  }

  _scheduleReload() {
    clearTimeout(this.reloadTimer);
    this.reloadTimer = setTimeout(() => this._loadMulti(), 350);
  }

  _selectedNames() {
    try {
      const parsed = window.PythonLiteral.parse(this.view._effective(this.leaf));
      return Array.isArray(parsed) ? parsed.map(String) : [];
    } catch (e) {
      return [];
    }
  }

  async _loadMulti() {
    const mode = DatasetPicker.MULTI_MODES[this.spec.mode];
    const base = this._base();

    this.count.textContent = "listing...";
    this.hint.textContent = "";

    const res = await window.apiGet(`${mode.endpoint}?base=${encodeURIComponent(base || "")}`);
    if (!res.ok) {
      this.items = [];
      this.count.textContent = `no ${mode.noun}`;
      this.body.innerHTML = `<p class="picker__empty">${res.error || `could not list ${mode.noun}`}</p>`;
      return;
    }

    this.items = (res[mode.key] || []).filter((d) => (this.spec.validOnly ? d.is_dataset : true));
    this.selected = new Set(this._selectedNames());
    this._renderItems();
    this._paintSummary();
  }

  _renderItems() {
    const mode = DatasetPicker.MULTI_MODES[this.spec.mode];
    const query = this.filter.value.trim().toLowerCase();

    this.body.innerHTML = "";

    if (!this.items.length) {
      this.body.appendChild(Object.assign(document.createElement("p"), { className: "picker__empty", textContent: `no ${mode.noun} in ${this._base()}` }));
      return;
    }

    let shown = 0;
    this.items.forEach((d) => {
      if (query && !d.name.toLowerCase().includes(query)) return;
      shown += 1;

      const row = document.createElement("label");
      row.className = "picker__item";
      row.classList.toggle("is-selected", this.selected.has(d.name));

      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = this.selected.has(d.name);
      box.addEventListener("change", () => {
        if (box.checked) this.selected.add(d.name);
        else this.selected.delete(d.name);
        row.classList.toggle("is-selected", box.checked);
        this._commitMulti();
      });

      const name = document.createElement("span");
      name.className = "picker__item-name";
      name.textContent = d.name;
      name.title = d.path || d.name;

      row.appendChild(box);
      row.appendChild(name);

      const badge = mode.badge(d);
      if (badge) {
        const pill = document.createElement("span");
        pill.className = `picker__badge picker__badge--${badge.tone}`;
        pill.textContent = badge.text;
        row.appendChild(pill);
      }

      this.body.appendChild(row);
    });

    if (!shown) {
      this.body.appendChild(Object.assign(document.createElement("p"), { className: "picker__empty", textContent: `no ${mode.noun} match "${query}"` }));
    }
  }

  _visibleNames() {
    const query = this.filter.value.trim().toLowerCase();
    return this.items.map((d) => d.name).filter((n) => (!query || n.toLowerCase().includes(query)));
  }

  _setAll(on) {
    this._visibleNames().forEach((n) => (on ? this.selected.add(n) : this.selected.delete(n)));
    this._renderItems();
    this._commitMulti();
  }

  _commitMulti() {
    const names = this.items.map((d) => d.name).filter((n) => this.selected.has(n));
    this.view._setValue(this.leaf, window.PythonLiteral.render(names));
    this._paintSummary();
  }

  _paintSummary() {
    const mode = DatasetPicker.MULTI_MODES[this.spec.mode];
    const picked = this.items.filter((d) => this.selected.has(d.name)).length;

    this.count.textContent = picked ? `${picked} of ${this.items.length} selected` : `all ${this.items.length} ${mode.noun}`;
    this.hint.textContent = picked ? "" : mode.hint;
    this.board.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
  }

  _reset() {
    if (this.spec.multi) {
      this.selected = new Set(this._selectedNames());
      this._renderItems();
      this._paintSummary();
      return;
    }
    this.custom.hidden = true;
    this.custom.value = "";
    this.custom.classList.remove("is-dirty");
    this._renderSingleOptions(this.options);
    if (this.spec.mode === "params") this._loadSingle();
  }

  _option(value, label) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = label;
    return opt;
  }

  _tail(path) {
    const trimmed = String(path).replace(/\/+$/, "");
    const cut = trimmed.lastIndexOf("/");
    return cut >= 0 ? trimmed.slice(cut + 1) : trimmed;
  }
}

window.DatasetPicker = DatasetPicker;
