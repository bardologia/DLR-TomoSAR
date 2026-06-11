"use strict";

class DatasetPicker {
  static CUSTOM = "__custom__";

  static MULTI_MODES = {
    datasets: { endpoint: "/api/fs/datasets", key: "datasets", noun: "datasets", hint: "none = all datasets in the queue", flag: (d) => (d.has_params ? "" : "  (no params)") },
    runs:     { endpoint: "/api/fs/runs",     key: "runs",     noun: "runs",     hint: "none = every run in the logs path", flag: (d) => (d.has_checkpoint ? "" : "  (no checkpoint)") },
  };

  constructor(view, leaf, spec) {
    this.view = view;
    this.leaf = leaf;
    this.spec = spec;
    this.el = null;
    this.select = null;
    this.custom = null;
    this.summary = null;
    this.panel = null;
    this.options = [];
    this.checks = new Map();
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
    await this._fetchDatasets(this._base());
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
    this._renderSingleOptions(items.map((d) => ({ value: d.path, label: d.name })));
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
    this.summary = document.createElement("button");
    this.summary.type = "button";
    this.summary.className = "cfg-edit__input picker__multi";
    this.summary.addEventListener("click", () => this._togglePanel());

    this.panel = document.createElement("div");
    this.panel.className = "picker__panel";
    this.panel.hidden = true;

    this.el.appendChild(this.summary);
    this.el.appendChild(this.panel);

    this._paintSummary();

    if (this.spec.baseFrom) {
      this.view._onDependency(this.spec.baseFrom, () => this._scheduleReload());
    }

    if (this.spec.open) {
      this.panel.hidden = false;
      this._loadMulti();
    }
  }

  _scheduleReload() {
    if (this.panel.hidden) return;
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

  _paintSummary() {
    const names = this._selectedNames();
    this.summary.textContent = names.length ? `${names.length} selected` : `all ${DatasetPicker.MULTI_MODES[this.spec.mode].noun}`;
    this.summary.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
  }

  async _togglePanel() {
    if (!this.panel.hidden) {
      this.panel.hidden = true;
      return;
    }
    this.panel.hidden = false;
    await this._loadMulti();
  }

  async _loadMulti() {
    const mode = DatasetPicker.MULTI_MODES[this.spec.mode];
    const base = this._base();
    this.panel.innerHTML = `<p class="picker__note">listing ${base || "(no base)"}...</p>`;

    const res = await window.apiGet(`${mode.endpoint}?base=${encodeURIComponent(base || "")}`);
    if (!res.ok) {
      this.panel.innerHTML = `<p class="picker__note">${res.error || `could not list ${mode.noun}`}</p>`;
      return;
    }

    const items = (res[mode.key] || []).filter((d) => (this.spec.validOnly ? d.is_dataset : true));
    const selected = new Set(this._selectedNames());

    this.panel.innerHTML = "";
    const head = document.createElement("div");
    head.className = "picker__panel-head";
    head.appendChild(window.LaunchWidgetDom.mini("All", () => this._setAll(true)));
    head.appendChild(window.LaunchWidgetDom.mini("None", () => this._setAll(false)));
    const hint = document.createElement("span");
    hint.className = "picker__note";
    hint.textContent = mode.hint;
    head.appendChild(hint);
    this.panel.appendChild(head);

    this.checks = new Map();
    items.forEach((d) => {
      const row = document.createElement("label");
      row.className = "picker__check";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = selected.has(d.name);
      box.addEventListener("change", () => this._commitMulti());
      const text = document.createElement("span");
      text.textContent = d.name + mode.flag(d);
      row.appendChild(box);
      row.appendChild(text);
      this.panel.appendChild(row);
      this.checks.set(d.name, box);
    });

    if (!items.length) {
      this.panel.appendChild(Object.assign(document.createElement("p"), { className: "picker__note", textContent: `no ${mode.noun} in ${res.base}` }));
    }
  }

  _setAll(on) {
    this.checks.forEach((box) => (box.checked = on));
    this._commitMulti();
  }

  _commitMulti() {
    const names = [];
    this.checks.forEach((box, name) => {
      if (box.checked) names.push(name);
    });
    this.view._setValue(this.leaf, window.PythonLiteral.render(names));
    this._paintSummary();
  }

  _reset() {
    if (this.spec.multi) {
      this._paintSummary();
      if (this.panel && !this.panel.hidden) this._loadMulti();
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
