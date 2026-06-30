"use strict";

class PythonLiteral {
  static parse(text) {
    const reader = new PythonLiteral(String(text));
    const value  = reader._value();
    reader._ws();
    if (reader.pos !== reader.text.length) throw new Error(`trailing input at ${reader.pos}`);
    return value;
  }

  static render(value) {
    if (value === null || value === undefined) return "None";
    if (value === true)  return "True";
    if (value === false) return "False";
    if (typeof value === "number") return String(value);
    if (typeof value === "string") return `'${value.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
    if (Array.isArray(value)) return `[${value.map((item) => PythonLiteral.render(item)).join(", ")}]`;
    return `{${Object.entries(value).map(([key, item]) => `${PythonLiteral.render(key)}: ${PythonLiteral.render(item)}`).join(", ")}}`;
  }

  constructor(text) {
    this.text = text;
    this.pos  = 0;
  }

  _ws() {
    while (this.pos < this.text.length && /\s/.test(this.text[this.pos])) this.pos += 1;
  }

  _value() {
    this._ws();
    const ch = this.text[this.pos];

    if (ch === "{") return this._dict();
    if (ch === "[") return this._seq("]");
    if (ch === "(") return this._seq(")");
    if (ch === "'" || ch === '"') return this._string(ch);
    return this._atom();
  }

  _dict() {
    this.pos += 1;
    const out = {};

    while (true) {
      this._ws();
      if (this.pos >= this.text.length) throw new Error("unterminated dict");
      if (this.text[this.pos] === "}") { this.pos += 1; return out; }

      const key = this._value();
      this._ws();
      if (this.text[this.pos] !== ":") throw new Error(`expected ':' at ${this.pos}`);
      this.pos += 1;

      out[String(key)] = this._value();
      this._ws();
      if (this.text[this.pos] === ",") this.pos += 1;
    }
  }

  _seq(close) {
    this.pos += 1;
    const out = [];

    while (true) {
      this._ws();
      if (this.pos >= this.text.length) throw new Error("unterminated sequence");
      if (this.text[this.pos] === close) { this.pos += 1; return out; }

      out.push(this._value());
      this._ws();
      if (this.text[this.pos] === ",") this.pos += 1;
    }
  }

  _string(quote) {
    this.pos += 1;
    let out = "";

    while (this.pos < this.text.length) {
      const ch = this.text[this.pos];
      if (ch === "\\") { out += this.text[this.pos + 1]; this.pos += 2; continue; }
      if (ch === quote) { this.pos += 1; return out; }
      out += ch;
      this.pos += 1;
    }
    throw new Error("unterminated string");
  }

  _atom() {
    const start = this.pos;
    while (this.pos < this.text.length && !/[\s,\]})\:]/.test(this.text[this.pos])) this.pos += 1;
    const token = this.text.slice(start, this.pos);

    if (token === "True")  return true;
    if (token === "False") return false;
    if (token === "None")  return null;
    if (token === "") throw new Error(`empty token at ${start}`);

    const number = Number(token);
    return Number.isNaN(number) ? token : number;
  }
}


class LaunchWidgetDom {
  static mini(label, onClick) {
    const btn       = document.createElement("button");
    btn.type        = "button";
    btn.className   = "btn btn--mini";
    btn.textContent = label;
    btn.addEventListener("click", onClick);
    return btn;
  }
}


class ModelTogglePanel {
  constructor(view, leaf) {
    this.view     = view;
    this.leaf     = leaf;
    this.families = view.modelFamilies || [];
    this.keys     = this.families.flatMap((family) => family.models.map((model) => model.key));
    this.chips    = new Map();
    this.countEl  = null;
  }

  build() {
    const root     = document.createElement("section");
    root.className = "model-panel";

    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = `<h3 class="special-head__name">Models in run</h3><span class="special-head__hint">toggled-off models are skipped</span>`;

    const count     = document.createElement("span");
    count.className = "model-panel__count";
    this.countEl    = count;

    head.appendChild(count);
    head.appendChild(LaunchWidgetDom.mini("All on",  () => this._emit(new Set())));
    head.appendChild(LaunchWidgetDom.mini("All off", () => this._emit(new Set(this.keys))));
    root.appendChild(head);

    const body     = document.createElement("div");
    body.className = "model-panel__families";
    this.families.forEach((family) => body.appendChild(this._family(family)));
    root.appendChild(body);

    this.view.controls[this.leaf.path] = { leaf: this.leaf, reset: () => this._paint() };
    this._paint();
    return root;
  }

  _family(family) {
    const block     = document.createElement("div");
    block.className = "model-family";

    const name       = document.createElement("div");
    name.className   = "model-family__name";
    name.textContent = family.family;

    const grid     = document.createElement("div");
    grid.className = "model-family__grid";
    family.models.forEach((model) => grid.appendChild(this._chip(model)));

    block.appendChild(name);
    block.appendChild(grid);
    return block;
  }

  _chip(model) {
    const chip     = document.createElement("button");
    chip.type      = "button";
    chip.className = "model-chip";
    chip.title     = `--${this.leaf.path} · ${model.key}`;
    chip.innerHTML = `<span class="model-chip__name">${model.name}</span><span class="model-chip__meta">${model.params || ""}</span>`;
    chip.addEventListener("click", () => this._toggle(model.key));
    this.chips.set(model.key, chip);
    return chip;
  }

  _skipped() {
    try {
      const parsed = PythonLiteral.parse(this.view._effective(this.leaf));
      return new Set(Array.isArray(parsed) ? parsed.map(String) : []);
    } catch (e) {
      return new Set();
    }
  }

  _toggle(key) {
    const skipped = this._skipped();
    if (skipped.has(key)) skipped.delete(key);
    else skipped.add(key);
    this._emit(skipped);
  }

  _emit(skipped) {
    const ordered = this.keys.filter((key) => skipped.has(key));
    this.view._setValue(this.leaf, PythonLiteral.render(ordered));
    this._paint();
  }

  _paint() {
    const skipped = this._skipped();

    this.chips.forEach((chip, key) => {
      const on = !skipped.has(key);
      chip.classList.toggle("is-on", on);
      chip.setAttribute("aria-pressed", String(on));
    });

    const active = this.keys.filter((key) => !skipped.has(key)).length;
    this.countEl.textContent = `${active}/${this.keys.length} active`;
  }
}


class ModelCardPanel {
  constructor(view, leaf) {
    this.view     = view;
    this.leaf     = leaf;
    this.families = view.modelFamilies || [];
    this.cards    = new Map();
    this.currentEl = null;
  }

  build() {
    const root     = document.createElement("section");
    root.className = "model-panel";

    const isAe     = this.leaf.path === "ae_model_name";
    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = isAe
      ? `<h3 class="special-head__name">Autoencoder</h3><span class="special-head__hint">the autoencoder to train</span>`
      : `<h3 class="special-head__name">Model</h3><span class="special-head__hint">the architecture to train</span>`;

    const current   = document.createElement("span");
    current.className = "model-panel__count";
    this.currentEl  = current;
    head.appendChild(current);
    root.appendChild(head);

    const body     = document.createElement("div");
    body.className = "model-panel__families";
    this.families.forEach((family) => body.appendChild(this._family(family)));
    root.appendChild(body);

    this.view.controls[this.leaf.path] = { leaf: this.leaf, reset: () => this._paint() };
    this._paint();
    return root;
  }

  _family(family) {
    const block     = document.createElement("div");
    block.className = "model-family";

    const name       = document.createElement("div");
    name.className   = "model-family__name";
    name.textContent = family.family;

    const grid     = document.createElement("div");
    grid.className = "model-pick__grid";
    family.models.forEach((model) => grid.appendChild(this._card(model)));

    block.appendChild(name);
    block.appendChild(grid);
    return block;
  }

  _card(model) {
    const card     = document.createElement("button");
    card.type      = "button";
    card.className = "model-pick";
    card.title     = `--${this.leaf.path} ${model.key}`;

    const badge = model.recommended ? `<span class="model-pick__badge">recommended</span>` : "";
    const meta  = [model.head, model.skip].filter(Boolean).join("  ·  ");

    card.innerHTML =
      `<span class="model-pick__top"><span class="model-pick__name">${model.name}</span>${badge}<span class="model-pick__params">${model.params || ""}</span></span>` +
      `<span class="model-pick__meta">${meta}</span>`;

    card.addEventListener("click", () => this._select(model.key));
    this.cards.set(model.key, card);
    return card;
  }

  _select(key) {
    this.view._setValue(this.leaf, key);
    this._paint();
  }

  _paint() {
    const current = this.view._effective(this.leaf);
    let label     = current;

    this.cards.forEach((card, key) => {
      const on = key === current;
      card.classList.toggle("is-on", on);
      card.setAttribute("aria-pressed", String(on));
      if (on) label = card.querySelector(".model-pick__name").textContent;
    });

    this.currentEl.textContent = label;
  }
}


class ExperimentBuilder {

  static MODES = [
    { key: "curriculum", label: "curriculum",  hint: "warmup x complete cross product, one training run each" },
    { key: "warmup",     label: "warmup only", hint: "curriculum disabled, check losses from the catalog and each trains alone as one trial" },
    { key: "secondary",  label: "secondaries", hint: "one training run per secondary-track selection" },
    { key: "patch",      label: "patch",       hint: "one training run per patch size, same model end to end" },
    { key: "presence",   label: "slot presence", hint: "slot-presence loss ablation crossed with both matching strategies, one training run per cell" },
    { key: "input",      label: "input channels", hint: "input-channel ablation across all tracks, one training run per input variant" },
  ];

  static PRESENCE_MATCHES = [
    { label: "sort", strategy: "sort_gt_by_mu" },
    { label: "hung", strategy: "hungarian_active" },
  ];

  static STRATEGIES = [
    { key: "uniform",     note: "each trial samples n_secondaries distinct tracks uniformly, n_trials distinct sets" },
    { key: "gaussian",    note: "indices drawn from Normal(mean, sigma) over the secondary list, mean and sigma required" },
    { key: "consecutive", note: "block of n_secondaries consecutive tracks, advanced block_step per trial until the stack ends" },
    { key: "spaced",      note: "n_secondaries tracks spaced 'spacing' apart, advanced block_step per trial until the stack ends" },
  ];

  static SECONDARY_FIELDS = [
    { key: "n_secondaries", strategies: ["uniform", "gaussian", "consecutive", "spaced"] },
    { key: "n_trials",      strategies: ["uniform", "gaussian"] },
    { key: "seed",          strategies: ["uniform", "gaussian"] },
    { key: "mean",          strategies: ["gaussian"] },
    { key: "sigma",         strategies: ["gaussian"] },
    { key: "block_step",    strategies: ["consecutive", "spaced"] },
    { key: "spacing",       strategies: ["spaced"] },
  ];

  constructor(view, byPath) {
    this.view         = view;
    this.trialsLeaf   = byPath.get("trials_enabled");
    this.modeLeaf     = byPath.get("trials_mode");
    this.warmupLeaf   = byPath.get("warmup_losses");
    this.completeLeaf = byPath.get("complete_losses");
    this.modelLeaf    = byPath.get("backbone_name");
    this.gpusLeaf     = byPath.get("gpus");

    this.presenceTrialsLeaf = byPath.get("presence_trials");
    this.presenceMatchLeaf  = byPath.get("presence_match_strategies");

    this.inputTrialsLeaf = byPath.get("input_trials");

    this.secondary = new Map();
    this.patch     = new Map();
    byPath.forEach((leaf) => {
      if (leaf.section === "secondary_trials") this.secondary.set(leaf.path.split(".").pop(), leaf);
      if (leaf.section === "patch_trials")     this.patch.set(leaf.path.split(".").pop(), leaf);
    });

    this.claimed = ["trials_enabled", "warmup_losses", "complete_losses"];
    if (this.modeLeaf) this.claimed.push("trials_mode");
    this.secondary.forEach((leaf) => this.claimed.push(leaf.path));
    this.patch.forEach((leaf) => this.claimed.push(leaf.path));
    if (this.presenceTrialsLeaf) this.claimed.push(this.presenceTrialsLeaf.path);
    if (this.presenceMatchLeaf)  this.claimed.push(this.presenceMatchLeaf.path);
    if (this.inputTrialsLeaf)    this.claimed.push(this.inputTrialsLeaf.path);

    this.terms          = this._termCatalog();
    this.variants       = { warmup: [], complete: [] };
    this.lists          = {};
    this.summaryEl      = null;
    this.namesEl        = null;
    this.hintEl         = null;
    this.root           = null;
    this.columnsEl      = null;
    this.columnEls      = {};
    this.secondaryEl    = null;
    this.strategySelect = null;
    this.strategyNoteEl = null;
    this.secondaryRows  = [];
    this.patchEl        = null;
    this.patchSizesEl   = null;
    this.warmupCatalogEl     = null;
    this.warmupCatalogGridEl = null;
    this.warmupCountEl       = null;
    this.warmupCustomEl      = null;
    this.warmupCustomHeadEl  = null;
    this.presenceEl     = null;
    this.inputEl        = null;
    this.inputCellsEl   = null;
    this.modeButtons    = new Map();
    this.modeEl         = null;
    this._paintSwitch   = null;
  }

  build() {
    this.root           = document.createElement("section");
    this.root.className = "exp-builder";

    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = `<h3 class="special-head__name">Experiment fan-out</h3>`;

    const hint     = document.createElement("span");
    hint.className = "special-head__hint";
    this.hintEl    = hint;
    head.appendChild(hint);

    const summary     = document.createElement("span");
    summary.className = "exp-builder__summary";
    this.summaryEl    = summary;
    head.appendChild(summary);
    if (this.modeLeaf) head.appendChild(this._modeSegment());
    head.appendChild(this._trialsSwitch());

    const body     = document.createElement("div");
    body.className = "exp-builder__body";

    const columns     = document.createElement("div");
    columns.className = "exp-builder__columns";
    this.columnsEl    = columns;
    this.columnEls.warmup   = columns.appendChild(this._column("warmup", "warmup losses"));
    this.columnEls.complete = columns.appendChild(this._column("complete", "complete losses"));
    body.appendChild(columns);

    if (this.modeLeaf) body.appendChild(this._warmupCatalogPanel());
    if (this.modeLeaf && this.secondary.size)     body.appendChild(this._secondaryPanel());
    if (this.modeLeaf && this.patch.size)         body.appendChild(this._patchPanel());
    if (this.modeLeaf && this.presenceTrialsLeaf) body.appendChild(this._presencePanel());
    if (this.modeLeaf && this.inputTrialsLeaf)    body.appendChild(this._inputPanel());

    const preview     = document.createElement("div");
    preview.className = "exp-builder__preview";
    preview.innerHTML = `<span class="exp-builder__preview-title">trial run names</span>`;
    const names       = document.createElement("div");
    names.className   = "exp-builder__names";
    this.namesEl      = names;
    preview.appendChild(names);
    body.appendChild(preview);

    this.root.appendChild(head);
    this.root.appendChild(body);

    [this.warmupLeaf, this.completeLeaf].forEach((leaf) => {
      this.view.controls[leaf.path] = { leaf, reset: () => this._reload() };
    });

    this._reload();
    return this.root;
  }

  refreshFromView() {
    if (!this.summaryEl) return;
    if (this._paintSwitch) this._paintSwitch();
    this._paintMode();
    this._paintSecondary();
    this._paintPatch();
    this._paintPresence();
    this._paintInput();
    this._paintWarmupCatalog();
    this._paintSummary();
    this._paintNames();
  }

  _mode() {
    if (!this.modeLeaf) return "curriculum";
    const value = this.view._effective(this.modeLeaf);
    return ExperimentBuilder.MODES.some((mode) => mode.key === value) ? value : "curriculum";
  }

  _strategy() {
    const leaf = this.secondary.get("strategy");
    if (!leaf) return "consecutive";
    const value = this.view._effective(leaf);
    return ExperimentBuilder.STRATEGIES.some((strategy) => strategy.key === value) ? value : "consecutive";
  }

  _modeSegment() {
    const leaf     = this.modeLeaf;
    const wrap     = document.createElement("div");
    wrap.className = "exp-mode";
    wrap.title     = `--${leaf.path}`;

    ExperimentBuilder.MODES.forEach((mode) => {
      const btn       = document.createElement("button");
      btn.type        = "button";
      btn.className   = "exp-mode__btn";
      btn.textContent = mode.label;
      btn.addEventListener("click", () => {
        this.view._setValue(leaf, mode.key);
        if (this.view._effective(this.trialsLeaf) !== "True") this.view._setValue(this.trialsLeaf, "True");
      });
      this.modeButtons.set(mode.key, btn);
      wrap.appendChild(btn);
    });

    this.modeEl = wrap;
    this.view.controls[leaf.path] = { leaf, reset: () => this._paintMode() };
    return wrap;
  }

  _secondaryPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">secondary selection</span>`;

    const strategyLeaf = this.secondary.get("strategy");
    if (strategyLeaf) {
      const select     = document.createElement("select");
      select.className = "run-select exp-secondary__strategy";
      select.title     = `--${strategyLeaf.path}`;
      ExperimentBuilder.STRATEGIES.forEach((strategy) => {
        const opt       = document.createElement("option");
        opt.value       = strategy.key;
        opt.textContent = strategy.key;
        select.appendChild(opt);
      });
      select.addEventListener("change", () => this.view._setValue(strategyLeaf, select.value));
      head.appendChild(select);

      this.strategySelect = select;
      this.view.controls[strategyLeaf.path] = { leaf: strategyLeaf, reset: () => this._paintSecondary() };
    }

    const note     = document.createElement("p");
    note.className = "exp-secondary__note";
    this.strategyNoteEl = note;

    const grid     = document.createElement("div");
    grid.className = "exp-secondary__grid";
    ExperimentBuilder.SECONDARY_FIELDS.forEach((field) => {
      const leaf = this.secondary.get(field.key);
      if (leaf) grid.appendChild(this._secondaryRow(field, leaf));
    });

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(grid);
    this.secondaryEl = panel;
    return panel;
  }

  _secondaryRow(field, leaf) {
    const row     = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title     = `--${leaf.path}`;

    const name     = document.createElement("div");
    name.className = "cfg-edit__name";
    name.innerHTML = `${field.key}<span>${leaf.type === "none" ? "float" : leaf.type}</span>`;

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.type       = "number";
    input.step       = leaf.type === "int" ? "1" : "any";
    input.value      = leaf.value === "None" ? "" : leaf.value;
    input.spellcheck = false;
    if (leaf.value === "None") input.placeholder = "required";
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value && input.value !== "");
      this.view._setValue(leaf, input.value);
    });

    row.appendChild(name);
    row.appendChild(input);

    this.secondaryRows.push({ strategies: field.strategies, row });
    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = leaf.value === "None" ? "" : leaf.value;
      input.classList.remove("is-dirty");
    } };
    return row;
  }

  _patchPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-patch";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">patch sweep</span>`;

    const ratioLeaf = this.patch.get("stride_ratio");
    if (ratioLeaf) head.appendChild(this._patchRatio(ratioLeaf));

    head.appendChild(LaunchWidgetDom.mini("Add size", () => {
      const sizes = this._patchSizes();
      sizes.push(sizes.length ? sizes[sizes.length - 1] : 64);
      this._emitPatchSizes(sizes);
    }));

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "each size trains the same model end to end; stride = round(size x stride_ratio)";

    const sizes     = document.createElement("div");
    sizes.className = "exp-patch__sizes";
    this.patchSizesEl = sizes;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(sizes);
    this.patchEl = panel;

    const sizesLeaf = this.patch.get("sizes");
    if (sizesLeaf) this.view.controls[sizesLeaf.path] = { leaf: sizesLeaf, reset: () => this._paintPatch() };
    return panel;
  }

  _patchRatio(leaf) {
    const wrap     = document.createElement("label");
    wrap.className = "exp-patch__ratio";
    wrap.title     = `--${leaf.path}`;
    wrap.innerHTML = `<span>stride ratio</span>`;

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.type       = "number";
    input.step       = "any";
    input.min        = "0";
    input.max        = "1";
    input.spellcheck = false;
    input.value      = leaf.value;
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value && input.value !== "");
      this.view._setValue(leaf, input.value);
    });

    wrap.appendChild(input);
    this.view.controls[leaf.path] = { leaf, reset: () => {
      input.value = leaf.value;
      input.classList.remove("is-dirty");
    } };
    return wrap;
  }

  _patchSizes() {
    const leaf = this.patch.get("sizes");
    if (!leaf) return [];

    let raw = null;
    try {
      raw = PythonLiteral.parse(this.view._effective(leaf));
    } catch (e) {
      raw = null;
    }
    if (!Array.isArray(raw)) return [];

    return raw.map((value) => Number(value)).filter((value) => Number.isFinite(value));
  }

  _emitPatchSizes(sizes) {
    const leaf = this.patch.get("sizes");
    if (!leaf) return;

    const clean = sizes.map((value) => Math.round(Number(value))).filter((value) => Number.isFinite(value) && value >= 1);
    this.view._setValue(leaf, PythonLiteral.render(clean));

    this._paintPatch();
    this._paintSummary();
    this._paintNames();
  }

  _patchChip(size, index, sizes) {
    const chip     = document.createElement("div");
    chip.className = "exp-patch__chip";

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input exp-patch__size";
    input.type       = "number";
    input.step       = "1";
    input.min        = "1";
    input.spellcheck = false;
    input.value      = String(size);
    input.title      = "patch size in pixels (square)";
    input.addEventListener("change", () => {
      const next  = sizes.slice();
      next[index] = Number(input.value);
      this._emitPatchSizes(next);
    });

    const remove = LaunchWidgetDom.mini("×", () => {
      const next = sizes.slice();
      next.splice(index, 1);
      this._emitPatchSizes(next);
    });
    remove.classList.add("exp-patch__remove");
    remove.title = "Remove size";

    chip.appendChild(input);
    chip.appendChild(remove);
    return chip;
  }

  _paintPatch() {
    if (!this.patchSizesEl) return;

    const sizes = this._patchSizes();
    this.patchSizesEl.innerHTML = "";
    sizes.forEach((size, index) => this.patchSizesEl.appendChild(this._patchChip(size, index, sizes)));
  }

  _presenceMatchMap() {
    if (!this.presenceMatchLeaf) return {};
    try {
      const raw = PythonLiteral.parse(this.view._effective(this.presenceMatchLeaf));
      return raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
    } catch (e) {
      return {};
    }
  }

  _presenceStrategies() {
    return Object.keys(this._presenceMatchMap());
  }

  _presenceCells() {
    if (!this.presenceTrialsLeaf) return [];
    try {
      const raw = PythonLiteral.parse(this.view._effective(this.presenceTrialsLeaf));
      return raw && typeof raw === "object" && !Array.isArray(raw) ? Object.keys(raw) : [];
    } catch (e) {
      return [];
    }
  }

  _presencePanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">slot-presence ablation</span>`;
    const reset    = LaunchWidgetDom.mini("reset matrix", () => this._resetPresence());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Toggle matching strategies and drop cells to trim the fan-out. Cell weights and adding new cells live in code (_default_presence_trials); use reset matrix to restore the full default.";

    const matchHead       = document.createElement("div");
    matchHead.className    = "exp-presence__sub";
    matchHead.textContent  = "matching strategies";

    const strategies          = document.createElement("div");
    strategies.className      = "exp-builder__names exp-presence__strategies";
    this.presenceStrategiesEl = strategies;

    const cellHead       = document.createElement("div");
    cellHead.className    = "exp-presence__sub";
    cellHead.textContent  = "loss-term cells";

    const cells          = document.createElement("div");
    cells.className      = "exp-builder__names exp-presence__cells";
    this.presenceCellsEl = cells;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(matchHead);
    panel.appendChild(strategies);
    panel.appendChild(cellHead);
    panel.appendChild(cells);
    this.presenceEl = panel;

    if (this.presenceTrialsLeaf) this.view.controls[this.presenceTrialsLeaf.path] = { leaf: this.presenceTrialsLeaf, reset: () => this._repaintPresence() };
    if (this.presenceMatchLeaf)  this.view.controls[this.presenceMatchLeaf.path]  = { leaf: this.presenceMatchLeaf,  reset: () => this._repaintPresence() };

    this._paintPresence();
    return panel;
  }

  _paintPresence() {
    if (this.presenceStrategiesEl) {
      this.presenceStrategiesEl.innerHTML = "";
      const active = this._presenceMatchMap();
      ExperimentBuilder.PRESENCE_MATCHES.forEach(({ label, strategy }) => {
        const on   = Object.prototype.hasOwnProperty.call(active, label);
        const chip = document.createElement("button");
        chip.type        = "button";
        chip.className   = "exp-name exp-presence__toggle" + (on ? " is-on" : "");
        chip.textContent = `${label} · ${strategy}`;
        chip.title       = on ? "Click to drop this matching strategy" : "Click to add this matching strategy";
        chip.addEventListener("click", () => this._toggleStrategy(label));
        this.presenceStrategiesEl.appendChild(chip);
      });
    }

    if (this.presenceCellsEl) {
      this.presenceCellsEl.innerHTML = "";
      this._presenceCells().forEach((cell) => this.presenceCellsEl.appendChild(this._presenceCellChip(cell)));
    }
  }

  _presenceCellChip(cell) {
    const chip     = document.createElement("span");
    chip.className = "exp-name exp-presence__cell";

    const label       = document.createElement("span");
    label.textContent = cell;
    chip.appendChild(label);

    const remove = LaunchWidgetDom.mini("×", () => this._removeCell(cell));
    remove.classList.add("exp-presence__remove");
    remove.title = "Remove cell";
    chip.appendChild(remove);
    return chip;
  }

  _emitPresence(leaf, value) {
    this.view._setValue(leaf, PythonLiteral.render(value));
    this._paintPresence();
    this._paintSummary();
    this._paintNames();
  }

  _removeCell(cell) {
    if (!this.presenceTrialsLeaf) return;
    let raw;
    try {
      raw = PythonLiteral.parse(this.view._effective(this.presenceTrialsLeaf));
    } catch (e) {
      return;
    }
    if (!raw || typeof raw !== "object" || Object.keys(raw).length <= 1) return;
    delete raw[cell];
    this._emitPresence(this.presenceTrialsLeaf, raw);
  }

  _toggleStrategy(label) {
    if (!this.presenceMatchLeaf) return;
    let raw;
    try {
      raw = PythonLiteral.parse(this.view._effective(this.presenceMatchLeaf));
    } catch (e) {
      raw = {};
    }
    if (!raw || typeof raw !== "object") raw = {};

    if (Object.prototype.hasOwnProperty.call(raw, label)) {
      if (Object.keys(raw).length <= 1) return;
      delete raw[label];
    } else {
      const canon = ExperimentBuilder.PRESENCE_MATCHES.find((entry) => entry.label === label);
      if (canon) raw[label] = canon.strategy;
    }
    this._emitPresence(this.presenceMatchLeaf, raw);
  }

  _resetPresence() {
    if (this.presenceTrialsLeaf) this.view._setValue(this.presenceTrialsLeaf, this.presenceTrialsLeaf.value);
    if (this.presenceMatchLeaf)  this.view._setValue(this.presenceMatchLeaf,  this.presenceMatchLeaf.value);
    this._repaintPresence();
  }

  _repaintPresence() {
    this._paintPresence();
    this._paintSummary();
    this._paintNames();
  }

  _inputTrials() {
    if (!this.inputTrialsLeaf) return {};
    try {
      const raw = PythonLiteral.parse(this.view._effective(this.inputTrialsLeaf));
      return raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
    } catch (e) {
      return {};
    }
  }

  _inputCells() {
    return Object.keys(this._inputTrials());
  }

  _inputSpecText(spec) {
    const on    = (key) => spec && Object.prototype.hasOwnProperty.call(spec, key) ? Boolean(spec[key]) : null;
    const amps  = [];
    if (on("use_primary")     !== false) amps.push("primary");
    if (on("use_secondaries") !== false) amps.push("secondaries");

    const parts = [];
    parts.push(amps.length ? `${amps.join("+")} amp` : "no amplitude");
    parts.push(on("use_interferograms") === false ? "no ifg" : "ifg");
    if (on("use_dem") === true) parts.push("dem");
    return parts.join(", ");
  }

  _inputPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">input-channel ablation</span>`;
    const reset    = LaunchWidgetDom.mini("reset variants", () => this._resetInput());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Every variant trains on all tracks; each toggles which input channels feed the model. Drop variants to trim the fan-out. Adding variants and channel representations live in code (_default_input_trials); use reset variants to restore the default.";

    const cellHead       = document.createElement("div");
    cellHead.className    = "exp-presence__sub";
    cellHead.textContent  = "input variants";

    const cells         = document.createElement("div");
    cells.className     = "exp-builder__names exp-presence__cells";
    this.inputCellsEl   = cells;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(cellHead);
    panel.appendChild(cells);
    this.inputEl = panel;

    if (this.inputTrialsLeaf) this.view.controls[this.inputTrialsLeaf.path] = { leaf: this.inputTrialsLeaf, reset: () => this._repaintInput() };

    this._paintInput();
    return panel;
  }

  _paintInput() {
    if (!this.inputCellsEl) return;
    this.inputCellsEl.innerHTML = "";
    const trials = this._inputTrials();
    Object.keys(trials).forEach((cell) => this.inputCellsEl.appendChild(this._inputCellChip(cell, trials[cell])));
  }

  _inputCellChip(cell, spec) {
    const chip     = document.createElement("span");
    chip.className = "exp-name exp-presence__cell";
    chip.title     = this._inputSpecText(spec);

    const label       = document.createElement("span");
    label.textContent = `${cell} · ${this._inputSpecText(spec)}`;
    chip.appendChild(label);

    const remove = LaunchWidgetDom.mini("×", () => this._removeInputCell(cell));
    remove.classList.add("exp-presence__remove");
    remove.title = "Remove variant";
    chip.appendChild(remove);
    return chip;
  }

  _removeInputCell(cell) {
    if (!this.inputTrialsLeaf) return;
    let raw;
    try {
      raw = PythonLiteral.parse(this.view._effective(this.inputTrialsLeaf));
    } catch (e) {
      return;
    }
    if (!raw || typeof raw !== "object" || Object.keys(raw).length <= 1) return;
    delete raw[cell];
    this.view._setValue(this.inputTrialsLeaf, PythonLiteral.render(raw));
    this._repaintInput();
  }

  _resetInput() {
    if (this.inputTrialsLeaf) this.view._setValue(this.inputTrialsLeaf, this.inputTrialsLeaf.value);
    this._repaintInput();
  }

  _repaintInput() {
    this._paintInput();
    this._paintSummary();
    this._paintNames();
  }

  _paintMode() {
    const mode = this._mode();

    this.modeButtons.forEach((btn, key) => btn.classList.toggle("is-on", key === mode));
    if (this.modeEl) this.modeEl.classList.toggle("is-dirty", this.view.dirty[this.modeLeaf.path] !== undefined);
    if (this.hintEl) this.hintEl.textContent = ExperimentBuilder.MODES.find((entry) => entry.key === mode).hint;

    if (this.columnsEl)          this.columnsEl.hidden          = mode !== "curriculum";
    if (this.columnEls.complete) this.columnEls.complete.hidden = mode !== "curriculum";
    if (this.warmupCatalogEl)    this.warmupCatalogEl.hidden    = mode !== "warmup";
    if (this.secondaryEl)        this.secondaryEl.hidden        = mode !== "secondary";
    if (this.patchEl)            this.patchEl.hidden            = mode !== "patch";
    if (this.presenceEl)         this.presenceEl.hidden         = mode !== "presence";
    if (this.inputEl)            this.inputEl.hidden            = mode !== "input";
  }

  _paintSecondary() {
    if (!this.secondaryEl) return;
    const strategy = this._strategy();

    if (this.strategySelect) this.strategySelect.value = strategy;
    this.strategyNoteEl.textContent = ExperimentBuilder.STRATEGIES.find((entry) => entry.key === strategy).note;
    this.secondaryRows.forEach((entry) => (entry.row.hidden = !entry.strategies.includes(strategy)));
  }

  _secondaryEffective(key) {
    const leaf = this.secondary.get(key);
    return leaf ? this.view._effective(leaf) : "?";
  }

  _termCatalog() {
    const prefix = "curriculum.complete.";
    const leaves = this.view.config.leaves.filter((leaf) => leaf.section === "curriculum.complete");

    const blocks = new Map();
    leaves.forEach((leaf) => {
      if (!blocks.has(leaf.block)) blocks.set(leaf.block, []);
      blocks.get(leaf.block).push(leaf);
    });

    const terms = [];
    blocks.forEach((blockLeaves) => {
      const short  = (leaf) => leaf.path.slice(prefix.length);
      const use    = blockLeaves.find((leaf) => leaf.type === "bool" && short(leaf).startsWith("use_"));
      const weight = blockLeaves.find((leaf) => short(leaf).startsWith("weight_") && (leaf.type === "float" || leaf.type === "int"));
      if (!use || !weight) return;

      const fallback = Number(weight.value);
      terms.push({ key: short(use).slice(4), useKey: short(use), weightKey: short(weight), defaultWeight: fallback > 0 ? fallback : 1.0 });
    });
    return terms;
  }

  _trialsSwitch() {
    const leaf     = this.trialsLeaf;
    const toggle   = document.createElement("button");
    toggle.type    = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.title     = `--${leaf.path}`;
    toggle.innerHTML = `<span class="switch__knob"></span>`;

    const paint = () => {
      const on = this.view._effective(leaf) === "True";
      toggle.classList.toggle("is-on", on);
      toggle.classList.toggle("is-dirty", this.view.dirty[leaf.path] !== undefined);
      toggle.setAttribute("aria-checked", String(on));
      this.root.classList.toggle("is-trials", on);
    };

    toggle.addEventListener("click", () => {
      const next = this.view._effective(leaf) === "True" ? "False" : "True";
      this.view._setValue(leaf, next);
      paint();
    });

    this._paintSwitch = paint;
    this.view.controls[leaf.path] = { leaf, reset: () => this._reload() };
    return toggle;
  }

  _column(which, title) {
    const col     = document.createElement("div");
    col.className = "exp-col";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">${title}</span><span class="exp-col__count"></span>`;
    head.appendChild(LaunchWidgetDom.mini("Add variant", () => {
      this.variants[which].push(this._blankVariant(which));
      this._emit(which);
    }));

    const list     = document.createElement("div");
    list.className = "exp-col__list";

    this.lists[which] = { list, count: head.querySelector(".exp-col__count") };
    col.appendChild(head);
    col.appendChild(list);
    return col;
  }

  _blankVariant(which) {
    const base  = this.terms.find((term) => term.key === "param_l1") || this.terms[0];
    const label = this._uniqueLabel(which, which === "warmup" ? "warm" : "exp");
    return { label, terms: [{ useKey: base.useKey, weightKey: base.weightKey, weight: base.defaultWeight }], extras: {} };
  }

  _uniqueLabel(which, seed) {
    const taken = new Set(this.variants[which].map((variant) => variant.label));
    if (!taken.has(seed)) return seed;

    let n = 2;
    while (taken.has(`${seed}${n}`)) n += 1;
    return `${seed}${n}`;
  }

  _variantsFrom(leaf) {
    let raw = null;
    try {
      raw = PythonLiteral.parse(this.view._effective(leaf));
    } catch (e) {
      raw = null;
    }
    if (!raw || Array.isArray(raw) || typeof raw !== "object") return [];

    return Object.entries(raw).map(([label, spec]) => {
      const entry  = spec && typeof spec === "object" && !Array.isArray(spec) ? spec : {};
      const terms  = [];
      const extras = {};
      const used   = new Set();

      this.terms.forEach((term) => {
        if (entry[term.useKey] !== true) return;
        const weight = typeof entry[term.weightKey] === "number" ? entry[term.weightKey] : term.defaultWeight;
        terms.push({ useKey: term.useKey, weightKey: term.weightKey, weight });
        used.add(term.useKey);
        used.add(term.weightKey);
      });

      Object.entries(entry).forEach(([key, value]) => {
        if (!used.has(key)) extras[key] = value;
      });

      return { label, terms, extras };
    });
  }

  _dictFor(variants) {
    const out = {};
    variants.forEach((variant) => {
      const entry = { ...variant.extras };
      variant.terms.forEach((term) => {
        entry[term.useKey]    = true;
        entry[term.weightKey] = Number(term.weight) || 0;
      });
      out[variant.label || "unnamed"] = entry;
    });
    return out;
  }

  _emit(which) {
    const leaf = which === "warmup" ? this.warmupLeaf : this.completeLeaf;
    this.view._setValue(leaf, PythonLiteral.render(this._dictFor(this.variants[which])));
    this._paintAll();
  }

  _reload() {
    this.variants.warmup   = this._variantsFrom(this.warmupLeaf);
    this.variants.complete = this._variantsFrom(this.completeLeaf);
    if (this._paintSwitch) this._paintSwitch();
    this._paintMode();
    this._paintSecondary();
    this._paintPatch();
    this._paintAll();
  }

  _paintAll() {
    ["warmup", "complete"].forEach((which) => {
      const { list, count } = this.lists[which];
      list.innerHTML = "";
      this.variants[which].forEach((variant, index) => list.appendChild(this._card(which, variant, index)));
      count.textContent = `${this.variants[which].length} variant${this.variants[which].length === 1 ? "" : "s"}`;
    });

    this._paintWarmupCatalog();
    this._paintSummary();
    this._paintNames();
  }

  _card(which, variant, index) {
    const card     = document.createElement("div");
    card.className = "exp-card";

    const top     = document.createElement("div");
    top.className = "exp-card__top";

    const label      = document.createElement("input");
    label.className  = "cfg-edit__input exp-card__label";
    label.value      = variant.label;
    label.spellcheck = false;
    label.title      = "variant label, used in the run name";
    label.addEventListener("change", () => {
      variant.label = label.value.trim() || variant.label;
      this._emit(which);
    });

    top.appendChild(label);
    top.appendChild(LaunchWidgetDom.mini("Duplicate", () => {
      const copy = JSON.parse(JSON.stringify(variant));
      copy.label = this._uniqueLabel(which, copy.label);
      this.variants[which].splice(index + 1, 0, copy);
      this._emit(which);
    }));
    top.appendChild(LaunchWidgetDom.mini("Remove", () => {
      this.variants[which].splice(index, 1);
      this._emit(which);
    }));
    card.appendChild(top);

    const terms     = document.createElement("div");
    terms.className = "exp-card__terms";
    variant.terms.forEach((term, ti) => terms.appendChild(this._termRow(which, variant, term, ti)));
    card.appendChild(terms);

    const extrasKeys = Object.keys(variant.extras);
    if (extrasKeys.length) {
      const note       = document.createElement("div");
      note.className   = "exp-card__extras";
      note.textContent = `raw keys kept: ${extrasKeys.join(", ")}`;
      card.appendChild(note);
    }

    const addTerm = LaunchWidgetDom.mini("Add term", () => {
      const unused = this.terms.find((term) => !variant.terms.some((vt) => vt.useKey === term.useKey));
      const pick   = unused || this.terms[0];
      variant.terms.push({ useKey: pick.useKey, weightKey: pick.weightKey, weight: pick.defaultWeight });
      this._emit(which);
    });
    addTerm.classList.add("exp-card__add");
    card.appendChild(addTerm);

    return card;
  }

  _termRow(which, variant, term, index) {
    const row     = document.createElement("div");
    row.className = "exp-term";

    const select     = document.createElement("select");
    select.className = "run-select exp-term__select";
    this.terms.forEach((option) => {
      const opt       = document.createElement("option");
      opt.value       = option.useKey;
      opt.textContent = option.key;
      if (option.useKey === term.useKey) opt.selected = true;
      select.appendChild(opt);
    });
    select.addEventListener("change", () => {
      const picked   = this.terms.find((option) => option.useKey === select.value);
      term.useKey    = picked.useKey;
      term.weightKey = picked.weightKey;
      term.weight    = picked.defaultWeight;
      this._emit(which);
    });

    const weight      = document.createElement("input");
    weight.className  = "cfg-edit__input exp-term__weight";
    weight.type       = "number";
    weight.step       = "any";
    weight.value      = String(term.weight);
    weight.spellcheck = false;
    weight.title      = "loss weight";
    weight.addEventListener("change", () => {
      term.weight = Number(weight.value) || 0;
      this._emit(which);
    });

    const remove = LaunchWidgetDom.mini("×", () => {
      variant.terms.splice(index, 1);
      this._emit(which);
    });
    remove.classList.add("exp-term__remove");
    remove.title = "Remove term";

    row.appendChild(select);
    row.appendChild(weight);
    row.appendChild(remove);
    return row;
  }

  _warmupCatalogPanel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-warmup";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">warmup loss sweep</span><span class="exp-col__count exp-warmup__count"></span>`;
    this.warmupCountEl = head.querySelector(".exp-warmup__count");

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Check a loss to train it alone as one trial. Expand a checked card to stack extra terms onto that trial.";

    const grid     = document.createElement("div");
    grid.className = "exp-warmup__grid";
    this.warmupCatalogGridEl = grid;

    const customHead       = document.createElement("div");
    customHead.className    = "exp-presence__sub exp-warmup__customhead";
    customHead.textContent  = "other variants";
    this.warmupCustomHeadEl = customHead;

    const custom     = document.createElement("div");
    custom.className = "exp-warmup__custom";
    this.warmupCustomEl = custom;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(grid);
    panel.appendChild(customHead);
    panel.appendChild(custom);
    this.warmupCatalogEl = panel;
    return panel;
  }

  _warmupBindings() {
    const byTerm = new Map();
    const custom = [];

    this.variants.warmup.forEach((variant, index) => {
      const base  = variant.terms[0];
      const known = base && this.terms.some((term) => term.useKey === base.useKey);
      if (known && !byTerm.has(base.useKey)) byTerm.set(base.useKey, index);
      else custom.push(index);
    });

    return { byTerm, custom };
  }

  _paintWarmupCatalog() {
    if (!this.warmupCatalogGridEl) return;

    const { byTerm, custom } = this._warmupBindings();

    this.warmupCatalogGridEl.innerHTML = "";
    this.terms.forEach((term) => {
      const index   = byTerm.has(term.useKey) ? byTerm.get(term.useKey) : -1;
      const variant = index >= 0 ? this.variants.warmup[index] : null;
      this.warmupCatalogGridEl.appendChild(this._warmupLossCard(term, variant, index));
    });

    this.warmupCustomEl.innerHTML = "";
    custom.forEach((index) => this.warmupCustomEl.appendChild(this._card("warmup", this.variants.warmup[index], index)));
    const hasCustom = custom.length > 0;
    this.warmupCustomEl.hidden     = !hasCustom;
    this.warmupCustomHeadEl.hidden = !hasCustom;

    if (this.warmupCountEl) {
      const n = this.variants.warmup.length;
      this.warmupCountEl.textContent = `${n} selected = ${n} trial${n === 1 ? "" : "s"}`;
    }
  }

  _warmupLossCard(term, variant, index) {
    const on       = !!variant;
    const card     = document.createElement("div");
    card.className = "exp-loss-card" + (on ? " is-on" : "");

    const head     = document.createElement("div");
    head.className = "exp-loss-card__head";

    const toggle     = document.createElement("button");
    toggle.type      = "button";
    toggle.className = "switch exp-loss-card__switch" + (on ? " is-on" : "");
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-checked", String(on));
    toggle.title     = on ? "Drop this loss from the sweep" : "Add this loss as a trial";
    toggle.innerHTML = `<span class="switch__knob"></span>`;
    toggle.addEventListener("click", () => this._toggleWarmupLoss(term, index));

    const name       = document.createElement("span");
    name.className   = "exp-loss-card__name";
    name.textContent = term.key;

    head.appendChild(toggle);
    head.appendChild(name);
    card.appendChild(head);

    if (!on) return card;

    const body     = document.createElement("div");
    body.className = "exp-loss-card__body";

    const labelRow     = document.createElement("label");
    labelRow.className = "exp-loss-card__labelrow";
    labelRow.innerHTML = `<span>label</span>`;
    const label        = document.createElement("input");
    label.className    = "cfg-edit__input";
    label.value        = variant.label;
    label.spellcheck   = false;
    label.title        = "variant label, used in the run name";
    label.addEventListener("change", () => {
      variant.label = label.value.trim() || variant.label;
      this._emit("warmup");
    });
    labelRow.appendChild(label);
    body.appendChild(labelRow);

    const baseRow     = document.createElement("div");
    baseRow.className = "exp-loss-card__base";
    baseRow.innerHTML = `<span class="exp-loss-card__weightlabel">weight</span>`;
    const baseWeight       = document.createElement("input");
    baseWeight.className    = "cfg-edit__input";
    baseWeight.type        = "number";
    baseWeight.step        = "any";
    baseWeight.value       = String(variant.terms[0].weight);
    baseWeight.spellcheck  = false;
    baseWeight.title       = "loss weight";
    baseWeight.addEventListener("change", () => {
      variant.terms[0].weight = Number(baseWeight.value) || 0;
      this._emit("warmup");
    });
    baseRow.appendChild(baseWeight);
    body.appendChild(baseRow);

    if (variant.terms.length > 1) {
      const extras     = document.createElement("div");
      extras.className = "exp-card__terms";
      variant.terms.forEach((extra, ti) => {
        if (ti === 0) return;
        extras.appendChild(this._termRow("warmup", variant, extra, ti));
      });
      body.appendChild(extras);
    }

    const extrasKeys = Object.keys(variant.extras);
    if (extrasKeys.length) {
      const kept       = document.createElement("div");
      kept.className   = "exp-card__extras";
      kept.textContent = `raw keys kept: ${extrasKeys.join(", ")}`;
      body.appendChild(kept);
    }

    const addTerm = LaunchWidgetDom.mini("Add term", () => {
      const unused = this.terms.find((candidate) => !variant.terms.some((vt) => vt.useKey === candidate.useKey));
      const pick   = unused || this.terms[0];
      variant.terms.push({ useKey: pick.useKey, weightKey: pick.weightKey, weight: pick.defaultWeight });
      this._emit("warmup");
    });
    addTerm.classList.add("exp-card__add");
    body.appendChild(addTerm);

    card.appendChild(body);
    return card;
  }

  _toggleWarmupLoss(term, index) {
    if (index >= 0) this.variants.warmup.splice(index, 1);
    else this.variants.warmup.push({ label: this._uniqueLabel("warmup", term.key), terms: [{ useKey: term.useKey, weightKey: term.weightKey, weight: term.defaultWeight }], extras: {} });

    this._emit("warmup");
  }

  _paintSummary() {
    const nWarm = this.variants.warmup.length;
    const nComp = this.variants.complete.length;
    const mode  = this._mode();

    let gpus = "";
    if (this.gpusLeaf) {
      try {
        const parsed = PythonLiteral.parse(this.view._effective(this.gpusLeaf));
        if (Array.isArray(parsed)) gpus = ` on ${parsed.length} GPU${parsed.length === 1 ? "" : "s"}`;
      } catch (e) {
        gpus = "";
      }
    }

    if (mode === "warmup") {
      this.summaryEl.textContent = `${nWarm} warmup loss${nWarm === 1 ? "" : "es"} = ${nWarm} trial${nWarm === 1 ? "" : "s"}${gpus}`;
      return;
    }

    if (mode === "secondary") {
      const strategy = this._strategy();
      const sampled  = strategy === "uniform" || strategy === "gaussian";
      const count    = sampled ? `${this._secondaryEffective("n_trials")} trials` : "trial count set by the stack";
      this.summaryEl.textContent = `${strategy}, ${this._secondaryEffective("n_secondaries")} secondaries, ${count}${gpus}`;
      return;
    }

    if (mode === "patch") {
      const n = this._patchSizes().length;
      this.summaryEl.textContent = `${n} patch size${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}${gpus}`;
      return;
    }

    if (mode === "presence") {
      const cells = this._presenceCells().length;
      const strat = this._presenceStrategies().length;
      const total = cells * strat;
      this.summaryEl.textContent = `${cells} cell${cells === 1 ? "" : "s"} x ${strat} matching = ${total} run${total === 1 ? "" : "s"}${gpus}`;
      return;
    }

    if (mode === "input") {
      const n = this._inputCells().length;
      this.summaryEl.textContent = `${n} input variant${n === 1 ? "" : "s"} = ${n} trial${n === 1 ? "" : "s"}, all tracks${gpus}`;
      return;
    }

    this.summaryEl.textContent = `${nWarm} warmup x ${nComp} complete = ${nWarm * nComp} trials${gpus}`;
  }

  _paintNames() {
    this.namesEl.innerHTML = "";
    const model = this.modelLeaf ? this.view._effective(this.modelLeaf) : "model";
    const mode  = this._mode();

    if (mode === "secondary") {
      const pattern       = document.createElement("span");
      pattern.className   = "exp-name";
      pattern.textContent = `${model}_sec-${this._strategy()}-tNN_<labels>`;
      this.namesEl.appendChild(pattern);

      const more       = document.createElement("span");
      more.className   = "exp-name exp-name--more";
      more.textContent = "one per selection, labels resolved from the dataset stack";
      this.namesEl.appendChild(more);
      return;
    }

    const names = [];
    if (mode === "patch") {
      this._patchSizes().forEach((size) => names.push(`${model}_p-${size}`));
    } else if (mode === "presence") {
      this._presenceCells().forEach((cell) => {
        this._presenceStrategies().forEach((strategy) => names.push(`${model}_pr-${cell}-${strategy}`));
      });
    } else if (mode === "input") {
      this._inputCells().forEach((cell) => names.push(`${model}_in-${cell}`));
    } else if (mode === "warmup") {
      this.variants.warmup.forEach((w) => names.push(`${model}_nc-${w.label}`));
    } else {
      this.variants.warmup.forEach((w) => {
        this.variants.complete.forEach((c) => names.push(`${model}_w-${w.label}_c-${c.label}`));
      });
    }

    const limit = 12;
    names.slice(0, limit).forEach((name) => {
      const chip       = document.createElement("span");
      chip.className   = "exp-name";
      chip.textContent = name;
      this.namesEl.appendChild(chip);
    });

    if (names.length > limit) {
      const more       = document.createElement("span");
      more.className   = "exp-name exp-name--more";
      more.textContent = `+${names.length - limit} more`;
      this.namesEl.appendChild(more);
    }
  }
}


class AblationBuilder {
  constructor(view, byPath) {
    this.view   = view;
    this.byPath = byPath;

    this.trialsLeaf   = byPath.get("trials_enabled");
    this.modeLeaf     = byPath.get("trials_mode");
    this.modelLeaf    = byPath.get("backbone_name");
    this.gpusLeaf     = byPath.get("gpus");
    this.featuresLeaf = byPath.get("ablation_features");
    this.catalogLeaf  = byPath.get("ablation_catalog");
    this.fullLeaf     = byPath.get("ablation_include_full");

    this.claimed = [];
    if (this.trialsLeaf)   this.claimed.push("trials_enabled");
    if (this.modeLeaf)     this.claimed.push("trials_mode");
    if (this.featuresLeaf) this.claimed.push(this.featuresLeaf.path);
    if (this.catalogLeaf)  this.claimed.push(this.catalogLeaf.path);
    if (this.fullLeaf)     this.claimed.push(this.fullLeaf.path);

    this.root         = null;
    this.summaryEl    = null;
    this.namesEl      = null;
    this.menuEl       = null;
    this.listEl       = null;
    this.fullEl       = null;
    this.datalistId   = "ablation-cfg-paths";
    this._paintSwitch = null;
  }

  get available() {
    return !!(this.featuresLeaf && this.catalogLeaf && this.trialsLeaf && this.modeLeaf);
  }

  build() {
    this.root           = document.createElement("section");
    this.root.className = "exp-builder exp-ablation-card";

    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = `<h3 class="special-head__name">Ablation study</h3>`;

    const hint       = document.createElement("span");
    hint.className   = "special-head__hint";
    hint.textContent = "full model degrades one feature at a time, in order, down to the baseline";
    head.appendChild(hint);

    const summary     = document.createElement("span");
    summary.className = "exp-builder__summary";
    this.summaryEl    = summary;
    head.appendChild(summary);

    head.appendChild(this._switch());

    const body     = document.createElement("div");
    body.className = "exp-builder__body";
    body.appendChild(this._panel());

    const preview     = document.createElement("div");
    preview.className = "exp-builder__preview";
    preview.innerHTML = `<span class="exp-builder__preview-title">trial run names</span>`;
    const names       = document.createElement("div");
    names.className   = "exp-builder__names";
    this.namesEl      = names;
    preview.appendChild(names);
    body.appendChild(preview);

    this.root.appendChild(head);
    this.root.appendChild(body);
    this.root.appendChild(this._pathDatalist());

    if (this.featuresLeaf) this.view.controls[this.featuresLeaf.path] = { leaf: this.featuresLeaf, reset: () => this._repaint() };
    if (this.fullLeaf)     this.view.controls[this.fullLeaf.path]     = { leaf: this.fullLeaf,     reset: () => this._repaint() };

    this._repaint();
    if (this._paintSwitch) this._paintSwitch();
    return this.root;
  }

  refreshFromView() {
    if (!this.summaryEl) return;
    if (this._paintSwitch) this._paintSwitch();
    this._paint();
    this._paintSummary();
    this._paintNames();
  }

  _active() {
    return this.view._effective(this.trialsLeaf) === "True" && this.view._effective(this.modeLeaf) === "ablation";
  }

  _switch() {
    const toggle     = document.createElement("button");
    toggle.type      = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.title     = "run an ablation study";
    toggle.innerHTML = `<span class="switch__knob"></span>`;

    const paint = () => {
      const on = this._active();
      toggle.classList.toggle("is-on", on);
      toggle.setAttribute("aria-checked", String(on));
      this.root.classList.toggle("is-trials", on);
    };

    toggle.addEventListener("click", () => {
      if (this._active()) {
        this.view._setValue(this.trialsLeaf, "False");
      } else {
        this.view._setValue(this.modeLeaf, "ablation");
        this.view._setValue(this.trialsLeaf, "True");
      }
      paint();
    });

    this._paintSwitch = paint;
    return toggle;
  }

  _catalog() {
    if (!this.catalogLeaf) return {};
    try {
      const raw = PythonLiteral.parse(this.view._effective(this.catalogLeaf));
      return raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
    } catch (e) {
      return {};
    }
  }

  _features() {
    if (!this.featuresLeaf) return [];
    try {
      const raw = PythonLiteral.parse(this.view._effective(this.featuresLeaf));
      return Array.isArray(raw) ? raw : [];
    } catch (e) {
      return [];
    }
  }

  _full() {
    if (!this.fullLeaf) return true;
    return this.view._effective(this.fullLeaf) === "True";
  }

  _specText(feature) {
    const degrade = feature && feature.degrade ? feature.degrade : {};
    const parts   = Object.entries(degrade).map(([path, value]) => `${path.split(".").pop()}=${PythonLiteral.render(value)}`);
    return parts.length ? parts.join(", ") : "no change";
  }

  _featureKeys(feature) {
    const keys = [];
    const seen = new Set();
    [feature.enable || {}, feature.degrade || {}].forEach((dict) => {
      Object.keys(dict).forEach((path) => {
        if (seen.has(path)) return;
        seen.add(path);
        keys.push(path);
      });
    });
    return keys;
  }

  _pathDatalist() {
    const list = document.createElement("datalist");
    list.id    = this.datalistId;
    [...this.byPath.keys()].forEach((path) => {
      const opt   = document.createElement("option");
      opt.value   = path;
      list.appendChild(opt);
    });
    return list;
  }

  _panel() {
    const panel     = document.createElement("div");
    panel.className = "exp-secondary exp-presence exp-ablation";

    const head     = document.createElement("div");
    head.className = "exp-col__head";
    head.innerHTML = `<span class="exp-col__name">ablation order</span>`;

    const full = LaunchWidgetDom.mini("full-model run", () => this._toggleFull());
    full.classList.add("exp-ablation__full");
    this.fullEl = full;
    head.appendChild(full);

    const reset = LaunchWidgetDom.mini("reset features", () => this._reset());
    reset.classList.add("exp-presence__reset");
    head.appendChild(reset);

    const note       = document.createElement("p");
    note.className   = "exp-secondary__note";
    note.textContent = "Run 0 is the full model with every selected feature enabled (the before values); each later run degrades one more feature in the order below to its after value, ending at the baseline with all of them degraded. Edit the before and after of any config path, add or remove paths, or rename the feature. Pick from the catalog, or add a custom feature by config path.";

    const menuHead      = document.createElement("div");
    menuHead.className   = "exp-presence__sub";
    menuHead.textContent = "feature catalog";

    const menu       = document.createElement("div");
    menu.className   = "exp-builder__names exp-ablation__menu";
    this.menuEl      = menu;

    const listHead      = document.createElement("div");
    listHead.className   = "exp-presence__sub";
    listHead.textContent = "degradation order (full to baseline)";

    const list       = document.createElement("div");
    list.className   = "exp-ablation__list";
    this.listEl      = list;

    panel.appendChild(head);
    panel.appendChild(note);
    panel.appendChild(menuHead);
    panel.appendChild(menu);
    panel.appendChild(listHead);
    panel.appendChild(list);
    panel.appendChild(this._customForm());
    return panel;
  }

  _customForm() {
    const form     = document.createElement("div");
    form.className = "exp-ablation__custom";

    const label = document.createElement("input");
    label.type        = "text";
    label.placeholder = "label";
    label.className   = "exp-ablation__field exp-ablation__field--label";

    const group = document.createElement("input");
    group.type        = "text";
    group.placeholder = "group";
    group.className   = "exp-ablation__field exp-ablation__field--group";

    const path = document.createElement("input");
    path.type        = "text";
    path.placeholder = "config.path";
    path.className   = "exp-ablation__field exp-ablation__field--path";
    path.setAttribute("list", this.datalistId);

    const before = document.createElement("input");
    before.type        = "text";
    before.placeholder = "before";
    before.className   = "exp-ablation__field exp-ablation__field--value";

    const after = document.createElement("input");
    after.type        = "text";
    after.placeholder = "after";
    after.className   = "exp-ablation__field exp-ablation__field--value";

    const add = LaunchWidgetDom.mini("add custom feature", () => {
      if (this._addCustom(label.value, group.value, path.value, before.value, after.value)) {
        label.value = "";
        group.value = "";
        path.value  = "";
        before.value = "";
        after.value  = "";
      }
    });

    form.appendChild(label);
    form.appendChild(group);
    form.appendChild(path);
    form.appendChild(before);
    form.appendChild(after);
    form.appendChild(add);
    return form;
  }

  _coerceLiteral(text) {
    const trimmed = String(text).trim();
    if (trimmed === "") return "";
    if (trimmed.toLowerCase() === "true")  return true;
    if (trimmed.toLowerCase() === "false") return false;
    if (trimmed.toLowerCase() === "none")  return null;
    if (!isNaN(Number(trimmed))) return Number(trimmed);
    return trimmed;
  }

  _addFeature(key) {
    const feature = this._catalog()[key];
    if (!feature) return;

    const features = this._features();
    if (features.some((entry) => entry.label === key)) return;

    features.push(feature);
    this._setFeatures(features);
  }

  _addCustom(label, group, path, before, after) {
    const cleanLabel = String(label).trim();
    const cleanGroup = String(group).trim() || "custom";
    const cleanPath  = String(path).trim();
    if (!cleanLabel || !cleanPath) {
      window.toast("custom feature needs a label and a config path", "error");
      return false;
    }

    const features = this._features();
    if (features.some((entry) => entry.label === cleanLabel)) {
      window.toast(`feature '${cleanLabel}' is already in the order`, "error");
      return false;
    }

    const enable  = {};
    const degrade = {};
    if (String(before).trim() !== "") enable[cleanPath]  = this._coerceLiteral(before);
    degrade[cleanPath] = this._coerceLiteral(after);

    features.push({ label: cleanLabel, group: cleanGroup, enable, degrade });
    this._setFeatures(features);
    return true;
  }

  _renameFeature(index, label) {
    const clean    = String(label).trim();
    if (!clean) return;
    const features = this._features();
    if (!features[index]) return;
    if (features.some((entry, i) => i !== index && entry.label === clean)) {
      window.toast(`feature '${clean}' is already in the order`, "error");
      this._repaint();
      return;
    }
    features[index].label = clean;
    this._setFeatures(features);
  }

  _setPathValue(index, side, path, value) {
    const features = this._features();
    const feature  = features[index];
    if (!feature) return;
    if (!feature[side]) feature[side] = {};
    feature[side][path] = value;
    this._setFeatures(features);
  }

  _addPath(index, path, before, after) {
    const cleanPath = String(path).trim();
    if (!cleanPath) {
      window.toast("a config path is required", "error");
      return false;
    }

    const features = this._features();
    const feature  = features[index];
    if (!feature) return false;
    if (!feature.enable)  feature.enable  = {};
    if (!feature.degrade) feature.degrade = {};
    if (String(before).trim() !== "") feature.enable[cleanPath] = this._coerceLiteral(before);
    feature.degrade[cleanPath] = this._coerceLiteral(after);
    this._setFeatures(features);
    return true;
  }

  _removePath(index, path) {
    const features = this._features();
    const feature  = features[index];
    if (!feature) return;
    if (feature.enable)  delete feature.enable[path];
    if (feature.degrade) delete feature.degrade[path];
    this._setFeatures(features);
  }

  _move(index, delta) {
    const features = this._features();
    const target   = index + delta;
    if (target < 0 || target >= features.length) return;

    const moved = features.splice(index, 1)[0];
    features.splice(target, 0, moved);
    this._setFeatures(features);
  }

  _remove(index) {
    const features = this._features();
    features.splice(index, 1);
    this._setFeatures(features);
  }

  _toggleFull() {
    if (!this.fullLeaf) return;
    const next = this._full() ? "False" : "True";
    this.view._setValue(this.fullLeaf, next);
    this._repaint();
  }

  _setFeatures(features) {
    if (!this.featuresLeaf) return;
    this.view._setValue(this.featuresLeaf, PythonLiteral.render(features));
    this._repaint();
  }

  _reset() {
    if (this.featuresLeaf) this.view._setValue(this.featuresLeaf, this.featuresLeaf.value);
    if (this.fullLeaf)     this.view._setValue(this.fullLeaf, this.fullLeaf.value);
    this._repaint();
  }

  _paint() {
    this._paintMenu();
    this._paintList();
    if (this.fullEl) this.fullEl.classList.toggle("is-on", this._full());
  }

  _paintMenu() {
    if (!this.menuEl) return;
    this.menuEl.innerHTML = "";

    const chosen  = new Set(this._features().map((feature) => feature.label));
    const catalog = this._catalog();

    Object.keys(catalog).forEach((key) => {
      const chip       = document.createElement("button");
      chip.type        = "button";
      chip.className   = "exp-name exp-ablation__option";
      chip.textContent = key;
      chip.title       = this._specText(catalog[key]);
      chip.disabled    = chosen.has(key);
      chip.classList.toggle("is-on", chosen.has(key));
      chip.addEventListener("click", () => this._addFeature(key));
      this.menuEl.appendChild(chip);
    });
  }

  _paintList() {
    if (!this.listEl) return;
    this.listEl.innerHTML = "";

    const features = this._features();

    features.forEach((feature, index) => this.listEl.appendChild(this._row(feature, index, features.length)));

    if (!features.length) {
      const empty       = document.createElement("p");
      empty.className   = "exp-ablation__empty";
      empty.textContent = "no features selected; pick from the catalog above";
      this.listEl.appendChild(empty);
    }
  }

  _row(feature, index, total) {
    const row     = document.createElement("div");
    row.className = "exp-ablation__row";

    const head     = document.createElement("div");
    head.className = "exp-ablation__rowhead";

    const isBaseline = index === total - 1;
    const step       = document.createElement("span");
    step.className   = "exp-ablation__step";
    step.textContent = isBaseline ? `${index + 1} base` : `${index + 1}`;
    head.appendChild(step);

    const label       = document.createElement("input");
    label.type        = "text";
    label.className   = "exp-ablation__labelinput";
    label.value       = feature.label || "";
    label.spellcheck  = false;
    label.title       = "feature label, used in the trial run name";
    label.addEventListener("change", () => this._renameFeature(index, label.value));
    head.appendChild(label);

    const group       = document.createElement("span");
    group.className   = "exp-ablation__group";
    group.textContent = feature.group || "custom";
    head.appendChild(group);

    const up = LaunchWidgetDom.mini("↑", () => this._move(index, -1));
    up.classList.add("exp-ablation__move");
    up.disabled = index === 0;
    head.appendChild(up);

    const down = LaunchWidgetDom.mini("↓", () => this._move(index, 1));
    down.classList.add("exp-ablation__move");
    down.disabled = index === total - 1;
    head.appendChild(down);

    const remove = LaunchWidgetDom.mini("×", () => this._remove(index));
    remove.classList.add("exp-presence__remove");
    remove.title = "remove feature";
    head.appendChild(remove);

    row.appendChild(head);

    const paths     = document.createElement("div");
    paths.className = "exp-ablation__paths";

    const keys = this._featureKeys(feature);
    if (!keys.length) {
      const none       = document.createElement("p");
      none.className   = "exp-ablation__pathnote";
      none.textContent = "no config paths yet — add one below";
      paths.appendChild(none);
    }
    keys.forEach((path) => paths.appendChild(this._pathRow(feature, index, path)));
    paths.appendChild(this._addPathForm(index));

    row.appendChild(paths);
    return row;
  }

  _pathRow(feature, index, path) {
    const item     = document.createElement("div");
    item.className = "exp-ablation__path";

    const name       = document.createElement("span");
    name.className   = "exp-ablation__pathname";
    name.textContent = path;
    name.title       = path;
    item.appendChild(name);

    const beforeVal = feature.enable ? feature.enable[path] : undefined;
    const before    = this._valueEditor(path, beforeVal, (v) => this._setPathValue(index, "enable", path, v));
    before.classList.add("exp-ablation__val", "exp-ablation__val--before");
    before.title = `before (full model) · --${path}`;
    item.appendChild(before);

    const arrow       = document.createElement("span");
    arrow.className   = "exp-ablation__arrow";
    arrow.textContent = "→";
    item.appendChild(arrow);

    const afterVal = feature.degrade ? feature.degrade[path] : undefined;
    const after    = this._valueEditor(path, afterVal, (v) => this._setPathValue(index, "degrade", path, v));
    after.classList.add("exp-ablation__val", "exp-ablation__val--after");
    after.title = `after (degraded) · --${path}`;
    item.appendChild(after);

    const rm = LaunchWidgetDom.mini("×", () => this._removePath(index, path));
    rm.classList.add("exp-ablation__pathremove");
    rm.title = "remove this config path";
    item.appendChild(rm);

    return item;
  }

  _addPathForm(index) {
    const form     = document.createElement("div");
    form.className = "exp-ablation__addpath";

    const path = document.createElement("input");
    path.type        = "text";
    path.placeholder = "config.path";
    path.className   = "exp-ablation__field exp-ablation__field--path";
    path.setAttribute("list", this.datalistId);

    let before  = this._addValueField(false, "before", "");
    let after   = this._addValueField(false, "after", "");
    let isModel = false;

    const swap = (next) => {
      const freshBefore = this._addValueField(next, "before", before.value);
      const freshAfter  = this._addValueField(next, "after", after.value);
      form.replaceChild(freshBefore, before);
      form.replaceChild(freshAfter, after);
      before  = freshBefore;
      after   = freshAfter;
      isModel = next;
    };

    path.addEventListener("input", () => {
      const next = path.value.trim() === "backbone_name" && !!(this.view.modelFamilies && this.view.modelFamilies.length);
      if (next !== isModel) swap(next);
    });

    const add = LaunchWidgetDom.mini("add path", () => {
      if (this._addPath(index, path.value, before.value, after.value)) {
        path.value = "";
        swap(false);
        before.value = "";
        after.value  = "";
      }
    });
    add.classList.add("exp-ablation__addbtn");

    form.appendChild(path);
    form.appendChild(before);
    form.appendChild(after);
    form.appendChild(add);
    return form;
  }

  _addValueField(isModel, placeholder, current) {
    if (isModel) {
      const select = this._backboneSelect(current, () => {});
      select.classList.add("exp-ablation__field", "exp-ablation__field--value");
      return select;
    }

    const input       = document.createElement("input");
    input.type        = "text";
    input.placeholder = placeholder;
    input.className   = "exp-ablation__field exp-ablation__field--value";
    input.value       = current === undefined || current === null ? "" : String(current);
    return input;
  }

  _valueEditor(path, value, onCommit) {
    const leaf = this.byPath.get(path);
    const type = leaf ? leaf.type : null;

    if (path === "backbone_name" && this.view.modelFamilies && this.view.modelFamilies.length) {
      return this._backboneSelect(value, onCommit);
    }

    if (type === "bool") {
      const btn       = document.createElement("button");
      btn.type        = "button";
      btn.className   = "switch exp-ablation__switch";
      btn.setAttribute("role", "switch");
      btn.innerHTML   = `<span class="switch__knob"></span>`;
      const paint = (state) => {
        btn.classList.toggle("is-on", state);
        btn.setAttribute("aria-checked", String(state));
      };
      paint(value === true);
      btn.addEventListener("click", () => {
        const next = !btn.classList.contains("is-on");
        paint(next);
        onCommit(next);
      });
      return btn;
    }

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input exp-ablation__input";
    input.spellcheck = false;

    if (type === "int" || type === "float") {
      input.type  = "number";
      input.step  = type === "int" ? "1" : "any";
      input.value = value === undefined || value === null ? "" : String(value);
      input.addEventListener("change", () => {
        const raw = input.value.trim();
        onCommit(raw === "" ? null : Number(raw));
      });
    } else {
      const structured = value !== null && typeof value === "object";
      input.type        = "text";
      input.value       = value === undefined || value === null ? "" : (structured ? PythonLiteral.render(value) : String(value));
      input.placeholder = value === undefined ? "unset" : "";
      input.addEventListener("change", () => onCommit(structured ? PythonLiteral.parse(input.value) : input.value));
    }
    return input;
  }

  _backboneSelect(value, onCommit) {
    const select     = document.createElement("select");
    select.className = "run-select exp-ablation__input exp-ablation__select";

    const current = value === undefined || value === null ? "" : String(value);
    const known   = new Set();

    (this.view.modelFamilies || []).forEach((family) => {
      const group   = document.createElement("optgroup");
      group.label   = family.family;
      family.models.forEach((model) => {
        const opt       = document.createElement("option");
        opt.value       = model.key;
        opt.textContent = model.recommended ? `${model.name} (recommended)` : model.name || model.key;
        group.appendChild(opt);
        known.add(model.key);
      });
      select.appendChild(group);
    });

    if (current && !known.has(current)) {
      const opt       = document.createElement("option");
      opt.value       = current;
      opt.textContent = current;
      select.insertBefore(opt, select.firstChild);
    }

    select.value = current;
    select.addEventListener("change", () => onCommit(select.value));
    return select;
  }

  _gpusSuffix() {
    if (!this.gpusLeaf) return "";
    try {
      const parsed = PythonLiteral.parse(this.view._effective(this.gpusLeaf));
      if (Array.isArray(parsed)) return ` on ${parsed.length} GPU${parsed.length === 1 ? "" : "s"}`;
    } catch (e) {
      return "";
    }
    return "";
  }

  _paintSummary() {
    if (!this.summaryEl) return;
    const n    = this._features().length;
    const runs = n + (this._full() ? 1 : 0);
    this.summaryEl.textContent = `${n} feature${n === 1 ? "" : "s"} = ${runs} run${runs === 1 ? "" : "s"} (full to baseline)${this._gpusSuffix()}`;
  }

  _paintNames() {
    if (!this.namesEl) return;
    this.namesEl.innerHTML = "";

    const model    = this.modelLeaf ? this.view._effective(this.modelLeaf) : "model";
    const features = this._features();
    const total    = features.length;
    const names    = [];

    if (this._full()) names.push(`${model}_abl-0-full`);
    features.forEach((feature, index) => {
      const step = index + 1;
      if (step === total) names.push(`${model}_abl-${step}-baseline`);
      else names.push(`${model}_abl-${step}-no_${feature.label}`);
    });

    const limit = 12;
    names.slice(0, limit).forEach((name) => {
      const chip       = document.createElement("span");
      chip.className   = "exp-name";
      chip.textContent = name;
      this.namesEl.appendChild(chip);
    });

    if (names.length > limit) {
      const more       = document.createElement("span");
      more.className   = "exp-name exp-name--more";
      more.textContent = `+${names.length - limit} more`;
      this.namesEl.appendChild(more);
    }
  }

  _repaint() {
    this._paint();
    this._paintSummary();
    this._paintNames();
  }
}


class NumberField {

  static SPECS = [
    { re: /(^|_)(lr|learning_rate)$/,                       log: true, min: 1e-6, max: 1e-1, presets: [1e-5, 5e-5, 1e-4, 3e-4, 1e-3, 1e-2] },
    { re: /weight_decay$|_wd$/,                             log: true, min: 1e-8, max: 1e-1, presets: [0, 1e-6, 1e-5, 1e-4, 1e-3] },
    { re: /eta_min$/,                                       log: true, min: 1e-9, max: 1e-2, presets: [0, 1e-7, 1e-6, 1e-5] },
    { re: /(^|_)epochs?$/,                                  int: true, min: 1,    max: 1000,  presets: [10, 50, 100, 200, 500] },
    { re: /warmup.*epoch|warmup_steps|^warmup$/,            int: true, min: 0,    max: 200,   presets: [0, 5, 10, 20, 50] },
    { re: /batch(_size)?$/,                                 int: true, min: 1,    max: 1024,  presets: [8, 16, 32, 64, 128, 256] },
    { re: /(num_)?workers$|n_workers$/,                     int: true, min: 0,    max: 64,    presets: [0, 2, 4, 8, 16, 32] },
    { re: /accumulation|grad_accum/,                        int: true, min: 1,    max: 64,    presets: [1, 2, 4, 8, 16] },
    { re: /patience$/,                                      int: true, min: 1,    max: 200,   presets: [5, 10, 20, 50, 100] },
    { re: /(patch|window|win|tile)(_size)?$/,               int: true, min: 8,    max: 512,   presets: [16, 32, 64, 96, 128, 256] },
    { re: /stride$/,                                        int: true, min: 1,    max: 512,   presets: [8, 16, 32, 64, 128] },
    { re: /seed$/,                                          int: true, min: 0,    max: 9999,  presets: [0, 1, 42, 123, 2024] },
    { re: /dropout|drop_rate|drop_path/,                    min: 0,    max: 1,    step: 0.01, presets: [0, 0.1, 0.2, 0.3, 0.5] },
    { re: /momentum$/,                                      min: 0,    max: 1,    step: 0.01, presets: [0, 0.9, 0.95, 0.99] },
    { re: /ema_decay|momentum_decay/,                       min: 0.9,  max: 1,    step: 0.001, presets: [0.99, 0.995, 0.999, 0.9999] },
    { re: /(ratio|fraction|frac|subsample|keep|prob|pct|percent)$/, min: 0, max: 1, step: 0.01, presets: [0.1, 0.25, 0.5, 0.75, 1.0] },
    { re: /^weight_|_weight$|^lambda|_lambda$|coeff|_scale$|gamma$|beta$/, min: 0, max: 10, step: 0.1, presets: [0, 0.1, 0.5, 1, 2, 5] },
    { re: /sigma$|_std$/,                                   min: 0,    max: 5,    step: 0.05, presets: [0.5, 1, 1.5, 2, 3] },
    { re: /dpi$/,                                           int: true, min: 72,   max: 600,   presets: [100, 150, 200, 300, 600] },
  ];

  constructor(view, leaf, short) {
    this.view    = view;
    this.leaf    = leaf;
    this.short   = short || leaf.path.split(".").pop();
    this.integer = leaf.type === "int";
    this.default = Number.isFinite(Number(leaf.value)) ? Number(leaf.value) : 0;
    this.log     = false;
    this.range   = this._infer();
    this.logMin  = 0;
    this.logMax  = 0;
    this.input   = null;
    this.slider  = null;
    this.chips   = new Map();
    this.reset   = () => this._paint();
  }

  build() {
    const el     = document.createElement("div");
    el.className = "numfield";

    const top     = document.createElement("div");
    top.className = "numfield__top";

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input numfield__input";
    input.type       = "number";
    input.step       = this.integer ? "1" : "any";
    input.spellcheck = false;
    this.input       = input;

    const slider     = document.createElement("input");
    slider.className = "numfield__slider";
    slider.type      = "range";
    this.slider      = slider;
    this._configureSlider();

    input.addEventListener("input", () => {
      const raw = input.value;
      const v   = Number(raw);
      if (raw !== "" && Number.isFinite(v)) slider.value = String(this._toSlider(v));
      const out = raw === "" ? "" : (v === this.default ? this.leaf.value : raw);
      this.view._setValue(this.leaf, out);
      this._mark();
    });
    top.appendChild(input);
    el.appendChild(top);

    const presets     = document.createElement("div");
    presets.className = "numfield__presets";
    this.range.presets.forEach((value) => presets.appendChild(this._chip(value)));
    top.appendChild(presets);

    this._paint();
    return { el, input, reset: this.reset };
  }

  _infer() {
    const spec = NumberField.SPECS.find((s) => s.re.test(this.short) || s.re.test(this.leaf.path));
    const r    = spec
      ? { min: spec.min, max: spec.max, step: spec.step || 1, log: Boolean(spec.log), presets: spec.presets.slice() }
      : this._fallback();

    this.log = r.log;
    r.min = Math.min(r.min, this.default);
    r.max = Math.max(r.max, this.default);
    r.presets.push(this.default);
    r.presets = this._cleanPresets(r.presets, r);
    return r;
  }

  _fallback() {
    if (!this.integer && this.default > 0 && this.default <= 1) {
      return { min: 0, max: 1, step: 0.01, log: false, presets: [0, 0.25, 0.5, 0.75, 1] };
    }
    const base    = Math.abs(this.default) || (this.integer ? 10 : 1);
    const max     = this._nice(base * 4);
    const min     = this.default < 0 ? -max : 0;
    const span    = max - min || 1;
    const step    = this.integer ? 1 : Math.pow(10, Math.floor(Math.log10(span)) - 2) || 0.01;
    const presets = [min, min + span * 0.25, min + span * 0.5, min + span * 0.75, max].map((x) => (this.integer ? Math.round(x) : this._nice(x)));
    return { min, max, step, log: false, presets };
  }

  _nice(x) {
    if (x === 0) return 0;
    const unit = Math.pow(10, Math.floor(Math.log10(Math.abs(x)))) / 10;
    return Math.round(x / unit) * unit;
  }

  _cleanPresets(list, r) {
    const within = list.filter((x) => Number.isFinite(x) && x >= r.min - 1e-9 && x <= r.max + 1e-9);
    const seen   = new Map();
    within.forEach((x) => {
      const key = this.integer ? String(Math.round(x)) : this._fmt(x);
      if (!seen.has(key)) seen.set(key, Number(key));
    });
    return [...seen.values()].sort((a, b) => a - b).slice(0, 8);
  }

  _configureSlider() {
    if (this.range.log) {
      this.logMin = this.range.min > 0 ? this.range.min : this.range.max / 1e6;
      this.logMax = this.range.max;
      this.slider.min  = "0";
      this.slider.max  = "1000";
      this.slider.step = "1";
      return;
    }
    this.slider.min  = String(this.range.min);
    this.slider.max  = String(this.range.max);
    this.slider.step = String(this.range.step);
  }

  _toSlider(v) {
    if (this.range.log) {
      if (v <= 0) return 0;
      const lo = Math.log(this.logMin);
      const hi = Math.log(this.logMax);
      const t  = 1000 * (Math.log(this._clamp(v, this.logMin, this.logMax)) - lo) / (hi - lo);
      return Math.round(this._clamp(t, 0, 1000));
    }
    return this._clamp(v, this.range.min, this.range.max);
  }

  _fromSlider() {
    if (this.range.log) {
      const lo = Math.log(this.logMin);
      const hi = Math.log(this.logMax);
      return Math.exp(lo + (Number(this.slider.value) / 1000) * (hi - lo));
    }
    return Number(this.slider.value);
  }

  _fmt(v) {
    if (this.integer) return String(Math.round(v));
    if (v === 0) return "0";
    return String(Number(v.toPrecision(this.log ? 2 : 6)));
  }

  _clamp(x, a, b) {
    return Math.min(b, Math.max(a, x));
  }

  _chip(value) {
    const chip       = document.createElement("button");
    chip.type        = "button";
    chip.className   = "numfield__chip";
    chip.textContent = this._fmt(value);
    chip.title       = `set ${this.short} = ${this._fmt(value)}`;
    chip.addEventListener("click", () => {
      this.input.value  = this._fmt(value);
      this.slider.value = String(this._toSlider(value));
      const out = value === this.default ? this.leaf.value : this._fmt(value);
      this.view._setValue(this.leaf, out);
      this._mark();
    });
    this.chips.set(value, chip);
    return chip;
  }

  _mark() {
    const cur   = Number(this.view._effective(this.leaf));
    const dirty = this.view.dirty[this.leaf.path] !== undefined;
    this.input.classList.toggle("is-dirty", dirty);
    this.chips.forEach((chip, key) => {
      const tol = this.integer ? 0.5 : Math.max(1e-12, Math.abs(cur) * 1e-6);
      chip.classList.toggle("is-active", Number.isFinite(cur) && Math.abs(Number(key) - cur) < tol);
    });
  }

  _paint() {
    const eff = this.view._effective(this.leaf);
    const v   = Number(eff);
    this.input.value = eff === "None" ? "" : eff;
    if (Number.isFinite(v)) this.slider.value = String(this._toSlider(v));
    this._mark();
  }
}


class GpuCardSelect {
  constructor(host, opts = {}) {
    this.host     = host;
    this.multi    = Boolean(opts.multi);
    this.onChange = typeof opts.onChange === "function" ? opts.onChange : () => {};
    this.selected = new Set((opts.initial || []).map(Number).filter(Number.isFinite));
    this.gpus     = [];
    this.chips    = new Map();
    this.manual   = null;
    this.loaded   = false;
  }

  async load() {
    let payload = null;
    try {
      payload = await window.apiGet("/api/system");
    } catch (e) {
      payload = null;
    }
    this.gpus   = payload && Array.isArray(payload.gpus) ? payload.gpus : [];
    this.loaded = true;
    this._render();
    return this;
  }

  value() {
    return [...this.selected].sort((a, b) => a - b);
  }

  set(indices) {
    this.selected = new Set((indices || []).map(Number).filter(Number.isFinite));
    if (this.loaded) this._render();
  }

  _indices() {
    const detected = this.gpus.map((gpu) => Number(gpu.index)).filter(Number.isFinite);
    const merged   = new Set([...detected, ...this.selected]);
    return [...merged].sort((a, b) => a - b);
  }

  _render() {
    this.host.innerHTML = "";
    this.chips.clear();
    this.host.classList.add("gpu-picker__board");

    const indices = this._indices();
    if (!indices.length) {
      this._renderManual();
      return;
    }

    this.host.classList.add("gpu-picker__board--chips");
    indices.forEach((index) => this.host.appendChild(this._chip(index)));
    this._paint();
  }

  _chip(index) {
    const info = this.gpus.find((gpu) => Number(gpu.index) === index) || null;

    const chip        = document.createElement("button");
    chip.type         = "button";
    chip.className    = "gpu-chip";
    chip.title        = `cuda:${index}`;
    chip.dataset.tone = info ? (info.others ? "busy" : info.mine ? "mine" : "free") : "offline";

    const name = info ? info.name : "not detected";
    const meta = [];
    if (info && info.util != null)      meta.push(`${info.util}%`);
    if (info && info.mem_total != null) meta.push(`${Math.round(info.mem_used || 0)}/${Math.round(info.mem_total)} MiB`);

    chip.innerHTML =
      `<span class="gpu-chip__id">GPU ${index}</span>` +
      `<span class="gpu-chip__name">${name}</span>` +
      `<span class="gpu-chip__meta">${meta.join("  ·  ")}</span>`;

    chip.addEventListener("click", () => this._toggle(index));
    this.chips.set(index, chip);
    return chip;
  }

  _toggle(index) {
    if (!this.multi) {
      this.selected = new Set([index]);
    } else if (this.selected.has(index)) {
      if (this.selected.size <= 1) return;
      this.selected.delete(index);
    } else {
      this.selected.add(index);
    }
    this._paint();
    this.onChange(this.value());
  }

  _paint() {
    this.chips.forEach((chip, index) => {
      const on = this.selected.has(index);
      chip.classList.toggle("is-on", on);
      chip.setAttribute("aria-pressed", String(on));
    });
  }

  _renderManual() {
    this.host.classList.remove("gpu-picker__board--chips");

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.value      = this.value().join(", ");
    input.spellcheck = false;
    input.title      = this.multi
      ? "no GPUs detected on this host — enter device indices, e.g. 0, 1"
      : "no GPUs detected on this host — enter a device index, e.g. 0";
    input.addEventListener("input", () => {
      const parts = input.value.split(",").map((token) => Number(token.trim())).filter(Number.isFinite);
      this.selected = new Set(this.multi ? parts : parts.slice(0, 1));
      this.onChange(this.value());
    });

    this.manual = input;
    this.host.appendChild(input);
  }
}


class GpuPicker {
  constructor(view, leaf) {
    this.view  = view;
    this.leaf  = leaf;
    this.multi = leaf.type === "list";
    this.el    = null;
    this.note  = null;
    this.cards = null;
    this.reset = () => this._syncFromLeaf();
  }

  build() {
    this.el           = document.createElement("div");
    this.el.className  = "picker gpu-picker";

    const board = document.createElement("div");
    this.el.appendChild(board);

    const note     = document.createElement("p");
    note.className = "picker__note";
    note.textContent = "detecting GPUs...";
    this.note      = note;
    this.el.appendChild(note);

    this.cards = new GpuCardSelect(board, {
      multi    : this.multi,
      initial  : this._parse(this.view._effective(this.leaf)),
      onChange : (indices) => this._emit(indices),
    });
    this.cards.load().then(() => this._paintNote());

    return { el: this.el, input: this.el, reset: this.reset };
  }

  _parse(raw) {
    if (this.multi) {
      try {
        const parsed = PythonLiteral.parse(raw);
        return Array.isArray(parsed) ? parsed.map(Number).filter(Number.isFinite) : [];
      } catch (e) {
        return [];
      }
    }
    const value = Number(raw);
    return Number.isFinite(value) ? [value] : [];
  }

  _emit(indices) {
    const value = this.multi ? PythonLiteral.render(indices) : (indices.length ? String(indices[0]) : "");
    this.view._setValue(this.leaf, value);
    this._paintNote();
  }

  _syncFromLeaf() {
    if (this.cards) this.cards.set(this._parse(this.leaf.value));
    this._paintNote();
  }

  _paintNote() {
    if (!this.note) return;
    const ordered = this.cards ? this.cards.value() : [];
    const head    = this.multi ? `${ordered.length} GPU${ordered.length === 1 ? "" : "s"}` : `GPU ${ordered[0]}`;
    this.note.textContent = ordered.length
      ? `${head} selected · CUDA_VISIBLE_DEVICES=${ordered.join(",")}`
      : (this.multi ? "select at least one GPU" : "select a GPU");
    this.note.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
  }
}



class MultiValueField {
  constructor(view, leaf, spec) {
    this.view  = view;
    this.leaf  = leaf;
    this.spec  = spec;
    this.el    = null;
    this.chips = null;
    this.input = null;
    this.count = null;
    this.reset = () => this._paint();
  }

  build() {
    this.el           = document.createElement("div");
    this.el.className = "picker multivalue";

    const chips     = document.createElement("div");
    chips.className = "multivalue__chips";
    this.chips      = chips;
    this.el.appendChild(chips);

    if (!this.spec.choices) {
      const entry     = document.createElement("input");
      entry.className = "cfg-edit__input multivalue__entry";
      entry.type      = "text";
      entry.spellcheck = false;
      entry.placeholder = this.spec.placeholder || "add value, Enter";
      entry.addEventListener("keydown", (event) => this._onKey(event));
      entry.addEventListener("blur", () => this._commitEntry());
      this.input = entry;
      this.el.appendChild(entry);
    }

    const note     = document.createElement("p");
    note.className = "picker__note";
    this.count     = note;
    this.el.appendChild(note);

    this._paint();
    return { el: this.el, input: this.input || this.el, reset: this.reset };
  }

  _values() {
    try {
      const parsed = PythonLiteral.parse(this.view._effective(this.leaf));
      return Array.isArray(parsed) ? parsed.slice() : [];
    } catch (e) {
      return [];
    }
  }

  _cast(token) {
    if (!this.spec.numeric) return token;
    const value = Number(token);
    if (!Number.isFinite(value)) return null;
    return this.spec.integer ? Math.trunc(value) : value;
  }

  _emit(values) {
    this.view._setValue(this.leaf, PythonLiteral.render(values));
    this._paint();
  }

  _onKey(event) {
    if (event.key !== "Enter" && event.key !== ",") return;
    event.preventDefault();
    this._commitEntry();
  }

  _commitEntry() {
    if (!this.input) return;
    const tokens = this.input.value.split(",").map((part) => part.trim()).filter(Boolean);
    if (!tokens.length) return;

    const values = this._values();
    tokens.forEach((token) => {
      const cast = this._cast(token);
      if (cast !== null && !values.some((existing) => existing === cast)) values.push(cast);
    });

    this.input.value = "";
    this._emit(values);
  }

  _toggleChoice(value) {
    const values = this._values();
    const index  = values.indexOf(value);
    if (index >= 0) values.splice(index, 1);
    else            values.push(value);

    const ordered = this.spec.choices.map((choice) => choice.value).filter((choice) => values.includes(choice));
    this._emit(ordered);
  }

  _removeValue(value) {
    this._emit(this._values().filter((existing) => existing !== value));
  }

  _paint() {
    const values = this._values();
    this.chips.innerHTML = "";

    if (this.spec.choices) {
      this.spec.choices.forEach((choice) => {
        const on        = values.includes(choice.value);
        const chip      = document.createElement("button");
        chip.type       = "button";
        chip.className  = "multivalue__choice" + (on ? " is-on" : "");
        chip.textContent = choice.label;
        chip.title      = `--${this.leaf.path} · ${choice.value}`;
        chip.setAttribute("aria-pressed", String(on));
        chip.addEventListener("click", () => this._toggleChoice(choice.value));
        this.chips.appendChild(chip);
      });
    } else {
      values.forEach((value) => {
        const chip      = document.createElement("span");
        chip.className  = "multivalue__chip";
        const label     = document.createElement("span");
        label.textContent = String(value);
        const remove    = document.createElement("button");
        remove.type     = "button";
        remove.className = "multivalue__x";
        remove.innerHTML = "&times;";
        remove.title    = "remove";
        remove.addEventListener("click", () => this._removeValue(value));
        chip.appendChild(label);
        chip.appendChild(remove);
        this.chips.appendChild(chip);
      });
    }

    if (this.count) {
      this.count.textContent = values.length
        ? `${values.length} value${values.length === 1 ? "" : "s"} · ${PythonLiteral.render(values)}`
        : (this.spec.empty || "select at least one value");
      this.count.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
    }
  }
}


class ConfigForm {
  constructor() {
    this.dirty            = {};
    this.controls         = {};
    this.dependents       = {};
    this.states           = [];
    this.panels           = new Map();
    this.bands            = [];
    this.gates            = [];
    this.gatedSections    = new Set();
    this.classColors      = new Map();
    this.query            = "";
    this.showAll          = false;
    this.overrideSections = { suppressed: new Set(), labels: new Map(), identical: new Set(), sections: new Set(), differ: new Map() };
    this.modelFamilies    = null;
    this.pinsEl           = null;
    this.nomatchEl        = null;
    this.countEl          = null;
    this.showAllBtn       = null;
    this.totalFields      = 0;
  }

  static PALETTE = ["#1d4fd8", "#0f766e", "#b45309", "#7c3aed", "#be185d", "#0e7490", "#4d7c0f", "#b91c1c"];

  static FIELD_TAXONOMY = [
    ["curve space", /curve|spectral|ssim/],
    ["param space", /^param/],
    ["slot presence", /presence|focal|active_normalization|active_weight|inactive_weight/],
    ["regularization", /smooth|_tv$/],
    ["physics", /total_power|moments|coherence_resyn|covariance_match|capon_|^physics_|wavelength|slant_range|look_angle|baseline|kz_values|height_axis/],
    ["schedule", /epoch|validation|scheduler|warmup|eta_min|abort_on_nonfinite/],
    ["early stopping", /^early_stop/],
    ["image autoencoder", /image_autoencoder|image_ae_finetune|image_ae_loss/],
    ["profile autoencoder", /profile_autoencoder|target_provider|ema_decay|ae_finetune|^pixel_subsample$|keep_empty/],
    ["embedding", /embedding/],
    ["probe", /^probe_/],
    ["outputs", /^save_(cubes|plots|animations)$/],
    ["identifiers", /identifier|output_tag$|^dataset_type$/],
    ["image stack", /^use_(primary|secondaries|interferograms|dem)$/],
    ["data", /batch|worker|patch|stride|azimuth|dataset|^use_amp$|accumulation/],
    ["model", /model|gauss/],
    ["source", /fusar|track_selection|polarisation|^base_directory$/],
    ["run", /^run_|^gpu|^seeds?$|^device|^log|dir|path/],
    ["reset", /^reset_/],
    ["beamforming", /beamforming|^filter_|^height_range$|^win_list$/],
    ["stitching", /stitch|cube/],
    ["figures", /cmap|dpi|intensity/],
  ];

  static STACK_PAIRS = [
    ["param space", "regularization"],
    ["schedule", "early stopping"],
    ["data", "run"],
  ];

  static OPTION_GATES = {
    benchmark: { field: "training_type", sections: {
      jepa:       ["jepa"],
      ae_loss:    ["profile_autoencoder"],
      size_match: ["backbone"],
      inference:  ["backbone", "jepa"],
    } },
    cross_validate: { field: "training_type", sections: {
      jepa:        ["jepa"],
      autoencoder: ["profile_autoencoder"],
      inference:   ["backbone", "jepa"],
    } },
    tune: { field: "training_type", sections: {
      jepa:          ["jepa"],
      ae_loss:       ["profile_autoencoder"],
      image_ae_loss: ["image_autoencoder"],
    } },
  };

  static EXPERIMENT_JEPA_CHOICES = {
    "jepa.profile_autoencoder_mode": ["frozen", "finetune"],
    "jepa.target_provider":          ["stopgrad", "ema", "live"],
  };

  static TUNE_CHOICES = {
    ...ConfigForm.EXPERIMENT_JEPA_CHOICES,
    "jepa.image_autoencoder_mode": ["frozen", "finetune"],
  };

  static NORM_PRESETS = ["min_max", "min_max_log1p", "robust_iqr", "robust_iqr_log1p", "fixed_div_pi", "zscore", "zscore_log1p"];

  static NORMALIZATION_DEFAULTS = {
    "normalization.pass_mag"   : "robust_iqr_log1p",
    "normalization.pass_phase" : "zscore",
    "normalization.ifg_mag"    : "robust_iqr_log1p",
    "normalization.ifg_phase"  : "zscore",
    "normalization.out_amp"    : "robust_iqr_log1p",
    "normalization.out_mu"     : "zscore",
    "normalization.out_sigma"  : "robust_iqr_log1p",
    "normalization.dem"        : "robust_iqr_log1p",
  };

  static NORMALIZATION_CHOICES = {
    "normalization.input_strategy":  ["per_slot", ...ConfigForm.NORM_PRESETS],
    "normalization.output_strategy": ["per_slot", ...ConfigForm.NORM_PRESETS],
    "normalization.pass_mag":        ["default",  ...ConfigForm.NORM_PRESETS],
    "normalization.pass_phase":      ["default",  ...ConfigForm.NORM_PRESETS],
    "normalization.ifg_mag":         ["default",  ...ConfigForm.NORM_PRESETS],
    "normalization.ifg_phase":       ["default",  ...ConfigForm.NORM_PRESETS],
    "normalization.out_amp":         ["default",  ...ConfigForm.NORM_PRESETS],
    "normalization.out_mu":          ["default",  ...ConfigForm.NORM_PRESETS],
    "normalization.out_sigma":       ["default",  ...ConfigForm.NORM_PRESETS],
    "normalization.dem":             ["default",  ...ConfigForm.NORM_PRESETS],
  };

  static CHOICES = {
    train_backbone: {
      "curriculum.warmup.param_matching":   ["hungarian", "sorted_gt"],
      "curriculum.complete.param_matching": ["hungarian", "sorted_gt"],
      ...ConfigForm.NORMALIZATION_CHOICES,
    },
    train_jepa: {
      profile_autoencoder_mode: ["frozen", "finetune"],
      image_autoencoder_mode:   ["frozen", "finetune"],
      target_provider:          ["stopgrad", "ema", "live"],
    },
    benchmark:      ConfigForm.EXPERIMENT_JEPA_CHOICES,
    cross_validate: ConfigForm.EXPERIMENT_JEPA_CHOICES,
    tune:           ConfigForm.TUNE_CHOICES,
  };

  static DATASET_PICKERS = {
    "paths.dataset_path":    { mode: "datasets", multi: false, baseFromParent: true, validOnly: true },
    "paths.parameters_path": { mode: "params", datasetFrom: "paths.dataset_path" },
  };

  static EXPERIMENT_PICKERS = {
    ...ConfigForm.DATASET_PICKERS,
    "jepa.profile_autoencoder_run": { mode: "runs", baseFrom: "jepa.profile_autoencoder_logdir", checkpointOnly: true },
  };

  static TUNE_PICKERS = {
    ...ConfigForm.EXPERIMENT_PICKERS,
    "jepa.image_autoencoder_run": { mode: "runs", baseFrom: "jepa.image_autoencoder_logdir", checkpointOnly: true },
  };

  static INFER_PICKERS = {
    run_filter: { mode: "runs", multi: true, baseFrom: "logs_dirs", multiBase: true },
  };

  static PICKERS = {
    extract_params: {
      dataset_filter: { mode: "datasets", multi: true, baseFrom: "dataset_base_path", validOnly: true },
    },
    train_backbone:             ConfigForm.DATASET_PICKERS,
    train_profile_autoencoder:  ConfigForm.DATASET_PICKERS,
    train_image_autoencoder:    ConfigForm.DATASET_PICKERS,
    train_jepa:              {
      ...ConfigForm.DATASET_PICKERS,
      profile_autoencoder_run: { mode: "runs", baseFrom: "profile_autoencoder_logdir", checkpointOnly: true },
      image_autoencoder_run:   { mode: "runs", baseFrom: "image_autoencoder_logdir",   checkpointOnly: true },
    },
    benchmark:         ConfigForm.EXPERIMENT_PICKERS,
    cross_validate:    ConfigForm.EXPERIMENT_PICKERS,
    tune:              ConfigForm.TUNE_PICKERS,
    infer_backbone:            ConfigForm.INFER_PICKERS,
    infer_profile_autoencoder: ConfigForm.INFER_PICKERS,
    infer_image_autoencoder:   ConfigForm.INFER_PICKERS,
    xray_weights: {
      run_filter: { mode: "runs", multi: true, baseFrom: "runs_dir", checkpointOnly: true },
    },
    analyze_preprocessing: {
      run_tags: { mode: "runs", multi: true, baseFrom: "runs_dir" },
    },
    analyze_param_extraction: {
      run_tags: { mode: "param_trials", multi: true, baseFrom: "params_dir" },
    },
    compare_trials: {
      run_tags: { mode: "runs_compare", multi: true, baseFrom: "runs_dir" },
    },
    compare_preprocessing_trials: {
      run_tags: { mode: "runs_compare", multi: true, baseFrom: "runs_dir" },
    },
    compare_param_extraction_trials: {
      run_tags: { mode: "param_trials", multi: true, baseFrom: "params_dir" },
    },
  };

  static MULTI_VALUE = {
    extract_params: {
      fit_k_values:      { numeric: true, integer: true, placeholder: "add K, Enter", empty: "select at least one K" },
      fit_lambda_values: { numeric: true, placeholder: "add lambda, Enter", empty: "select at least one lambda" },
      fit_modes:         { empty: "select at least one fit mode", choices: [
        { value: "sigma",        label: "sigma only" },
        { value: "sigma_amp",    label: "sigma + amplitude" },
        { value: "sigma_amp_mu", label: "sigma + amplitude + mean" },
      ] },
    },
  };

  static GPU_FIELDS = ["gpu", "gpus", "gpu_device_ids"];

  _buildToolbar(cfg) {
    const bar = document.createElement("div");
    bar.className = "cfg-toolbar";

    const search = document.createElement("input");
    search.className = "cfg-search";
    search.type = "search";
    search.placeholder = `Filter ${cfg.leaves.length} fields...`;
    search.spellcheck = false;
    search.addEventListener("input", () => {
      this.query = search.value.trim().toLowerCase();
      this._applyVisibility(true);
    });

    const count = document.createElement("span");
    count.className = "cfg-toolbar__count";
    this.countEl = count;

    const showAll = document.createElement("button");
    showAll.className = "btn btn--mini cfg-showall";
    this.showAllBtn = showAll;
    this.totalFields = cfg.leaves.length - this.overrideSections.identical.size;
    showAll.addEventListener("click", () => {
      this.showAll = !this.showAll;
      this._refreshShowAll();
      this._applyVisibility(true);
    });

    const reset = document.createElement("button");
    reset.className = "btn btn--mini";
    reset.textContent = "Reset all";
    reset.addEventListener("click", () => this._resetAll());

    bar.appendChild(search);
    bar.appendChild(count);
    bar.appendChild(showAll);
    bar.appendChild(reset);
    this._refreshShowAll();
    return bar;
  }

  _refreshShowAll() {
    if (!this.showAllBtn) return;
    this.showAllBtn.classList.toggle("is-on", this.showAll);
    this.showAllBtn.textContent = this.showAll ? "Essentials only" : `Show all ${this.totalFields} fields`;
  }

  _detectOverrideSections(leaves) {
    const byPath   = new Map(leaves.map((l) => [l.path, l.value]));
    const identical = new Set();
    const sections  = new Set();
    const differ    = new Map();

    leaves.forEach((leaf) => {
      if (!/\.warmup(\.|$)/.test(leaf.path)) return;
      const counterpart = leaf.path.replace(/\.warmup(\.|$)/, ".complete$1");
      if (!byPath.has(counterpart)) return;
      sections.add(leaf.section);
      if (byPath.get(counterpart) === leaf.value) identical.add(leaf.path);
      else differ.set(leaf.section, (differ.get(leaf.section) || 0) + 1);
    });

    return { identical, sections, differ };
  }

  _shownCount(leaves) {
    return leaves.filter((l) => !this.overrideSections.identical.has(l.path)).length;
  }

  _buildPins(pinned) {
    const panel = document.createElement("section");
    panel.className = "launch-pins";

    const head = document.createElement("header");
    head.className = "launch-pins__head";
    head.innerHTML = `<h3 class="launch-pins__name">Run essentials</h3><span class="launch-pins__hint">check these before every launch</span>`;

    const grid = document.createElement("div");
    grid.className = "launch-pins__grid";
    pinned.forEach((leaf) => grid.appendChild(this._buildRow(leaf, "", true)));

    panel.appendChild(head);
    panel.appendChild(grid);
    this.pinsEl = panel;
    return panel;
  }

  _renderBands(host, claimed) {
    const grouped = new Map();
    this.config.leaves.forEach((leaf) => {
      if (claimed.has(leaf.path)) return;
      if (!grouped.has(leaf.section)) grouped.set(leaf.section, []);
      grouped.get(leaf.section).push(leaf);
    });

    const sections = [...grouped.keys()];
    const nonRoot  = sections.filter((s) => s !== "");
    const promote  = nonRoot.length > 1 && new Set(nonRoot.map((s) => s.split(".")[0])).size === 1;
    const bandKey  = (s) => (s === "" ? "run" : s.split(".").slice(0, promote ? 2 : 1).join("."));

    const bandMap = new Map();
    sections.forEach((s) => {
      const k = bandKey(s);
      if (!bandMap.has(k)) bandMap.set(k, []);
      bandMap.get(k).push(s);
    });

    const wall = document.createElement("div");
    wall.className = "launch-bands";
    bandMap.forEach((bandSections, key) => wall.appendChild(this._buildBand(key, bandSections, grouped)));
    host.appendChild(wall);

    this._wireSectionGates(grouped);

    const empty = document.createElement("p");
    empty.className = "cfg-note launch-nomatch";
    empty.textContent = "No fields match this filter.";
    empty.hidden = true;
    this.nomatchEl = empty;
    host.appendChild(empty);
  }

  _classColor(sectionClass) {
    if (!this.classColors.has(sectionClass)) {
      this.classColors.set(sectionClass, ConfigForm.PALETTE[this.classColors.size % ConfigForm.PALETTE.length]);
    }
    return this.classColors.get(sectionClass);
  }

  _buildBand(key, bandSections, grouped) {
    const ordered = [...bandSections].sort((a, b) => a.split(".").length - b.split(".").length);
    const nFields = bandSections.reduce((n, s) => n + this._shownCount(grouped.get(s)), 0);
    const rootClass = grouped.get(ordered[0])[0].section_class;

    const band = document.createElement("section");
    band.className = "launch-band";
    band.style.setProperty("--cc", this._classColor(rootClass));

    const head = document.createElement("header");
    head.className = "band-head";
    head.tabIndex = 0;
    head.setAttribute("role", "button");
    head.setAttribute("aria-expanded", "false");
    head.innerHTML =
      `<span class="band-head__chev">&rsaquo;</span>` +
      `<span class="cc-dot" aria-hidden="true"></span>` +
      `<h3 class="band-head__name">${key}</h3>` +
      `<span class="edit-badge" hidden></span>` +
      `<span class="band-head__class">${rootClass}</span>` +
      `<span class="band-head__count">${nFields} fields</span>`;
    head.addEventListener("click", () => this._toggleBand(band, head));
    head.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      this._toggleBand(band, head);
    });

    const body = document.createElement("div");
    body.className = "band-body";
    band.appendChild(head);
    band.appendChild(body);

    const holders = new Map();
    const bandChildren = document.createElement("div");
    bandChildren.className = "band-children";

    ordered.forEach((section) => {
      const leaves = grouped.get(section);
      const isBandRoot = section === "" || section === key;

      if (isBandRoot) {
        const grid = this._buildFieldsGrid(section, leaves, "band-fields");
        this._packBandColumns(grid);
        body.appendChild(grid);
        this.panels.set(section, { el: grid, badge: null });
        holders.set(section, bandChildren);
        return;
      }

      const parentPath = section.split(".").slice(0, -1).join(".");
      const host = holders.get(parentPath) || bandChildren;
      const sub = this._buildSubPanel(section, leaves);
      host.appendChild(sub.el);
      this.panels.set(section, { el: sub.el, badge: sub.badge });
      holders.set(section, sub.children);
    });

    body.appendChild(bandChildren);
    this.bands.push({ el: band, head, badge: head.querySelector(".edit-badge"), sections: bandSections });
    return band;
  }

  _toggleBand(band, head) {
    const open = band.classList.toggle("is-open");
    head.setAttribute("aria-expanded", String(open));
  }

  _setBandOpen(band, head, open) {
    band.classList.toggle("is-open", open);
    head.setAttribute("aria-expanded", String(open));
  }

  _sectionEdited(section) {
    return this.states.some(({ leaf }) => leaf.section === section && this.dirty[leaf.path] !== undefined);
  }

  _buildSubPanel(section, leaves) {
    const isOverride = this.overrideSections.sections.has(section);
    const name  = section.split(".").pop();
    const label = isOverride ? `${name} overrides` : name;
    const shown = this._shownCount(leaves);

    const el = document.createElement("section");
    el.className = "sub-panel";
    if (isOverride) el.classList.add("sub-panel--override");
    el.dataset.section = section;
    el.title = isOverride ? `${section} — only the fields that differ from complete` : section;
    el.style.setProperty("--cc", this._classColor(leaves[0].section_class));

    const head = document.createElement("button");
    head.type = "button";
    head.className = "sub-panel__head";
    head.setAttribute("aria-expanded", "false");
    head.innerHTML =
      `<span class="sub-panel__chev">&rsaquo;</span>` +
      `<span class="cc-dot" aria-hidden="true"></span>` +
      `<h4 class="sub-panel__name">${label}</h4>` +
      `<span class="edit-badge" hidden></span>` +
      `<span class="sub-panel__class">${leaves[0].section_class}</span>` +
      `<span class="sub-panel__count">${shown}${isOverride ? " differ" : " fields"}</span>`;
    head.addEventListener("click", () => {
      const open = el.classList.toggle("is-open");
      head.setAttribute("aria-expanded", String(open));
    });

    const body = this._buildFieldsGrid(section, leaves, "sub-panel__body");

    const children = document.createElement("div");
    children.className = "band-children band-children--nested";

    el.appendChild(head);
    if (isOverride) {
      const note = document.createElement("p");
      note.className = "sub-panel__note";
      note.textContent = "Inherits the complete stage; fields edited here override the complete values.";
      el.appendChild(note);
    }
    el.appendChild(body);
    el.appendChild(children);
    return { el, badge: head.querySelector(".edit-badge"), children };
  }

  _buildFieldsGrid(section, leaves, className) {
    const grid = document.createElement("div");
    grid.className = className;
    grid.dataset.section = section;

    const blocks = new Map();
    leaves.forEach((leaf) => {
      if (!blocks.has(leaf.block)) blocks.set(leaf.block, []);
      blocks.get(leaf.block).push(leaf);
    });

    const shortName = (leaf) => (section ? leaf.path.slice(section.length + 1) : leaf.path);
    const isTermBlock = (blockLeaves) => {
      const lead = blockLeaves[0];
      const name = shortName(lead);
      return lead.type === "bool" && lead.editable && name.startsWith("use_") && blockLeaves.some((l) => shortName(l).startsWith("weight_"));
    };
    const isGateBlock = (blockLeaves) => {
      const lead = blockLeaves[0];
      const name = shortName(lead);
      return isTermBlock(blockLeaves) || (lead.type === "bool" && lead.editable && name !== "enabled" && name.endsWith("_enabled"));
    };
    const appendCell = (host, blockLeaves) => {
      if (!isGateBlock(blockLeaves)) {
        blockLeaves.forEach((leaf) => host.appendChild(this._buildRow(leaf, section)));
        return;
      }
      const cell = document.createElement("div");
      cell.className = "band-block";
      this._buildBlock(blockLeaves, section, cell);
      host.appendChild(cell);
    };

    const blockTitle = (blockLeaves) => {
      const names = blockLeaves.map(shortName);
      const lead = names[0];
      if (isGateBlock(blockLeaves)) {
        if (lead.endsWith("_enabled")) return lead.slice(0, -"_enabled".length);
        if (lead.startsWith("use_")) return lead.slice(4);
      }
      if (names.length < 2) return null;

      const tokenLists = names.map((n) => n.split("_"));

      const prefix = [];
      for (let i = 0; i < tokenLists[0].length; i++) {
        const token = tokenLists[0][i];
        if (tokenLists.every((tokens) => tokens[i] === token)) prefix.push(token);
        else break;
      }
      if (prefix.join("_").length >= 3) return prefix.join("_");

      const suffix = [];
      for (let i = 1; i <= Math.min(...tokenLists.map((t) => t.length)); i++) {
        const token = tokenLists[0][tokenLists[0].length - i];
        if (tokenLists.every((tokens) => tokens[tokens.length - i] === token)) suffix.unshift(token);
        else break;
      }
      if (suffix.join("_").length >= 3) return suffix.join("_");

      const counts = new Map();
      tokenLists.forEach((tokens) => new Set(tokens).forEach((token) => counts.set(token, (counts.get(token) || 0) + 1)));
      const majority = [...counts.entries()].filter(([token, n]) => token.length >= 3 && n > names.length / 2).sort((a, b) => b[1] - a[1] || b[0].length - a[0].length);
      if (majority.length) return majority[0][0];

      return taxonomyTitle(names);
    };

    const taxonomyTitle = (names) => {
      const order  = ConfigForm.FIELD_TAXONOMY.map(([title]) => title);
      const counts = new Map();
      names.forEach((name) => {
        const rule = ConfigForm.FIELD_TAXONOMY.find(([, pattern]) => pattern.test(name));
        if (rule) counts.set(rule[0], (counts.get(rule[0]) || 0) + 1);
      });
      if (!counts.size) return null;
      return [...counts.entries()].sort((a, b) => b[1] - a[1] || order.indexOf(a[0]) - order.indexOf(b[0]))[0][0];
    };

    const makeGroupEl = (title) => {
      const group = document.createElement("div");
      group.className = "field-group";
      if (title) {
        const heading = document.createElement("div");
        heading.className = "field-group__title";
        heading.textContent = title;
        group.appendChild(heading);
      }
      const inner = document.createElement("div");
      inner.className = "field-group__grid";
      group.appendChild(inner);
      return { group, inner };
    };

    const makeGroup = (title, members) => {
      if (!members.length) return null;
      const { group, inner } = makeGroupEl(title);
      members.forEach((blockLeaves) => appendCell(inner, blockLeaves));
      return group;
    };

    const appendWithStack = (named) => {
      const stackedBelow = new Set();
      ConfigForm.STACK_PAIRS.forEach(([top, bottom]) => {
        if (named.get(top) && named.get(bottom)) stackedBelow.add(bottom);
      });

      named.forEach((el, title) => {
        if (el === null || stackedBelow.has(title)) return;

        const pair = ConfigForm.STACK_PAIRS.find(([top, bottom]) => top === title && stackedBelow.has(bottom));
        if (pair) {
          const stack = document.createElement("div");
          stack.className = "field-group-stack";
          stack.appendChild(el);
          stack.appendChild(named.get(pair[1]));
          grid.appendChild(stack);
          return;
        }

        grid.appendChild(el);
      });
    };

    const termCount = [...blocks.values()].filter(isTermBlock).length;

    const SLOT_PRESENCE = /presence|focal|active_normalization|active_weight|inactive_weight/;

    if (termCount >= 2) {
      const buckets = { curve: [], param: [], slot: [], reg: [], general: [] };
      blocks.forEach((blockLeaves) => {
        if (!isTermBlock(blockLeaves)) {
          if (shortName(blockLeaves[0]).startsWith("param")) buckets.param.push(blockLeaves);
          else if (SLOT_PRESENCE.test(shortName(blockLeaves[0]))) buckets.slot.push(blockLeaves);
          else buckets.general.push(blockLeaves);
          return;
        }
        const label = shortName(blockLeaves[0]).slice(4);
        if (label.startsWith("param")) buckets.param.push(blockLeaves);
        else if (/curve|spectral|ssim/.test(label)) buckets.curve.push(blockLeaves);
        else if (SLOT_PRESENCE.test(label)) buckets.slot.push(blockLeaves);
        else buckets.reg.push(blockLeaves);
      });

      grid.classList.add("is-grouped");
      const named = new Map();
      named.set("curve space", makeGroup("curve space", buckets.curve));
      named.set("param space", makeGroup("param space", buckets.param));
      named.set("slot presence", makeGroup("slot presence", buckets.slot));
      named.set("regularization", makeGroup("regularization", buckets.reg));
      named.set("general", makeGroup("general", buckets.general));
      appendWithStack(named);
      return grid;
    }

    if (blocks.size >= 3) {
      grid.classList.add("is-grouped");
      blocks.forEach((blockLeaves) => {
        grid.appendChild(makeGroup(blockTitle(blockLeaves), [blockLeaves]));
      });
      return grid;
    }

    const plainBlocks = [];
    const gateBlocks = [];
    blocks.forEach((blockLeaves) => (isGateBlock(blockLeaves) ? gateBlocks : plainBlocks).push(blockLeaves));
    const plainLeaves = plainBlocks.flat();

    if (plainLeaves.length >= 7) {
      const classified = new Map();
      plainLeaves.forEach((leaf) => {
        const rule = ConfigForm.FIELD_TAXONOMY.find(([, pattern]) => pattern.test(shortName(leaf)));
        const title = rule ? rule[0] : "general";
        if (!classified.has(title)) classified.set(title, []);
        classified.get(title).push(leaf);
      });

      if (classified.size >= 2 || gateBlocks.length) {
        grid.classList.add("is-grouped");
        const named = new Map();
        [...ConfigForm.FIELD_TAXONOMY.map(([title]) => title), "general"].forEach((title) => {
          if (!classified.has(title)) return;
          const { group, inner } = makeGroupEl(title);
          classified.get(title).forEach((leaf) => inner.appendChild(this._buildRow(leaf, section)));
          named.set(title, group);
        });
        appendWithStack(named);
        gateBlocks.forEach((blockLeaves) => grid.appendChild(makeGroup(blockTitle(blockLeaves), [blockLeaves])));
        return grid;
      }
    }

    if (blocks.size >= 2) {
      grid.classList.add("is-grouped");
      blocks.forEach((blockLeaves) => {
        grid.appendChild(makeGroup(blockTitle(blockLeaves), [blockLeaves]));
      });
      return grid;
    }

    blocks.forEach((blockLeaves) => appendCell(grid, blockLeaves));
    return grid;
  }

  _rowHeight(row) {
    if (row.classList.contains("cfg-edit__row--num"))   return 2.2;
    if (row.classList.contains("cfg-edit__row--board")) return 3.0;
    if (row.classList.contains("cfg-edit__row--wide"))  return 2.0;
    return 1.0;
  }

  _groupHeight(el) {
    let h = el.querySelectorAll(".field-group__title").length * 1.2 + 0.4;
    el.querySelectorAll(".cfg-edit__row").forEach((row) => (h += this._rowHeight(row)));
    return h;
  }

  _packBandColumns(grid) {
    if (!grid.classList.contains("is-grouped")) return;
    const items = [...grid.children];
    if (items.length < 2) return;

    const ncols  = Math.min(3, items.length);
    const total  = items.reduce((sum, el) => sum + this._groupHeight(el), 0);
    const target = total / ncols;

    const cols = Array.from({ length: ncols }, () => {
      const col = document.createElement("div");
      col.className = "field-col";
      return { el: col, h: 0 };
    });

    let ci = 0;
    items.forEach((el) => {
      cols[ci].el.appendChild(el);
      cols[ci].h += this._groupHeight(el);
      if (ci < ncols - 1 && cols[ci].h >= target) ci++;
    });

    grid.classList.remove("is-grouped");
    grid.classList.add("is-packed");
    cols.forEach((col) => grid.appendChild(col.el));
  }

  _buildBlock(blockLeaves, section, body) {
    const shortName = (leaf) => (section ? leaf.path.slice(section.length + 1) : leaf.path);
    const lead = blockLeaves[0];
    const leadName = shortName(lead);
    const hasWeight = blockLeaves.some((l) => shortName(l).startsWith("weight_"));
    const isTermGate = lead.type === "bool" && lead.editable && leadName.startsWith("use_") && hasWeight;
    const isBlockGate = lead.type === "bool" && lead.editable && leadName !== "enabled" && leadName.endsWith("_enabled");

    if (!isTermGate && !isBlockGate) {
      blockLeaves.forEach((leaf) => body.appendChild(this._buildRow(leaf, section)));
      return;
    }

    const rest = blockLeaves.slice(1);
    let weight = null;
    if (isTermGate) {
      const index = rest.findIndex((l) => shortName(l).startsWith("weight_") && (l.type === "float" || l.type === "int"));
      if (index >= 0) weight = rest.splice(index, 1)[0];
    }

    const label = isTermGate ? leadName.slice(4) : leadName;
    body.appendChild(this._buildGateRow(lead, label));

    const gatedRows = [];

    if (weight) {
      body.appendChild(this._buildWeightRow(weight));
      gatedRows.push(this.states[this.states.length - 1]);
    }

    rest.forEach((leaf) => {
      const dependent = body.appendChild(this._buildRow(leaf, section));
      dependent.classList.add("cfg-edit__row--dependent");
      gatedRows.push(this.states[this.states.length - 1]);
    });

    this.gates.push({ leaf: lead, states: gatedRows, sections: [] });
  }

  _buildGateRow(lead, label) {
    const row = document.createElement("div");
    row.className = "cfg-edit__row cfg-edit__row--bool cfg-edit__row--gate";
    row.title = `--${lead.path}`;

    const name = document.createElement("div");
    name.className = "cfg-edit__name";
    name.textContent = label;
    row.appendChild(name);

    const toggle = this._switchControl(lead);
    row.appendChild(toggle.el);
    this.controls[lead.path] = { leaf: lead, reset: toggle.reset };
    this.states.push({ leaf: lead, row, section: lead.section });
    return row;
  }

  _buildWeightRow(weight) {
    const row = document.createElement("div");
    row.className = "cfg-edit__row cfg-edit__row--dependent cfg-edit__row--gateweight";
    row.title = `--${weight.path}`;

    const name = document.createElement("div");
    name.className = "cfg-edit__name";
    name.textContent = "weight";
    row.appendChild(name);

    const control = this._numberControl(weight);
    control.input.title = `--${weight.path}`;
    row.appendChild(control.input);

    this.controls[weight.path] = { leaf: weight, reset: control.reset };
    this.states.push({ leaf: weight, row, section: weight.section });
    return row;
  }

  _buildRow(leaf, section, pinned = false) {
    const short = section ? leaf.path.slice(section.length + 1) : leaf.path;

    const row = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title = `--${leaf.path}`;

    const label = document.createElement("div");
    label.className = "cfg-edit__name";
    label.textContent = short;
    label.title = `${leaf.type} · --${leaf.path}`;
    row.appendChild(label);

    let control;
    const pickerSpec = leaf.editable ? this._pickerSpec(leaf) : null;
    const multiSpec  = leaf.editable ? this._multiValueSpec(leaf) : null;
    const choices    = leaf.editable ? this._choicesFor(leaf) : null;
    if (leaf.editable && this._isGpuField(leaf) && window.GpuPicker) {
      control = new window.GpuPicker(this, leaf).build();
      row.classList.add("cfg-edit__row--board");
    } else if (multiSpec && window.MultiValueField) {
      control = new window.MultiValueField(this, leaf, multiSpec).build();
      row.classList.add("cfg-edit__row--board");
    } else if (pickerSpec && window.DatasetPicker) {
      control = new window.DatasetPicker(this, leaf, pickerSpec).build();
      row.classList.add(pickerSpec.multi ? "cfg-edit__row--board" : "cfg-edit__row--wide");
    } else if (choices) {
      control = this._choiceControl(leaf, choices);
      row.classList.add("cfg-edit__row--choice");
    } else if (!leaf.editable) {
      control = this._textControl(leaf);
      control.input.disabled = true;
      control.input.classList.add("is-locked");
      control.input.title = "not overridable from the command line";
    } else if (leaf.type === "bool") {
      control = this._switchControl(leaf);
      row.classList.add("cfg-edit__row--bool");
    } else if (leaf.type === "int" || leaf.type === "float") {
      control = new window.NumberField(this, leaf, short).build();
      row.classList.add("cfg-edit__row--num");
    } else {
      control = this._textControl(leaf);
    }

    row.appendChild(control.el);
    this.controls[leaf.path] = { leaf, reset: control.reset };
    this.states.push({ leaf, row, section: leaf.section, pinned });
    return row;
  }

  _wireSectionGates(grouped) {
    grouped.forEach((leaves, section) => {
      const lead = leaves[0];
      const leadName = section ? lead.path.slice(section.length + 1) : lead.path;
      if (leadName !== "enabled" || lead.type !== "bool" || !lead.editable) return;

      const states = this.states.filter((s) => s.leaf.section === section && s.leaf !== lead);
      const sections = [...this.panels.keys()].filter((name) => name.startsWith(`${section}.complete`));

      if (states.length || sections.length) {
        this.gates.push({ leaf: lead, states, sections });
      }
    });
  }

  _effective(leaf) {
    return this.dirty[leaf.path] !== undefined ? this.dirty[leaf.path] : leaf.value;
  }

  _leafByPath(path) {
    return this.config ? this.config.leaves.find((leaf) => leaf.path === path) : null;
  }

  _optionGate() {
    return ConfigForm.OPTION_GATES[this.key] || null;
  }

  _optionValue(gate) {
    const leaf = this._leafByPath(gate.field);
    return leaf ? this._effective(leaf) : null;
  }

  _sectionOptionGated(section, gate, value) {
    const allowed = gate.sections[section.split(".")[0]];
    return Boolean(allowed) && !allowed.includes(value);
  }

  _pickerSpec(leaf) {
    const specs = ConfigForm.PICKERS[this.key];
    return specs ? specs[leaf.path] : null;
  }

  _multiValueSpec(leaf) {
    const specs = ConfigForm.MULTI_VALUE[this.key];
    return specs ? specs[leaf.path] : null;
  }

  _isGpuField(leaf) {
    if (leaf.type !== "list" && leaf.type !== "int") return false;
    return ConfigForm.GPU_FIELDS.includes(leaf.path.split(".").pop());
  }

  _choicesFor(leaf) {
    if (Array.isArray(leaf.choices) && leaf.choices.length) return leaf.choices;
    const map = ConfigForm.CHOICES[this.key];
    return map ? map[leaf.path] || null : null;
  }

  _choiceControl(leaf, choices) {
    const select = document.createElement("select");
    select.className = "cfg-edit__input picker__select";

    const current = String(leaf.value);
    const options = choices.includes(current) ? choices : [current, ...choices];
    const resolved = ConfigForm.NORMALIZATION_DEFAULTS[leaf.path];
    options.forEach((value) => {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value === "default" && resolved ? `${resolved} · per-slot` : value;
      select.appendChild(opt);
    });
    select.value = current;

    select.addEventListener("change", () => {
      select.classList.toggle("is-dirty", select.value !== leaf.value);
      this._setValue(leaf, select.value);
      this._fireDependents(leaf.path, select.value);
    });

    const reset = () => {
      select.value = leaf.value;
      select.classList.remove("is-dirty");
    };
    return { el: select, input: select, reset };
  }

  _onDependency(path, fn) {
    (this.dependents[path] = this.dependents[path] || []).push(fn);
  }

  _fireDependents(path, value) {
    (this.dependents[path] || []).forEach((fn) => fn(value));
  }

  _setValue(leaf, value) {
    const changed = value !== leaf.value && value !== "";
    if (changed) this.dirty[leaf.path] = value;
    else delete this.dirty[leaf.path];
    this._refresh();
  }

  _textControl(leaf) {
    const input = document.createElement("input");
    input.className = "cfg-edit__input";
    input.value = leaf.value;
    input.spellcheck = false;
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value);
      this._setValue(leaf, input.value);
      this._fireDependents(leaf.path, input.value);
    });
    const reset = () => {
      input.value = leaf.value;
      input.classList.remove("is-dirty");
    };
    return { el: input, input, reset };
  }

  _numberControl(leaf) {
    const input = document.createElement("input");
    input.className = "cfg-edit__input";
    input.type = "number";
    input.step = leaf.type === "int" ? "1" : "any";
    input.value = leaf.value;
    input.spellcheck = false;
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value && input.value !== "");
      this._setValue(leaf, input.value);
    });
    const reset = () => {
      input.value = leaf.value;
      input.classList.remove("is-dirty");
    };
    return { el: input, input, reset };
  }

  _switchControl(leaf) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.innerHTML = `<span class="switch__knob"></span>`;

    const paint = () => {
      const on = this._effective(leaf) === "True";
      toggle.classList.toggle("is-on", on);
      toggle.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
      toggle.setAttribute("aria-checked", String(on));
    };
    toggle.addEventListener("click", () => {
      const next = this._effective(leaf) === "True" ? "False" : "True";
      this._setValue(leaf, next);
      paint();
    });
    paint();

    const reset = () => paint();
    return { el: toggle, input: toggle, reset };
  }

  _resetField(path) {
    const control = this.controls[path];
    delete this.dirty[path];
    if (control) control.reset();
    this._refresh();
  }

  _resetAll() {
    this.dirty = {};
    Object.values(this.controls).forEach((c) => c.reset());
    this._refresh();
  }

  _refreshGates() {
    this.gatedSections = new Set();
    this.states.forEach(({ row }) => {
      delete row.dataset.gated;
    });

    this.gates.forEach((gate) => {
      const open = this._effective(gate.leaf) === "True";
      if (!open) {
        gate.states.forEach(({ row }) => (row.dataset.gated = "1"));
        gate.sections.forEach((section) => this.gatedSections.add(section));
      }
    });

    const optionGate = this._optionGate();
    if (optionGate) {
      const value = this._optionValue(optionGate);
      this.states.forEach(({ leaf, row }) => {
        if (this._sectionOptionGated(leaf.section, optionGate, value)) row.dataset.gated = "1";
      });
      [...this.panels.keys()].forEach((section) => {
        if (this._sectionOptionGated(section, optionGate, value)) this.gatedSections.add(section);
      });
    }

    this._applyVisibility();
  }

  _applyVisibility(sync = false) {
    const rowVisible = new Map();
    this.states.forEach(({ leaf, row, pinned }) => {
      const matches    = !this.query || leaf.path.toLowerCase().includes(this.query);
      const dirty      = this.dirty[leaf.path] !== undefined;
      const identical  = this.overrideSections.identical.has(leaf.path);
      const inOverride = this.overrideSections.sections.has(leaf.section);

      let disclosed;
      if (this.query)            disclosed = matches;
      else if (dirty || pinned)  disclosed = true;
      else if (inOverride)       disclosed = true;
      else if (identical)        disclosed = false;
      else                       disclosed = this.showAll;

      if (matches && disclosed && row.dataset.gated !== "1") rowVisible.set(row, true);
      else if (!rowVisible.has(row)) rowVisible.set(row, false);
    });
    rowVisible.forEach((visible, row) => (row.hidden = !visible));

    const sectionHasRows = new Map();
    this.states.forEach(({ leaf, row, pinned }) => {
      if (pinned) return;
      const prior = sectionHasRows.get(leaf.section) || false;
      sectionHasRows.set(leaf.section, prior || rowVisible.get(row));
    });

    const orderedSections = [...this.panels.keys()].sort((a, b) => b.split(".").length - a.split(".").length);
    const sectionVisible = new Map();
    orderedSections.forEach((section) => {
      const childVisible = orderedSections.some((other) => other !== section && other.startsWith(section ? `${section}.` : ".") && sectionVisible.get(other));
      const isOverride = this.overrideSections.sections.has(section);
      const visible = !this.gatedSections.has(section) && ((sectionHasRows.get(section) || false) || childVisible || isOverride);
      sectionVisible.set(section, visible);
      const el = this.panels.get(section).el;
      el.hidden = !visible;
      if (el.classList.contains("sub-panel")) {
        const subWant = visible && (Boolean(this.query) || this.showAll || this._sectionEdited(section) || childVisible);
        const subHead = el.querySelector(".sub-panel__head");
        if (sync) { el.classList.toggle("is-open", subWant); if (subHead) subHead.setAttribute("aria-expanded", String(subWant)); }
        else if (subWant) { el.classList.add("is-open"); if (subHead) subHead.setAttribute("aria-expanded", "true"); }
      }
    });

    let anyVisible = false;
    this.bands.forEach((band) => {
      const visible = band.sections.some((section) => sectionVisible.get(section));
      band.el.hidden = !visible;
      anyVisible = anyVisible || visible;

      const edited   = band.sections.some((section) => this._sectionEdited(section));
      const wantOpen = visible && (Boolean(this.query) || this.showAll || edited);
      if (sync) this._setBandOpen(band.el, band.head, wantOpen);
      else if (wantOpen && !band.el.classList.contains("is-open")) this._setBandOpen(band.el, band.head, true);
    });

    if (this.pinsEl) {
      const rows = [...this.pinsEl.querySelectorAll(".cfg-edit__row")];
      this.pinsEl.hidden = rows.length > 0 && rows.every((row) => row.hidden);
      anyVisible = anyVisible || !this.pinsEl.hidden;
    }

    const none = this.nomatchEl;
    if (none) none.hidden = anyVisible;
  }

  _refreshBadges() {
    const counts = new Map();
    this.states.forEach(({ leaf, pinned }) => {
      if (pinned) return;
      if (this.dirty[leaf.path] !== undefined) counts.set(leaf.section, (counts.get(leaf.section) || 0) + 1);
    });

    this.panels.forEach(({ badge }, section) => {
      if (!badge) return;
      const n = counts.get(section) || 0;
      badge.hidden = n === 0;
      badge.textContent = n ? `${n} edited` : "";
    });

    this.bands.forEach((band) => {
      const n = band.sections.reduce((sum, section) => sum + (counts.get(section) || 0), 0);
      band.badge.hidden = n === 0;
      band.badge.textContent = n ? `${n} edited` : "";
    });
  }
}

window.PythonLiteral     = PythonLiteral;
window.GpuCardSelect     = GpuCardSelect;
window.GpuPicker         = GpuPicker;
window.NumberField       = NumberField;
window.MultiValueField   = MultiValueField;
window.LaunchWidgetDom   = LaunchWidgetDom;
window.ModelTogglePanel  = ModelTogglePanel;
window.ModelCardPanel    = ModelCardPanel;
window.ExperimentBuilder = ExperimentBuilder;
window.AblationBuilder   = AblationBuilder;
window.ConfigForm        = ConfigForm;
