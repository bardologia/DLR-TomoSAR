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
      `<span class="model-pick__meta">${meta}</span>` +
      `<span class="model-pick__when">${model.when || ""}</span>`;

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


window.PythonLiteral     = PythonLiteral;
window.GpuCardSelect     = GpuCardSelect;
window.GpuPicker         = GpuPicker;
window.NumberField       = NumberField;
window.LaunchWidgetDom   = LaunchWidgetDom;
window.ModelTogglePanel  = ModelTogglePanel;
window.ModelCardPanel    = ModelCardPanel;
window.ExperimentBuilder = ExperimentBuilder;
