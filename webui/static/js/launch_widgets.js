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


class ExperimentBuilder {
  constructor(view, byPath) {
    this.view         = view;
    this.trialsLeaf   = byPath.get("trials_enabled");
    this.warmupLeaf   = byPath.get("warmup_losses");
    this.completeLeaf = byPath.get("complete_losses");
    this.modelLeaf    = byPath.get("model_name");
    this.gpusLeaf     = byPath.get("gpus");
    this.terms        = this._termCatalog();
    this.variants     = { warmup: [], complete: [] };
    this.lists        = {};
    this.summaryEl    = null;
    this.namesEl      = null;
    this.root         = null;
    this._paintSwitch = null;
  }

  build() {
    this.root           = document.createElement("section");
    this.root.className = "exp-builder";

    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = `<h3 class="special-head__name">Experiment fan-out</h3><span class="special-head__hint">warmup x complete cross product, one training run each</span>`;

    const summary     = document.createElement("span");
    summary.className = "exp-builder__summary";
    this.summaryEl    = summary;
    head.appendChild(summary);
    head.appendChild(this._trialsSwitch());

    const body     = document.createElement("div");
    body.className = "exp-builder__body";

    const columns     = document.createElement("div");
    columns.className = "exp-builder__columns";
    columns.appendChild(this._column("warmup", "warmup losses"));
    columns.appendChild(this._column("complete", "complete losses"));
    body.appendChild(columns);

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
    this._paintSummary();
    this._paintNames();
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
    this._paintAll();
  }

  _paintAll() {
    ["warmup", "complete"].forEach((which) => {
      const { list, count } = this.lists[which];
      list.innerHTML = "";
      this.variants[which].forEach((variant, index) => list.appendChild(this._card(which, variant, index)));
      count.textContent = `${this.variants[which].length} variant${this.variants[which].length === 1 ? "" : "s"}`;
    });

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

  _paintSummary() {
    const nWarm = this.variants.warmup.length;
    const nComp = this.variants.complete.length;

    let gpus = "";
    if (this.gpusLeaf) {
      try {
        const parsed = PythonLiteral.parse(this.view._effective(this.gpusLeaf));
        if (Array.isArray(parsed)) gpus = ` on ${parsed.length} GPU${parsed.length === 1 ? "" : "s"}`;
      } catch (e) {
        gpus = "";
      }
    }

    this.summaryEl.textContent = `${nWarm} warmup x ${nComp} complete = ${nWarm * nComp} trials${gpus}`;
  }

  _paintNames() {
    this.namesEl.innerHTML = "";
    const model = this.modelLeaf ? this.view._effective(this.modelLeaf) : "model";

    const names = [];
    this.variants.warmup.forEach((w) => {
      this.variants.complete.forEach((c) => names.push(`${model}_w-${w.label}_c-${c.label}`));
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
}


window.PythonLiteral     = PythonLiteral;
window.ModelTogglePanel  = ModelTogglePanel;
window.ExperimentBuilder = ExperimentBuilder;
