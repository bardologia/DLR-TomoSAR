"use strict";

class ScriptPanel {
  constructor(refs, runConsole, project) {
    this.gridEl = refs.grid;
    this.filterEl = refs.filters;
    this.drawer = refs.drawer;
    this.runConsole = runConsole;
    this.project = project;
    this.scripts = [];
    this.detail = null;
    this.filter = "All";
    this.dirty = {};
    this._wireDrawer();
  }

  async load() {
    const data = await window.apiGet("/api/scripts");
    this.scripts = data.scripts || [];
    this._renderFilters();
    this._renderGrid();
  }

  _categories() {
    const cats = ["All", ...new Set(this.scripts.map((s) => s.category))];
    return cats;
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

    items.forEach((s, i) => {
      const card = document.createElement("button");
      card.className = "script-card reveal";
      card.style.transitionDelay = `${i * 0.04}s`;
      card.innerHTML =
        `<span class="script-card__glow"></span>` +
        `<div class="script-card__top"><span class="script-card__cat">${s.category}</span>` +
        `<span class="script-card__file">${s.file}</span></div>` +
        `<h3 class="script-card__title">${s.title}</h3>` +
        `<p class="script-card__purpose">${s.purpose}</p>` +
        `<div class="script-card__foot"><span>${s.n_constants} config values</span>` +
        `<span class="arrow">inspect &rarr;</span></div>`;
      card.addEventListener("click", () => this.open(s.key));
      this.gridEl.appendChild(card);
    });

    window.revealScan();
  }

  async open(key) {
    const detail = await window.apiGet(`/api/scripts/${key}`);
    if (detail.error) {
      window.toast("could not load script", "error");
      return;
    }
    this.detail = detail;
    this.dirty = {};
    this._fillDrawer(detail);
    this.drawer.classList.add("is-open");
    this.drawer.setAttribute("aria-hidden", "false");
  }

  close() {
    this.drawer.classList.remove("is-open");
    this.drawer.setAttribute("aria-hidden", "true");
  }

  _fillDrawer(d) {
    document.getElementById("drawer-kicker").textContent = `${d.category} · ${d.file}`;
    document.getElementById("drawer-title").textContent = d.title;
    document.getElementById("drawer-purpose").textContent = d.purpose;

    this._buildRun(d);
    this._buildConfig(d);
    this._buildSource(d);
    this._setPane("config");
  }

  _buildRun(d) {
    const host = document.getElementById("drawer-run");
    host.innerHTML = "";

    const interpreters = (this.project.interpreters || []);
    const field = document.createElement("div");
    field.className = "run-field";
    const label = document.createElement("label");
    label.textContent = "Interpreter";
    const select = document.createElement("select");
    select.className = "run-select";
    select.id = "run-interpreter";
    interpreters.forEach((it) => {
      const opt = document.createElement("option");
      opt.value = it.path;
      opt.textContent = `${it.label}  ·  ${it.path}`;
      if (it.path === this.project.preferred) opt.selected = true;
      select.appendChild(opt);
    });
    field.appendChild(label);
    field.appendChild(select);

    const cmd = document.createElement("div");
    cmd.className = "run-field";
    const cmdLabel = document.createElement("label");
    cmdLabel.textContent = "Command";
    const cmdRow = document.createElement("div");
    cmdRow.className = "run-command";
    const code = document.createElement("code");
    code.textContent = d.command;
    const copy = document.createElement("button");
    copy.className = "btn btn--mini";
    copy.textContent = "Copy";
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(d.command).then(() => window.toast("command copied", "ok"));
    });
    cmdRow.appendChild(code);
    cmdRow.appendChild(copy);
    cmd.appendChild(cmdLabel);
    cmd.appendChild(cmdRow);

    const actions = document.createElement("div");
    actions.className = "run-actions";
    const launch = document.createElement("button");
    launch.className = "btn btn--primary";
    launch.innerHTML = "&#9654;&nbsp; Launch run";
    launch.addEventListener("click", () => {
      const interp = document.getElementById("run-interpreter").value;
      this.runConsole.launch(d.key, interp, d.title);
      this.close();
    });
    actions.appendChild(launch);

    host.appendChild(field);
    host.appendChild(cmd);
    host.appendChild(actions);
  }

  _buildConfig(d) {
    const host = document.getElementById("pane-config");
    host.innerHTML = "";

    if (!d.constants.length) {
      host.innerHTML = `<p class="cfg-note">This script exposes no editable constant block.</p>`;
      return;
    }

    const note = document.createElement("p");
    note.className = "cfg-note";
    note.textContent = `${d.constants.length} top-level constants · edits rewrite ${d.file} after a timestamped backup`;
    host.appendChild(note);

    const grid = document.createElement("div");
    grid.className = "cfg-edit";

    d.constants.forEach((c) => {
      const row = document.createElement("div");
      row.className = "cfg-edit__row";
      const name = document.createElement("div");
      name.className = "cfg-edit__name";
      name.innerHTML = `${c.name}<span>${c.type}</span>`;
      const input = document.createElement("input");
      input.className = "cfg-edit__input";
      input.value = c.value;
      input.spellcheck = false;
      input.addEventListener("input", () => {
        const changed = input.value !== c.value;
        input.classList.toggle("is-dirty", changed);
        if (changed) this.dirty[c.name] = input.value;
        else delete this.dirty[c.name];
      });
      row.appendChild(name);
      row.appendChild(input);
      grid.appendChild(row);
    });

    host.appendChild(grid);

    const actions = document.createElement("div");
    actions.className = "run-actions";
    actions.style.marginTop = "18px";
    const apply = document.createElement("button");
    apply.className = "btn btn--ghost";
    apply.textContent = "Apply configuration";
    apply.addEventListener("click", () => this._applyConfig(d.key));
    const reset = document.createElement("button");
    reset.className = "btn btn--mini";
    reset.textContent = "Revert edits";
    reset.addEventListener("click", () => this.open(d.key));
    actions.appendChild(apply);
    actions.appendChild(reset);
    host.appendChild(actions);
  }

  async _applyConfig(key) {
    if (!Object.keys(this.dirty).length) {
      window.toast("no edits to apply", "error");
      return;
    }
    const res = await window.apiPost(`/api/scripts/${key}/config`, { values: this.dirty });
    if (res.ok) {
      window.toast(`updated ${res.changed.length} value(s) · backup ${res.backup}`, "ok");
      this.open(key);
    } else {
      window.toast(res.error || "apply failed", "error");
    }
  }

  _buildSource(d) {
    const code = document.getElementById("source-code");
    code.textContent = d.source;
    code.className = "language-python";
    if (window.hljs) {
      try {
        window.hljs.highlightElement(code);
      } catch (e) {}
    }
  }

  _setPane(pane) {
    document.querySelectorAll(".drawer__tab").forEach((t) => t.classList.toggle("is-active", t.dataset.pane === pane));
    document.getElementById("pane-config").classList.toggle("is-active", pane === "config");
    document.getElementById("pane-source").classList.toggle("is-active", pane === "source");
  }

  _wireDrawer() {
    document.getElementById("drawer-close").addEventListener("click", () => this.close());
    document.getElementById("drawer-scrim").addEventListener("click", () => this.close());
    document.querySelectorAll(".drawer__tab").forEach((t) => {
      t.addEventListener("click", () => this._setPane(t.dataset.pane));
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.drawer.classList.contains("is-open")) this.close();
    });
  }
}

window.ScriptPanel = ScriptPanel;
