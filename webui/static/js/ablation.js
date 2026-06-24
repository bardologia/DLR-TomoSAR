"use strict";

class AblationView {
  constructor(runConsole, project) {
    this.runConsole = runConsole;
    this.project    = project || {};
    this.host       = document.getElementById("ablation-root");
    this.key        = "train_backbone";

    this.detail   = null;
    this.config   = null;
    this.byPath   = new Map();
    this.dirty    = {};
    this.controls = {};
    this.builder  = null;
    this.detach   = true;

    this.loaded  = false;
    this.loading = false;

    this.interpEl  = null;
    this.detachEl  = null;
    this.countEl   = null;
    this.launchBtn = null;
    this.cmdEl     = null;
  }

  async enter() {
    if (!this.host || this.loaded || this.loading) return;
    this.loading = true;
    await this._load();
    this.loading = false;
  }

  async _load() {
    this.host.innerHTML = `<p class="ablation__note">Loading backbone training configuration…</p>`;

    const detail = await window.apiGet(`/api/scripts/${this.key}`);
    if (!detail || detail.error) {
      this.host.innerHTML = `<p class="ablation__note ablation__note--error">Could not load the backbone training script.</p>`;
      return;
    }
    this.detail = detail;

    const cfg = await window.apiGet(`/api/scripts/${this.key}/config`);
    if (!cfg || !cfg.ok) {
      this.host.innerHTML = `<p class="ablation__note ablation__note--error">${(cfg && cfg.error) || "Could not resolve configuration."}</p>`;
      return;
    }
    this.config = cfg;
    this.byPath = new Map(cfg.leaves.map((leaf) => [leaf.path, leaf]));

    if (!this.byPath.get("ablation_features")) {
      this.host.innerHTML = `<p class="ablation__note ablation__note--error">This build exposes no ablation configuration.</p>`;
      return;
    }

    this.loaded = true;
    this._render();
    this._arm();
    this._refresh();
  }

  _render() {
    this.host.innerHTML = "";

    const bar     = document.createElement("div");
    bar.className = "ablation__bar";

    const interp     = document.createElement("select");
    interp.className = "ablation__interp";
    (this.project.interpreters || []).forEach((it) => {
      const opt       = document.createElement("option");
      opt.value       = it.path;
      opt.textContent = it.label || it.path;
      if (it.path === this.detail.preferred) opt.selected = true;
      interp.appendChild(opt);
    });
    this.interpEl = interp;
    bar.appendChild(interp);

    const detach = window.LaunchWidgetDom.mini(this.detach ? "detached" : "attached", () => {
      this.detach = !this.detach;
      detach.textContent = this.detach ? "detached" : "attached";
      detach.classList.toggle("is-on", this.detach);
    });
    detach.classList.toggle("is-on", this.detach);
    this.detachEl = detach;
    bar.appendChild(detach);

    const count     = document.createElement("span");
    count.className = "ablation__count";
    this.countEl    = count;
    bar.appendChild(count);

    const launch     = document.createElement("button");
    launch.type      = "button";
    launch.className = "btn btn--primary ablation__launch";
    launch.addEventListener("click", () => this._launch());
    this.launchBtn = launch;
    bar.appendChild(launch);

    this.host.appendChild(bar);

    this.builder = new window.AblationBuilder(this, this.byPath);
    this.host.appendChild(this.builder.build());

    const cmd     = document.createElement("pre");
    cmd.className = "ablation__cmd";
    this.cmdEl    = cmd;
    this.host.appendChild(cmd);
  }

  _arm() {
    const trials = this.byPath.get("trials_enabled");
    const mode   = this.byPath.get("trials_mode");
    if (mode)   this._setValue(mode, "ablation");
    if (trials) this._setValue(trials, "True");
  }

  _effective(leaf) {
    return this.dirty[leaf.path] !== undefined ? this.dirty[leaf.path] : leaf.value;
  }

  _setValue(leaf, value) {
    const changed = value !== leaf.value && value !== "";
    if (changed) this.dirty[leaf.path] = value;
    else delete this.dirty[leaf.path];
    this._refresh();
  }

  _commandText() {
    let text = this.detail.command || "python main/train.py";
    Object.entries(this.dirty).forEach(([path, value]) => {
      const raw      = String(value);
      const shown    = raw.length > 80 ? `${raw.slice(0, 77)}…` : raw;
      const rendered = /\s/.test(shown) ? `'${shown}'` : shown;
      text += ` \\\n  --${path} ${rendered}`;
    });
    return text;
  }

  _active() {
    const trials = this.byPath.get("trials_enabled");
    const mode   = this.byPath.get("trials_mode");
    return trials && mode && this._effective(trials) === "True" && this._effective(mode) === "ablation";
  }

  _refresh() {
    if (this.builder) this.builder.refreshFromView();

    const n = Object.keys(this.dirty).length;
    if (this.countEl) this.countEl.textContent = n ? `${n} override${n === 1 ? "" : "s"}` : "all defaults";
    if (this.cmdEl) this.cmdEl.textContent = this._commandText();

    if (this.launchBtn) {
      const active = this._active();
      this.launchBtn.disabled    = !active;
      this.launchBtn.textContent = active ? "Launch ablation study" : "Enable the ablation switch to launch";
    }
  }

  _launch() {
    if (!this._active()) return;
    const interp = this.interpEl ? this.interpEl.value : this.detail.preferred;
    this.runConsole.launch(this.key, interp, "Ablation study", { ...this.dirty }, null, this.detach);
    window.router.go("console");
  }
}

window.AblationView = AblationView;
