"use strict";

class AblationView extends ConfigForm {
  constructor(runConsole, project) {
    super();
    this.runConsole = runConsole;
    this.project    = project || {};
    this.host       = document.getElementById("ablation-root");
    this.key        = "train_backbone";

    this.detail = null;
    this.config = null;
    this.byPath = new Map();
    this.builder = null;
    this.detach  = true;

    this.loaded  = false;
    this.loading = false;

    this.configHost = null;
    this.interpEl   = null;
    this.detachEl   = null;
    this.topCountEl = null;
    this.launchBtn  = null;
    this.cmdEl      = null;
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

    const models = await window.apiGet("/api/backbones");
    this.modelFamilies = (models && models.families) || [];

    this.loaded = true;
    this._render();
  }

  _render() {
    this.host.innerHTML = "";

    this.dirty         = {};
    this.controls      = {};
    this.states        = [];
    this.panels        = new Map();
    this.bands         = [];
    this.gates         = [];
    this.gatedSections = new Set();
    this.classColors   = new Map();
    this.query         = "";
    this.showAll       = false;

    this.host.appendChild(this._buildBar());

    const configHost     = document.createElement("div");
    configHost.className  = "ablation__config";
    this.configHost      = configHost;
    this.host.appendChild(configHost);

    const cmd     = document.createElement("pre");
    cmd.className = "ablation__cmd";
    this.cmdEl    = cmd;
    this.host.appendChild(cmd);

    this._arm();
    this._renderConfig();
    this._refresh();
  }

  _buildBar() {
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
      this._refresh();
    });
    detach.classList.toggle("is-on", this.detach);
    this.detachEl = detach;
    bar.appendChild(detach);

    const count     = document.createElement("span");
    count.className = "ablation__count";
    this.topCountEl = count;
    bar.appendChild(count);

    const launch     = document.createElement("button");
    launch.type      = "button";
    launch.className = "btn btn--primary ablation__launch";
    launch.addEventListener("click", () => this._launch());
    this.launchBtn = launch;
    bar.appendChild(launch);

    return bar;
  }

  _arm() {
    const trials = this.byPath.get("trials_enabled");
    const mode   = this.byPath.get("trials_mode");
    if (mode)   this._setValue(mode, "ablation");
    if (trials) this._setValue(trials, "True");
  }

  _renderConfig() {
    const host   = this.configHost;
    host.innerHTML = "";
    const cfg    = this.config;
    const byPath = this.byPath;

    this.overrideSections = this._detectOverrideSections(cfg.leaves);

    host.appendChild(this._buildToolbar(cfg));

    const modelNameLeaf = byPath.get("backbone_name");
    const cardPanel     = modelNameLeaf && this.modelFamilies && this.modelFamilies.length ? new window.ModelCardPanel(this, modelNameLeaf) : null;

    const pinned  = (this.detail.essentials || []).map((path) => byPath.get(path)).filter(Boolean).filter((leaf) => !(cardPanel && leaf.path === modelNameLeaf.path));
    const claimed = new Set(pinned.map((leaf) => leaf.path));
    if (cardPanel) claimed.add(modelNameLeaf.path);

    this.builder = new window.AblationBuilder(this, byPath);
    this.builder.claimed.forEach((path) => claimed.add(path));

    if (byPath.get("trials_enabled") && byPath.get("warmup_losses") && byPath.get("complete_losses")) {
      const fanout = new window.ExperimentBuilder(this, byPath);
      fanout.claimed.forEach((path) => claimed.add(path));
    }

    if (pinned.length) host.appendChild(this._buildPins(pinned));
    if (cardPanel) host.appendChild(cardPanel.build());
    host.appendChild(this.builder.build());

    this._renderBands(host, claimed);
  }

  _active() {
    const trials = this.byPath.get("trials_enabled");
    const mode   = this.byPath.get("trials_mode");
    return Boolean(trials && mode && this._effective(trials) === "True" && this._effective(mode) === "ablation");
  }

  _refresh() {
    const active = this._active();
    const n      = Object.keys(this.dirty).length;
    const label  = n ? `${n} override${n === 1 ? "" : "s"}` : "all defaults";

    if (this.topCountEl) this.topCountEl.textContent = label;
    if (this.countEl)    this.countEl.textContent    = label;
    if (this.cmdEl)      this.cmdEl.textContent       = this._commandText();

    if (this.launchBtn) {
      this.launchBtn.disabled    = !active;
      this.launchBtn.textContent = active ? "Launch ablation study" : "Enable the ablation switch to launch";
    }

    if (this.builder) this.builder.refreshFromView();
    this._refreshBadges();
    this._refreshGates();
  }

  _commandText() {
    let text = (this.detail && this.detail.command) || "python main/train_backbone.py";
    Object.entries(this.dirty).forEach(([path, value]) => {
      const raw      = String(value);
      const rendered = /^[\w@%+=:,./-]+$/.test(raw) ? raw : `'${raw.replace(/'/g, `'\\''`)}'`;
      text += ` \\\n  --${path} ${rendered}`;
    });
    if (this.detach) text += ` \\\n  --detach`;
    return text;
  }

  _launch() {
    if (!this._active()) return;
    const interp = this.interpEl ? this.interpEl.value : this.detail.preferred;
    this.runConsole.launch(this.key, interp, "Ablation study", { ...this.dirty }, null, this.detach);
    window.router.go("console");
  }
}

window.AblationView = AblationView;
