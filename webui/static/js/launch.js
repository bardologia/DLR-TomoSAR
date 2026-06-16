"use strict";

class LaunchView {
  static PALETTE = ["#1d4fd8", "#0f766e", "#b45309", "#7c3aed", "#be185d", "#0e7490", "#4d7c0f", "#b91c1c"];

  static FOLLOW_INFER = new Set(["train_backbone", "train_jepa"]);

  static FIELD_TAXONOMY = [
    ["curve space", /curve|spectral|ssim/],
    ["param space", /^param/],
    ["regularization", /smooth|_tv$/],
    ["physics", /total_power|moments|coherence_resyn|covariance_match|capon_|^physics_|wavelength|slant_range|look_angle|baseline|kz_values/],
    ["schedule", /epoch|validation|scheduler|warmup|eta_min/],
    ["early stopping", /^early_stop/],
    ["image autoencoder", /image_autoencoder|image_ae_finetune/],
    ["profile autoencoder", /profile_autoencoder|target_provider|ema_decay|ae_finetune|^pixel_subsample$|keep_empty/],
    ["embedding", /embedding/],
    ["probe", /^probe_/],
    ["identifiers", /identifier|output_tag$|^dataset_type$/],
    ["data", /batch|worker|patch|stride|azimuth|dataset|^use_amp$|accumulation/],
    ["model", /model|gauss/],
    ["source", /fusar|track_selection|polarisation|^base_directory$/],
    ["run", /^run_|^gpu|^seed$|^device|^log|dir|path/],
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

  static TRAINING_TYPES = [
    ["backbone", "Backbone"],
    ["autoencoder", "Profile AE"],
    ["jepa", "JEPA"],
  ];

  static MODEL_KEY_TYPE = {
    train_backbone:             "backbone",
    train_profile_autoencoder:  "autoencoder",
    train_image_autoencoder:    "image_autoencoder",
    train_jepa:                 "jepa",
  };

  static JEPA_MEANINGS = {
    profile_only: {
      title  : "JEPA · backbone + profile autoencoder",
      label  : "Backbone + profile autoencoder",
      summary: "Predicts the profile embedding from the raw image stack, then decodes it back to normalized profiles through the profile autoencoder decoder. Select a profile autoencoder run; leave the image autoencoder unset.",
      flow   : [
        { kind: "input",  glyph: "stack",   label: "Image stack",          sub: "primary · secondaries · interferograms" },
        { kind: "model",  glyph: "net",     label: "Backbone",             sub: "predictor" },
        { kind: "latent", glyph: "vector",  label: "Normalized embeddings" },
        { kind: "model",  glyph: "decoder", label: "Profile AE decoder",   sub: "frozen / fine-tuned" },
        { kind: "output", glyph: "curve",   label: "Normalized profiles" },
      ],
    },
    image_only: {
      title  : "JEPA · image autoencoder + backbone",
      label  : "Image autoencoder + backbone",
      summary: "Encodes the image stack with the image autoencoder front-end, then the backbone maps that latent straight to the Gaussian-mixture profile parameters. Select an image autoencoder run; leave the profile autoencoder unset.",
      flow   : [
        { kind: "input",  glyph: "stack",   label: "Image stack",        sub: "primary · secondaries · interferograms" },
        { kind: "model",  glyph: "encoder", label: "Image AE encoder",   sub: "frozen / fine-tuned" },
        { kind: "latent", glyph: "vector",  label: "Image embedding",    sub: "2D latent" },
        { kind: "model",  glyph: "net",     label: "Backbone",           sub: "predictor" },
        { kind: "output", glyph: "params",  label: "Params array",       sub: "amp, μ, σ, amp, μ, σ, …" },
      ],
    },
    image_profile: {
      title  : "JEPA · image autoencoder + backbone + profile autoencoder",
      label  : "Image autoencoder + backbone + profile autoencoder",
      summary: "Encodes the image stack with the image autoencoder front-end, the backbone predicts the profile embedding from that latent, and the profile autoencoder decoder reconstructs the normalized profiles. Select both an image and a profile autoencoder run.",
      flow   : [
        { kind: "input",  glyph: "stack",   label: "Image stack",          sub: "primary · secondaries · interferograms" },
        { kind: "model",  glyph: "encoder", label: "Image AE encoder",     sub: "frozen / fine-tuned" },
        { kind: "latent", glyph: "vector",  label: "Image embedding",      sub: "2D latent" },
        { kind: "model",  glyph: "net",     label: "Backbone",             sub: "predictor" },
        { kind: "latent", glyph: "vector",  label: "Normalized embeddings" },
        { kind: "model",  glyph: "decoder", label: "Profile AE decoder",   sub: "frozen / fine-tuned" },
        { kind: "output", glyph: "curve",   label: "Normalized profiles" },
      ],
    },
  };

  static MODEL_MEANINGS = {
    backbone: {
      title  : "Backbone",
      summary: "Supervised regression that maps the SAR image stack straight to the Gaussian-mixture profile parameters.",
      flow   : [
        { kind: "input",  glyph: "stack",  label: "Image stack",  sub: "primary · secondaries · interferograms" },
        { kind: "model",  glyph: "net",    label: "Backbone",     sub: "supervised network" },
        { kind: "output", glyph: "params", label: "Params array", sub: "amp, μ, σ, amp, μ, σ, …" },
      ],
    },
    autoencoder: {
      title  : "Profile autoencoder",
      summary: "Learns the latent profile (output) space: encodes the fitted normalized profiles into embeddings and reconstructs them.",
      flow   : [
        { kind: "input",  glyph: "curve",   label: "Normalized profiles", sub: "fitted targets" },
        { kind: "model",  glyph: "encoder", label: "Encoder" },
        { kind: "latent", glyph: "vector",  label: "Normalized embeddings" },
        { kind: "model",  glyph: "decoder", label: "Decoder" },
        { kind: "output", glyph: "curve",   label: "Reconstructed profiles" },
      ],
    },
    image_autoencoder: {
      title  : "Image autoencoder",
      summary: "Learns the latent input space: encodes the SAR image stack into a 2D embedding and reconstructs it. The encoder later serves as a JEPA front-end.",
      flow   : [
        { kind: "input",  glyph: "stack",   label: "Image stack",            sub: "primary · secondaries · interferograms" },
        { kind: "model",  glyph: "encoder", label: "Encoder" },
        { kind: "latent", glyph: "vector",  label: "Image embedding",        sub: "2D latent" },
        { kind: "model",  glyph: "decoder", label: "Decoder" },
        { kind: "output", glyph: "stack",   label: "Reconstructed stack" },
      ],
    },
    jepa: LaunchView.JEPA_MEANINGS.profile_only,
  };

  static MODEL_COLORS = { input: "#1d4fd8", model: "#16191b", latent: "#a16207", output: "#0f766e", calc: "#7c3aed" };

  static MODEL_TAGS = { input: "input", model: "network", latent: "latent", output: "output", calc: "result" };

  static PROCESS_MEANINGS = {
    pre_process: {
      title  : "Pre-process",
      summary: "Ingests the raw F-SAR products, beamforms the full-stack Capon tomogram, and forms the interferometric image stack.",
      flow   : [
        { kind: "input", glyph: "stack", label: "Raw F-SAR products", sub: "single-look complex stack" },
        { kind: "model", glyph: "beam",  label: "Beamform", tag: "operation", sub: "Capon + interferograms" },
        [
          { kind: "output", glyph: "cube",  label: "Capon tomogram", sub: "full stack" },
          { kind: "output", glyph: "stack", label: "Image stack",    sub: "primary · secondaries · interferograms" },
        ],
      ],
    },
    extract_params: {
      title  : "Extract Parameters",
      summary: "Fits a per-pixel Gaussian mixture to the Capon tomogram to build the supervised profile-parameter targets.",
      flow   : [
        { kind: "input",  glyph: "cube",   label: "Capon tomogram",      sub: "reflectivity profiles" },
        { kind: "model",  glyph: "fit",    label: "Gaussian-mixture fit", tag: "operation", sub: "per pixel" },
        { kind: "output", glyph: "params", label: "Params array",        sub: "amp, μ, σ, amp, μ, σ, …" },
      ],
    },
    infer: {
      title  : "Inference",
      summary: "Runs a trained model over the image stack with a sliding window, stitches the predicted cube, and renders reports.",
      flow   : [
        { kind: "input",  glyph: "run",    label: "Trained run",      sub: "checkpoint + image stack" },
        { kind: "model",  glyph: "net",    label: "Predict",          sub: "sliding window" },
        { kind: "calc",   glyph: "cube",   label: "Stitched cube",    sub: "predicted profiles" },
        { kind: "output", glyph: "report", label: "Reports & figures", sub: "metrics · plots · animations" },
      ],
    },
  };

  static TYPE_TABS = {
    benchmark:      { field: "training_type", options: LaunchView.TRAINING_TYPES },
    cross_validate: { field: "training_type", options: LaunchView.TRAINING_TYPES },
    tune:           { field: "training_type", options: LaunchView.TRAINING_TYPES },
  };

  static EXPERIMENT_JEPA_CHOICES = {
    "jepa.profile_autoencoder_mode": ["frozen", "finetune"],
    "jepa.target_provider":          ["stopgrad", "ema", "live"],
  };

  static CHOICES = {
    train_jepa: {
      profile_autoencoder_mode: ["frozen", "finetune"],
      image_autoencoder_mode:   ["frozen", "finetune"],
      target_provider:          ["stopgrad", "ema", "live"],
    },
    benchmark:      LaunchView.EXPERIMENT_JEPA_CHOICES,
    cross_validate: LaunchView.EXPERIMENT_JEPA_CHOICES,
    tune:           LaunchView.EXPERIMENT_JEPA_CHOICES,
  };

  static DATASET_PICKERS = {
    "paths.dataset_path":    { mode: "datasets", multi: false, baseFromParent: true, validOnly: true },
    "paths.parameters_path": { mode: "params", datasetFrom: "paths.dataset_path" },
  };

  static EXPERIMENT_PICKERS = {
    ...LaunchView.DATASET_PICKERS,
    "jepa.profile_autoencoder_run": { mode: "runs", baseFrom: "jepa.profile_autoencoder_logdir", checkpointOnly: true },
  };

  static PICKERS = {
    extract_params: {
      dataset_filter: { mode: "datasets", multi: true, baseFrom: "dataset_base_path", validOnly: true },
    },
    train_backbone:             LaunchView.DATASET_PICKERS,
    train_profile_autoencoder:  LaunchView.DATASET_PICKERS,
    train_image_autoencoder:    LaunchView.DATASET_PICKERS,
    train_jepa:              {
      ...LaunchView.DATASET_PICKERS,
      profile_autoencoder_run: { mode: "runs", baseFrom: "profile_autoencoder_logdir", checkpointOnly: true },
      image_autoencoder_run:   { mode: "runs", baseFrom: "image_autoencoder_logdir",   checkpointOnly: true },
    },
    benchmark:         LaunchView.EXPERIMENT_PICKERS,
    cross_validate:    LaunchView.EXPERIMENT_PICKERS,
    tune:              LaunchView.EXPERIMENT_PICKERS,
    infer: {
      run_filter: { mode: "runs", multi: true, baseFrom: "logs_dir" },
    },
  };

  constructor(runConsole, project) {
    this.runConsole = runConsole;
    this.project = project;
    this.key = null;
    this.detail = null;
    this.config = null;
    this.dirty = {};
    this.controls = {};
    this.dependents = {};
    this.states = [];
    this.panels = new Map();
    this.bands = [];
    this.gates = [];
    this.gatedSections = new Set();
    this.classColors = new Map();
    this.query = "";
    this.modelFamilies = null;
    this.pinsEl = null;
    this.builder = null;
    this.detach = true;
    this.cmdEl = null;
    this.manifestEl = null;
    this.launchBtn = null;
    this.countEl = null;
    this.active = false;
    this.loadSeq = 0;
    this._wireTabs();
    this._wireKeys();
  }

  async show(key) {
    this.active = true;
    if (key === this.key && this.config) return;

    const seq = ++this.loadSeq;
    this.key = key;
    this.detail = null;
    this.config = null;
    this.dirty = {};
    this.controls = {};
    this.dependents = {};
    this.states = [];
    this.panels = new Map();
    this.bands = [];
    this.gates = [];
    this.gatedSections = new Set();
    this.classColors = new Map();
    this.query = "";
    this.modelFamilies = null;
    this.pinsEl = null;
    this.builder = null;
    this.detach = true;

    this._renderSkeleton();

    const detail = await window.apiGet(`/api/scripts/${key}`);
    if (seq !== this.loadSeq) return;
    if (detail.error) {
      window.toast("Could not load script", "error");
      window.router.go("scripts");
      return;
    }
    this.detail = detail;
    this._renderHead(detail);
    this._renderRail();
    this._renderSource(detail);
    this._setPane("config");

    const cfg = await window.apiGet(`/api/scripts/${key}/config`);
    if (seq !== this.loadSeq) return;
    if (!cfg.ok) {
      this._renderConfigError(cfg.error || "could not resolve configuration");
      return;
    }
    this.config = cfg;

    if (cfg.leaves.some((leaf) => leaf.path === "skip_models" || leaf.path === "backbone_name" || leaf.path === "ae_model_name")) {
      const models = await window.apiGet(this._modelFamiliesEndpoint());
      if (seq !== this.loadSeq) return;
      this.modelFamilies = (models && models.families) || [];
    }

    this._renderConfig(cfg);
    this._refresh();
  }

  _modelFamiliesEndpoint() {
    if (this.key === "train_image_autoencoder")   return "/api/image-autoencoders";
    if (this.key === "train_profile_autoencoder") return "/api/profile-autoencoders";
    return "/api/backbones";
  }

  leave() {
    this.active = false;
  }

  _renderSkeleton() {
    document.getElementById("launch-kicker").textContent = "";
    document.getElementById("launch-variants").innerHTML = "";
    document.getElementById("launch-title").textContent = "Loading...";
    document.getElementById("launch-purpose").textContent = "";
    document.getElementById("launch-facts").innerHTML = "";
    document.getElementById("launch-rail").innerHTML = "";

    const host = document.getElementById("launch-config");
    host.innerHTML = "";
    const wall = document.createElement("div");
    wall.className = "launch-skeleton";
    for (let i = 0; i < 4; i++) {
      const block = document.createElement("div");
      block.className = "launch-skeleton__panel";
      wall.appendChild(block);
    }
    host.appendChild(wall);
  }

  _renderHead(d) {
    this._renderVariants(d);
    document.getElementById("launch-kicker").textContent = d.group_title ? `${d.group_title} · ${d.file}` : `${d.category} · ${d.file}`;
    document.getElementById("launch-title").textContent = d.title;
    document.getElementById("launch-purpose").textContent = d.purpose;
    this._renderFacts();
  }

  _renderVariants(d) {
    const host = document.getElementById("launch-variants");
    host.innerHTML = "";

    const variants = d.variants || [];
    if (variants.length < 2) {
      host.hidden = true;
      return;
    }
    host.hidden = false;

    variants.forEach((v) => {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "launch__variant" + (v.key === d.key ? " is-active" : "");
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", String(v.key === d.key));
      tab.textContent = v.label;
      if (v.key !== d.key) tab.addEventListener("click", () => window.router.go(`launch/${v.key}`));
      host.appendChild(tab);
    });
  }

  _renderTypeTab(spec, leaf) {
    const host = document.getElementById("launch-variants");
    host.hidden = false;
    host.innerHTML = "";

    const paint = () => {
      const value = this._effective(leaf);
      [...host.children].forEach((child) => child.classList.toggle("is-active", child.dataset.value === value));
    };

    spec.options.forEach(([value, label]) => {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "launch__variant";
      tab.dataset.value = value;
      tab.setAttribute("role", "tab");
      tab.textContent = label;
      tab.addEventListener("click", () => {
        this._setValue(leaf, value);
        paint();
        this._paintModelCard(LaunchView.MODEL_MEANINGS[value]);
      });
      host.appendChild(tab);
    });

    paint();
  }

  _buildModelCard(meaning) {
    const card = document.createElement("section");
    card.className = "modelcard";
    card.id = "launch-model-card";
    this.modelCardEl = card;
    this._paintModelCard(meaning);
    return card;
  }

  _buildJepaModesCard() {
    const card = document.createElement("section");
    card.className = "modelcard";
    card.id = "launch-model-card";
    this.modelCardEl = card;

    const modes = [
      LaunchView.JEPA_MEANINGS.profile_only,
      LaunchView.JEPA_MEANINGS.image_only,
      LaunchView.JEPA_MEANINGS.image_profile,
    ];

    const rows = modes.map((mode) =>
      `<div class="modelmode">` +
      `<div class="modelmode__head"><span class="modelmode__label">${mode.label}</span>` +
      `<span class="modelmode__sub">${mode.summary}</span></div>` +
      `<div class="modelflow">${this._modelDiagram(mode.flow)}</div></div>`
    ).join("");

    card.innerHTML =
      `<div class="modelcard__head"><span class="modelcard__kicker">What JEPA can do</span>` +
      `<p class="modelcard__summary">JEPA trains in three modes, selected by which autoencoder runs you provide. All three possibilities are shown below.</p></div>` +
      `<div class="modelmodes">${rows}</div>`;

    return card;
  }

  _paintModelCard(meaning) {
    const card = this.modelCardEl;
    if (!card) return;
    if (!meaning) {
      card.hidden = true;
      return;
    }
    card.hidden = false;

    card.innerHTML =
      `<div class="modelcard__head"><span class="modelcard__kicker">What ${meaning.title} does</span>` +
      `<p class="modelcard__summary">${meaning.summary}</p></div>` +
      `<div class="modelflow">${this._modelDiagram(meaning.flow)}</div>`;
  }

  _modelDiagram(flow) {
    const SLOT = 200, PAD = 26, BH = 164, GH = 44, TOP = 14;
    const columns = flow.map((col) => (Array.isArray(col) ? col : [col]));
    const maxM    = columns.reduce((m, col) => Math.max(m, col.length), 1);
    const width   = columns.length * SLOT + PAD * 2;
    const height  = TOP * 2 + maxM * BH;

    const laid = columns.map((col, i) => {
      const cx       = PAD + i * SLOT + SLOT / 2;
      const groupTop = TOP + (maxM * BH - col.length * BH) / 2;
      return col.map((node, j) => ({ node, cx, cy: groupTop + j * BH + GH + 8 }));
    });

    let links = "";
    for (let i = 1; i < laid.length; i++) {
      laid[i - 1].forEach((src) => laid[i].forEach((tgt) => {
        const x1 = src.cx + 54, x2 = tgt.cx - 54, mx = (x1 + x2) / 2;
        links += `<path class="mflow-link" d="M${x1} ${src.cy} C ${mx} ${src.cy}, ${mx} ${tgt.cy}, ${x2} ${tgt.cy}" fill="none" marker-end="url(#mflow-arrow)"/>`;
      }));
    }

    let nodes = "";
    laid.flat().forEach(({ node, cx, cy }) => {
      const color = LaunchView.MODEL_COLORS[node.kind] || "#16191b";
      nodes += this._glyph(node.glyph, cx, cy, color) + this._caption(node, cx, cy, color, SLOT);
    });

    return `<svg class="mflow-svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="data flow">` +
      `<defs><marker id="mflow-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">` +
      `<path d="M0 0L10 5L0 10z" fill="#9aa196"/></marker></defs>` +
      links + nodes + `</svg>`;
  }

  _caption(node, cx, cy, color, slot) {
    const tag = node.tag || LaunchView.MODEL_TAGS[node.kind] || node.kind;
    const sub = node.sub ? `<span class="mflow-sub">${node.sub}</span>` : "";
    const w   = slot - 28;
    return `<foreignObject x="${cx - w / 2}" y="${cy + 50}" width="${w}" height="64">` +
      `<div xmlns="http://www.w3.org/1999/xhtml" class="mflow-cap">` +
      `<span class="mflow-tag" style="color:${color}">${tag}</span>` +
      `<span class="mflow-name">${node.label}</span>${sub}</div></foreignObject>`;
  }

  _glyph(glyph, cx, cy, color) {
    const tint = color + "14";

    if (glyph === "stack") {
      const w = 52, h = 58, x = cx - w / 2, y = cy - h / 2;
      let rows = "";
      [18, 32, 46].forEach((dy) => { rows += `<line x1="${x + 10}" y1="${y + dy}" x2="${x + w - 10}" y2="${y + dy}" stroke="${color}" stroke-width="1" opacity="0.45"/>`; });
      return `<g fill="${tint}" stroke="${color}" stroke-width="1.6">` +
        `<rect x="${x + 14}" y="${y - 14}" width="${w}" height="${h}" rx="4" opacity="0.4"/>` +
        `<rect x="${x + 7}"  y="${y - 7}"  width="${w}" height="${h}" rx="4" opacity="0.7"/>` +
        `<rect x="${x}"      y="${y}"      width="${w}" height="${h}" rx="4"/>${rows}</g>`;
    }

    if (glyph === "net") {
      const s = 84, x = cx - s / 2, y = cy - s / 2, lx = cx - 18, rx = cx + 18, rows = [cy - 22, cy, cy + 22];
      let lines = "", dots = "";
      rows.forEach((ly) => rows.forEach((ry) => { lines += `<line x1="${lx}" y1="${ly}" x2="${rx}" y2="${ry}" stroke="#fff" stroke-width="0.8" opacity="0.32"/>`; }));
      rows.forEach((ry) => { dots += `<circle cx="${lx}" cy="${ry}" r="4" fill="#fff"/><circle cx="${rx}" cy="${ry}" r="4" fill="#fff"/>`; });
      return `<g><rect x="${x}" y="${y}" width="${s}" height="${s}" rx="11" fill="${color}"/>${lines}${dots}</g>`;
    }

    if (glyph === "encoder" || glyph === "decoder") {
      const hw = 36, lh = glyph === "encoder" ? 44 : 18, rh = glyph === "encoder" ? 18 : 44;
      const pts = `${cx - hw},${cy - lh} ${cx + hw},${cy - rh} ${cx + hw},${cy + rh} ${cx - hw},${cy + lh}`;
      let dots = "";
      [-14, 0, 14].forEach((dx) => { dots += `<circle cx="${cx + dx}" cy="${cy}" r="3.4" fill="#fff" opacity="0.85"/>`; });
      return `<g><polygon points="${pts}" fill="${color}"/>${dots}</g>`;
    }

    if (glyph === "vector") {
      const w = 32, h = 88, x = cx - w / 2, y = cy - h / 2, cells = 5, ch = h / cells;
      let segs = "";
      for (let k = 1; k < cells; k++) segs += `<line x1="${x}" y1="${y + ch * k}" x2="${x + w}" y2="${y + ch * k}" stroke="${color}" stroke-width="1" opacity="0.5"/>`;
      return `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${tint}" stroke="${color}" stroke-width="1.6"/>${segs}</g>`;
    }

    if (glyph === "params") {
      const cw = 15, h = 34, y = cy - h / 2, xs = [0, 18, 36, 61, 79, 97].map((o) => cx - 56 + o), op = [1, 0.7, 0.45, 1, 0.7, 0.45];
      let cells = "";
      xs.forEach((x, k) => { cells += `<rect x="${x}" y="${y}" width="${cw}" height="${h}" rx="3" fill="${color}" opacity="${op[k]}"/>`; });
      cells += `<text x="${cx + 64}" y="${cy + 6}" font-family="JetBrains Mono, monospace" font-size="18" fill="${color}">…</text>`;
      return `<g>${cells}</g>`;
    }

    if (glyph === "curve") {
      const ax = cx - 38, ay0 = cy - 32, ay1 = cy + 28, axr = cx + 38;
      const axis  = `<path d="M${ax} ${ay0} L${ax} ${ay1} L${axr} ${ay1}" fill="none" stroke="${color}" stroke-width="1.2" opacity="0.4"/>`;
      const curve = `<path d="M${ax + 4} ${ay1 - 2} C ${cx - 16} ${ay1 - 2}, ${cx - 11} ${cy - 30}, ${cx} ${cy - 30} C ${cx + 11} ${cy - 30}, ${cx + 16} ${ay1 - 2}, ${axr - 4} ${ay1 - 2} Z" fill="${tint}" stroke="${color}" stroke-width="2"/>`;
      return `<g>${axis}${curve}</g>`;
    }

    if (glyph === "cube") {
      const u = 26;
      const top   = `<polygon points="${cx},${cy - 28} ${cx + u},${cy - 13} ${cx},${cy + 2} ${cx - u},${cy - 13}" fill="${color}" opacity="0.20"/>`;
      const left  = `<polygon points="${cx - u},${cy - 13} ${cx},${cy + 2} ${cx},${cy + 32} ${cx - u},${cy + 17}" fill="${color}" opacity="0.11"/>`;
      const right = `<polygon points="${cx + u},${cy - 13} ${cx},${cy + 2} ${cx},${cy + 32} ${cx + u},${cy + 17}" fill="${color}" opacity="0.05"/>`;
      return `<g stroke="${color}" stroke-width="1.5" stroke-linejoin="round">${top}${left}${right}</g>`;
    }

    if (glyph === "beam") {
      const s = 84;
      let arcs = "";
      [16, 28, 40].forEach((r, k) => { arcs += `<path d="M${cx - 24} ${cy - r} A ${r} ${r} 0 0 1 ${cx - 24} ${cy + r}" fill="none" stroke="#fff" stroke-width="1.6" opacity="${0.5 - k * 0.12}"/>`; });
      return `<g><rect x="${cx - s / 2}" y="${cy - s / 2}" width="${s}" height="${s}" rx="11" fill="${color}"/>${arcs}<circle cx="${cx - 24}" cy="${cy}" r="3.2" fill="#fff"/></g>`;
    }

    if (glyph === "fit") {
      const s = 84;
      const curve = `<path d="M${cx - 28} ${cy + 22} C ${cx - 12} ${cy + 22}, ${cx - 8} ${cy - 20}, ${cx} ${cy - 20} C ${cx + 8} ${cy - 20}, ${cx + 12} ${cy + 22}, ${cx + 28} ${cy + 22}" fill="none" stroke="#fff" stroke-width="1.8" opacity="0.9"/>`;
      let dots = "";
      [[-16, 8], [-4, -12], [9, -7], [19, 12]].forEach(([dx, dy]) => { dots += `<circle cx="${cx + dx}" cy="${cy + dy}" r="2.2" fill="#fff" opacity="0.85"/>`; });
      return `<g><rect x="${cx - s / 2}" y="${cy - s / 2}" width="${s}" height="${s}" rx="11" fill="${color}"/>${curve}${dots}</g>`;
    }

    if (glyph === "run") {
      const s = 78, x = cx - s / 2, y = cy - s / 2, lx = cx - 16, rx = cx + 16, rows = [cy - 18, cy, cy + 18];
      let lines = "", dots = "";
      rows.forEach((ly) => rows.forEach((ry) => { lines += `<line x1="${lx}" y1="${ly}" x2="${rx}" y2="${ry}" stroke="${color}" stroke-width="0.8" opacity="0.3"/>`; }));
      rows.forEach((ry) => { dots += `<circle cx="${lx}" cy="${ry}" r="3.4" fill="${color}"/><circle cx="${rx}" cy="${ry}" r="3.4" fill="${color}"/>`; });
      const badge = `<circle cx="${cx + 28}" cy="${cy - 28}" r="9" fill="${color}"/><path d="M${cx + 23.5} ${cy - 28} l3 3 l5 -6" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>`;
      return `<g><rect x="${x}" y="${y}" width="${s}" height="${s}" rx="10" fill="${tint}" stroke="${color}" stroke-width="1.6"/>${lines}${dots}${badge}</g>`;
    }

    if (glyph === "report") {
      const w = 52, h = 66, x = cx - w / 2, y = cy - h / 2;
      let lines = "";
      [-18, -10, -2].forEach((dy) => { lines += `<line x1="${cx - 16}" y1="${cy + dy}" x2="${cx + 16}" y2="${cy + dy}" stroke="${color}" stroke-width="1.3" opacity="0.55"/>`; });
      let bars = "";
      [[-16, 14], [-4, 22], [8, 11]].forEach(([dx, bh]) => { bars += `<rect x="${cx + dx}" y="${cy + 26 - bh}" width="7" height="${bh}" rx="1.5" fill="${color}" opacity="0.85"/>`; });
      return `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${tint}" stroke="${color}" stroke-width="1.6"/>${lines}${bars}</g>`;
    }

    const r = 30;
    return `<rect x="${cx - r}" y="${cy - r}" width="${r * 2}" height="${r * 2}" rx="6" fill="${tint}" stroke="${color}" stroke-width="1.6"/>`;
  }

  _renderFacts() {
    const host = document.getElementById("launch-facts");
    host.innerHTML = "";
    const facts = [["entry config", this.detail.config_class || "none"]];
    if (this.config) {
      facts.push(["fields", String(this.config.leaves.length)]);
      facts.push(["sections", String(this.panels.size)]);
    }
    facts.forEach(([term, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      host.appendChild(dt);
      host.appendChild(dd);
    });
  }

  _renderRail() {
    const host = document.getElementById("launch-rail");
    host.innerHTML = "";

    const interp = document.createElement("div");
    interp.className = "rail-block";
    interp.innerHTML = `<span class="rail-block__label">Interpreter</span>`;
    const select = document.createElement("select");
    select.className = "run-select";
    select.id = "launch-interpreter";
    (this.project.interpreters || []).forEach((it) => {
      const opt = document.createElement("option");
      opt.value = it.path;
      opt.textContent = `${it.label}  ·  ${it.path}`;
      if (it.path === this.detail.preferred) opt.selected = true;
      select.appendChild(opt);
    });
    interp.appendChild(select);

    let follow = null;
    let followSelect = null;
    if (this.detail && LaunchView.FOLLOW_INFER.has(this.detail.key)) {
      follow = document.createElement("div");
      follow.className = "rail-block";
      follow.innerHTML = `<span class="rail-block__label">After run</span>`;
      const fsel = document.createElement("select");
      fsel.className = "run-select";
      fsel.id = "launch-follow";
      [["", "nothing"], ["infer", "Inference"]].forEach(([value, label]) => {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = label;
        fsel.appendChild(opt);
      });
      follow.appendChild(fsel);
      followSelect = fsel;
    }

    const detach = document.createElement("div");
    detach.className = "rail-block";
    detach.innerHTML = `<span class="rail-block__label">Execution</span>`;

    const detachRow = document.createElement("div");
    detachRow.className = "rail-detach";
    const detachToggle = document.createElement("button");
    detachToggle.type = "button";
    detachToggle.className = "switch";
    detachToggle.setAttribute("role", "switch");
    detachToggle.innerHTML = `<span class="switch__knob"></span>`;
    const detachLabel = document.createElement("span");
    detachLabel.className = "rail-detach__label";
    detachLabel.textContent = "Detach from server";

    const hint = document.createElement("p");
    hint.className = "rail-detach__hint";

    const paintDetach = () => {
      detachToggle.classList.toggle("is-on", this.detach);
      detachToggle.setAttribute("aria-checked", String(this.detach));
      hint.textContent = this.detach
        ? "The run survives a lost connection or a console restart. Output goes to logs/<script>_<stamp>.out in the repo."
        : "Output streams to this console. The run dies if the console server goes down.";
      if (followSelect) {
        followSelect.disabled = this.detach;
        followSelect.title = this.detach ? "unavailable for detached runs" : "";
        if (this.detach) followSelect.value = "";
      }
      this._refresh();
    };
    detachToggle.addEventListener("click", () => {
      this.detach = !this.detach;
      paintDetach();
    });

    detachRow.appendChild(detachToggle);
    detachRow.appendChild(detachLabel);
    detach.appendChild(detachRow);
    detach.appendChild(hint);
    paintDetach();

    const cmd = document.createElement("div");
    cmd.className = "rail-block";
    const cmdHead = document.createElement("div");
    cmdHead.className = "rail-block__row";
    cmdHead.innerHTML = `<span class="rail-block__label">Command</span>`;
    const copy = document.createElement("button");
    copy.className = "btn btn--mini";
    copy.textContent = "Copy";
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(this._commandText()).then(() => window.toast("Command copied", "ok"));
    });
    cmdHead.appendChild(copy);
    const code = document.createElement("code");
    code.className = "rail-command";
    this.cmdEl = code;
    cmd.appendChild(cmdHead);
    cmd.appendChild(code);

    const manifest = document.createElement("div");
    manifest.className = "rail-block";
    manifest.innerHTML = `<span class="rail-block__label">Overrides</span>`;
    const list = document.createElement("div");
    list.className = "rail-manifest";
    this.manifestEl = list;
    manifest.appendChild(list);

    const actions = document.createElement("div");
    actions.className = "rail-block rail-block--actions";
    const launch = document.createElement("button");
    launch.className = "btn btn--primary rail-launch";
    launch.addEventListener("click", () => this._launch());
    this.launchBtn = launch;
    actions.appendChild(launch);

    host.appendChild(interp);
    if (follow) host.appendChild(follow);
    host.appendChild(detach);
    host.appendChild(cmd);
    host.appendChild(manifest);
    host.appendChild(actions);
    this._refresh();
  }

  _renderConfigError(message) {
    const host = document.getElementById("launch-config");
    host.innerHTML = "";
    const panel = document.createElement("div");
    panel.className = "launch-error";
    const text = document.createElement("pre");
    text.className = "launch-error__text";
    text.textContent = message;
    const retry = document.createElement("button");
    retry.className = "btn btn--ghost";
    retry.textContent = "Retry resolution";
    retry.addEventListener("click", () => this.show(this.key));
    panel.appendChild(text);
    panel.appendChild(retry);
    host.appendChild(panel);
  }

  _renderConfig(cfg) {
    const host = document.getElementById("launch-config");
    host.innerHTML = "";

    if (!cfg.leaves.length) {
      host.innerHTML = `<p class="cfg-note">${cfg.config_class} exposes no configuration fields.</p>`;
      return;
    }

    const byPath  = new Map(cfg.leaves.map((leaf) => [leaf.path, leaf]));

    const typeTab  = LaunchView.TYPE_TABS[this.key];
    const typeLeaf = typeTab ? byPath.get(typeTab.field) : null;

    const modelType = LaunchView.MODEL_KEY_TYPE[this.key] || (typeLeaf ? this._effective(typeLeaf) : null);
    const meaning   = (modelType && LaunchView.MODEL_MEANINGS[modelType]) || LaunchView.PROCESS_MEANINGS[this.key] || null;

    if (this.key === "train_jepa") host.appendChild(this._buildJepaModesCard());
    else if (meaning)              host.appendChild(this._buildModelCard(meaning));

    host.appendChild(this._buildToolbar(cfg));

    if (typeLeaf) this._renderTypeTab(typeTab, typeLeaf);

    const modelNameLeaf = byPath.get("backbone_name") || byPath.get("ae_model_name");
    const cardPanel     = modelNameLeaf && this.modelFamilies && this.modelFamilies.length ? new window.ModelCardPanel(this, modelNameLeaf) : null;

    const pinned  = (this.detail.essentials || []).map((path) => byPath.get(path)).filter(Boolean).filter((leaf) => !(cardPanel && modelNameLeaf && leaf.path === modelNameLeaf.path)).filter((leaf) => !(typeLeaf && leaf.path === typeLeaf.path));
    const claimed = new Set(pinned.map((leaf) => leaf.path));
    if (cardPanel) claimed.add(modelNameLeaf.path);
    if (typeLeaf) claimed.add(typeLeaf.path);

    const modelLeaf  = byPath.get("skip_models");
    const modelPanel = modelLeaf && this.modelFamilies && this.modelFamilies.length ? new window.ModelTogglePanel(this, modelLeaf) : null;
    if (modelPanel) claimed.add("skip_models");

    if (byPath.get("trials_enabled") && byPath.get("warmup_losses") && byPath.get("complete_losses")) {
      const candidate = new window.ExperimentBuilder(this, byPath);
      if (candidate.terms.length) {
        this.builder = candidate;
        candidate.claimed.forEach((path) => claimed.add(path));
      }
    }

    if (pinned.length) host.appendChild(this._buildPins(pinned));
    if (cardPanel) host.appendChild(cardPanel.build());
    if (this.builder) host.appendChild(this.builder.build());
    if (modelPanel) host.appendChild(modelPanel.build());

    const grouped = new Map();
    cfg.leaves.forEach((leaf) => {
      if (claimed.has(leaf.path)) return;
      if (!grouped.has(leaf.section)) grouped.set(leaf.section, []);
      grouped.get(leaf.section).push(leaf);
    });

    const sections = [...grouped.keys()];
    const nonRoot = sections.filter((s) => s !== "");
    const promote = nonRoot.length > 1 && new Set(nonRoot.map((s) => s.split(".")[0])).size === 1;
    const bandKey = (s) => (s === "" ? "run" : s.split(".").slice(0, promote ? 2 : 1).join("."));

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
    host.appendChild(empty);

    this._renderFacts();
  }

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
      this._applyVisibility();
    });

    const count = document.createElement("span");
    count.className = "cfg-toolbar__count";
    this.countEl = count;

    const reset = document.createElement("button");
    reset.className = "btn btn--mini";
    reset.textContent = "Reset all";
    reset.addEventListener("click", () => this._resetAll());

    bar.appendChild(search);
    bar.appendChild(count);
    bar.appendChild(reset);
    return bar;
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

  _classColor(sectionClass) {
    if (!this.classColors.has(sectionClass)) {
      this.classColors.set(sectionClass, LaunchView.PALETTE[this.classColors.size % LaunchView.PALETTE.length]);
    }
    return this.classColors.get(sectionClass);
  }

  _buildBand(key, bandSections, grouped) {
    const ordered = [...bandSections].sort((a, b) => a.split(".").length - b.split(".").length);
    const nFields = bandSections.reduce((n, s) => n + grouped.get(s).length, 0);
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

  _buildSubPanel(section, leaves) {
    const label = section.split(".").pop();

    const el = document.createElement("section");
    el.className = "sub-panel";
    el.dataset.section = section;
    el.title = section;
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
      `<span class="sub-panel__count">${leaves.length} fields</span>`;
    head.addEventListener("click", () => {
      const open = el.classList.toggle("is-open");
      head.setAttribute("aria-expanded", String(open));
    });

    const body = this._buildFieldsGrid(section, leaves, "sub-panel__body");

    const children = document.createElement("div");
    children.className = "band-children band-children--nested";

    el.appendChild(head);
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
      const order  = LaunchView.FIELD_TAXONOMY.map(([title]) => title);
      const counts = new Map();
      names.forEach((name) => {
        const rule = LaunchView.FIELD_TAXONOMY.find(([, pattern]) => pattern.test(name));
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
      LaunchView.STACK_PAIRS.forEach(([top, bottom]) => {
        if (named.get(top) && named.get(bottom)) stackedBelow.add(bottom);
      });

      named.forEach((el, title) => {
        if (el === null || stackedBelow.has(title)) return;

        const pair = LaunchView.STACK_PAIRS.find(([top, bottom]) => top === title && stackedBelow.has(bottom));
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

    if (termCount >= 2) {
      const buckets = { curve: [], param: [], reg: [], general: [] };
      blocks.forEach((blockLeaves) => {
        if (!isTermBlock(blockLeaves)) {
          if (shortName(blockLeaves[0]).startsWith("param")) buckets.param.push(blockLeaves);
          else buckets.general.push(blockLeaves);
          return;
        }
        const label = shortName(blockLeaves[0]).slice(4);
        if (label.startsWith("param")) buckets.param.push(blockLeaves);
        else if (/curve|spectral|ssim/.test(label)) buckets.curve.push(blockLeaves);
        else buckets.reg.push(blockLeaves);
      });

      grid.classList.add("is-grouped");
      const named = new Map();
      named.set("curve space", makeGroup("curve space", buckets.curve));
      named.set("param space", makeGroup("param space", buckets.param));
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
        const rule = LaunchView.FIELD_TAXONOMY.find(([, pattern]) => pattern.test(shortName(leaf)));
        const title = rule ? rule[0] : "general";
        if (!classified.has(title)) classified.set(title, []);
        classified.get(title).push(leaf);
      });

      if (classified.size >= 2 || gateBlocks.length) {
        grid.classList.add("is-grouped");
        const named = new Map();
        [...LaunchView.FIELD_TAXONOMY.map(([title]) => title), "general"].forEach((title) => {
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
    const row = this._buildGateRow(lead, weight, label);
    body.appendChild(row);

    const gatedRows = rest.map((leaf) => {
      const dependent = body.appendChild(this._buildRow(leaf, section));
      dependent.classList.add("cfg-edit__row--dependent");
      return this.states[this.states.length - 1];
    });

    this.gates.push({ leaf: lead, states: gatedRows, sections: [], weightEl: row.querySelector(".term-weight") });
  }

  _buildGateRow(lead, weight, label) {
    const row = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title = weight ? `--${lead.path} · --${weight.path}` : `--${lead.path}`;

    const name = document.createElement("div");
    name.className = "cfg-edit__name";
    name.textContent = label;
    row.appendChild(name);

    const cell = document.createElement("div");
    cell.className = "term-control";

    const toggle = this._switchControl(lead);
    cell.appendChild(toggle.el);
    this.controls[lead.path] = { leaf: lead, reset: toggle.reset };
    this.states.push({ leaf: lead, row, section: lead.section });

    if (weight) {
      const wrap = document.createElement("div");
      wrap.className = "term-weight";
      const tag = document.createElement("span");
      tag.className = "term-weight__tag";
      tag.textContent = "weight";
      const control = this._numberControl(weight);
      control.input.title = `--${weight.path}`;
      wrap.appendChild(tag);
      wrap.appendChild(control.input);
      cell.appendChild(wrap);
      this.controls[weight.path] = { leaf: weight, reset: control.reset };
      this.states.push({ leaf: weight, row, section: weight.section });
    }

    row.appendChild(cell);
    return row;
  }

  _buildRow(leaf, section, pinned = false) {
    const short = section ? leaf.path.slice(section.length + 1) : leaf.path;

    const row = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title = `--${leaf.path}`;

    const label = document.createElement("div");
    label.className = "cfg-edit__name";
    label.innerHTML = `${short}<span>${leaf.type}</span>`;
    row.appendChild(label);

    let control;
    const pickerSpec = leaf.editable ? this._pickerSpec(leaf) : null;
    const choices    = leaf.editable ? this._choicesFor(leaf) : null;
    if (pickerSpec && window.DatasetPicker) {
      control = new window.DatasetPicker(this, leaf, pickerSpec).build();
      row.classList.add(pickerSpec.multi ? "cfg-edit__row--board" : "cfg-edit__row--wide");
    } else if (choices) {
      control = this._choiceControl(leaf, choices);
    } else if (!leaf.editable) {
      control = this._textControl(leaf);
      control.input.disabled = true;
      control.input.classList.add("is-locked");
      control.input.title = "not overridable from the command line";
    } else if (leaf.type === "bool") {
      control = this._switchControl(leaf);
    } else if (leaf.type === "int" || leaf.type === "float") {
      control = this._numberControl(leaf);
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
        this.gates.push({ leaf: lead, states, sections, weightEl: null });
      }
    });
  }

  _effective(leaf) {
    return this.dirty[leaf.path] !== undefined ? this.dirty[leaf.path] : leaf.value;
  }

  _leafByPath(path) {
    return this.config ? this.config.leaves.find((leaf) => leaf.path === path) : null;
  }

  _pickerSpec(leaf) {
    const specs = LaunchView.PICKERS[this.key];
    return specs ? specs[leaf.path] : null;
  }

  _choicesFor(leaf) {
    const map = LaunchView.CHOICES[this.key];
    return map ? map[leaf.path] || null : null;
  }

  _choiceControl(leaf, choices) {
    const select = document.createElement("select");
    select.className = "cfg-edit__input picker__select";

    const current = String(leaf.value);
    const options = choices.includes(current) ? choices : [current, ...choices];
    options.forEach((value) => {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value;
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
      if (gate.weightEl) gate.weightEl.hidden = !open;
      if (!open) {
        gate.states.forEach(({ row }) => (row.dataset.gated = "1"));
        gate.sections.forEach((section) => this.gatedSections.add(section));
      }
    });
    this._applyVisibility();
  }

  _applyVisibility() {
    const rowVisible = new Map();
    this.states.forEach(({ leaf, row }) => {
      const hit = !this.query || leaf.path.toLowerCase().includes(this.query);
      if (hit && row.dataset.gated !== "1") rowVisible.set(row, true);
      else if (!rowVisible.has(row)) rowVisible.set(row, false);
    });
    rowVisible.forEach((visible, row) => (row.hidden = !visible));

    const sectionHasRows = new Map();
    this.states.forEach(({ leaf, row }) => {
      const prior = sectionHasRows.get(leaf.section) || false;
      sectionHasRows.set(leaf.section, prior || rowVisible.get(row));
    });

    const orderedSections = [...this.panels.keys()].sort((a, b) => b.split(".").length - a.split(".").length);
    const sectionVisible = new Map();
    orderedSections.forEach((section) => {
      const childVisible = orderedSections.some((other) => other !== section && other.startsWith(section ? `${section}.` : ".") && sectionVisible.get(other));
      const visible = !this.gatedSections.has(section) && ((sectionHasRows.get(section) || false) || childVisible);
      sectionVisible.set(section, visible);
      const el = this.panels.get(section).el;
      el.hidden = !visible;
      if (this.query && visible && el.classList.contains("sub-panel")) el.classList.add("is-open");
    });

    let anyVisible = false;
    this.bands.forEach((band) => {
      const visible = band.sections.some((section) => sectionVisible.get(section));
      band.el.hidden = !visible;
      anyVisible = anyVisible || visible;
      if (this.query && visible && !band.el.classList.contains("is-open")) this._toggleBand(band.el, band.head);
    });

    if (this.pinsEl) {
      const rows = [...this.pinsEl.querySelectorAll(".cfg-edit__row")];
      this.pinsEl.hidden = rows.length > 0 && rows.every((row) => row.hidden);
      anyVisible = anyVisible || !this.pinsEl.hidden;
    }

    const none = document.querySelector(".launch-nomatch");
    if (none) none.hidden = anyVisible;
  }

  _commandText() {
    if (!this.detail) return "";
    let text = this.detail.command;
    Object.entries(this.dirty).forEach(([path, value]) => {
      const rendered = /^[\w@%+=:,./-]+$/.test(value) ? value : `'${String(value).replace(/'/g, `'\\''`)}'`;
      text += ` --${path} ${rendered}`;
    });
    if (this.detach) text += " --detach";
    return text;
  }

  _refresh() {
    const n = Object.keys(this.dirty).length;

    if (this.cmdEl) this.cmdEl.textContent = this._commandText();
    if (this.countEl) this.countEl.textContent = n ? `${n} override${n > 1 ? "s" : ""}` : "all defaults";

    if (this.launchBtn) {
      this.launchBtn.classList.toggle("is-armed", n > 0);
      this.launchBtn.innerHTML = n
        ? `&#9654;&nbsp; Launch run <small>${n} override${n > 1 ? "s" : ""}</small>`
        : `&#9654;&nbsp; Launch run <small>all defaults</small>`;
    }

    if (this.manifestEl) this._renderManifest();
    if (this.builder) this.builder.refreshFromView();
    this._refreshBadges();
    this._refreshGates();
  }

  _renderManifest() {
    this.manifestEl.innerHTML = "";
    const entries = Object.entries(this.dirty);

    if (!entries.length) {
      const empty = document.createElement("p");
      empty.className = "rail-manifest__empty";
      empty.textContent = "No overrides. Every field launches at its default.";
      this.manifestEl.appendChild(empty);
      return;
    }

    entries.forEach(([path, value]) => {
      const control = this.controls[path];
      const item = document.createElement("button");
      item.type = "button";
      item.className = "rail-override";
      item.title = "Remove override";
      const from = control ? control.leaf.value : "";
      item.innerHTML =
        `<span class="rail-override__path">${path}</span>` +
        `<span class="rail-override__change">${from} &rarr; <b>${value}</b></span>` +
        `<span class="rail-override__x" aria-hidden="true">&times;</span>`;
      item.addEventListener("click", () => this._resetField(path));
      this.manifestEl.appendChild(item);
    });
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

  _launch() {
    if (!this.detail) return;
    const interp = document.getElementById("launch-interpreter").value;
    const followEl = document.getElementById("launch-follow");
    const follow = this.detach ? "" : (followEl ? followEl.value : "");
    this.runConsole.launch(this.detail.key, interp, this.detail.title, { ...this.dirty }, follow, this.detach);
  }

  _renderSource(d) {
    const code = document.getElementById("launch-source-code");
    code.textContent = d.source;
    code.className = "language-python";
    if (window.hljs) {
      try {
        window.hljs.highlightElement(code);
      } catch (e) {}
    }
  }

  _setPane(pane) {
    document.querySelectorAll(".launch__tab").forEach((t) => {
      const active = t.dataset.pane === pane;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", String(active));
    });
    document.getElementById("launch-config").classList.toggle("is-active", pane === "config");
    document.getElementById("launch-source").classList.toggle("is-active", pane === "source");
  }

  _wireTabs() {
    document.querySelectorAll(".launch__tab").forEach((t) => {
      t.addEventListener("click", () => this._setPane(t.dataset.pane));
    });
  }

  _wireKeys() {
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape" || !this.active) return;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
        document.activeElement.blur();
        return;
      }
      window.router.go("scripts");
    });
  }
}

window.LaunchView = LaunchView;
