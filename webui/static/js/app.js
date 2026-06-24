"use strict";

window.apiGet = async function (url) {
  const res = await fetch(url);
  if (!res.ok && res.status >= 500) return { error: `server ${res.status}` };
  return res.json();
};

window.apiPost = async function (url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
};

let _toastTimer = null;
window.toast = function (message, kind) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.className = "toast is-show" + (kind ? ` is-${kind}` : "");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => {
    el.className = "toast";
  }, 3200);
};

class RevealObserver {
  constructor() {
    this.io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("is-in");
            this.io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -6% 0px" }
    );
  }

  scan() {
    document.querySelectorAll(".reveal:not(.is-in)").forEach((el) => this.io.observe(el));
  }
}

class App {
  constructor() {
    this.project = null;
    this.scenes = [];
    this.reveal = new RevealObserver();
    window.revealScan = () => this.reveal.scan();
  }

  async init() {
    this._initScenes();
    await this._loadProject();
    this._initComponents();
    this._initRouter();
    this.reveal.scan();
  }

  _initScenes() {
    const radar = document.getElementById("radar");
    const server = document.getElementById("server-anim");
    try {
      if (radar && window.TomoScene) this.scenes.push(new window.TomoScene(radar));
    } catch (e) {}
    try {
      if (server && window.ServerScene) {
        window.serverScene = new window.ServerScene(server);
        this.scenes.push(window.serverScene);
      }
    } catch (e) {}
  }

  async _loadProject() {
    try {
      this.project = await window.apiGet("/api/project");
      this._setStatus(true);
    } catch (e) {
      this.project = { interpreters: [], counts: {} };
      this._setStatus(false);
      return;
    }
    this._initStatus();
  }

  _setStatus(ok) {
    const wrap = document.getElementById("nav-status");
    const text = document.getElementById("status-text");
    wrap.classList.toggle("is-ok", ok);
    wrap.classList.toggle("is-down", !ok);
    text.textContent = ok ? "backend live" : "offline";
  }

  _initStatus() {
    const footRoot = document.getElementById("footer-root");
    if (footRoot) footRoot.textContent = this.project.repo_root || "";

    this.statusBoard = new window.StatusBoard({
      host: document.getElementById("status-host"),
      sum: document.getElementById("status-sum"),
      board: document.getElementById("status-board"),
    });
    this.statusBoard.start();
  }

  _initComponents() {
    this.runConsole = new window.RunConsole({
      list: document.getElementById("job-list"),
      tiles: document.getElementById("console-tiles"),
      hint: document.getElementById("console-hint"),
    });
    this.runConsole.refresh();

    this.scriptPanel = new window.ScriptPanel({
      grid: document.getElementById("script-grid"),
      filters: document.getElementById("script-filters"),
    });
    this.scriptPanel.load();
    window.scriptPanel = this.scriptPanel;

    this.launchView = new window.LaunchView(this.runConsole, this.project || {});

    this.equationView = new window.EquationView(
      document.getElementById("eq-tabs"),
      document.getElementById("eq-grid")
    );
    this.equationView.load();

    this.flowView = new window.FlowView(document.getElementById("flowx"));
    this._initModelMode();

    this.physicsLossView = new window.PhysicsLossView(document.getElementById("phys"));

    this.pipelineFlow = new window.PipelineFlow(document.getElementById("flow"));
    this.pipelineFlow.load();

    this.modelGallery = new window.ModelGallery(
      document.getElementById("model-list"),
      document.getElementById("model-detail")
    );
    this.modelGallery.load();
    this._initModelZoo();

    this.configBrowser = new window.ConfigBrowser(
      document.getElementById("config-list"),
      document.getElementById("config-detail"),
      document.getElementById("config-search")
    );
    this.configBrowser.load();

    this.tensorboardView = new window.TensorboardView();

    this.feedTuner = new window.FeedTuner();
    window.feedTuner = this.feedTuner;

    this.resultsView = new window.ResultsView(
      document.getElementById("results-list"),
      document.getElementById("results-detail")
    );

    this.tomogramView = new window.TomogramView({
      strip         : document.getElementById("cube-strip"),
      stage         : document.getElementById("cube-stage"),
      deck          : document.getElementById("cube-deck"),
      topdown       : document.getElementById("cube-topdown"),
      cross         : document.getElementById("cube-cross"),
      coords        : document.getElementById("cube-coords"),
      back          : document.getElementById("cube-back"),
      hint          : document.getElementById("cube-hint"),
      slices        : document.getElementById("cube-slices"),
      slicesAt      : document.getElementById("cube-slices-at"),
      sources       : document.getElementById("cube-sources"),
      profiles      : document.getElementById("cube-profiles"),
      profAt        : document.getElementById("cube-prof-at"),
      profMetrics   : document.getElementById("cube-prof-metrics"),
      progress      : document.getElementById("cube-progress"),
      progressFill  : document.getElementById("cube-progress-fill"),
      progressLabel : document.getElementById("cube-progress-label"),
      spaceBtns     : [...document.querySelectorAll(".cube-space[data-space]")],
      profModeBtns  : [...document.querySelectorAll("#cube-prof-mode .cube-space")],
      modeBtns      : [...document.querySelectorAll(".cube-mode[data-view]")],
      views         : [...document.querySelectorAll(".cube-view[data-view]")],
      elevGrid      : document.getElementById("cube-elev-grid"),
      elevAt        : document.getElementById("cube-elev-at"),
      elevFill      : document.getElementById("cube-elev-fill"),
      ePanels       : [...document.querySelectorAll(".cube-eplane")].map((root) => ({
        root,
        source : root.dataset.source,
        canvas : root.querySelector("canvas"),
      })),
      panels        : [...document.querySelectorAll(".cube-slice")].map((root) => ({
        root,
        axis   : root.dataset.axis,
        source : root.dataset.source,
        canvas : root.querySelector("canvas"),
        ref    : root.querySelector(".cube-ref"),
        metric : root.querySelector(".cube-slice__metric"),
      })),
      profPanels    : [...document.querySelectorAll(".cube-prof")].map((root) => ({
        root,
        source : root.dataset.source,
        canvas : root.querySelector("canvas"),
      })),
    });
  }

  _initModelZoo() {
    const wrap = document.getElementById("model-zoo");
    if (!wrap) return;
    wrap.querySelectorAll(".model-zoo__btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.classList.contains("is-active")) return;
        wrap.querySelectorAll(".model-zoo__btn").forEach((b) => b.classList.toggle("is-active", b === btn));
        this.modelGallery.reload(btn.dataset.endpoint, btn.dataset.zoo);
      });
    });
  }

  _initModelMode() {
    const wrap  = document.getElementById("model-mode");
    const views = [...document.querySelectorAll(".model-view")];
    if (!wrap) return;
    wrap.querySelectorAll(".model-mode__btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        wrap.querySelectorAll(".model-mode__btn").forEach((b) => b.classList.toggle("is-active", b === btn));
        const mode = btn.dataset.mode;
        views.forEach((v) => v.classList.toggle("is-hidden", v.dataset.view !== mode));
        if (mode === "walkthrough") this.flowView.load();
      });
    });
  }

  _initRouter() {
    this.router = new window.Router((route, param) => this._onRoute(route, param));
    window.router = this.router;
    this.router.start();
  }

  _onRoute(route, param) {
    if (route === "home") {
      requestAnimationFrame(() => this.scenes.forEach((s) => s.resize && s.resize()));
    }
    if (route === "launch") {
      if (param) this.launchView.show(param);
      else this.router.go("scripts");
    } else {
      this.launchView.leave();
    }
    if (route === "physics") this.physicsLossView.load();
    if (route === "tensorboard") this.tensorboardView.enter();
    else this.tensorboardView.leave();
    if (route === "results") this.resultsView.enter();
    if (route === "feedtuner") this.feedTuner.enter();
    if (route === "cube") this.tomogramView.enter();
    if (route === "console") this.runConsole.onShow();
    setTimeout(() => this.reveal.scan(), 60);
  }
}

document.addEventListener("DOMContentLoaded", () => new App().init());
