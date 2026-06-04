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
    if (radar && window.RadarScene) this.scenes.push(new window.RadarScene(radar));
    const spectrum = document.getElementById("spectrum");
    if (spectrum && window.SpectrumScene) {
      this.scenes.push(new window.SpectrumScene(spectrum, document.getElementById("spectrum-readout")));
    }
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
    this._fillHero();
  }

  _setStatus(ok) {
    const wrap = document.getElementById("nav-status");
    const text = document.getElementById("status-text");
    wrap.classList.toggle("is-ok", ok);
    wrap.classList.toggle("is-down", !ok);
    text.textContent = ok ? "backend live" : "offline";
  }

  _fillHero() {
    const c = this.project.counts || {};
    const stats = [
      { v: c.scripts || 0, l: "entry points" },
      { v: c.models || 0, l: "architectures" },
      { v: c.pipelines || 0, l: "pipelines" },
      { v: "3K", l: "params / pixel" },
    ];
    const host = document.getElementById("hero-stats");
    host.innerHTML = "";
    stats.forEach((s) => {
      const block = document.createElement("div");
      block.innerHTML = `<dt>${s.v}</dt><dd>${s.l}</dd>`;
      host.appendChild(block);
    });

    const footRoot = document.getElementById("footer-root");
    if (footRoot) footRoot.textContent = this.project.repo_root || "";

    this._buildHomeJump();
  }

  _buildHomeJump() {
    const host = document.getElementById("home-jump");
    if (!host) return;
    const links = [
      { r: "model", t: "Signal model", d: "equations" },
      { r: "pipelines", t: "Pipelines", d: "six stages" },
      { r: "architectures", t: "Architectures", d: "ten backbones" },
      { r: "scripts", t: "Scripts", d: "launch runs" },
    ];
    host.innerHTML = "";
    links.forEach((l) => {
      const a = document.createElement("a");
      a.className = "home-jump__item";
      a.href = `#/${l.r}`;
      a.dataset.route = l.r;
      a.innerHTML = `<span class="home-jump__t">${l.t}</span><span class="home-jump__d">${l.d}</span>`;
      host.appendChild(a);
    });
  }

  _initComponents() {
    this.runConsole = new window.RunConsole({
      list: document.getElementById("job-list"),
      out: document.getElementById("console-out"),
      title: document.getElementById("console-title"),
      stop: document.getElementById("console-stop"),
      clear: document.getElementById("console-clear"),
    });
    this.runConsole.refresh();

    this.scriptPanel = new window.ScriptPanel(
      {
        grid: document.getElementById("script-grid"),
        filters: document.getElementById("script-filters"),
        drawer: document.getElementById("drawer"),
      },
      this.runConsole,
      this.project || {}
    );
    this.scriptPanel.load();
    window.scriptPanel = this.scriptPanel;

    this.equationView = new window.EquationView(
      document.getElementById("eq-tabs"),
      document.getElementById("eq-grid")
    );
    this.equationView.load();

    this.pipelineFlow = new window.PipelineFlow(document.getElementById("flow"));
    this.pipelineFlow.load();

    this.modelGallery = new window.ModelGallery(
      document.getElementById("model-list"),
      document.getElementById("model-detail")
    );
    this.modelGallery.load();

    this.configBrowser = new window.ConfigBrowser(
      document.getElementById("config-list"),
      document.getElementById("config-detail"),
      document.getElementById("config-search")
    );
    this.configBrowser.load();
  }

  _initRouter() {
    this.router = new window.Router((route) => this._onRoute(route));
    window.router = this.router;
    this.router.start();
  }

  _onRoute(route) {
    if (route === "home") {
      requestAnimationFrame(() => this.scenes.forEach((s) => s.resize && s.resize()));
    }
    setTimeout(() => this.reveal.scan(), 60);
  }
}

document.addEventListener("DOMContentLoaded", () => new App().init());
