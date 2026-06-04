"use strict";

class PipelineFlow {
  constructor(hostEl) {
    this.hostEl = hostEl;
    this.pipelines = [];
    this.animator = null;
    this._wireModal();
  }

  async load() {
    const data = await window.apiGet("/api/pipelines");
    this.pipelines = data.pipelines || [];
    this._render();
  }

  _render() {
    this.hostEl.innerHTML = "";

    this.pipelines.forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "flow-row flow-row--play reveal";
      row.style.transitionDelay = `${i * 0.05}s`;

      const lead = document.createElement("div");
      lead.className = "flow-row__lead";

      const idx = String(i + 1).padStart(2, "0");
      let runHtml = "";
      if (p.script) {
        runHtml = `<button class="flow-row__run" data-script="${p.script}">main/${p.script}.py &rarr;</button>`;
      } else {
        runHtml = `<span class="flow-row__nomap">built into training</span>`;
      }

      lead.innerHTML =
        `<span class="flow-row__idx">${idx}</span>` +
        `<div class="flow-row__body"><h3 class="flow-row__name">${p.name}</h3>` +
        `<p class="flow-row__blurb">${p.blurb}</p>${runHtml}</div>`;

      const stages = document.createElement("div");
      stages.className = "flow-row__stages";
      p.stages.forEach((s, j) => {
        if (j > 0) {
          const arrow = document.createElement("span");
          arrow.className = "stage-arrow";
          arrow.textContent = "→";
          stages.appendChild(arrow);
        }
        const chip = document.createElement("span");
        chip.className = "stage";
        chip.style.transitionDelay = `${j * 0.07}s`;
        chip.textContent = s;
        stages.appendChild(chip);
      });

      const play = document.createElement("span");
      play.className = "flow-row__play";
      play.innerHTML = "&#9658; watch the process";
      stages.appendChild(play);

      row.appendChild(lead);
      row.appendChild(stages);
      row.addEventListener("click", (e) => {
        if (e.target.closest(".flow-row__run")) return;
        this._openProcess(p);
      });
      this.hostEl.appendChild(row);
    });

    this.hostEl.querySelectorAll(".flow-row__run").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (window.scriptPanel) window.scriptPanel.open(btn.dataset.script);
      });
    });

    window.revealScan();
  }

  _openProcess(p) {
    const modal = document.getElementById("proc-modal");
    document.getElementById("proc-kicker").textContent = p.script ? `main/${p.script}.py` : "pipeline";
    document.getElementById("proc-title").textContent = p.name + " pipeline";

    const stages = document.getElementById("proc-stages");
    stages.innerHTML = "";
    p.stages.forEach((s, j) => {
      if (j > 0) stages.insertAdjacentHTML("beforeend", '<span class="stage-arrow">→</span>');
      stages.insertAdjacentHTML("beforeend", `<span class="stage">${s}</span>`);
    });

    modal.classList.add("is-open");
    modal.setAttribute("aria-hidden", "false");

    if (!this.animator) {
      this.animator = new window.ProcessAnimator(document.getElementById("proc-canvas"), document.getElementById("proc-caption"));
    }
    requestAnimationFrame(() => this.animator.start(p.key));
  }

  _closeProcess() {
    const modal = document.getElementById("proc-modal");
    modal.classList.remove("is-open");
    modal.setAttribute("aria-hidden", "true");
    if (this.animator) this.animator.stop();
  }

  _wireModal() {
    document.getElementById("proc-close").addEventListener("click", () => this._closeProcess());
    document.getElementById("proc-scrim").addEventListener("click", () => this._closeProcess());
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && document.getElementById("proc-modal").classList.contains("is-open")) this._closeProcess();
    });
  }
}

window.PipelineFlow = PipelineFlow;
