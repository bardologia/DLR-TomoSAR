"use strict";

class RunConsole {
  constructor(refs) {
    this.listEl = refs.list;
    this.outEl = refs.out;
    this.titleEl = refs.title;
    this.stopBtn = refs.stop;
    this.clearBtn = refs.clear;
    this.jobs = [];
    this.activeId = null;
    this.source = null;

    this.stopBtn.addEventListener("click", () => this._stopActive());
    this.clearBtn.addEventListener("click", () => this._clearOut());
  }

  async refresh() {
    const data = await window.apiGet("/api/jobs");
    this.jobs = data.jobs || [];
    this._renderList();
  }

  async launch(scriptKey, interpreter, label, overrides) {
    const res = await window.apiPost("/api/run", { script_key: scriptKey, interpreter, overrides: overrides || {} });
    if (!res.ok) {
      window.toast(res.error || "Launch failed", "error");
      return null;
    }
    window.toast(`Launched ${label || scriptKey}`, "ok");
    await this.refresh();
    this.select(res.job_id);
    if (window.router) window.router.go("console");
    return res.job_id;
  }

  select(jobId) {
    if (this.source) {
      this.source.close();
      this.source = null;
    }
    this.activeId = jobId;
    const job = this.jobs.find((j) => j.job_id === jobId);
    this.titleEl.textContent = job ? `${job.command}` : jobId;
    this.outEl.innerHTML = "";
    this._renderList();

    const running = job && job.status === "running";
    this.stopBtn.disabled = !running;

    this.source = new EventSource(`/api/jobs/${jobId}/stream`);
    this.source.onmessage = (ev) => this._onEvent(ev);
    this.source.onerror = () => {
      if (this.source) {
        this.source.close();
        this.source = null;
      }
    };
  }

  _onEvent(ev) {
    let data;
    try {
      data = JSON.parse(ev.data);
    } catch (e) {
      return;
    }

    if (data.type === "line") {
      this._appendLine(data.text, "");
    } else if (data.type === "status") {
      if (data.status === "running") {
        this._appendLine(`process started (pid ${data.pid})`, "status");
      } else {
        const verdict = data.code === 0 ? "status" : "err";
        this._appendLine(`process ${data.status} (exit ${data.code})`, verdict);
        this.stopBtn.disabled = true;
        this.refresh();
      }
    } else if (data.type === "end") {
      if (this.source) {
        this.source.close();
        this.source = null;
      }
    }
  }

  _appendLine(text, kind) {
    const atBottom = this.outEl.scrollHeight - this.outEl.scrollTop - this.outEl.clientHeight < 60;
    const span = document.createElement("span");
    span.className = "console__line is-new" + (kind ? ` console__line--${kind}` : "");
    span.textContent = text;
    this.outEl.appendChild(span);
    if (atBottom) this.outEl.scrollTop = this.outEl.scrollHeight;
  }

  _clearOut() {
    this.outEl.innerHTML = "";
  }

  async _stopActive() {
    if (!this.activeId) return;
    const res = await window.apiPost(`/api/jobs/${this.activeId}/stop`, {});
    if (res.ok) window.toast("Stop signal sent", "ok");
    else window.toast(res.error || "Could not stop", "error");
  }

  _renderList() {
    this.listEl.innerHTML = "";
    if (!this.jobs.length) {
      const empty = document.createElement("li");
      empty.className = "job-list__empty";
      empty.textContent = "No jobs yet.";
      this.listEl.appendChild(empty);
      return;
    }

    this.jobs.forEach((job) => {
      const item = document.createElement("li");
      item.className = "job-item" + (job.job_id === this.activeId ? " is-active" : "");
      item.innerHTML =
        `<div class="job-item__top"><span class="job-item__name">${job.script}</span>` +
        `<span class="badge badge--${job.status}">${job.status}</span></div>` +
        `<div class="job-item__meta">${job.started.replace("T", " ")} · pid ${job.pid}</div>`;
      item.addEventListener("click", () => this.select(job.job_id));
      this.listEl.appendChild(item);
    });
  }
}

window.RunConsole = RunConsole;
