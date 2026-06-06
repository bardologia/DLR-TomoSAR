"use strict";

class ConsoleTile {
  constructor(job, manager, host) {
    this.job = job;
    this.manager = manager;
    this.source = null;
    this.opened = false;

    this.root = document.createElement("div");
    this.root.className = "console-tile";

    const bar = document.createElement("div");
    bar.className = "console-tile__bar";

    this.nameEl = document.createElement("span");
    this.nameEl.className = "console-tile__name";
    this.nameEl.textContent = job.script;
    this.nameEl.title = job.command;

    this.metaEl = document.createElement("span");
    this.metaEl.className = "console-tile__meta";
    this.metaEl.textContent = job.pid ? `pid ${job.pid}` : "queued";

    this.badgeEl = document.createElement("span");
    this.badgeEl.className = `badge badge--${job.status}`;
    this.badgeEl.textContent = job.status;

    this.stopBtn = document.createElement("button");
    this.stopBtn.className = "btn btn--mini btn--danger";
    this.stopBtn.addEventListener("click", () => this.manager.stop(this.job.job_id));

    this.closeBtn = document.createElement("button");
    this.closeBtn.className = "btn btn--mini";
    this.closeBtn.textContent = "Close";
    this.closeBtn.addEventListener("click", () => this.manager.close(this.job.job_id));

    bar.append(this.nameEl, this.metaEl, this.badgeEl, this.stopBtn, this.closeBtn);

    this.outEl = document.createElement("div");
    this.outEl.className = "console-tile__out";

    this.root.append(bar, this.outEl);
    host.appendChild(this.root);

    this.term = new Terminal({
      cols: 120,
      rows: 24,
      convertEol: true,
      disableStdin: true,
      cursorBlink: false,
      cursorInactiveStyle: "none",
      scrollback: 10000,
      fontFamily: '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace',
      fontSize: 12,
      lineHeight: 1.25,
      theme: {
        background: "#0e1216",
        foreground: "#dde4e7",
        cursor: "#0e1216",
        selectionBackground: "#2a3a52",
      },
    });
    this.fitAddon = new FitAddon.FitAddon();
    this.term.loadAddon(this.fitAddon);

    this.setStatus(job.status);
    this._connect();
  }

  _connect() {
    this.source = new EventSource(`/api/jobs/${this.job.job_id}/stream`);
    this.source.onmessage = (ev) => this._onEvent(ev);
    this.source.onerror = () => this._disconnect();
  }

  _disconnect() {
    if (this.source) {
      this.source.close();
      this.source = null;
    }
  }

  _onEvent(ev) {
    let data;
    try {
      data = JSON.parse(ev.data);
    } catch (e) {
      return;
    }

    if (data.type === "chunk") {
      this.term.write(data.data);
    } else if (data.type === "line") {
      this.term.writeln(data.text);
    } else if (data.type === "status") {
      if (data.status === "running") {
        this._note(`process started (pid ${data.pid})`, "36");
        this.metaEl.textContent = `pid ${data.pid}`;
        this.setStatus("running");
      } else if (data.status === "scheduled") {
        this._note(`scheduled to run after ${data.after || "the current job"}`, "33");
      } else if (data.status === "cancelled") {
        this._note("cancelled before start", "33");
        this.setStatus("cancelled");
        this.manager.refresh();
      } else {
        this._note(`process ${data.status} (exit ${data.code})`, data.code === 0 ? "36" : "31");
        this.setStatus(data.status);
        this.manager.refresh();
      }
    } else if (data.type === "end") {
      this._disconnect();
    }
  }

  _note(text, color) {
    this.term.write(`\r\n\x1b[2;${color}m── ${text} ──\x1b[0m\r\n`);
  }

  setStatus(status) {
    this.job.status = status;
    this.badgeEl.className = `badge badge--${status}`;
    this.badgeEl.textContent = status;
    this.stopBtn.textContent = status === "scheduled" ? "Cancel" : "Stop";
    this.stopBtn.disabled = status !== "running" && status !== "scheduled";
  }

  fit() {
    const w = this.outEl.clientWidth;
    const h = this.outEl.clientHeight;
    if (!w || !h) return;

    if (!this.opened) {
      this.term.open(this.outEl);
      this.opened = true;
    }

    let font = Math.max(8, Math.min(16, (w - 16) / 72));
    this.term.options.fontSize = font;

    let dims = this.fitAddon.proposeDimensions();
    if (dims && dims.cols && dims.cols < 120) {
      font = Math.max(7, (font * dims.cols) / 120);
      this.term.options.fontSize = font;
      dims = this.fitAddon.proposeDimensions();
    }
    if (!dims || !dims.cols || !dims.rows) return;

    this.term.resize(Math.min(120, dims.cols), Math.max(2, dims.rows));
    this.term.scrollToBottom();
  }

  dispose() {
    this._disconnect();
    this.term.dispose();
    this.root.remove();
  }
}

class RunConsole {
  constructor(refs) {
    this.listEl = refs.list;
    this.tilesEl = refs.tiles;
    this.hintEl = refs.hint;
    this.jobs = [];
    this.tiles = new Map();
    this.dismissed = new Set();
    this._fitTimer = null;

    window.addEventListener("resize", () => this._queueFit());
  }

  async refresh() {
    const data = await window.apiGet("/api/jobs");
    this.jobs = data.jobs || [];
    this.jobs
      .filter((j) => j.status === "running" && !this.dismissed.has(j.job_id) && !this.tiles.has(j.job_id))
      .forEach((j) => this.open(j.job_id));
    this._renderList();
  }

  async launch(scriptKey, interpreter, label, overrides, followUp, detach) {
    const res = await window.apiPost("/api/run", { script_key: scriptKey, interpreter, overrides: overrides || {}, follow_up: followUp || null, detach: !!detach });
    if (!res.ok) {
      window.toast(res.error || "Launch failed", "error");
      return null;
    }
    window.toast(`Launched ${label || scriptKey}`, "ok");
    if (window.router) window.router.go("console");
    await this.refresh();
    this.open(res.job_id);
    return res.job_id;
  }

  open(jobId) {
    if (this.tiles.has(jobId)) return;
    const job = this.jobs.find((j) => j.job_id === jobId);
    if (!job) return;
    this.dismissed.delete(jobId);
    this.tiles.set(jobId, new ConsoleTile(job, this, this.tilesEl));
    this._layout();
    this._renderList();
  }

  close(jobId) {
    const tile = this.tiles.get(jobId);
    if (!tile) return;
    tile.dispose();
    this.tiles.delete(jobId);
    this.dismissed.add(jobId);
    this._layout();
    this._renderList();
  }

  toggle(jobId) {
    if (this.tiles.has(jobId)) this.close(jobId);
    else this.open(jobId);
  }

  async stop(jobId) {
    const res = await window.apiPost(`/api/jobs/${jobId}/stop`, {});
    if (res.ok) window.toast("Stop signal sent", "ok");
    else window.toast(res.error || "Could not stop", "error");
  }

  onShow() {
    this._queueFit();
  }

  _layout() {
    const n = this.tiles.size;
    this.hintEl.style.display = n ? "none" : "flex";
    const cols = n <= 1 ? 1 : n <= 4 ? 2 : 3;
    const rows = Math.max(1, Math.ceil(n / cols));
    this.tilesEl.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    this.tilesEl.style.gridTemplateRows = `repeat(${rows}, minmax(0, 1fr))`;
    this._queueFit();
  }

  _queueFit() {
    clearTimeout(this._fitTimer);
    this._fitTimer = setTimeout(() => this.tiles.forEach((t) => t.fit()), 120);
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

    const followers = new Map();
    this.jobs.forEach((j) => {
      if (j.follow_of) followers.set(j.follow_of, j);
    });

    this.jobs.forEach((job) => {
      if (job.follow_of) return;
      const item = document.createElement("li");
      item.className = "job-item" + (this.tiles.has(job.job_id) ? " is-active" : "");
      item.innerHTML =
        `<div class="job-item__top"><span class="job-item__name">${job.script}</span>` +
        `<span class="badge badge--${job.status}">${job.status}</span></div>` +
        `<div class="job-item__meta">${job.started.replace("T", " ")} · pid ${job.pid}</div>`;
      item.addEventListener("click", () => this.toggle(job.job_id));

      const next = followers.get(job.job_id);
      if (next) {
        const sub = document.createElement("div");
        sub.className = "job-item__follow" + (this.tiles.has(next.job_id) ? " is-active" : "");
        sub.innerHTML =
          `<span class="job-item__arrow" aria-hidden="true">&#8627;</span>` +
          `<span class="job-item__name">${next.script}</span>` +
          `<span class="badge badge--${next.status}">${next.status}</span>`;
        sub.addEventListener("click", (ev) => {
          ev.stopPropagation();
          this.toggle(next.job_id);
        });
        item.appendChild(sub);
      }

      this.listEl.appendChild(item);
    });
  }
}

window.RunConsole = RunConsole;
