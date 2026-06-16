"use strict";

const FT_DEFAULTS = {
  synthetic:           { batch: "512,1024,2048,4096", workers: "0,2,4,6,8", prefetch: "2,4,8,16", timed: 60, warmup: 8, paths: false },
  profile_autoencoder: { batch: "256,512,1024,2048,4096", workers: "0,2,4,6,8", prefetch: "2,4,8,16", timed: 80, warmup: 8, paths: true },
  image_autoencoder:   { batch: "4,8,16,32,64", workers: "0,4,8,12,16", prefetch: "2,4,8", timed: 40, warmup: 5, paths: true },
};

const FT_WORKER_COLORS = ["#6f9bff", "#4cc4e6", "#2dd4bf", "#7ce3a1", "#d8e36b", "#f0a85a"];

class FtChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.pad = { l: 54, r: 16, t: 16, b: 34 };
  }

  _resize() {
    const ratio = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth || 480;
    const height = this.canvas.clientHeight || 240;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.w = width;
    this.h = height;
  }

  _frame(yMax, yLabel, xLabels) {
    const ctx = this.ctx;
    const { l, r, t, b } = this.pad;
    const plotW = this.w - l - r;
    const plotH = this.h - t - b;

    ctx.clearRect(0, 0, this.w, this.h);

    ctx.strokeStyle = "rgba(111, 155, 255, 0.10)";
    ctx.fillStyle = "#8d979d";
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.lineWidth = 1;

    const rows = 5;
    for (let i = 0; i <= rows; i++) {
      const y = t + (plotH * i) / rows;
      ctx.beginPath();
      ctx.moveTo(l, y);
      ctx.lineTo(l + plotW, y);
      ctx.stroke();
      const value = yMax * (1 - i / rows);
      ctx.textAlign = "right";
      ctx.fillText(this._fmt(value), l - 8, y + 3);
    }

    ctx.textAlign = "center";
    xLabels.forEach((label, i) => {
      const x = xLabels.length === 1 ? l + plotW / 2 : l + (plotW * i) / (xLabels.length - 1);
      ctx.fillText(label, x, t + plotH + 18);
    });

    ctx.save();
    ctx.translate(13, t + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = "#6f9bff";
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    return { l, t, plotW, plotH };
  }

  _fmt(value) {
    if (value >= 1000) return (value / 1000).toFixed(value >= 10000 ? 0 : 1) + "k";
    if (value >= 100) return value.toFixed(0);
    return value.toFixed(value >= 10 ? 0 : 1);
  }

  empty(message) {
    this._resize();
    this.ctx.clearRect(0, 0, this.w, this.h);
    this.ctx.fillStyle = "#6b747a";
    this.ctx.font = '12px "JetBrains Mono", monospace';
    this.ctx.textAlign = "center";
    this.ctx.fillText(message, this.w / 2, this.h / 2);
  }

  throughput(rows) {
    this._resize();
    const main = rows.filter((row) => row.phase === "main" && row.status === "ok");
    if (!main.length) {
      this.empty("awaiting results");
      return;
    }

    const batches = [...new Set(main.map((row) => row.batch_size))].sort((a, b) => a - b);
    const workers = [...new Set(main.map((row) => row.num_workers))].sort((a, b) => a - b);
    const yMax = Math.max(...main.map((row) => row.end_to_end_samples_per_s)) * 1.1 || 1;

    const geo = this._frame(yMax, "samples / s", batches.map(String));
    const ctx = this.ctx;

    const xAt = (batch) => {
      const i = batches.indexOf(batch);
      return batches.length === 1 ? geo.l + geo.plotW / 2 : geo.l + (geo.plotW * i) / (batches.length - 1);
    };
    const yAt = (value) => geo.t + geo.plotH * (1 - value / yMax);

    workers.forEach((worker, wi) => {
      const color = FT_WORKER_COLORS[wi % FT_WORKER_COLORS.length];
      const series = main
        .filter((row) => row.num_workers === worker)
        .sort((a, b) => a.batch_size - b.batch_size);
      if (!series.length) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      series.forEach((row, i) => {
        const x = xAt(row.batch_size);
        const y = yAt(row.end_to_end_samples_per_s);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;

      ctx.fillStyle = color;
      series.forEach((row) => {
        ctx.beginPath();
        ctx.arc(xAt(row.batch_size), yAt(row.end_to_end_samples_per_s), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    });

    this._legend(workers.map((worker, wi) => ({ label: `${worker}w`, color: FT_WORKER_COLORS[wi % FT_WORKER_COLORS.length] })), geo);
  }

  utilization(rows) {
    this._resize();
    const main = rows.filter((row) => row.phase === "main" && row.status === "ok");
    if (!main.length) {
      this.empty("awaiting results");
      return;
    }

    const ordered = main
      .slice()
      .sort((a, b) => a.batch_size - b.batch_size || a.num_workers - b.num_workers);
    const labels = ordered.map((row) => `${row.batch_size}/${row.num_workers}`);
    const geo = this._frame(100, "percent", labels);
    const ctx = this.ctx;

    const yAt = (value) => geo.t + geo.plotH * (1 - value / 100);
    const slot = geo.plotW / ordered.length;
    const barW = Math.max(3, Math.min(22, slot * 0.45));

    ordered.forEach((row, i) => {
      const cx = geo.l + slot * (i + 0.5);
      const top = yAt(row.gpu_util_mean);

      const gradient = ctx.createLinearGradient(0, top, 0, geo.t + geo.plotH);
      gradient.addColorStop(0, "#2dd4bf");
      gradient.addColorStop(1, "rgba(45, 212, 191, 0.18)");
      ctx.fillStyle = gradient;
      ctx.shadowColor = "rgba(45, 212, 191, 0.6)";
      ctx.shadowBlur = 8;
      ctx.fillRect(cx - barW / 2, top, barW, geo.t + geo.plotH - top);
      ctx.shadowBlur = 0;
    });

    ctx.strokeStyle = "#f0a85a";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ordered.forEach((row, i) => {
      const cx = geo.l + slot * (i + 0.5);
      const y = yAt(row.data_wait_fraction * 100);
      if (i === 0) ctx.moveTo(cx, y);
      else ctx.lineTo(cx, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#f0a85a";
    ordered.forEach((row, i) => {
      const cx = geo.l + slot * (i + 0.5);
      ctx.beginPath();
      ctx.arc(cx, yAt(row.data_wait_fraction * 100), 2.4, 0, Math.PI * 2);
      ctx.fill();
    });

    this._legend([
      { label: "GPU util", color: "#2dd4bf" },
      { label: "data wait", color: "#f0a85a" },
    ], geo);
  }

  _legend(items, geo) {
    const ctx = this.ctx;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = "left";
    let x = geo.l + 4;
    const y = geo.t + 6;
    items.forEach((item) => {
      ctx.fillStyle = item.color;
      ctx.fillRect(x, y - 6, 9, 3);
      ctx.fillStyle = "#aeb6bb";
      ctx.fillText(item.label, x + 13, y);
      x += 22 + ctx.measureText(item.label).width;
    });
  }
}

class FeedTuner {
  constructor() {
    this.results = [];
    this.jobId = null;
    this.source = null;
    this.buffer = "";
    this.meta = null;
    this.expected = 0;
    this.ready = false;
  }

  enter() {
    if (!this.ready) this._wire();
    requestAnimationFrame(() => this._draw());
  }

  _wire() {
    this.ready = true;

    this.deck = document.getElementById("ft-deck");
    this.modes = [...document.querySelectorAll("#ft-modes .ft-mode")];
    this.paths = document.getElementById("ft-paths");
    this.inputs = {
      dataset: document.getElementById("ft-dataset"),
      params: document.getElementById("ft-params"),
      gpu: document.getElementById("ft-gpu"),
      model: document.getElementById("ft-model"),
      batch: document.getElementById("ft-batch"),
      workers: document.getElementById("ft-workers"),
      prefetch: document.getElementById("ft-prefetch"),
      timed: document.getElementById("ft-timed"),
      warmup: document.getElementById("ft-warmup"),
      amp: document.getElementById("ft-amp"),
    };
    this.runBtn = document.getElementById("ft-run");
    this.stopBtn = document.getElementById("ft-stop");
    this.hint = document.getElementById("ft-hint");

    this.idle = document.getElementById("ft-readout-idle");
    this.chips = document.getElementById("ft-chips");
    this.progress = document.getElementById("ft-progress");
    this.progressFill = document.getElementById("ft-progress-fill");
    this.progressLabel = document.getElementById("ft-progress-label");
    this.reco = document.getElementById("ft-reco");
    this.tableBody = document.getElementById("ft-table-body");
    this.log = document.getElementById("ft-log");

    this.throughputChart = new FtChart(document.getElementById("ft-chart-throughput"));
    this.utilChart = new FtChart(document.getElementById("ft-chart-util"));

    this.mode = "synthetic";
    this.modes.forEach((btn) => btn.addEventListener("click", () => this._selectMode(btn.dataset.mode)));
    this.runBtn.addEventListener("click", () => this._run());
    this.stopBtn.addEventListener("click", () => this._stop());
    window.addEventListener("resize", () => this._draw());

    this._applyDefaults("synthetic");
  }

  _selectMode(mode) {
    this.mode = mode;
    this.modes.forEach((btn) => btn.classList.toggle("is-active", btn.dataset.mode === mode));
    this._applyDefaults(mode);
  }

  _applyDefaults(mode) {
    const preset = FT_DEFAULTS[mode];
    this.inputs.batch.value = preset.batch;
    this.inputs.workers.value = preset.workers;
    this.inputs.prefetch.value = preset.prefetch;
    this.inputs.timed.value = preset.timed;
    this.inputs.warmup.value = preset.warmup;
    this.paths.hidden = !preset.paths;
    this.hint.textContent = preset.paths
      ? "Point the paths at a preprocessing run on this machine, then run the sweep."
      : "Synthetic mode needs no dataset — it runs anywhere with a GPU.";
  }

  _overrides() {
    const overrides = {
      mode: this.mode,
      gpu: String(this.inputs.gpu.value || "0"),
      batch_sizes: this.inputs.batch.value,
      worker_counts: this.inputs.workers.value,
      prefetch_factors: this.inputs.prefetch.value,
      timed_batches: String(this.inputs.timed.value || "60"),
      warmup_batches: String(this.inputs.warmup.value || "8"),
      use_amp: this.inputs.amp.checked ? "true" : "false",
    };
    if (this.inputs.model.value.trim()) overrides.model_name = this.inputs.model.value.trim();
    if (FT_DEFAULTS[this.mode].paths) {
      if (this.inputs.dataset.value.trim()) overrides["paths.dataset_path"] = this.inputs.dataset.value.trim();
      if (this.inputs.params.value.trim()) overrides["paths.parameters_path"] = this.inputs.params.value.trim();
    }
    return overrides;
  }

  async _run() {
    if (this.jobId) return;

    this._reset();
    const res = await window.apiPost("/api/run", { script_key: "tune_dataloader", overrides: this._overrides() });
    if (!res.ok) {
      window.toast(res.error || "Launch failed", "error");
      this._appendLog(`launch failed: ${res.error || "unknown error"}`);
      return;
    }

    this.jobId = res.job_id;
    this.runBtn.classList.add("is-busy");
    this.stopBtn.hidden = false;
    this.idle.hidden = true;
    this.progress.hidden = false;
    window.toast("Feed sweep launched", "ok");
    this._connect();
  }

  async _stop() {
    if (!this.jobId) return;
    await window.apiPost(`/api/jobs/${this.jobId}/stop`, {});
    window.toast("Stop signal sent", "ok");
  }

  _reset() {
    this.results = [];
    this.buffer = "";
    this.meta = null;
    this.expected = 0;
    this.tableBody.innerHTML = "";
    this.log.textContent = "";
    this.chips.hidden = true;
    this.chips.innerHTML = "";
    this.reco.hidden = true;
    this.progressFill.style.width = "0%";
    this.progressLabel.textContent = "";
    this._draw();
  }

  _connect() {
    this.source = new EventSource(`/api/jobs/${this.jobId}/stream`);
    this.source.onmessage = (ev) => this._onEvent(ev);
    this.source.onerror = () => this._disconnect();
  }

  _disconnect() {
    if (this.source) {
      this.source.close();
      this.source = null;
    }
    this.jobId = null;
    this.runBtn.classList.remove("is-busy");
    this.stopBtn.hidden = true;
  }

  _onEvent(ev) {
    let data;
    try {
      data = JSON.parse(ev.data);
    } catch (e) {
      return;
    }

    if (data.type === "chunk") this._consume(data.data);
    else if (data.type === "status" && data.status && data.status !== "running") this._appendLog(`process ${data.status}${data.code != null ? ` (exit ${data.code})` : ""}`);
    else if (data.type === "end") this._disconnect();
  }

  _consume(text) {
    this.buffer += text;
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop();
    lines.forEach((line) => this._line(line));
  }

  _line(line) {
    const clean = line.replace(/\x1b\[[0-9;]*m/g, "").replace(/[\r]/g, "");
    const marker = clean.indexOf("@TUNE ");
    if (marker < 0) {
      if (clean.trim()) this._appendLog(clean);
      return;
    }

    const rest = clean.slice(marker + 6);
    const space = rest.indexOf(" ");
    if (space < 0) return;

    const kind = rest.slice(0, space);
    let payload;
    try {
      payload = JSON.parse(rest.slice(space + 1));
    } catch (e) {
      return;
    }
    this._tune(kind, payload);
  }

  _tune(kind, payload) {
    if (kind === "meta") this._onMeta(payload);
    else if (kind === "result") this._onResult(payload);
    else if (kind === "recommendation") this._onRecommendation(payload);
    else if (kind === "done") this._onDone(payload);
  }

  _onMeta(meta) {
    this.meta = meta;
    this.expected = meta.n_specs || 0;
    this.idle.hidden = true;
    this.chips.hidden = false;
    this.chips.innerHTML = [
      ["mode", meta.mode],
      ["device", meta.device],
      ["model", meta.model_name],
      ["params", this._compact(meta.parameters)],
      ["sample", meta.sample],
    ]
      .map(([key, value]) => `<span class="ft-chip"><b>${key}</b>${value}</span>`)
      .join("");
    this._appendLog(`target ready — ${meta.item_source}`);
  }

  _onResult(record) {
    this.results.push(record);
    this._appendRow(record);
    this._tickProgress(record);
    this._draw();
  }

  _tickProgress(record) {
    const done = this.results.length;
    const total = record.phase === "refine" ? done : Math.max(this.expected, done);
    const pct = total ? Math.min(100, Math.round((done / total) * 100)) : 0;
    this.progressFill.style.width = `${record.phase === "refine" ? 100 : pct}%`;
    this.progressLabel.textContent = record.phase === "refine"
      ? `refining prefetch & pin-memory · ${done} configs measured`
      : `main sweep · ${done}/${this.expected || "?"} configs`;
  }

  _onRecommendation(payload) {
    const rec = payload.recommendation || {};
    const final = payload.final || {};
    if (!rec.found) {
      this.reco.hidden = false;
      this.reco.innerHTML = `<div class="ft-reco__head">No successful configuration</div>`;
      return;
    }

    const verdict = final.cpu_bound
      ? "CPU-bound — the input pipeline is the ceiling. Beyond these settings, lower the per-item cost."
      : "GPU saturated — this configuration keeps the device fed.";

    this.reco.hidden = false;
    this.reco.className = "ft-reco" + (final.cpu_bound ? " is-cpu" : " is-good");
    this.reco.innerHTML = `
      <div class="ft-reco__head">
        <span class="ft-reco__dot"></span>
        <span class="ft-reco__title">Recommended configuration</span>
        <span class="ft-reco__verdict">${verdict}</span>
      </div>
      <div class="ft-reco__grid">
        ${this._recoCell("batch size", final.batch_size)}
        ${this._recoCell("workers", final.num_workers)}
        ${this._recoCell("prefetch", final.prefetch_factor)}
        ${this._recoCell("pin memory", final.pin_memory)}
        ${this._recoCell("GPU util", `${(rec.gpu_util_mean || 0).toFixed(0)}%`)}
        ${this._recoCell("data wait", `${(100 * (rec.data_wait_fraction || 0)).toFixed(1)}%`)}
      </div>`;
  }

  _recoCell(label, value) {
    return `<div class="ft-reco__cell"><span class="ft-reco__k">${label}</span><span class="ft-reco__v">${value}</span></div>`;
  }

  _onDone(payload) {
    this._appendLog(`done — results at ${payload.results_path}`);
    this.progressLabel.textContent = "sweep complete";
    window.toast("Feed sweep complete", "ok");
  }

  _appendRow(record) {
    const row = document.createElement("tr");
    row.className = "ft-row ft-row--" + record.phase + (record.status !== "ok" ? " ft-row--bad" : "");
    if (record.status !== "ok") {
      row.innerHTML = `<td>${record.phase}</td><td>${record.batch_size}</td><td>${record.num_workers}</td><td colspan="8">${record.status}</td>`;
    } else {
      const wait = 100 * record.data_wait_fraction;
      row.innerHTML =
        `<td>${record.phase}</td><td>${record.batch_size}</td><td>${record.num_workers}</td>` +
        `<td>${record.prefetch_factor}</td><td>${record.pin_memory ? "on" : "off"}</td>` +
        `<td>${this._compact(record.loader_only_samples_per_s)}</td>` +
        `<td>${this._compact(record.compute_ceiling_samples_per_s)}</td>` +
        `<td class="ft-num">${this._compact(record.end_to_end_samples_per_s)}</td>` +
        `<td>${record.feed_ratio.toFixed(2)}</td>` +
        `<td class="${wait > 100 * (this.meta ? this.meta.wait_target : 0.05) ? "ft-warn" : "ft-ok"}">${wait.toFixed(1)}%</td>` +
        `<td>${record.gpu_util_mean.toFixed(0)}%</td>`;
    }
    this.tableBody.appendChild(row);
  }

  _appendLog(text) {
    this.log.textContent += text + "\n";
    this.log.scrollTop = this.log.scrollHeight;
  }

  _compact(value) {
    if (value == null || isNaN(value)) return "–";
    if (value >= 1000) return (value / 1000).toFixed(value >= 10000 ? 0 : 1) + "k";
    return Number(value).toFixed(value >= 10 ? 0 : 1);
  }

  _draw() {
    if (!this.throughputChart) return;
    this.throughputChart.throughput(this.results);
    this.utilChart.utilization(this.results);
  }
}

window.FeedTuner = FeedTuner;
