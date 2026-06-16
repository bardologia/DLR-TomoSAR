"use strict";

class StatusBoard {
  constructor(els) {
    this.els = els;
    this.built = false;
    this.gpuEls = [];
    this.coreEls = [];
    this.histMax = 144;
    this.hist = { cpu: [], ram: [], gpus: [] };
  }

  start() {
    this._poll();
    setInterval(() => { if (!document.hidden) this._poll(); }, 2500);
    this._pollJobs();
    setInterval(() => { if (!document.hidden) this._pollJobs(); }, 5000);
  }

  async _poll() {
    let sys;
    try {
      sys = await window.apiGet("/api/system");
    } catch (e) {
      return;
    }
    if (!sys || sys.error) return;
    if (!this.built) this._build(sys);
    this._update(sys);
  }

  async _pollJobs() {
    let data;
    try {
      data = await window.apiGet("/api/jobs");
    } catch (e) {
      return;
    }
    this._updateJobs(data.jobs || []);
  }

  _build(sys) {
    this.built = true;
    const gpus = sys.gpus || [];
    const cores = (sys.cpu && sys.cpu.cores) || [];
    this.hist.gpus = gpus.map(() => ({ u: [], m: [] }));

    const gpuCards = gpus.length
      ? gpus.map((g, i) =>
          `<article class="gcard" data-gpu="${i}">` +
          `<header class="gcard__head"><span class="gcard__idx">gpu ${g.index != null ? g.index : i}</span><span class="gcard__name">${this._esc(g.name || "unknown")}</span><span class="gcard__who"></span><span class="gcard__temp">--</span></header>` +
          `<div class="gcard__row"><span class="gcard__pct">--</span><span class="gcard__unit">% util</span><span class="gcard__vram">--</span></div>` +
          `<canvas class="gcard__graph"></canvas>` +
          `<footer class="gcard__foot"><span class="gcard__power">--</span><span class="gcard__legend"><i class="lg lg--util"></i>util<i class="lg lg--vram"></i>vram</span></footer>` +
          `</article>`
        ).join("")
      : `<div class="sboard__empty">no CUDA devices visible to the backend</div>`;

    const coreCells = cores.map((_, i) => `<i class="cpu__cell" data-core="${i}" title="core ${i}"></i>`).join("");

    const lim = (sys.alerts && sys.alerts.limits) || {};
    const limitCells = [
      [lim.cpu_alert != null ? `${Math.round(lim.cpu_alert)}%` : "--", "cpu alert"],
      [lim.load_ratio != null ? `${lim.load_ratio.toFixed(1)} / core` : "--", "load limit"],
      [lim.ram_warn != null ? `${Math.round(lim.ram_warn)}%` : "--", "ram warn"],
      [lim.ram_kill != null ? `${Math.round(lim.ram_kill)}%` : "--", "ram kill"],
      [lim.interval != null ? `${Math.round(lim.interval)} s` : "--", "interval"],
      [lim.cooldown != null ? `${Math.round(lim.cooldown)} s` : "--", "kill cooldown"],
    ].map(([v, k]) => `<div><dt>${v}</dt><dd>${k}</dd></div>`).join("");

    this.els.board.innerHTML =
      `<section class="sboard sboard--alerts" id="sb-alerts" aria-label="Alerts" hidden></section>` +

      `<section class="sboard sboard--gpualarm" id="sb-gpu-guard" aria-label="GPU intrusion alarm" hidden></section>` +

      `<section class="sboard sboard--wd" aria-label="Resource watchdog">` +
      `<div class="wd__state">` +
      `<button type="button" class="wd__nuke" id="sb-nuke" title="Kill every process running under your user">` +
      `<span class="wd__nuke-sym" aria-hidden="true">&#9762;</span><span class="wd__nuke-txt">NUKE</span>` +
      `</button>` +
      `<i class="wd__light" id="sb-wd-light" aria-hidden="true"></i><span class="wd__label">watchdog</span><span class="wd__mode" id="sb-wd-mode">--</span></div>` +
      `<span class="wd__status" id="sb-wd-status">--</span>` +
      `<dl class="wd__limits">${limitCells}</dl>` +
      `</section>` +

      `<section class="sboard sboard--gpus" aria-label="CUDA devices">` +
      `<header class="sboard__cap"><span>cuda devices</span><span class="sboard__n">${gpus.length}</span></header>` +
      `<div class="sboard__gpugrid">${gpuCards}</div>` +
      `</section>` +

      `<section class="sboard sboard--cpu" aria-label="Processor">` +
      `<header class="sboard__cap"><span>processor</span><span class="sboard__n">${sys.cpu ? sys.cpu.count : 0} cores</span></header>` +
      `<div class="cpu__top">` +
      `<div class="cpu__big"><span class="cpu__pct" id="sb-cpu-pct">--</span><span class="cpu__unit">% busy</span></div>` +
      `<dl class="cpu__load"><div><dt id="sb-load1">--</dt><dd>load 1m</dd></div><div><dt id="sb-load5">--</dt><dd>5m</dd></div><div><dt id="sb-load15">--</dt><dd>15m</dd></div></dl>` +
      `</div>` +
      `<canvas class="sboard__graph" id="sb-cpu-graph"></canvas>` +
      `<div class="sboard__metric"><span>avg usage</span><span id="sb-cpu-avg">--</span></div>` +
      `<div class="bar"><i class="bar__fill" id="sb-cpu-bar"></i></div>` +
      `<div class="sboard__metric"><span>active cores</span><span id="sb-cpu-active">--</span></div>` +
      `<div class="bar"><i class="bar__fill bar__fill--cores" id="sb-cores-bar"></i></div>` +
      `<div class="cpu__grid" id="sb-cores">${coreCells}</div>` +
      `</section>` +

      `<section class="sboard sboard--mem" aria-label="Memory">` +
      `<header class="sboard__cap"><span>memory</span><span class="sboard__n" id="sb-mem-total"></span></header>` +
      `<div class="sboard__metric"><span>ram</span><span id="sb-ram-txt">--</span></div>` +
      `<div class="bar"><i class="bar__fill" id="sb-ram-bar"></i></div>` +
      `<div class="sboard__metric"><span>swap</span><span id="sb-swap-txt">--</span></div>` +
      `<div class="bar"><i class="bar__fill" id="sb-swap-bar"></i></div>` +
      `<canvas class="sboard__graph sboard__graph--mem" id="sb-mem-graph"></canvas>` +
      `</section>` +

      `<section class="sboard sboard--disk" aria-label="Storage">` +
      `<header class="sboard__cap"><span>storage</span><span class="sboard__n" id="sb-disk-total"></span></header>` +
      `<div class="sboard__metric"><span class="sboard__path" id="sb-disk-path"></span><span id="sb-disk-txt">--</span></div>` +
      `<div class="bar"><i class="bar__fill" id="sb-disk-bar"></i></div>` +
      `<div class="sboard__metric"><span>free &middot; all users</span><span id="sb-disk-free">--</span></div>` +
      `<div class="sboard__metric"><span class="sboard__path" id="sb-disk-user-path">${this._esc(sys.user || "user")} &middot; total</span><span id="sb-disk-user">--</span></div>` +
      `<div class="bar"><i class="bar__fill bar__fill--user" id="sb-disk-user-bar"></i></div>` +
      `<div class="sboard__metric"><span class="sboard__path" id="sb-disk-repo-path">dlr root</span><span id="sb-disk-repo">--</span></div>` +
      `<div class="bar"><i class="bar__fill bar__fill--repo" id="sb-disk-repo-bar"></i></div>` +
      `</section>` +

      `<section class="sboard sboard--procs" aria-label="Processes">` +
      `<header class="sboard__cap"><span>processes &middot; ${this._esc(sys.user || "user")}</span><span class="sboard__n" id="sb-proc-n"></span></header>` +
      `<div class="ptable">` +
      `<div class="ptable__row ptable__row--head"><span>pid</span><span>cpu%</span><span>mem</span><span class="ptable__gpu">gpu</span><span>s</span><span>command</span></div>` +
      `<div class="ptable__body" id="sb-procs"></div>` +
      `</div>` +
      `</section>` +

      `<section class="sboard sboard--jobs" aria-label="Jobs">` +
      `<header class="sboard__cap"><span>jobs</span><span class="sboard__n" id="sb-jobs-n">0</span></header>` +
      `<ul class="sboard__jobs" id="sb-jobs"><li class="sboard__empty">no runs yet</li></ul>` +
      `</section>`;

    this.gpuEls = [...this.els.board.querySelectorAll(".gcard")].map((card) => ({
      pct: card.querySelector(".gcard__pct"),
      vramTxt: card.querySelector(".gcard__vram"),
      temp: card.querySelector(".gcard__temp"),
      who: card.querySelector(".gcard__who"),
      power: card.querySelector(".gcard__power"),
      graph: card.querySelector(".gcard__graph"),
    }));
    this.coreEls = [...this.els.board.querySelectorAll(".cpu__cell")];

    this._wireNuke();

    if (!window.REDUCED_MOTION && window.gsap) {
      gsap.from(this.els.board.querySelectorAll(".sboard"), { opacity: 0, y: 16, duration: 0.7, stagger: 0.08, ease: "expo.out" });
    }
  }

  _wireNuke() {
    const btn = document.getElementById("sb-nuke");
    if (!btn) return;

    btn.addEventListener("click", async () => {
      const ok = window.confirm("NUKE: this kills EVERY process running under your user (training runs, shells, jobs). The web UI is spared. Continue?");
      if (!ok) return;

      btn.disabled = true;
      btn.classList.add("is-firing");
      try {
        const res = await window.apiPost("/api/system/nuke");
        if (res && res.ok) {
          window.toast(`nuke: terminated ${res.signalled}, force-killed ${res.killed}`, "ok");
        } else {
          window.toast(`nuke failed: ${(res && res.error) || "unknown error"}`, "error");
        }
      } catch (e) {
        window.toast("nuke failed: network error", "error");
      } finally {
        btn.disabled = false;
        btn.classList.remove("is-firing");
      }
    });
  }

  _update(sys) {
    if (window.serverScene && window.serverScene.feed) window.serverScene.feed(sys);
    this._renderWatchdog(sys.alerts || {});
    this._renderAlerts(sys.alerts || {});
    this._renderGpuGuard(sys.gpu_guard || {});
    const cpu = sys.cpu || {};
    const mem = sys.mem || {};
    const disk = sys.disk || {};
    const gpus = sys.gpus || [];

    if (this.els.host) this.els.host.textContent = sys.host || "server";
    if (this.els.sum) {
      const bits = [];
      if (sys.uptime) bits.push(`up ${this._uptime(sys.uptime)}`);
      if (cpu.count) bits.push(`${cpu.count} cores`);
      bits.push(`${gpus.length} CUDA device${gpus.length === 1 ? "" : "s"}`);
      if (mem.total) bits.push(`${this._tb(mem.total)} ram`);
      this.els.sum.textContent = bits.join(" · ");
    }

    this._push(this.hist.cpu, cpu.total || 0);
    if (mem.total) this._push(this.hist.ram, (100 * (mem.total - mem.available)) / mem.total);

    gpus.forEach((g, i) => {
      const el = this.gpuEls[i];
      const h = this.hist.gpus[i];
      if (!el || !h) return;
      const util = g.util != null ? g.util : 0;
      const memPct = g.mem_total ? (100 * g.mem_used) / g.mem_total : 0;
      this._push(h.u, util);
      this._push(h.m, memPct);

      el.pct.textContent = Math.round(util);
      el.vramTxt.innerHTML = `<b>${this._gb(g.mem_used * 1048576)}</b> / ${this._gb(g.mem_total * 1048576)} GB`;
      el.temp.textContent = g.temp != null ? `${Math.round(g.temp)}°C` : "--";
      el.temp.className = "gcard__temp" + (g.temp >= 85 ? " is-danger" : g.temp >= 70 ? " is-warn" : "");

      const holders = (g.holders || []).join(", ");
      if (holders) {
        el.who.textContent = holders;
        el.who.className = "gcard__who " + (g.others ? "is-others" : "is-mine");
      } else if (g.stale) {
        el.who.textContent = "stale memory";
        el.who.className = "gcard__who is-stale";
      } else {
        el.who.textContent = "";
        el.who.className = "gcard__who";
      }
      el.power.textContent = g.power != null ? `${Math.round(g.power)}${g.power_limit ? ` / ${Math.round(g.power_limit)}` : ""} W` : "--";
      this._spark(el.graph, [
        { data: h.m, color: "15, 118, 110", fill: 0.10 },
        { data: h.u, color: "29, 79, 216", fill: 0.16 },
      ]);
    });

    const pctEl = document.getElementById("sb-cpu-pct");
    if (pctEl) pctEl.textContent = Math.round(cpu.total || 0);
    const load = cpu.load || [];
    ["sb-load1", "sb-load5", "sb-load15"].forEach((id, i) => {
      const el = document.getElementById(id);
      if (el && load[i] != null) el.textContent = load[i].toFixed(1);
    });
    const cores = cpu.cores || [];
    cores.forEach((u, i) => {
      const cell = this.coreEls[i];
      if (!cell) return;
      const a = 0.05 + Math.min(1, u / 100) * 0.85;
      cell.style.background = `rgba(29, 79, 216, ${a.toFixed(3)})`;
      cell.title = `core ${i} · ${Math.round(u)}%`;
    });

    if (cores.length) {
      const avg    = cores.reduce((s, u) => s + u, 0) / cores.length;
      const active = cores.filter((u) => u >= 50).length;
      this._bar("sb-cpu-bar", avg);
      this._bar("sb-cores-bar", (100 * active) / cores.length);
      this._txt("sb-cpu-avg", `<b>${avg.toFixed(1)}</b> %`);
      this._txt("sb-cpu-active", `<b>${active}</b> / ${cores.length} dispatched`);
    }
    this._spark(document.getElementById("sb-cpu-graph"), [{ data: this.hist.cpu, color: "29, 79, 216", fill: 0.16 }]);

    if (mem.total) {
      const used = mem.total - mem.available;
      this._bar("sb-ram-bar", (100 * used) / mem.total);
      this._txt("sb-ram-txt", `<b>${this._gb(used)}</b> / ${this._gb(mem.total)} GB`);
      this._txt("sb-mem-total", `${this._tb(mem.total)}`);
      const swapUsed = (mem.swap_total || 0) - (mem.swap_free || 0);
      this._bar("sb-swap-bar", mem.swap_total ? (100 * swapUsed) / mem.swap_total : 0);
      this._txt("sb-swap-txt", mem.swap_total ? `<b>${this._gb(swapUsed)}</b> / ${this._gb(mem.swap_total)} GB` : "none");
      this._spark(document.getElementById("sb-mem-graph"), [{ data: this.hist.ram, color: "15, 118, 110", fill: 0.12 }]);
    }

    if (disk.total) {
      this._bar("sb-disk-bar", (100 * disk.used) / disk.total);
      this._txt("sb-disk-txt", `<b>${this._tb(disk.used)}</b> / ${this._tb(disk.total)}`);
      this._txt("sb-disk-total", this._tb(disk.total));
      this._txt("sb-disk-free", `<b>${this._tb(disk.free)}</b>`);
      const path = document.getElementById("sb-disk-path");
      if (path) path.textContent = disk.path || "";

      const userPath = document.getElementById("sb-disk-user-path");
      if (userPath && disk.user_path) userPath.title = disk.user_path;
      this._txt("sb-disk-user", disk.user_used != null ? `<b>${this._tb(disk.user_used)}</b>` : "scanning&hellip;");
      this._bar("sb-disk-user-bar", disk.user_used ? (100 * disk.user_used) / disk.total : 0);

      const repoPath = document.getElementById("sb-disk-repo-path");
      if (repoPath && disk.repo_path) repoPath.title = disk.repo_path;
      this._txt("sb-disk-repo", disk.repo_used != null ? `<b>${this._tb(disk.repo_used)}</b>` : "scanning&hellip;");
      this._bar("sb-disk-repo-bar", disk.repo_used ? (100 * disk.repo_used) / disk.total : 0);
    }

    this._renderProcs(sys.procs || []);
  }

  _renderWatchdog(alerts) {
    const light = document.getElementById("sb-wd-light");
    const mode = document.getElementById("sb-wd-mode");
    const status = document.getElementById("sb-wd-status");
    if (!light || !mode || !status) return;

    const armed = !!alerts.armed;
    const active = (alerts.active || []).length;

    light.classList.toggle("is-armed", armed);
    mode.textContent = armed ? "armed" : "offline";
    mode.classList.toggle("is-off", !armed);
    status.textContent = active ? `${active} active alert${active === 1 ? "" : "s"}` : "all clear";
    status.classList.toggle("is-alert", active > 0);
  }

  _renderAlerts(alerts) {
    const box = document.getElementById("sb-alerts");
    if (!box) return;

    const active = alerts.active || [];
    const events = (alerts.events || []).slice(-3);
    if (!active.length && !events.length) {
      box.hidden = true;
      box.innerHTML = "";
      return;
    }

    const chips = active.map((a) =>
      `<span class="alert alert--${a.level === "danger" ? "danger" : "warn"}">${this._esc(a.message)}</span>`
    ).join("");
    const log = events.map((e) =>
      `<span class="alert alert--event"><b>${this._esc(e.time.replace("T", " "))}</b> ${this._esc(e.message)}</span>`
    ).join("");

    box.hidden = false;
    box.innerHTML = `<header class="sboard__cap"><span>alerts</span><span class="sboard__n">${active.length} active</span></header><div class="alert__list">${chips}${log}</div>`;
  }

  _renderGpuGuard(guard) {
    const box = document.getElementById("sb-gpu-guard");
    if (!box) return;

    if (typeof guard.count === "number") {
      if (this._guardCount == null) this._guardCount = guard.count;
      else if (guard.count > this._guardCount) { this._guardCount = guard.count; this._alarm(false); }
    }
    if (typeof guard.critical === "number") {
      if (this._critCount == null) this._critCount = guard.critical;
      else if (guard.critical > this._critCount) { this._critCount = guard.critical; this._alarm(true); }
    }

    const active = guard.active || [];
    const gpus = guard.gpus || [];
    const events = (guard.events || []).slice(-5).reverse();
    const crit = active.filter((a) => a.status === "critical").length;

    if (!active.length && !events.length) {
      box.hidden = true;
      box.innerHTML = "";
      box.classList.remove("is-firing", "is-critical");
      return;
    }

    const context = gpus.length
      ? `<div class="gpuctx">` + gpus.map((g) => {
          const who = g.free ? "free" : (g.holders || []).map((h) => this._esc(h.user)).join(", ");
          const kind = g.free ? "is-free" : g.others ? (g.mine ? "is-clash" : "is-others") : "is-mine";
          return `<span class="gpuctx__cell ${kind}" title="gpu ${g.index} &middot; ${Math.round(g.util || 0)}% util">gpu ${g.index} &middot; ${who}</span>`;
        }).join("") + `</div>`
      : "";

    const chips = active.map((a) => {
      if (a.status === "critical") {
        return `<span class="alert alert--critical">&#9760; CRITICAL &middot; your pid(s) <b>${this._esc((a.dead_pids || a.mine_pids || []).join(", "))}</b> on gpu ${a.gpu_index} <b>${this._esc(a.gpu_name || "")}</b> DIED after <b>${this._esc(a.user)}</b> (pid ${a.pid}) invaded it</span>`;
      }
      return `<span class="alert alert--danger">gpu ${a.gpu_index} <b>${this._esc(a.gpu_name || "")}</b> invaded by <b>${this._esc(a.user)}</b> &middot; pid ${a.pid} &middot; ${Math.round(a.mem_mib)} MiB &middot; clashing with your pids ${this._esc((a.mine_pids || []).join(", "))}</span>`;
    }).join("");

    const log = events.map((e) => {
      const stamp = (e.critical_at || e.since || "").replace("T", " ");
      const tag = e.status === "critical" ? `<b class="alert__crit">process killed &middot;</b> ` : "";
      return `<span class="alert alert--event"><b>${this._esc(stamp)}</b> ${tag}${this._esc(e.user)} on gpu ${e.gpu_index} (pid ${e.pid})</span>`;
    }).join("");

    box.hidden = false;
    box.classList.toggle("is-firing", active.length > 0);
    box.classList.toggle("is-critical", crit > 0);
    const cap = crit ? `${crit} critical &middot; ${active.length} active` : `${active.length} active`;
    box.innerHTML = `<header class="sboard__cap"><span>&#9888; gpu intrusion alarm</span><span class="sboard__n">${cap}</span></header>${context}<div class="alert__list">${chips}${log}</div>`;
  }

  _alarm(critical) {
    window.toast && window.toast(critical ? "CRITICAL: your process died after a GPU intrusion" : "GPU intrusion: another user is on a GPU you are using", "error");
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      this._actx = this._actx || new Ctx();
      const ctx = this._actx;
      if (ctx.state === "suspended") ctx.resume();
      const t0 = ctx.currentTime;
      const beeps = critical ? [0, 0.16, 0.32, 0.48, 0.64] : [0, 0.2, 0.4];
      const freq = critical ? 1320 : 880;
      beeps.forEach((dt) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "square";
        osc.frequency.setValueAtTime(freq, t0 + dt);
        gain.gain.setValueAtTime(0.0001, t0 + dt);
        gain.gain.exponentialRampToValueAtTime(0.25, t0 + dt + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dt + 0.13);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(t0 + dt);
        osc.stop(t0 + dt + 0.15);
      });
    } catch (e) {}
  }

  _renderProcs(procs) {
    const body = document.getElementById("sb-procs");
    const n = document.getElementById("sb-proc-n");
    if (!body) return;
    if (n) n.textContent = String(procs.length);

    if (!procs.length) {
      body.innerHTML = `<div class="sboard__empty">no processes</div>`;
      return;
    }

    body.innerHTML = procs.map((p) => {
      const cls = p.cpu >= 100 ? "is-hot" : p.cpu >= 25 ? "is-mid" : "";
      const run = p.state === "R" ? " is-run" : "";
      return (
        `<div class="ptable__row${run}">` +
        `<span class="ptable__pid">${p.pid}</span>` +
        `<span class="ptable__cpu ${cls}">${p.cpu.toFixed(1)}</span>` +
        `<span>${this._mb(p.rss)}</span>` +
        `<span class="ptable__gpu">${p.gpu ? this._mb(p.gpu * 1048576) : "--"}</span>` +
        `<span class="ptable__state">${this._esc(p.state)}</span>` +
        `<span class="ptable__cmd" title="${this._esc(p.cmd)}">${this._esc(p.cmd)}</span>` +
        `</div>`
      );
    }).join("");
  }

  _updateJobs(jobs) {
    const list = document.getElementById("sb-jobs");
    const n = document.getElementById("sb-jobs-n");
    if (!list) return;
    const running = jobs.filter((j) => j.status === "running").length;
    if (n) n.textContent = running > 0 ? `${running} running` : String(jobs.length);

    if (!jobs.length) {
      list.innerHTML = `<li class="sboard__empty">no runs yet</li>`;
      return;
    }

    const followers = new Map();
    jobs.forEach((j) => {
      if (j.follow_of) followers.set(j.follow_of, j);
    });

    const row = (j, follow) => {
      const name = this._esc(String(j.command || "").split("/").pop() || "job");
      const cls =
        j.status === "running" ? "is-run" :
        j.status === "failed" ? "is-fail" :
        j.status === "scheduled" ? "is-sched" :
        j.status === "cancelled" ? "is-cancel" : "is-done";
      const mark = follow ? `<span class="sboard__jarrow" aria-hidden="true">&#8627;</span>` : "";
      return `<li class="sboard__job ${cls}${follow ? " sboard__job--follow" : ""}">${mark}<span class="sboard__jdot" aria-hidden="true"></span><span class="sboard__jname">${name}</span><span class="sboard__jstate">${this._esc(j.status)}</span></li>`;
    };

    list.innerHTML = jobs.filter((j) => !j.follow_of).slice(0, 8).map((j) => {
      const next = followers.get(j.job_id);
      return row(j, false) + (next ? row(next, true) : "");
    }).join("");
  }

  _push(arr, v) {
    arr.push(Math.max(0, Math.min(100, v)));
    if (arr.length > this.histMax) arr.shift();
  }

  _spark(cv, series) {
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(20, 30, 40, 0.08)";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach((f) => {
      const y = Math.round(h * f) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    });

    const step = w / (this.histMax - 1);
    series.forEach((s) => {
      const d = s.data;
      if (d.length < 2) return;
      const x0 = w - (d.length - 1) * step;
      const py = (v) => h - 1.5 - (v / 100) * (h - 3);

      ctx.beginPath();
      d.forEach((v, i) => {
        const x = x0 + i * step;
        if (i === 0) ctx.moveTo(x, py(v));
        else ctx.lineTo(x, py(v));
      });
      ctx.strokeStyle = `rgba(${s.color}, 0.9)`;
      ctx.lineWidth = 1.4;
      ctx.stroke();

      ctx.lineTo(w, h);
      ctx.lineTo(x0, h);
      ctx.closePath();
      ctx.fillStyle = `rgba(${s.color}, ${s.fill})`;
      ctx.fill();
    });
  }

  _bar(id, pct) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    el.classList.toggle("is-hot", pct >= 90);
  }

  _txt(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
  }

  _gb(bytes) {
    return (bytes / 1073741824).toFixed(1);
  }

  _mb(bytes) {
    const gb = bytes / 1073741824;
    if (gb >= 1) return `${gb.toFixed(1)}G`;
    return `${Math.round(bytes / 1048576)}M`;
  }

  _tb(bytes) {
    const tb = bytes / 1099511627776;
    return tb >= 1 ? `${tb.toFixed(2)} TB` : `${this._gb(bytes)} GB`;
  }

  _uptime(sec) {
    const d = Math.floor(sec / 86400);
    const h = Math.floor((sec % 86400) / 3600);
    const m = Math.floor((sec % 3600) / 60);
    if (d > 0) return `${d}d ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
    return `${h}:${String(m).padStart(2, "0")}`;
  }

  _esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
}

window.StatusBoard = StatusBoard;
