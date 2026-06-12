"use strict";

class FlowView {
  constructor(root) {
    this.root    = root;
    this.flows   = [];
    this.flow    = null;
    this.cursor  = -1;
    this.shown   = -1;
    this.playing = false;
    this.timer   = null;
    this.iterTimer = null;
    this.mjReady = null;
    this.loaded  = false;
    this.lineSeq = 0;
    this.groups  = [];
    this.lastAnimate = true;

    this.COLOR = { measured: "#6ea8ff", intermediate: "#f5b971", calculated: "#4fd6c4", final: "#c4a3ff" };
    this.ROLES = ["measured", "intermediate", "calculated", "final"];

    this.SLOTW = "28%";
    this.SLOTS = { s0: "1.5%", s1: "36%", s2: "70.5%", enter: "102%", exit: "-30%" };
    this.LSLOT = [
      { op: 0.65, scale: 0.85 },
      { op: 0.80, scale: 0.92 },
      { op: 1.00, scale: 1.00 },
      { op: 0.50, scale: 0.80 },
      { op: 0.40, scale: 0.70 },
    ];
    this.LTOPS = { none: [12, 31, 50, 69, 88], bottom: [12, 31, 50, 62, 70] };
    this.stackTimer = null;

    this.GUIDE = {
      deramp    : { sketch: "phasor",   tip: "Each pass is multiplied by the conjugate DEM phase, removing terrain topography and leaving only the sub-resolution elevation signal." },
      interf    : { sketch: "unit",     tip: "The master-secondary conjugate product is normalised to a unit phasor, then re-weighted by the clipped secondary amplitude as an SNR proxy." },
      beamform  : { sketch: "spectrum", tip: "A Capon minimum-variance beamformer turns the interferometric stack into an elevation power spectrum, sharp at each scatterer's height." },
      concat    : { sketch: "concat",   tip: "The per-worker azimuth subsections are concatenated back into one continuous beamformed tomogram." },
      condition : { sketch: "profile",  tip: "Each elevation profile is the tomogram magnitude, floored at a relative threshold and normalised by its own maximum." },
      smooth    : { sketch: "smooth",   tip: "A width-5 uniform filter smooths the profile so noise spikes are not mistaken for real layers." },
      peakfind  : { sketch: "peaks",    tip: "Peaks are kept only when their prominence clears a fraction of the maximum and they sit far enough apart." },
      select    : { sketch: "select",   tip: "The K most prominent peaks seed the components; any remaining slots are filled from the residual maxima." },
      objective : { sketch: "mixfit",   tip: "The loss is the squared mismatch between the K-Gaussian mixture and the normalised measured profile." },
      adam      : { sketch: "converge", tip: "The component widths sigma are fit to the profile by gradient descent, while the means and amplitudes stay fixed at the peak maxima found during initialisation." },
      bestk     : { sketch: "argmin",   tip: "Every candidate order K is scored with a complexity penalty; the smallest K that still fits the profile wins." },
      assemble  : { sketch: "sort",     tip: "The winning components are rescaled, sorted by ascending elevation, and packed into the interleaved 3K target vector." },
      quality   : { sketch: "heatmap",  tip: "A per-pixel coefficient of determination summarises how well the mixture fits, written out as a spatial quality map." },

      stack     : { sketch: "channels", tip: "Primary, secondary, and interferogram passes are written into one complex buffer indexed by pass." },
      patchgrid : { sketch: "tiling",   tip: "A strided sliding window tiles the crop into overlapping patches, reflected at the borders." },
      represent : { sketch: "c2r",      tip: "Each complex pass becomes real channels: magnitude and phase, or magnitude-normalised real and imaginary parts." },
      assemble_in : { sketch: "channels", tip: "Every enabled source is concatenated along the channel axis, with the optional DEM channel last." },
      augment   : { sketch: "augment",  tip: "Flips and rotations transform input and target together; additive noise perturbs only the input." },
      fitstats  : { sketch: "hist",     tip: "Per-channel location and scale are fitted on the training split using percentile min-max." },
      normalise : { sketch: "normalize",tip: "The fitted statistics standardise every split into the dimensionless tensor the network reads." },
      target    : { sketch: "target",   tip: "The configured subset of Gaussian parameters becomes the supervised regression target." },

      forward   : { sketch: "network",  tip: "The network maps one normalised patch to all per-pixel Gaussian parameters in a single pass." },
      clamp     : { sketch: "clamp",    tip: "Predictions are clamped to physical bounds with a leaky straight-through clamp so gradients survive saturation." },
      reconstruct:{ sketch: "mixfit",   tip: "Predicted parameters are evaluated on the elevation axis; the residual against ground truth drives the curve loss." },
      loss      : { sketch: "lossbars", tip: "Curve-space and parameter-space terms, each scaled by a fixed normaliser, are summed into one weighted objective." },
      gradclip  : { sketch: "clip",     tip: "Gradients are rescaled together so their global norm never exceeds the clipping threshold." },
      adamw     : { sketch: "converge", tip: "AdamW applies bias-corrected adaptive moments with decoupled weight decay; the loss falls over the epoch loop." },
      checkpoint: { sketch: "argmin",   tip: "The model is evaluated on the validation split; the best epoch is checkpointed and early stopping reverts to it." },

      predict   : { sketch: "network",  tip: "The trained model predicts parameters for every patch of the sliding-window grid over the scene." },
      recon     : { sketch: "mixfit",   tip: "Each patch's parameters are evaluated to a spectrum, with amplitudes rectified at zero." },
      window    : { sketch: "window",   tip: "A separable Hann window de-emphasises patch borders so overlapping predictions blend without seams." },
      ola       : { sketch: "overlapadd",tip: "Windowed patches are scattered into value and weight accumulators at their grid positions." },
      finalise  : { sketch: "cube",     tip: "Accumulated values divided by accumulated weights give the dense, seam-free prediction cube." },
      metrics   : { sketch: "heatmap",  tip: "The stitched cube is scored at physical scale; R-squared summarises overall reconstruction quality." },

      sample    : { sketch: "sample",   tip: "A multivariate TPE sampler proposes a joint hyperparameter vector across parallel workers." },
      trial     : { sketch: "converge", tip: "Each trial trains a full model for the epoch budget and returns its best validation loss." },
      prune     : { sketch: "prune",    tip: "A trial is stopped early once its reported loss exceeds the running median of completed trials." },
      best      : { sketch: "argmin",   tip: "The study is topped up in chunks; the minimising configuration is exported and rewritten after every trial." },
    };
  }

  async load() {
    if (this.loaded) return;
    this.loaded = true;
    const data = await window.apiGet("/api/flows");
    this.flows = data.flows || [];
    this._buildShell();
    if (this.flows.length) this._selectFlow(this.flows[0].key);
  }

  _buildShell() {
    this.root.innerHTML = "";
    this.root.classList.add("cine");

    const bar = document.createElement("div");
    bar.className = "cine__bar";
    this.selEl = document.createElement("div");
    this.selEl.className = "cine__pick";
    this.flows.forEach((f) => {
      const b = document.createElement("button");
      b.className = "cine__pickbtn";
      b.textContent = f.name;
      b.dataset.key = f.key;
      b.addEventListener("click", () => this._selectFlow(f.key));
      this.selEl.appendChild(b);
    });
    const legend = document.createElement("div");
    legend.className = "cine__legend";
    this.ROLES.forEach((role) => {
      const item = document.createElement("span");
      item.className = "cine__leg cine__leg--" + role;
      item.innerHTML = `<i></i>${role}`;
      legend.appendChild(item);
    });
    bar.appendChild(this.selEl);
    bar.appendChild(legend);

    this.stage = document.createElement("div");
    this.stage.className = "cine__stage";
    this.stage.innerHTML = `
      <svg class="cine__wires"><defs>
        <marker id="cine-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="context-stroke"></path>
        </marker></defs></svg>
      <div class="cine__head">
        <span class="cine__stepno"></span>
        <div class="cine__headtxt"><h3 class="cine__title"></h3><span class="cine__phase"></span></div>
      </div>
      <div class="cine__trail"></div>
      <div class="cine__scene"></div>
      <p class="cine__note"></p>`;

    this.wires   = this.stage.querySelector(".cine__wires");
    this.stepNo  = this.stage.querySelector(".cine__stepno");
    this.headEl  = this.stage.querySelector(".cine__head");
    this.titleEl = this.stage.querySelector(".cine__title");
    this.phaseEl = this.stage.querySelector(".cine__phase");
    this.trailEl = this.stage.querySelector(".cine__trail");
    this.sceneEl = this.stage.querySelector(".cine__scene");
    this.noteEl  = this.stage.querySelector(".cine__note");

    this.ctrl = document.createElement("div");
    this.ctrl.className = "cine__ctrl";
    this.ctrl.innerHTML = `
      <button class="cine__btn" data-act="restart" title="Restart">&#8635;</button>
      <button class="cine__btn" data-act="prev" title="Previous step">&#9664;</button>
      <button class="cine__btn cine__btn--play" data-act="play" title="Play / pause">&#9654;</button>
      <button class="cine__btn" data-act="next" title="Next step">&#9654;&#9654;</button>
      <div class="cine__track"><div class="cine__trackfill"></div></div>
      <span class="cine__count">--</span>
      <button class="cine__btn cine__btn--full" data-act="full" title="Full screen">&#9974;</button>`;
    this.ctrl.querySelectorAll(".cine__btn").forEach((b) => b.addEventListener("click", () => this._onCtrl(b.dataset.act)));
    this.trackFill = this.ctrl.querySelector(".cine__trackfill");
    this.countEl   = this.ctrl.querySelector(".cine__count");
    this.playBtn   = this.ctrl.querySelector(".cine__btn--play");

    document.addEventListener("fullscreenchange", () => {
      this.stage.classList.toggle("is-full", document.fullscreenElement === this.stage);
      clearTimeout(this._rz); this._rz = setTimeout(() => this._drawWires(false), 130);
    });
    window.addEventListener("resize", () => { clearTimeout(this._rz); this._rz = setTimeout(() => this._drawWires(false), 160); });

    this.root.appendChild(bar);
    this.root.appendChild(this.stage);
    this.root.appendChild(this.ctrl);
  }

  _selectFlow(key) {
    this._pause();
    this.flow = this.flows.find((f) => f.key === key) || this.flows[0];
    [...this.selEl.children].forEach((b) => b.classList.toggle("is-active", b.dataset.key === this.flow.key));
    this.byId = {};
    this.flow.nodes.forEach((n) => (this.byId[n.id] = n));
    this._buildTrail();
    this._reset();
  }

  _buildTrail() {
    this.trailEl.innerHTML = "";
    this.flow.steps.forEach((step, i) => {
      if (i) { const con = document.createElement("span"); con.className = "cine__trailcon"; this.trailEl.appendChild(con); }
      const out = this.byId[step.outputs[step.outputs.length - 1]] || this.byId[step.outputs[0]];
      const chip = document.createElement("span");
      chip.className = "cine__chip cine__chip--" + (out ? out.role : "intermediate");
      chip.dataset.step = i;
      this.trailEl.appendChild(chip);
      if (out) { chip.style.color = this.COLOR[out.role]; this._typeset(chip, out.tex, false); }
      chip.addEventListener("click", () => { this._pause(); this._go(i); });
    });
  }

  _reset() {
    this._pause();
    clearTimeout(this.iterTimer);
    clearTimeout(this.stackTimer);
    this.cursor = -1;
    this.shown  = -1;
    this.groups = [];
    this.sceneEl.innerHTML = "";
    this._clearWires();
    this.stepNo.textContent  = "00";
    this.titleEl.textContent = this.flow.name;
    this.phaseEl.textContent = "press play to walk the pipeline step by step";
    this.noteEl.textContent  = this.flow.blurb;
    [...this.trailEl.querySelectorAll(".cine__chip")].forEach((c) => c.classList.remove("is-done", "is-active"));
    this._progress();
  }

  _onCtrl(act) {
    if (act === "restart") return this._reset();
    if (act === "prev")    { this._pause(); return this._go(Math.max(-1, this.cursor - 1)); }
    if (act === "next")    { this._pause(); return this._go(Math.min(this.flow.steps.length - 1, this.cursor + 1)); }
    if (act === "play")    return this.playing ? this._pause() : this._play();
    if (act === "full")    {
      if (document.fullscreenElement === this.stage) document.exitFullscreen && document.exitFullscreen();
      else this.stage.requestFullscreen && this.stage.requestFullscreen().catch(() => {});
    }
  }

  _play() {
    if (this.cursor >= this.flow.steps.length - 1) this._reset();
    this.playing = true;
    this.playBtn.innerHTML = "&#10073;&#10073;";
    this.playBtn.classList.add("is-on");
    this._tick(this.cursor < 0 ? 800 : 0);
  }

  _pause() {
    this.playing = false;
    if (this.playBtn) { this.playBtn.innerHTML = "&#9654;"; this.playBtn.classList.remove("is-on"); }
    clearTimeout(this.timer);
  }

  _tick(delay) {
    if (!this.playing) return;
    if (this.cursor >= this.flow.steps.length - 1) return this._pause();
    this.timer = setTimeout(() => {
      this._go(this.cursor + 1);
      const step  = this.flow.steps[this.cursor];
      const lines = step ? (step.lines || []).length : 1;
      const base  = step && step.iterative ? 4200 : 3000;
      this._tick(base + Math.max(0, lines - 1) * 1900);
    }, delay || 0);
  }

  _go(i) {
    if (i < 0) return this._reset();
    const reduced = window.REDUCED_MOTION;
    const slide   = !reduced && Math.abs(i - this.shown) === 1 && this.groups.length > 0;
    this.lastAnimate = !reduced;
    this.cursor = i;
    this._renderWindow(i, slide);
    this.shown = i;
    this._setHead(this.flow.steps[i], i, !reduced);
    this._setTrail(i);
    this._progress();
  }

  _renderWindow(c, slide) {
    clearTimeout(this.iterTimer);
    clearTimeout(this.stackTimer);
    const N    = this.flow.steps.length;
    const want = [c - 1, c, c + 1].filter((x) => x >= 0 && x < N);
    const have = new Map(this.groups.map((g) => [g.i, g]));

    const keep = want.map((si) => {
      let g = have.get(si);
      if (!g) { g = this._buildGroup(si); this._setGeom(g, si > this.shown ? "enter" : "exit", 0, false); }
      return g;
    });

    this.groups.forEach((g) => {
      if (want.indexOf(g.i) >= 0) return;
      this._setGeom(g, g.i < c ? "exit" : "enter", 0, slide);
      this._removeLater(g, slide);
    });

    const place = () => keep.forEach((g) => {
      const off = g.i - c;
      g.el.classList.toggle("is-current", off === 0);
      this._setGeom(g, off < 0 ? "s0" : off > 0 ? "s2" : "s1", 1, slide);
      if (off === 0) {
        g.ready.then(() => setTimeout(() => {
          if (this.cursor !== g.i) return;
          this._stackRun(g);
          if (g.step.iterative && g.iterEl) this._iterRun(g.step, g.iterEl, this.lastAnimate);
        }, slide ? 720 : 0));
      } else {
        if (g.step.iterative && g.iterEl) this._iterIdle(g.step, g.iterEl);
      }
    });
    if (slide) requestAnimationFrame(place); else place();

    this.groups = keep;
    this._clearWires();
    Promise.all(keep.map((g) => g.ready)).then(() => setTimeout(() => this._drawWires(slide), slide ? 740 : 0));
  }

  _buildGroup(i) {
    const step    = this.flow.steps[i];
    const grp     = document.createElement("div");
    grp.className = "cine__grp";
    grp.dataset.step = i;

    const tipTop = (step.lines || []).length > 1 ? false : i % 2 === 0;
    const g = this.GUIDE[step.id];
    let tip = null, sketchSvg = null;
    if (g) {
      tip = document.createElement("div");
      tip.className = "cine__gtip " + (tipTop ? "is-top" : "is-bottom");
      tip.innerHTML = `<div class="cine__gsketch"></div><p class="cine__gtiptext">${this._esc(g.tip)}</p>`;
      sketchSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      sketchSvg.setAttribute("viewBox", "0 0 240 150");
      sketchSvg.classList.add("cine__skl");
      sketchSvg.innerHTML = this._sketchMarkup(g.sketch);
      tip.querySelector(".cine__gsketch").appendChild(sketchSvg);
    }

    const eq = document.createElement("div");
    eq.className = "cine__eq";

    const lines = [];
    (step.lines || []).forEach((terms) => {
      const line = document.createElement("div");
      line.className = "cine__line";
      const uid = ++this.lineSeq;
      lines.push({ line, terms, uid });
      eq.appendChild(line);
    });

    let iterEl = null;
    if (step.iterative) { iterEl = document.createElement("div"); iterEl.className = "cine__giter"; eq.appendChild(iterEl); }

    if (tip) grp.appendChild(tip);
    grp.appendChild(eq);
    this.sceneEl.appendChild(grp);

    const ready = Promise.all(lines.map(({ line, terms, uid }) =>
      this._typeset(line, this._composeLine(terms, uid), true).then(() => this._colorize(terms, uid))));

    const group = { el: grp, step, i, lines, iterEl, tipTop, tipEl: tip, sketchSvg, stackK: 0 };
    if (lines.length > 1 && !window.REDUCED_MOTION && window.gsap) {
      eq.classList.add("is-stack");
      this._stackPlace(group, false);
    }
    if (iterEl) this._iterIdle(step, iterEl);
    if (tip && this.lastAnimate && window.gsap) window.gsap.fromTo(tip, { opacity: 0, y: tipTop ? -10 : 10 }, { opacity: 1, y: 0, duration: 0.5, delay: 0.15, ease: "power2.out" });
    group.ready = ready.then(() => {
      if (this.lastAnimate) this._writeIn(group);
      if (group.sketchSvg && this.lastAnimate) this._animateSketch(group.sketchSvg);
    });
    return group;
  }

  _writeIn(group) {
    if (!window.gsap) return;
    const svgs = group.el.querySelectorAll(".cine__line svg");
    window.gsap.fromTo(svgs, { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.5, stagger: 0.08, ease: "power2.out" });
  }

  _setGeom(group, slot, op, animate) {
    const left = this.SLOTS[slot];
    group.el.style.width = this.SLOTW;
    if (animate && window.gsap) window.gsap.to(group.el, { left: left, opacity: op, duration: 0.62, ease: "power3.inOut" });
    else { group.el.style.left = left; group.el.style.opacity = op; }
  }

  _stackRun(group) {
    clearTimeout(this.stackTimer);
    if (group.lines.length < 2 || !this.lastAnimate || !window.gsap) return;
    this.stackTimer = setTimeout(() => this._stackAdvance(group), 1500);
  }

  _stackAdvance(group) {
    if (this.cursor !== group.i || !group.el.isConnected) return;
    if (group.stackK >= group.lines.length - 1) return;
    group.stackK += 1;
    this._stackPlace(group, true);
    this.stackTimer = setTimeout(() => this._stackAdvance(group), 2100);
  }

  _stackPlace(group, animate) {
    const tops = this.LTOPS[group.tipEl ? "bottom" : "none"];
    group.lines.forEach(({ line }, li) => {
      const o = Math.max(0, Math.min(4, li - group.stackK + 2));
      const s = this.LSLOT[o];
      const props = { top: tops[o] + "%", yPercent: -50, scale: s.scale, opacity: s.op };
      if (animate) window.gsap.to(line, Object.assign(props, { duration: 0.6, ease: "power3.inOut" }));
      else window.gsap.set(line, props);
    });
  }

  _removeLater(group, animate) {
    const el = group.el;
    if (animate) setTimeout(() => el.remove(), 700);
    else el.remove();
  }

  _composeLine(terms, uid) {
    return terms.map((t, ti) => (t.role ? `\\cssId{flx-${uid}-${ti}-${t.id}}{${t.tex}}` : t.tex)).join(" ");
  }

  _colorize(terms, uid) {
    terms.forEach((t, ti) => {
      if (!t.role) return;
      const el = document.getElementById(`flx-${uid}-${ti}-${t.id}`);
      if (el) el.style.color = this.COLOR[t.role] || "#e7edf0";
    });
  }

  _clearWires() {
    while (this.wires.childNodes.length > 1) this.wires.removeChild(this.wires.lastChild);
  }

  _wireHost() {
    const host = this.stage.getBoundingClientRect();
    this.wires.setAttribute("width", host.width);
    this.wires.setAttribute("height", host.height);
    this.wires.setAttribute("viewBox", `0 0 ${host.width} ${host.height}`);
    return host;
  }

  _drawWires(animate) {
    this._clearWires();
    if (!this.groups.length) return;
    const host = this._wireHost();
    this.groups.forEach((g) => this._tipLink(g, host, animate));
  }

  _tipLink(group, host, animate) {
    const eq  = group.el.querySelector(".cine__eq");
    const tip = group.tipEl;
    if (!eq || !tip) return;
    const er = eq.getBoundingClientRect();
    const tr = tip.getBoundingClientRect();
    if (!er.width || !tr.width) return;

    const stacked = eq.classList.contains("is-stack");
    const cx = er.left + er.width / 2 - host.left;
    let ya, yb;
    if (group.tipTop) { ya = (stacked ? er.top + er.height * 0.40 : er.top - 3) - host.top;    yb = tr.bottom - host.top + 4; }
    else              { ya = (stacked ? er.top + er.height * 0.60 : er.bottom + 3) - host.top; yb = tr.top - host.top - 4; }

    const link = document.createElementNS("http://www.w3.org/2000/svg", "path");
    link.setAttribute("class", "cine__tiplink");
    link.setAttribute("d", `M ${cx} ${ya} V ${yb}`);
    link.setAttribute("marker-end", "url(#cine-arrow)");
    this.wires.appendChild(link);
    if (animate && window.gsap) window.gsap.fromTo(link, { opacity: 0 }, { opacity: 1, duration: 0.4, delay: 0.2 });
  }

  _setHead(step, i, animate) {
    const set = () => {
      this.stepNo.textContent  = String(i + 1).padStart(2, "0");
      this.titleEl.innerHTML   = this._esc(step.title) + (step.iterative ? ` <span class="cine__loop">iterative</span>` : "");
      this.phaseEl.textContent = step.phase ? "phase " + step.phase : "";
      this.noteEl.textContent  = step.note || "";
    };
    if (animate && window.gsap) window.gsap.fromTo([this.headEl, this.noteEl], { opacity: 0, y: 6 }, { opacity: 1, y: 0, duration: 0.4, onStart: set });
    else set();
  }

  _setTrail(i) {
    [...this.trailEl.querySelectorAll(".cine__chip")].forEach((c) => {
      const s = Number(c.dataset.step);
      c.classList.toggle("is-done", s < i);
      c.classList.toggle("is-active", s === i);
    });
  }

  _animateSketch(svg) {
    if (!window.gsap) return;
    svg.querySelectorAll(".skl-draw, .skl-axis").forEach((el, k) => {
      let len = 0; try { len = el.getTotalLength(); } catch (e) {}
      if (!len) return;
      el.style.strokeDasharray = len; el.style.strokeDashoffset = len;
      window.gsap.to(el, { strokeDashoffset: 0, duration: 0.7, delay: 0.2 + k * 0.06, ease: "power2.out" });
    });
    window.gsap.fromTo(svg.querySelectorAll(".skl-pop, .skl-dash"), { opacity: 0 }, { opacity: 1, duration: 0.4, delay: 0.6, stagger: 0.05 });
  }

  _iterIdle(step, el) {
    if (!el) return;
    const it = step.iterative;
    const unit = it.unit || "t";
    el.className = "cine__giter is-on is-idle";
    el.innerHTML = `<span class="cine__iterlab">iterating</span><span class="cine__itercount">${this._esc(unit)} = 1 &hellip; ${it.steps}</span>`;
  }

  _iterRun(step, el, animate) {
    if (!el) return;
    const it = step.iterative;
    const unit = it.unit || "t";
    const sym  = it.symbol || "x";
    el.className = "cine__giter is-on";
    el.innerHTML = `<span class="cine__iterlab">iterating</span><span class="cine__itercount"></span><span class="cine__iterval"></span>`;
    const cnt = el.querySelector(".cine__itercount");
    const val = el.querySelector(".cine__iterval");
    const trace = it.trace || [];
    const last  = trace.length ? trace[trace.length - 1] : "";
    if (!animate || !trace.length) { cnt.textContent = `${unit} = ${it.steps}`; val.textContent = `${sym} → ${last}`; return; }
    let k = 0;
    clearTimeout(this.iterTimer);
    const tick = () => {
      if (!el.isConnected) return;
      const frac = trace.length > 1 ? k / (trace.length - 1) : 1;
      cnt.textContent = `${unit} = ${Math.round(1 + frac * (it.steps - 1))}`;
      val.textContent = `${sym} → ${trace[Math.min(k, trace.length - 1)]}`;
      val.classList.remove("is-bump"); void val.offsetWidth; val.classList.add("is-bump");
      k += 1;
      if (k < trace.length) this.iterTimer = setTimeout(tick, 460);
    };
    tick();
  }

  _progress() {
    const n = this.flow.steps.length;
    this.countEl.textContent   = `${Math.max(0, this.cursor + 1)} / ${n}`;
    this.trackFill.style.width = `${((this.cursor + 1) / n) * 100}%`;
  }

  _sketchMarkup(type) {
    const M = {
      phasor: `
        <line class="skl-axis" x1="120" y1="22" x2="120" y2="128"/>
        <line class="skl-axis" x1="46" y1="75" x2="194" y2="75"/>
        <circle class="skl-dash c-faint" cx="120" cy="75" r="42" fill="none"/>
        <line class="skl-draw c-meas" x1="120" y1="75" x2="159" y2="50"/>
        <line class="skl-draw c-cal" x1="120" y1="75" x2="148" y2="42"/>`,
      unit: `
        <line class="skl-axis" x1="112" y1="24" x2="112" y2="126"/>
        <line class="skl-axis" x1="44" y1="75" x2="180" y2="75"/>
        <circle class="skl-draw c-faint" cx="112" cy="75" r="40" fill="none"/>
        <line class="skl-draw c-cal" x1="112" y1="75" x2="140" y2="47"/>
        <circle class="skl-pop f-cal" cx="140" cy="47" r="4"/>`,
      spectrum: `
        <line class="skl-axis" x1="30" y1="22" x2="30" y2="126"/>
        <line class="skl-axis" x1="30" y1="126" x2="214" y2="126"/>
        <path class="skl-draw c-cal" d="M34 122 L98 120 C 114 120 118 38 126 38 C 134 38 138 120 152 120 L 210 122"/>`,
      concat: `
        <rect class="skl-draw c-faint" x="30" y="36" width="40" height="32" fill="none"/>
        <rect class="skl-draw c-faint" x="82" y="36" width="40" height="32" fill="none"/>
        <rect class="skl-draw c-faint" x="134" y="36" width="40" height="32" fill="none"/>
        <rect class="skl-draw c-fin" x="30" y="92" width="144" height="32" fill="none"/>
        <line class="skl-dash c-faint" x1="78" y1="92" x2="78" y2="124"/>
        <line class="skl-dash c-faint" x1="126" y1="92" x2="126" y2="124"/>`,
      profile: `
        <line class="skl-axis" x1="30" y1="20" x2="30" y2="126"/>
        <line class="skl-axis" x1="30" y1="126" x2="214" y2="126"/>
        <path class="skl-draw c-mid" d="M30 120 C 66 118 66 54 98 66 C 122 76 130 38 150 68 C 172 100 192 92 214 110"/>
        <line class="skl-dash c-faint" x1="30" y1="92" x2="214" y2="92"/>`,
      smooth: `
        <line class="skl-axis" x1="30" y1="20" x2="30" y2="126"/>
        <line class="skl-axis" x1="30" y1="126" x2="214" y2="126"/>
        <polyline class="skl-draw c-faint" points="30,116 48,66 64,106 82,58 100,102 118,62 136,108 154,70 172,110 190,78 208,114"/>
        <path class="skl-draw c-cal" d="M30 110 C 72 82 112 84 152 92 C 182 98 200 104 214 106"/>`,
      peaks: `
        <line class="skl-axis" x1="30" y1="20" x2="30" y2="126"/>
        <line class="skl-axis" x1="30" y1="126" x2="214" y2="126"/>
        <path class="skl-draw c-mid" d="M30 122 C 62 118 64 56 94 66 C 118 74 126 42 148 68 C 170 98 192 62 214 96"/>
        <circle class="skl-pop f-cal" cx="94" cy="66" r="4"/>
        <circle class="skl-pop f-cal" cx="148" cy="68" r="4"/>
        <line class="skl-dash c-faint" x1="94" y1="66" x2="94" y2="100"/>
        <line class="skl-dash c-faint" x1="148" y1="68" x2="148" y2="100"/>`,
      select: `
        <line class="skl-axis" x1="30" y1="126" x2="214" y2="126"/>
        <rect class="skl-pop f-mid" x="44" y="78" width="20" height="48"/>
        <rect class="skl-pop f-cal" x="82" y="46" width="20" height="80"/>
        <rect class="skl-pop f-cal" x="120" y="58" width="20" height="68"/>
        <rect class="skl-pop f-faint" x="158" y="96" width="20" height="30"/>
        <rect class="skl-draw c-cal" x="78" y="42" width="66" height="88" rx="4" fill="none"/>`,
      mixfit: `
        <line class="skl-axis" x1="30" y1="20" x2="30" y2="126"/>
        <line class="skl-axis" x1="30" y1="126" x2="214" y2="126"/>
        <path class="skl-draw c-mid" d="M30 120 C 64 118 74 70 98 72 C 122 74 126 50 150 72 C 174 96 192 90 214 110"/>
        <path class="skl-dash c-cal" d="M46 124 C 72 124 80 80 98 80 C 116 80 124 124 150 124"/>
        <path class="skl-dash c-cal" d="M112 124 C 136 124 142 62 154 62 C 168 62 174 124 198 124"/>`,
      converge: `
        <line class="skl-axis" x1="30" y1="20" x2="30" y2="126"/>
        <line class="skl-axis" x1="30" y1="126" x2="214" y2="126"/>
        <path class="skl-draw c-cal" d="M32 30 C 72 108 120 120 214 122"/>
        <line class="skl-dash c-faint" x1="30" y1="122" x2="214" y2="122"/>`,
      argmin: `
        <line class="skl-axis" x1="30" y1="126" x2="214" y2="126"/>
        <rect class="skl-pop f-faint" x="40" y="46" width="22" height="80"/>
        <rect class="skl-pop f-cal"   x="76" y="92" width="22" height="34"/>
        <rect class="skl-pop f-faint" x="112" y="84" width="22" height="42"/>
        <rect class="skl-pop f-faint" x="148" y="66" width="22" height="60"/>
        <rect class="skl-pop f-faint" x="184" y="52" width="22" height="74"/>`,
      sort: `
        <line class="skl-axis" x1="30" y1="120" x2="100" y2="120"/>
        <rect class="skl-pop f-mid" x="38" y="62" width="16" height="58"/>
        <rect class="skl-pop f-mid" x="60" y="42" width="16" height="78"/>
        <rect class="skl-pop f-mid" x="82" y="86" width="16" height="34"/>
        <path class="skl-dash c-cal" d="M110 84 H 134" marker-end="url(#cine-arrow)"/>
        <line class="skl-axis" x1="142" y1="120" x2="212" y2="120"/>
        <rect class="skl-pop f-cal" x="150" y="92" width="16" height="28"/>
        <rect class="skl-pop f-cal" x="172" y="66" width="16" height="54"/>
        <rect class="skl-pop f-cal" x="194" y="42" width="16" height="78"/>`,
      channels: `
        <rect class="skl-draw c-meas" x="44" y="34" width="152" height="15" fill="none"/>
        <rect class="skl-draw c-mid"  x="44" y="55" width="152" height="15" fill="none"/>
        <rect class="skl-draw c-mid"  x="44" y="76" width="152" height="15" fill="none"/>
        <rect class="skl-draw c-cal"  x="44" y="97" width="152" height="15" fill="none"/>
        <rect class="skl-draw c-fin"  x="44" y="118" width="152" height="15" fill="none"/>`,
      tiling: `
        <rect class="skl-axis" x="34" y="26" width="172" height="98" fill="none"/>
        <line class="skl-dash c-faint" x1="91" y1="26" x2="91" y2="124"/>
        <line class="skl-dash c-faint" x1="148" y1="26" x2="148" y2="124"/>
        <line class="skl-dash c-faint" x1="34" y1="58" x2="206" y2="58"/>
        <line class="skl-dash c-faint" x1="34" y1="91" x2="206" y2="91"/>
        <rect class="skl-pop f-cal" x="91" y="58" width="57" height="33" rx="2" style="opacity:0.5"/>`,
      c2r: `
        <line class="skl-axis" x1="58" y1="26" x2="58" y2="118"/>
        <line class="skl-axis" x1="22" y1="72" x2="96" y2="72"/>
        <line class="skl-draw c-meas" x1="58" y1="72" x2="88" y2="46"/>
        <circle class="skl-pop f-meas" cx="88" cy="46" r="3.5"/>
        <rect class="skl-pop f-mid" x="142" y="58" width="18" height="60"/>
        <rect class="skl-pop f-cal" x="176" y="84" width="18" height="34"/>`,
      augment: `
        <rect class="skl-draw c-mid" x="42" y="44" width="58" height="58" fill="none"/>
        <path class="skl-draw c-mid" d="M52 92 L71 58 L90 92"/>
        <path class="skl-dash c-faint" d="M112 73 H 132"/>
        <rect class="skl-draw c-cal" x="150" y="44" width="58" height="58" fill="none"/>
        <path class="skl-draw c-cal" d="M160 58 L179 92 L198 58"/>`,
      hist: `
        <line class="skl-axis" x1="30" y1="120" x2="214" y2="120"/>
        <rect class="skl-pop f-faint" x="54" y="96" width="16" height="24"/>
        <rect class="skl-pop f-faint" x="74" y="74" width="16" height="46"/>
        <rect class="skl-pop f-mid" x="94" y="50" width="16" height="70"/>
        <rect class="skl-pop f-mid" x="114" y="44" width="16" height="76"/>
        <rect class="skl-pop f-faint" x="134" y="64" width="16" height="56"/>
        <rect class="skl-pop f-faint" x="154" y="92" width="16" height="28"/>
        <line class="skl-dash c-cal" x1="50" y1="38" x2="50" y2="120"/>
        <line class="skl-dash c-cal" x1="178" y1="38" x2="178" y2="120"/>`,
      normalize: `
        <line class="skl-axis" x1="30" y1="118" x2="214" y2="118"/>
        <path class="skl-draw c-faint" d="M96 118 C 128 118 134 70 152 70 C 170 70 176 118 208 118"/>
        <path class="skl-draw c-cal" d="M44 118 C 70 118 78 44 96 44 C 114 44 122 118 148 118"/>
        <path class="skl-dash c-faint" d="M150 56 H 110" marker-end="url(#cine-arrow)"/>`,
      target: `
        <rect class="skl-pop f-faint" x="40" y="62" width="18" height="22" rx="2"/>
        <rect class="skl-pop f-fin" x="62" y="62" width="18" height="22" rx="2"/>
        <rect class="skl-pop f-faint" x="84" y="62" width="18" height="22" rx="2"/>
        <rect class="skl-pop f-fin" x="106" y="62" width="18" height="22" rx="2"/>
        <rect class="skl-pop f-faint" x="128" y="62" width="18" height="22" rx="2"/>
        <rect class="skl-pop f-fin" x="150" y="62" width="18" height="22" rx="2"/>
        <rect class="skl-pop f-faint" x="172" y="62" width="18" height="22" rx="2"/>`,
      network: `
        <rect class="skl-pop f-faint" x="32" y="52" width="14" height="44"/>
        <path class="skl-draw c-cal" d="M50 40 L98 92 L142 92 L190 40"/>
        <path class="skl-draw c-cal" d="M50 108 L98 96 L142 96 L190 108"/>
        <rect class="skl-pop f-fin" x="194" y="52" width="14" height="44"/>`,
      clamp: `
        <line class="skl-axis" x1="30" y1="28" x2="30" y2="120"/>
        <line class="skl-axis" x1="30" y1="120" x2="214" y2="120"/>
        <line class="skl-dash c-faint" x1="30" y1="46" x2="214" y2="46"/>
        <line class="skl-dash c-faint" x1="30" y1="100" x2="214" y2="100"/>
        <path class="skl-draw c-cal" d="M30 100 H 78 L 150 46 H 214"/>`,
      lossbars: `
        <line class="skl-axis" x1="30" y1="120" x2="214" y2="120"/>
        <rect class="skl-pop f-mid" x="50" y="58" width="18" height="62"/>
        <rect class="skl-pop f-cal" x="78" y="78" width="18" height="42"/>
        <rect class="skl-pop f-fin" x="106" y="66" width="18" height="54"/>
        <rect class="skl-pop f-mid" x="134" y="90" width="18" height="30"/>
        <rect class="skl-draw c-cal" x="170" y="42" width="20" height="78" fill="none"/>`,
      clip: `
        <circle class="skl-draw c-faint" cx="120" cy="74" r="42" fill="none"/>
        <line class="skl-axis" x1="78" y1="74" x2="162" y2="74"/>
        <line class="skl-axis" x1="120" y1="32" x2="120" y2="116"/>
        <line class="skl-dash c-faint" x1="120" y1="74" x2="192" y2="40"/>
        <line class="skl-draw c-cal" x1="120" y1="74" x2="156" y2="57"/>
        <circle class="skl-pop f-cal" cx="156" cy="57" r="3.5"/>`,
      window: `
        <line class="skl-axis" x1="30" y1="120" x2="214" y2="120"/>
        <path class="skl-draw c-cal" d="M30 120 C 80 120 90 40 120 40 C 150 40 160 120 210 120"/>`,
      overlapadd: `
        <line class="skl-axis" x1="30" y1="118" x2="214" y2="118"/>
        <path class="skl-dash c-faint" d="M40 118 C 70 118 78 60 100 60 C 122 60 130 118 160 118"/>
        <path class="skl-dash c-faint" d="M84 118 C 114 118 122 60 144 60 C 166 60 174 118 204 118"/>
        <path class="skl-draw c-cal" d="M40 70 C 90 56 154 56 204 70"/>`,
      cube: `
        <rect class="skl-draw c-fin" x="56" y="56" width="78" height="78" fill="none"/>
        <rect class="skl-draw c-faint" x="92" y="30" width="78" height="78" fill="none"/>
        <line class="skl-draw c-faint" x1="56" y1="56" x2="92" y2="30"/>
        <line class="skl-draw c-faint" x1="134" y1="56" x2="170" y2="30"/>
        <line class="skl-draw c-faint" x1="56" y1="134" x2="92" y2="108"/>
        <line class="skl-draw c-faint" x1="134" y1="134" x2="170" y2="108"/>`,
      sample: `
        <rect class="skl-axis" x="34" y="26" width="172" height="98" fill="none"/>
        <circle class="skl-pop f-faint" cx="68" cy="92" r="4"/>
        <circle class="skl-pop f-faint" cx="104" cy="58" r="4"/>
        <circle class="skl-pop f-faint" cx="150" cy="100" r="4"/>
        <circle class="skl-pop f-faint" cx="176" cy="50" r="4"/>
        <circle class="skl-pop f-faint" cx="120" cy="78" r="4"/>
        <circle class="skl-pop f-cal" cx="138" cy="70" r="5.5"/>`,
      prune: `
        <line class="skl-axis" x1="30" y1="26" x2="30" y2="120"/>
        <line class="skl-axis" x1="30" y1="120" x2="214" y2="120"/>
        <line class="skl-dash c-faint" x1="30" y1="76" x2="214" y2="76"/>
        <path class="skl-draw c-cal" d="M32 34 C 74 96 130 110 210 114"/>
        <path class="skl-draw c-faint" d="M32 40 C 70 64 92 70 110 72"/>
        <line class="skl-draw c-faint" x1="104" y1="66" x2="116" y2="78"/>
        <line class="skl-draw c-faint" x1="116" y1="66" x2="104" y2="78"/>`,
      heatmap: (() => {
        const v = [[0.9, 0.7, 0.95, 0.6, 0.85], [0.8, 0.5, 0.9, 0.99, 0.7], [0.6, 0.88, 0.4, 0.92, 0.8], [0.95, 0.7, 0.85, 0.6, 0.9]];
        let s = "";
        v.forEach((row, r) => row.forEach((c, q) => { s += `<rect class="skl-pop f-cal" x="${34 + q * 34}" y="${28 + r * 24}" width="30" height="20" rx="2" style="opacity:${0.25 + c * 0.7}"/>`; }));
        return s;
      })(),
    };
    return M[type] || "";
  }

  _whenMathJax() {
    if (this.mjReady) return this.mjReady;
    this.mjReady = new Promise((resolve) => {
      const check = () => { if (window.MathJax && window.MathJax.tex2svgPromise) resolve(); else setTimeout(check, 120); };
      check();
    }).then(() => { if (!document.getElementById("MJX-SVG-styles")) document.head.appendChild(window.MathJax.svgStylesheet()); });
    return this.mjReady;
  }

  _typeset(el, tex, display) {
    return this._whenMathJax().then(() => window.MathJax.tex2svgPromise(tex, { display: !!display })).then((node) => {
      el.textContent = ""; el.appendChild(node);
    }).catch(() => { el.textContent = ""; });
  }

  _esc(s) {
    return String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }
}

window.FlowView = FlowView;
