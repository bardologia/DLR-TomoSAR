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
      { op: 0.00, scale: 0.80 },
      { op: 0.00, scale: 0.80 },
    ];
    this.LTOPS = [10, 24, 38, 50, 58];
    this.stackTimer = null;

    this.GUIDE = {
      slc_load   : { sketch: "channels", tip: "Two PyRat products meet here: the master range-Doppler image and each secondary already co-registered to it, plus its DEM phase." },
      baselines  : { sketch: "baseline", tip: "Track positions collapse to one offset per pass relative to the reference; drift beyond five metres and the run aborts." },
      deramp     : { sketch: "phasor",   tip: "Multiplying by the conjugate DEM phasor erases the terrain the radar already knows, leaving only the metres-scale height signal." },
      crossprod  : { sketch: "phasor",   tip: "Conjugating master against secondary subtracts the DEM phase outright, since the conjugation flips its sign inside the product." },
      phasor     : { sketch: "unit",     tip: "Stripped of magnitude, the product keeps direction but not strength, so an amplitude swing between passes cannot pose as signal." },
      clip       : { sketch: "clip",     tip: "A hard ceiling of 1.25 stops a single bright corner reflector from dominating the per-pass weighting." },
      interf     : { sketch: "unit",     tip: "The clipped amplitude returns as the modulus, stamping a bounded confidence onto the surviving elevation phase." },
      subdivide  : { sketch: "tiling",   tip: "Too large for memory, the azimuth crop is split into bands, each handed to its own isolated PyRat worker." },
      covariance : { sketch: "covmat",   tip: "A 20-by-10 Boxcar window blurs the stack into the per-pixel covariance matrix that Capon then inverts." },
      capon      : { sketch: "spectrum", tip: "Capon's minimum-variance estimator sharpens the stack into an elevation spectrum that peaks exactly where scatterers sit." },
      concat     : { sketch: "concat",   tip: "The worker bands are stitched edge to edge back into one continuous tomogram." },

      threshold  : { sketch: "profile",  tip: "The profile is the tomogram's magnitude, floored below a fraction of its peak and clipped past a fixed height bin." },
      activity   : { sketch: "clamp",    tip: "Quiet pixels skip the fit entirely: only those whose peak clears the activity floor earn a mixture." },
      pnorm      : { sketch: "normalize",tip: "Scaling by the peak frees the loss from absolute brightness, so a faint pixel and a bright one are judged alike." },
      peakfind   : { sketch: "peaks",    tip: "A peak survives only if it stands proud of the noise and keeps its distance, all read off the raw profile." },
      geometry   : { sketch: "select",   tip: "One elevation span fixes everything: the seed width, the minimum peak spacing, and the bounds Adam may wander between." },
      residfill  : { sketch: "peaks",    tip: "Short on peaks, the fit suppresses the neighbourhood of each one and drafts replacements from the maxima that remain." },
      seed       : { sketch: "select",   tip: "The loudest peaks lock in the means and amplitudes; from here on, only the widths are free to move." },
      objective  : { sketch: "mixfit",   tip: "The loss is simply how far the K-Gaussian guess sits from the normalised profile, squared and summed." },
      adam       : { sketch: "converge", tip: "Three thousand clamped Adam steps, compiled into a single kernel, settle the widths while means and amplitudes hold fixed." },
      scoreK     : { sketch: "lossbars", tip: "Each candidate order must pay for itself: fit error plus a penalty that grows with the amplitude it spends." },
      selectK    : { sketch: "argmin",   tip: "The cheapest penalised order wins, and a tie always breaks toward the simpler model." },
      rescale    : { sketch: "normalize",tip: "Amplitudes step back into real backscatter units for storage; the means and widths were never rescaled." },
      assemble   : { sketch: "sort",     tip: "Survivors line up by ascending height and slot into the interleaved target, empty components falling to the end as zeros." },
      quality    : { sketch: "heatmap",  tip: "An R-squared per pixel, measured against the very profile that was fit, becomes a spatial map of trust." },
      diagnostics: { sketch: "argmin",   tip: "Two read-outs watch from the sidelines, flagging shaky order choices and weak peak-to-floor contrast, never touching the fit." },

      splitgeom  : { sketch: "tiling",   tip: "Azimuth is cut into three contiguous bands, 70/15/15, so train, validation and test never share ground." },
      localslice : { sketch: "tiling",   tip: "Absolute scene coordinates shift by the crop origin into zero-based slices straight into the memory-mapped arrays." },
      secselect  : { sketch: "channels", tip: "A handful of flight labels pick the secondaries, and the same indices carry across to their interferograms." },
      stack      : { sketch: "channels", tip: "Master, secondaries and interferograms drop into one complex buffer, each pass at its own index." },
      patchgrid  : { sketch: "tiling",   tip: "A strided window walks the crop, and a ceiling on the count guarantees the final patch still reaches the edge." },
      padgeom    : { sketch: "tiling",   tip: "Whatever the grid overshoots is padded symmetrically, with edge patches mirroring their own overhang back inward." },
      extract    : { sketch: "tiling",   tip: "Every patch is a fresh copy, never a view, reflect-padded in a single stroke before it leaves." },
      represent  : { sketch: "c2r",      tip: "Complex turns real: magnitude and phase, or magnitude with normalised real and imaginary parts, depending on the source." },
      assemble_in: { sketch: "channels", tip: "Sources line up along the channel axis, primary then secondaries then interferograms, with the DEM tacked on last." },
      target     : { sketch: "target",   tip: "From the interleaved ground truth, only the requested parameter roles are kept as the supervised target." },
      augment_geo: { sketch: "augment",  tip: "Flips and rotations hit input and target with the identical transform, keeping geometry in lockstep, all before normalisation." },
      slotkeys   : { sketch: "channels", tip: "Each channel earns a slot tag that decides which normalisation recipe it will later be fed." },
      fitstats   : { sketch: "hist",     tip: "Location and scale are learned on the train split alone, z-scored, with an optional log squeeze for heavy-tailed channels." },
      normalise  : { sketch: "normalize",tip: "One fitted statistic per slot flattens every split into the unitless tensor the network expects." },
      noise      : { sketch: "augment",  tip: "A small Gaussian perturbation roughens only the normalised input; the regression target stays clean." },
      denorm     : { sketch: "normalize",tip: "Reading predictions back to physical units inverts the same statistics, capping the exponential so nothing overflows." },

      forward    : { sketch: "network",  tip: "One pass, one patch in, every per-pixel Gaussian parameter out." },
      tdenorm    : { sketch: "normalize",tip: "Log-coded amplitude and width channels uncompress through a capped exp-minus-one, back into physical units." },
      clamp      : { sketch: "clamp",    tip: "Bounds are enforced with a leaky clamp, so saturated amplitudes and widths still leak a sliver of gradient." },
      renorm     : { sketch: "normalize",tip: "Clamped physical values fold back into training space, so the parameter losses speak the same units as the labels." },
      reconstruct: { sketch: "mixfit",   tip: "Parameters become a curve again on the elevation axis, and its gap from ground truth is what the curve losses chase." },
      residual   : { sketch: "mixfit",   tip: "One residual, computed once, is shared out to every pointwise curve term that follows." },
      curvepoint : { sketch: "lossbars", tip: "Four ways to punish the same residual: squared, absolute, Huber's hybrid, and Charbonnier's smooth L1." },
      curveshape : { sketch: "lossbars", tip: "Three terms care about shape, not size, scoring cosine alignment, spectral coherence, and structural similarity." },
      physgeom   : { sketch: "spectrum", tip: "Every physics term hangs off one Fourier operator built from the vertical wavenumber and the perpendicular baseline." },
      physmoments: { sketch: "spectrum", tip: "Relative checks on total power and the first three moments, mass, centroid and spread, keep the profile honest." },
      physcov    : { sketch: "covmat",   tip: "Re-synthesised coherence and covariance pit the two profiles' multibaseline statistics against each other." },
      physcapon  : { sketch: "spectrum", tip: "A Capon spectrum rebuilt from the prediction is matched against the measured one, closing the loop." },
      paramterms : { sketch: "lossbars", tip: "Mean, width and amplitude errors plus a smoothness penalty, with empty slots masked down to amplitude alone." },
      composite  : { sketch: "lossbars", tip: "Every active term, rescaled by a fixed normaliser to comparable size, sums into the one number that gets backpropagated." },
      gradclip   : { sketch: "clip",     tip: "All gradients shrink by a single shared factor the instant their global norm threatens the clip threshold." },
      adamw      : { sketch: "converge", tip: "Bias-corrected moments with decoupled decay nudge the weights, and across the epoch loop the loss drifts down." },
      schedule   : { sketch: "converge", tip: "A warmup ramp and a cosine decay multiply into the live learning rate; midway, a curriculum swap changes the objective itself." },
      checkpoint : { sketch: "argmin",   tip: "Validation crowns a best epoch and checkpoints it; patience running out rewinds the model back to it." },

      load       : { sketch: "channels", tip: "The architecture is rebuilt verbatim from its config, then its weights, elevation axis and norm stats are reloaded." },
      predict    : { sketch: "network",  tip: "The trained model sweeps the sliding-window grid, predicting parameters for every patch in the scene." },
      idenorm    : { sketch: "clamp",    tip: "At inference the clamp is hard, with no leaky slope, pinning amplitude, mean and width to their physical bounds." },
      align      : { sketch: "sort",     tip: "Ground-truth slots sort by height while the prediction stays put, so the metrics test whether the network learned the order." },
      recon      : { sketch: "mixfit",   tip: "Each patch's parameters expand back into a spectrum, with negative amplitudes rectified to zero." },
      window     : { sketch: "window",   tip: "A separable Hann taper dims patch edges so overlapping tiles blend into each other without a seam." },
      ola        : { sketch: "overlapadd",tip: "Windowed tiles add into a value buffer and their windows into a weight buffer, in any order, to the same result." },
      finalise   : { sketch: "cube",     tip: "Dividing value by weight, with uncovered pixels falling to zero, yields the dense cube, trimmed of its padding." },
      pixelmaps  : { sketch: "heatmap",  tip: "Five maps fall out per pixel: error, R-squared, cosine similarity, and how far the predicted peak bin drifted." },
      globalcurve: { sketch: "lossbars", tip: "Four scene-wide numbers, MSE, RMSE, overall R-squared and PSNR, sum up the reconstruction at a glance." },
      elevssim   : { sketch: "heatmap",  tip: "Sliced by elevation, the cube is scored bin by bin and by structural similarity across the slices." },
      paramslot  : { sketch: "lossbars", tip: "Beyond curves, the metrics weigh predicted means and widths, their ordering, and how slots assign to ground truth." },
      reduced    : { sketch: "spectrum", tip: "A Capon tomogram rebuilt from only the reduced baseline subset becomes the bar the network must clear, pixel by pixel." },

      spacelr    : { sketch: "sample",   tip: "Learning rates and weight decays span decades log-uniformly; dropout is sampled on a plain interval." },
      spacearch  : { sketch: "sample",   tip: "Five categorical dials set the feature widths, bottleneck, activation, normalisation, and how the decoder upsamples." },
      merge      : { sketch: "sample",   tip: "Both blocks fuse into one space that a multivariate TPE sampler explores with their correlations intact." },
      tpesplit   : { sketch: "density",  tip: "Past trials split at a loss quantile into winners and losers, and a density is fit over each group." },
      tpeacq     : { sketch: "density",  tip: "The next guess maximises the winners-over-losers density ratio, the TPE stand-in for expected improvement." },
      liar       : { sketch: "sample",   tip: "Trials still running are handed a pessimistic placeholder score, so parallel workers do not all chase one point." },
      trialsetup : { sketch: "network",  tip: "Every trial clones the base config and overrides its epoch budget, patience and seed before training." },
      trial      : { sketch: "converge", tip: "A trial is a full training run; its score is the best validation loss it reaches within the budget." },
      prune      : { sketch: "prune",    tip: "Fall behind the running median, once past the startup and warmup gates, and the trial is cut short." },
      best       : { sketch: "argmin",   tip: "Only the missing trials are dispatched, in chunks, and the running champion is rewritten to disk after each one." },
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
    });

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
    this.groups.forEach((g) => this._killSketchLoop(g));
    this.groups = [];
    this.sceneEl.innerHTML = "";
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
      this._setGeom(g, off < 0 ? "s0" : off > 0 ? "s2" : "s1", off > 0 ? 0.45 : 1, slide);
      this._setTip(g, off <= 0, slide);
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
  }

  _buildGroup(i) {
    const step    = this.flow.steps[i];
    const grp     = document.createElement("div");
    grp.className = "cine__grp";
    grp.dataset.step = i;

    const g = this.GUIDE[step.id];
    let tip = null, sketchSvg = null;
    if (g) {
      tip = document.createElement("div");
      tip.className = "cine__gtip";
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

    if (tip) tip.style.opacity = "0";
    const group = { el: grp, step, i, lines, iterEl, tipEl: tip, sketchSvg, sketchType: g ? g.sketch : null, sketchLoop: null, stackK: 0, tipShown: false };
    if (lines.length > 1 && !window.REDUCED_MOTION && window.gsap) {
      eq.classList.add("is-stack");
      this._stackPlace(group, false);
    }
    if (iterEl) this._iterIdle(step, iterEl);
    group.ready = ready.then(() => {
      if (this.lastAnimate) this._writeIn(group);
    });
    return group;
  }

  _setTip(group, shown, slide) {
    if (!group.tipEl || group.tipShown === shown) return;
    group.tipShown = shown;
    const el = group.tipEl;
    if (!window.gsap || window.REDUCED_MOTION) { el.style.opacity = shown ? "1" : "0"; return; }
    if (shown) {
      window.gsap.fromTo(el, { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.5, delay: slide ? 0.62 : 0.2, ease: "power2.out" });
      if (group.sketchSvg) {
        this._animateSketch(group.sketchSvg);
        this._killSketchLoop(group);
        group.sketchLoop = this._sketchLoop(group.sketchSvg, group.sketchType);
      }
    } else {
      window.gsap.to(el, { opacity: 0, y: 0, duration: 0.25 });
      this._killSketchLoop(group);
    }
  }

  _sketchLoop(svg, type) {
    if (!svg || !window.gsap || window.REDUCED_MOTION) return null;
    const NS = "http://www.w3.org/2000/svg";
    const ORBIT = { phasor: 1, unit: 1, clip: 1 };
    const SCAN  = { spectrum: 1, profile: 1, mixfit: 1, smooth: 1, peaks: 1, density: 1, normalize: 1, window: 1, overlapadd: 1 };

    if (ORBIT[type]) {
      const circ = svg.querySelector("circle");
      const cx = circ ? +circ.getAttribute("cx") : 120;
      const cy = circ ? +circ.getAttribute("cy") : 75;
      const els = svg.querySelectorAll("line.c-cal, circle.f-cal");
      if (!els.length) return null;
      return window.gsap.fromTo(els, { rotation: 0 },
        { rotation: 360, svgOrigin: cx + " " + cy, duration: 7, ease: "none", repeat: -1, delay: 0.8 });
    }

    if (type === "converge") {
      const path = svg.querySelector("path.c-cal") || svg.querySelector("path.skl-draw");
      let len = 0; try { len = path && path.getTotalLength(); } catch (e) {}
      if (!len) return null;
      const dot = document.createElementNS(NS, "circle");
      dot.setAttribute("r", "4.5"); dot.setAttribute("class", "skl-pop f-cal"); dot.setAttribute("data-loop", "1");
      svg.appendChild(dot);
      const st = { p: 0 };
      return window.gsap.to(st, { p: 1, duration: 2.8, ease: "power1.inOut", repeat: -1, repeatDelay: 0.5, delay: 0.6,
        onUpdate() { const pt = path.getPointAtLength(st.p * len); dot.setAttribute("cx", pt.x); dot.setAttribute("cy", pt.y); } });
    }

    const pops = svg.querySelectorAll(".skl-pop");
    if (!SCAN[type] && pops.length) {
      return window.gsap.to(pops, { opacity: "-=0.4", duration: 1.0, ease: "sine.inOut",
        repeat: -1, yoyo: true, stagger: { each: 0.1, from: "start" }, delay: 1.1 });
    }
    const beam = document.createElementNS(NS, "line");
    beam.setAttribute("class", "skl-beam"); beam.setAttribute("data-loop", "1");
    beam.setAttribute("y1", "20"); beam.setAttribute("y2", "130");
    svg.appendChild(beam);
    return window.gsap.fromTo(beam, { attr: { x1: 34, x2: 34 } },
      { attr: { x1: 208, x2: 208 }, duration: 2.6, ease: "sine.inOut", repeat: -1, yoyo: true, delay: 0.7 });
  }

  _killSketchLoop(group) {
    if (!group) return;
    if (group.sketchLoop) { group.sketchLoop.kill(); group.sketchLoop = null; }
    if (group.sketchSvg) group.sketchSvg.querySelectorAll("[data-loop]").forEach((e) => e.remove());
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
    group.lines.forEach(({ line }, li) => {
      const o = Math.max(0, Math.min(4, li - group.stackK + 2));
      const s = this.LSLOT[o];
      const props = { top: this.LTOPS[o] + "%", yPercent: -50, scale: s.scale, opacity: s.op };
      if (animate) window.gsap.to(line, Object.assign(props, { duration: 0.6, ease: "power3.inOut" }));
      else window.gsap.set(line, props);
    });
  }

  _removeLater(group, animate) {
    this._killSketchLoop(group);
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
      baseline: `
        <line class="skl-axis" x1="30" y1="100" x2="210" y2="100"/>
        <line class="skl-dash c-faint" x1="96" y1="100" x2="96" y2="70"/>
        <line class="skl-dash c-faint" x1="140" y1="100" x2="140" y2="118"/>
        <line class="skl-dash c-faint" x1="186" y1="100" x2="186" y2="58"/>
        <circle class="skl-pop f-meas" cx="50" cy="100" r="4.5"/>
        <circle class="skl-pop f-cal" cx="96" cy="70" r="4"/>
        <circle class="skl-pop f-cal" cx="140" cy="118" r="4"/>
        <circle class="skl-pop f-cal" cx="186" cy="58" r="4"/>`,
      covmat: (() => {
        let s = "";
        for (let r = 0; r < 4; r++) for (let q = 0; q < 4; q++) {
          const cls = r === q ? "f-cal" : (Math.abs(r - q) === 1 ? "f-mid" : "f-faint");
          const op = r === q ? 1 : (Math.abs(r - q) === 1 ? 0.55 : 0.28);
          s += `<rect class="skl-pop ${cls}" x="${72 + q * 24}" y="${30 + r * 24}" width="20" height="20" rx="2" style="opacity:${op}"/>`;
        }
        return s;
      })(),
      density: `
        <line class="skl-axis" x1="30" y1="120" x2="214" y2="120"/>
        <path class="skl-draw c-cal" d="M30 120 C 70 120 78 50 100 50 C 122 50 130 120 152 120"/>
        <path class="skl-draw c-faint" d="M94 120 C 130 120 140 64 162 64 C 184 64 192 120 214 120"/>
        <line class="skl-dash c-mid" x1="126" y1="34" x2="126" y2="120"/>`,
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
