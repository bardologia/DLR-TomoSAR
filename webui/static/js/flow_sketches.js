"use strict";
window.FLOW_SKETCHES = {
slc_load: {
  tip: "The master loads as an RGI-SLC range-Doppler image while each secondary arrives as an INF-SLC already co-registered and carrying its own DEM-predicted phase.",
  build(svg) {
    svg.innerHTML = `
      <rect id="slc_load-master" x="30" y="30" width="56" height="90" rx="3" class="skl-pop f-meas" style="opacity:0.85"/>
      <line class="skl-axis" x1="30" y1="60" x2="86" y2="60"/>
      <line class="skl-axis" x1="30" y1="90" x2="86" y2="90"/>
      <g id="slc_load-stack">
        <rect id="slc_load-sec1" x="150" y="24" width="52" height="80" rx="3" class="skl-pop f-faint" style="opacity:0.5"/>
        <rect id="slc_load-sec2" x="142" y="32" width="52" height="80" rx="3" class="skl-pop f-faint" style="opacity:0.6"/>
        <rect id="slc_load-sec3" x="134" y="40" width="52" height="80" rx="3" class="skl-pop f-meas" style="opacity:0.85"/>
      </g>
      <path id="slc_load-reg" class="skl-dash c-mid" d="M86 75 C108 60, 116 58, 134 70" style="opacity:0" fill="none"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    tl.set(svg.querySelector('#slc_load-master'), { opacity: 0, x: -14 })
      .set([svg.querySelector('#slc_load-sec1'), svg.querySelector('#slc_load-sec2'), svg.querySelector('#slc_load-sec3')], { opacity: 0, x: 18 })
      .set(svg.querySelector('#slc_load-reg'), { opacity: 0 })
      .to(svg.querySelector('#slc_load-master'), { opacity: 0.85, x: 0, duration: 0.6, ease: "power2.out" }, 0.1)
      .to(svg.querySelector('#slc_load-sec1'), { opacity: 0.5, x: 0, duration: 0.4 }, 0.5)
      .to(svg.querySelector('#slc_load-sec2'), { opacity: 0.6, x: 0, duration: 0.4 }, 0.7)
      .to(svg.querySelector('#slc_load-sec3'), { opacity: 0.85, x: 0, duration: 0.4 }, 0.9)
      .to(svg.querySelector('#slc_load-reg'), { opacity: 0.9, duration: 0.7, ease: "sine.inOut" }, 1.3)
      .to(svg.querySelector('#slc_load-sec3'), { y: 5, duration: 0.4, ease: "power1.inOut", yoyo: true, repeat: 1 }, 1.4);
    return tl;
  },
},
baselines: {
  tip: "Horizontal and vertical track positions are averaged over the azimuth window and re-expressed relative to track 0, aborting if any per-track position std exceeds 5 metres.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="118" x2="210" y2="118"/>
      <line class="skl-axis" x1="40" y1="118" x2="40" y2="24"/>
      <line id="baselines-ref" class="skl-draw c-faint" x1="40" y1="100" x2="40" y2="100"/>
      <circle id="baselines-t0" cx="40" cy="100" r="4" class="skl-pop f-faint"/>
      <circle id="baselines-t1" cx="92" cy="70" r="3.5" class="skl-pop f-cal"/>
      <circle id="baselines-t2" cx="138" cy="92" r="3.5" class="skl-pop f-cal"/>
      <circle id="baselines-t3" cx="186" cy="48" r="3.5" class="skl-pop f-cal"/>
      <line id="baselines-b1" class="skl-dash c-cal" x1="40" y1="100" x2="92" y2="70" style="opacity:0"/>
      <line id="baselines-b2" class="skl-dash c-cal" x1="40" y1="100" x2="138" y2="92" style="opacity:0"/>
      <line id="baselines-b3" class="skl-dash c-cal" x1="40" y1="100" x2="186" y2="48" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const dots = [svg.querySelector('#baselines-t1'), svg.querySelector('#baselines-t2'), svg.querySelector('#baselines-t3')];
    const lines = [svg.querySelector('#baselines-b1'), svg.querySelector('#baselines-b2'), svg.querySelector('#baselines-b3')];
    tl.set(dots, { scale: 0, transformOrigin: "center" })
      .set(lines, { opacity: 0 })
      .set(svg.querySelector('#baselines-t0'), { scale: 1, transformOrigin: "center" })
      .to(svg.querySelector('#baselines-t0'), { scale: 1.5, duration: 0.3, yoyo: true, repeat: 1 }, 0.1)
      .to(dots[0], { scale: 1, duration: 0.35, ease: "back.out(2)" }, 0.5)
      .to(lines[0], { opacity: 0.9, duration: 0.4 }, 0.55)
      .to(dots[1], { scale: 1, duration: 0.35, ease: "back.out(2)" }, 0.9)
      .to(lines[1], { opacity: 0.9, duration: 0.4 }, 0.95)
      .to(dots[2], { scale: 1, duration: 0.35, ease: "back.out(2)" }, 1.3)
      .to(lines[2], { opacity: 0.9, duration: 0.4 }, 1.35)
      .to(dots, { opacity: 0.5, duration: 0.3, stagger: 0.05 }, 1.9)
      .to(dots, { opacity: 1, duration: 0.3, stagger: 0.05 }, 2.2);
    return tl;
  },
},
deramp: {
  tip: "Multiplying each secondary by exp(j phi_DEM,i) rotates its phase so the later conjugation cancels the DEM-predicted terrain ramp, leaving only sub-resolution elevation structure.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="122" x2="214" y2="122"/>
      <line id="deramp-ramp" class="skl-dash c-faint" x1="28" y1="116" x2="214" y2="40"/>
      <path id="deramp-sig" class="skl-draw c-meas" d="M28 104 q12 -14 24 0 t24 0 t24 0 t24 0 t24 0 t24 0 t24 0" fill="none"/>
      <path id="deramp-flat" class="skl-draw c-mid" d="M28 92 q12 -10 24 0 t24 0 t24 0 t24 0 t24 0 t24 0 t24 0" fill="none" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    tl.set(svg.querySelector('#deramp-flat'), { opacity: 0 })
      .set(svg.querySelector('#deramp-sig'), { opacity: 1, rotation: 0, svgOrigin: "28 122" })
      .set(svg.querySelector('#deramp-ramp'), { opacity: 0.25 })
      .to(svg.querySelector('#deramp-ramp'), { attr: { x2: 214, y2: 116 }, opacity: 0.15, duration: 0.9, ease: "power2.inOut" }, 0.4)
      .to(svg.querySelector('#deramp-sig'), { rotation: 11.5, opacity: 0, duration: 0.9, ease: "power2.inOut" }, 0.4)
      .fromTo(svg.querySelector('#deramp-flat'), { opacity: 0 }, { opacity: 1, duration: 0.6 }, 1.0);
    return tl;
  },
},
crossprod: {
  tip: "Conjugating the deramped secondary against the master subtracts its phase, and because conjugation flips the DEM sign the term phi_DEM,i is effectively removed from arg(c_i).",
  build(svg) {
    svg.innerHTML = `
      <circle class="skl-axis" cx="78" cy="74" r="34" fill="none"/>
      <line class="skl-axis" x1="44" y1="74" x2="112" y2="74"/>
      <line class="skl-axis" x1="78" y1="40" x2="78" y2="108"/>
      <line id="crossprod-m" class="skl-draw c-meas" x1="78" y1="74" x2="105" y2="54"/>
      <line id="crossprod-s" class="skl-draw c-mid" x1="78" y1="74" x2="100" y2="96"/>
      <text x="160" y="78" text-anchor="middle" style="fill:#4a5a6b;font-size:14px;font-family:sans-serif">=</text>
      <line id="crossprod-c" class="skl-draw c-cal" x1="190" y1="74" x2="214" y2="50" style="opacity:0"/>
      <circle class="skl-axis" cx="190" cy="74" r="20" fill="none" style="opacity:0.4"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    tl.set(svg.querySelector('#crossprod-m'), { rotation: 0, svgOrigin: "78 74" })
      .set(svg.querySelector('#crossprod-s'), { rotation: 0, svgOrigin: "78 74" })
      .set(svg.querySelector('#crossprod-c'), { opacity: 0, rotation: 0, svgOrigin: "190 74" })
      .to(svg.querySelector('#crossprod-s'), { rotation: -52, duration: 0.8, ease: "sine.inOut" }, 0.4)
      .to(svg.querySelector('#crossprod-c'), { opacity: 1, duration: 0.4 }, 1.1)
      .fromTo(svg.querySelector('#crossprod-c'), { rotation: 30 }, { rotation: 0, duration: 0.6, ease: "back.out(1.6)" }, 1.1);
    return tl;
  },
},
phasor: {
  tip: "Dividing by |c_i| floored at 1e-30 collapses every cross-product onto the unit circle, equalising inter-pass amplitude while null pixels go to zero instead of NaN.",
  build(svg) {
    svg.innerHTML = `
      <circle id="phasor-unit" cx="120" cy="76" r="42" fill="none" class="skl-dash c-faint" style="opacity:0.5"/>
      <line class="skl-axis" x1="70" y1="76" x2="170" y2="76"/>
      <line class="skl-axis" x1="120" y1="28" x2="120" y2="124"/>
      <line id="phasor-v1" class="skl-draw c-mid" x1="120" y1="76" x2="146" y2="48"/>
      <circle id="phasor-h1" cx="146" cy="48" r="3.5" class="skl-pop f-cal" style="opacity:0"/>
      <line id="phasor-v2" class="skl-draw c-mid" x1="120" y1="76" x2="92" y2="98"/>
      <circle id="phasor-h2" cx="92" cy="98" r="3.5" class="skl-pop f-cal" style="opacity:0"/>
      <line id="phasor-v3" class="skl-draw c-mid" x1="120" y1="76" x2="152" y2="92"/>
      <circle id="phasor-h3" cx="152" cy="92" r="3.5" class="skl-pop f-cal" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const v1 = svg.querySelector('#phasor-v1'), v2 = svg.querySelector('#phasor-v2'), v3 = svg.querySelector('#phasor-v3');
    const h1 = svg.querySelector('#phasor-h1'), h2 = svg.querySelector('#phasor-h2'), h3 = svg.querySelector('#phasor-h3');
    tl.set(v1, { attr: { x2: 146, y2: 48 } }).set(v2, { attr: { x2: 92, y2: 98 } }).set(v3, { attr: { x2: 152, y2: 92 } })
      .set([h1, h2, h3], { opacity: 0 })
      .to(v1, { attr: { x2: 149.7, y2: 41.5 }, duration: 0.8, ease: "power2.out" }, 0.3)
      .to(v2, { attr: { x2: 84.5, y2: 105 }, duration: 0.8, ease: "power2.out" }, 0.3)
      .to(v3, { attr: { x2: 157, y2: 99 }, duration: 0.8, ease: "power2.out" }, 0.3)
      .to([h1, h2, h3], { opacity: 1, duration: 0.4, stagger: 0.1 }, 1.0)
      .to(svg.querySelector('#phasor-unit'), { opacity: 0.9, duration: 0.5, yoyo: true, repeat: 1 }, 1.0);
    return tl;
  },
},
clip: {
  tip: "The secondary amplitude is capped at c_max = 1.25 so a single bright corner reflector or artefact cannot dominate the per-pass weight.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="120" x2="214" y2="120"/>
      <line id="clip-cap" class="skl-dash c-fin" x1="30" y1="56" x2="214" y2="56"/>
      <rect id="clip-b1" x="44" y="92" width="18" height="28" class="skl-pop f-mid"/>
      <rect id="clip-b2" x="72" y="78" width="18" height="42" class="skl-pop f-mid"/>
      <rect id="clip-b3" x="100" y="56" width="18" height="64" class="skl-pop f-mid"/>
      <rect id="clip-b4" x="128" y="100" width="18" height="20" class="skl-pop f-mid"/>
      <rect id="clip-b5" x="156" y="56" width="18" height="64" class="skl-pop f-mid"/>
      <rect id="clip-b6" x="184" y="84" width="18" height="36" class="skl-pop f-mid"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const b3 = svg.querySelector('#clip-b3'), b5 = svg.querySelector('#clip-b5');
    tl.set(svg.querySelector('#clip-cap'), { opacity: 0.4 })
      .fromTo(b3, { attr: { y: 24, height: 96 } }, { attr: { y: 24, height: 96 }, duration: 0.01 }, 0)
      .fromTo(b5, { attr: { y: 32, height: 88 } }, { attr: { y: 32, height: 88 }, duration: 0.01 }, 0)
      .to(svg.querySelector('#clip-cap'), { opacity: 1, attr: { "stroke-width": 2 }, duration: 0.5 }, 0.5)
      .to(b3, { attr: { y: 56, height: 64 }, duration: 0.6, ease: "power3.in" }, 1.0)
      .to(b5, { attr: { y: 56, height: 64 }, duration: 0.6, ease: "power3.in" }, 1.05);
    return tl;
  },
},
interf: {
  tip: "The clipped amplitude A_i is re-attached as the modulus of the unit phasor, producing an interferogram whose argument is the residual elevation phase and whose magnitude is a bounded SNR proxy.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="120" y1="30" x2="120" y2="122"/>
      <line class="skl-axis" x1="74" y1="76" x2="166" y2="76"/>
      <circle id="interf-unit" cx="120" cy="76" r="20" fill="none" class="skl-dash c-faint" style="opacity:0.5"/>
      <circle id="interf-out" cx="120" cy="76" r="40" fill="none" class="skl-dash c-faint" style="opacity:0.3"/>
      <line id="interf-p" class="skl-draw c-mid" x1="120" y1="76" x2="138" y2="67"/>
      <circle id="interf-tip" cx="138" cy="67" r="3.5" class="skl-pop f-cal"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const p = svg.querySelector('#interf-p'), tip = svg.querySelector('#interf-tip');
    tl.set(p, { attr: { x2: 138, y2: 67 } }).set(tip, { attr: { cx: 138, cy: 67 } })
      .to(svg.querySelector('#interf-out'), { opacity: 0.7, duration: 0.5 }, 0.2)
      .to(p, { attr: { x2: 156, y2: 58 }, duration: 0.7, ease: "back.out(1.4)" }, 0.5)
      .to(tip, { attr: { cx: 156, cy: 58 }, duration: 0.7, ease: "back.out(1.4)" }, 0.5)
      .to(p, { rotation: 360, svgOrigin: "120 76", duration: 2.0, ease: "none" }, 1.3)
      .to(tip, { rotation: 360, svgOrigin: "120 76", duration: 2.0, ease: "none" }, 1.3);
    return tl;
  },
},
subdivide: {
  tip: "A crop above W_max = 1000 lines is split into M non-overlapping azimuth subsections processed by a worker plan resolved from the core budget B = floor(C f_effort).",
  build(svg) {
    svg.innerHTML = `
      <rect id="subdivide-strip" x="40" y="30" width="40" height="106" rx="2" class="skl-pop f-meas" style="opacity:0.8"/>
      <text x="108" y="80" text-anchor="middle" style="fill:#4a5a6b;font-size:12px;font-family:sans-serif">&#8594;</text>
      <g id="subdivide-tiles">
        <rect class="subdivide-tile skl-pop f-mid" x="134" y="30" width="76" height="22" rx="2"/>
        <rect class="subdivide-tile skl-pop f-mid" x="134" y="56" width="76" height="22" rx="2"/>
        <rect class="subdivide-tile skl-pop f-mid" x="134" y="82" width="76" height="22" rx="2"/>
        <rect class="subdivide-tile skl-pop f-mid" x="134" y="108" width="76" height="22" rx="2"/>
      </g>
      <circle class="subdivide-worker skl-pop f-cal" cx="146" cy="41" r="3"/>
      <circle class="subdivide-worker skl-pop f-cal" cx="146" cy="67" r="3"/>
      <circle class="subdivide-worker skl-pop f-cal" cx="146" cy="93" r="3"/>
      <circle class="subdivide-worker skl-pop f-cal" cx="146" cy="119" r="3"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const tiles = svg.querySelectorAll('.subdivide-tile');
    const workers = svg.querySelectorAll('.subdivide-worker');
    tl.set(tiles, { opacity: 0, x: -30 })
      .set(workers, { scale: 0, transformOrigin: "center" })
      .set(svg.querySelector('#subdivide-strip'), { scaleY: 1, transformOrigin: "60px 83px" })
      .to(svg.querySelector('#subdivide-strip'), { scaleY: 1.04, duration: 0.3, yoyo: true, repeat: 1 }, 0.1)
      .to(tiles, { opacity: 0.85, x: 0, duration: 0.4, stagger: 0.12, ease: "power2.out" }, 0.5)
      .to(workers, { scale: 1, duration: 0.3, stagger: 0.12, ease: "back.out(2.5)" }, 1.1)
      .to(workers, { scale: 1.5, duration: 0.4, stagger: { each: 0.08, repeat: 1, yoyo: true } }, 1.7);
    return tl;
  },
},
covariance: {
  tip: "A 20x10 px Boxcar window slides over the interferometric stack, averaging outer products into the per-pixel sample covariance R-hat that the Capon estimator later inverts.",
  build(svg) {
    svg.innerHTML = `
      <rect id="covariance-box" x="40" y="34" width="40" height="20" fill="none" class="skl-draw c-mid" style="stroke-width:2"/>
      <g id="covariance-mat" transform="translate(150,40)">
        <rect class="cov-cell skl-pop f-faint" x="0" y="0" width="20" height="20"/>
        <rect class="cov-cell skl-pop f-faint" x="22" y="0" width="20" height="20"/>
        <rect class="cov-cell skl-pop f-faint" x="0" y="22" width="20" height="20"/>
        <rect class="cov-cell skl-pop f-faint" x="22" y="22" width="20" height="20"/>
        <rect class="cov-cell skl-pop f-faint" x="44" y="0" width="20" height="20"/>
        <rect class="cov-cell skl-pop f-faint" x="44" y="22" width="20" height="20"/>
        <rect class="cov-cell skl-pop f-faint" x="0" y="44" width="20" height="20"/>
        <rect class="cov-cell skl-pop f-faint" x="22" y="44" width="20" height="20"/>
        <rect class="cov-cell skl-pop f-faint" x="44" y="44" width="20" height="20"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.4 });
    const box = svg.querySelector('#covariance-box');
    const cells = svg.querySelectorAll('.cov-cell');
    tl.set(box, { attr: { x: 40, y: 30 } })
      .set(cells, { opacity: 0.25 })
      .to(box, { attr: { x: 86, y: 30 }, duration: 0.5, ease: "none" }, 0)
      .to(cells[0], { opacity: 1, fill: "rgba(62,200,192,0.7)", duration: 0.3 }, 0.4)
      .to(box, { attr: { x: 40, y: 62 }, duration: 0.5, ease: "none" }, 0.6)
      .to([cells[1], cells[2], cells[4]], { opacity: 1, fill: "rgba(62,200,192,0.5)", duration: 0.3, stagger: 0.1 }, 1.0)
      .to(box, { attr: { x: 86, y: 62 }, duration: 0.5, ease: "none" }, 1.4)
      .to([cells[3], cells[5], cells[6], cells[7], cells[8]], { opacity: 1, fill: "rgba(62,200,192,0.45)", duration: 0.3, stagger: 0.08 }, 1.8);
    return tl;
  },
},
capon: {
  tip: "Over a uniform elevation grid spanning [x_min, x_max] the minimum-variance estimator evaluates 1/(a^H R^-1 a), sweeping the steering vector until a sharp peak locks onto the true height.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="36" y1="120" x2="212" y2="120"/>
      <path id="capon-spec" class="skl-draw c-cal" fill="none" d="M36 116 L70 113 L100 108 L120 60 L140 108 L172 114 L212 117"/>
      <line id="capon-scan" class="skl-draw c-mid" x1="120" y1="28" x2="120" y2="120" style="opacity:0.7"/>
      <circle id="capon-peak" cx="120" cy="60" r="4" class="skl-pop f-fin"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const scan = svg.querySelector('#capon-scan');
    const spec = svg.querySelector('#capon-spec');
    const peak = svg.querySelector('#capon-peak');
    tl.set(scan, { attr: { x1: 36, x2: 36 } })
      .set(spec, { attr: { d: "M36 116 L70 116 L100 116 L120 116 L140 116 L172 116 L212 116" } })
      .set(peak, { opacity: 0, attr: { cx: 36, cy: 116 } })
      .to(scan, { attr: { x1: 212, x2: 212 }, duration: 1.6, ease: "sine.inOut" }, 0)
      .to(spec, { attr: { d: "M36 116 L70 113 L100 108 L120 60 L140 108 L172 114 L212 117" }, duration: 1.6, ease: "power2.out" }, 0)
      .to(scan, { attr: { x1: 120, x2: 120 }, duration: 0.4, ease: "power2.out" }, 1.7)
      .to(peak, { opacity: 1, attr: { cx: 120, cy: 60 }, duration: 0.4, ease: "back.out(2)" }, 1.9)
      .fromTo(peak, { scale: 1, transformOrigin: "center" }, { scale: 1.6, duration: 0.35, yoyo: true, repeat: 1 }, 2.3);
    return tl;
  },
},
concat: {
  tip: "Each worker's HDF5 subsection is reassembled along azimuth, stacking the DEM on axis 0 and the tomogram on axis 1 into the full-stack outputs.",
  build(svg) {
    svg.innerHTML = `
      <g id="concat-pieces">
        <rect class="concat-tile skl-pop f-cal" x="44" y="34" width="58" height="20" rx="2"/>
        <rect class="concat-tile skl-pop f-cal" x="44" y="58" width="58" height="20" rx="2"/>
        <rect class="concat-tile skl-pop f-cal" x="44" y="82" width="58" height="20" rx="2"/>
        <rect class="concat-tile skl-pop f-cal" x="44" y="106" width="58" height="20" rx="2"/>
      </g>
      <rect id="concat-out" x="150" y="34" width="58" height="92" rx="3" fill="none" class="skl-dash c-fin" style="opacity:0.4"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const tiles = svg.querySelectorAll('.concat-tile');
    const targY = [34, 57, 80, 103];
    tl.set(tiles, { x: 0, y: 0, opacity: 0.9 })
      .set(svg.querySelector('#concat-out'), { opacity: 0.3 });
    tiles.forEach((t, i) => {
      tl.to(t, { x: 106, y: targY[i] - (34 + i * 24), duration: 0.6, ease: "power2.inOut" }, 0.3 + i * 0.35);
    });
    tl.to(svg.querySelector('#concat-out'), { opacity: 0.95, attr: { "stroke-width": 2 }, duration: 0.5 }, 2.0);
    return tl;
  },
},
threshold: {
  tip: "Magnitude samples below the relative floor t_f times the profile peak are zeroed and everything past index H_tr is dropped before the loss or R-squared ever see the profile.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="120" x2="214" y2="120"/>
      <line id="threshold-floor" class="skl-dash c-mid" x1="24" y1="96" x2="180" y2="96"/>
      <line id="threshold-trunc" class="skl-dash c-faint" x1="180" y1="20" x2="180" y2="124"/>
      <path id="threshold-raw" class="skl-draw c-meas" d="M24 118 L40 110 L52 70 L64 104 L82 112 L98 44 L114 100 L132 108 L150 78 L168 106 L186 90 L204 86" style="opacity:0.28"/>
      <path id="threshold-keep" class="skl-draw c-cal" d="M24 120 L40 120 L52 70 L64 104 L82 120 L98 44 L114 100 L132 120 L150 78 L168 120 L180 120"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const keep = svg.querySelector('#threshold-keep');
    const trunc = svg.querySelector('#threshold-trunc');
    const floor = svg.querySelector('#threshold-floor');
    tl.fromTo(svg.querySelector('#threshold-raw'), { opacity: 0.55 }, { opacity: 0.28, duration: 0.8 }, 0)
      .fromTo(floor, { attr: { x2: 24 }, opacity: 0 }, { attr: { x2: 180 }, opacity: 1, duration: 0.7 }, 0.2)
      .fromTo(keep, { opacity: 0 }, { opacity: 1, duration: 0.9, ease: "power2.out" }, 0.9)
      .fromTo(trunc, { opacity: 0, attr: { y1: 124 } }, { opacity: 1, attr: { y1: 20 }, duration: 0.6 }, 1.4)
      .to(svg.querySelector('#threshold-raw'), { opacity: 0.05, x: 6, duration: 0.5 }, 1.4);
    return tl;
  },
},
activity: {
  tip: "A pixel is fitted only when its profile maximum clears the activity threshold tau_a of 1e-3; otherwise it is skipped with parameters left at zero and scale fixed to one.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <line id="activity-gate" class="skl-dash c-mid" x1="24" y1="58" x2="214" y2="58"/>
      <text x="28" y="52" fill="#f5b971" style="font:italic 9px serif">tau_a</text>
      <path id="activity-pass" class="skl-draw c-cal" d="M40 118 L52 110 L66 44 L80 112 L92 118" />
      <circle id="activity-passdot" cx="66" cy="44" r="4" class="skl-pop f-cal"/>
      <path id="activity-fail" class="skl-draw c-meas" d="M150 118 L162 112 L176 88 L190 113 L202 118" style="opacity:0.4"/>
      <circle id="activity-faildot" cx="176" cy="88" r="4" class="skl-pop f-faint"/>
      <text id="activity-skip" x="158" y="76" fill="#4a5a6b" style="font:9px sans-serif">skip</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    tl.set(svg.querySelector('#activity-skip'), { opacity: 0 });
    tl.fromTo(svg.querySelector('#activity-passdot'), { scale: 0, transformOrigin: "66px 44px" }, { scale: 1, duration: 0.4, ease: "back.out(2)" }, 0.3)
      .to(svg.querySelector('#activity-passdot'), { fill: "#4fd6c4", scale: 1.4, duration: 0.3, yoyo: true, repeat: 1, transformOrigin: "66px 44px" }, 0.7)
      .to(svg.querySelector('#activity-pass'), { opacity: 1, duration: 0.3 }, 0.7)
      .fromTo(svg.querySelector('#activity-faildot'), { scale: 0, transformOrigin: "176px 88px" }, { scale: 1, duration: 0.4 }, 1.1)
      .to([svg.querySelector('#activity-fail'), svg.querySelector('#activity-faildot')], { opacity: 0.18, duration: 0.5 }, 1.5)
      .fromTo(svg.querySelector('#activity-skip'), { opacity: 0, y: -4 }, { opacity: 0.9, y: 0, duration: 0.5 }, 1.5);
    return tl;
  },
},
pnorm: {
  tip: "Dividing every bin by the per-pixel maximum lifts the tallest peak to exactly one, so the MSE and complexity penalty are comparable across pixels of any backscatter level.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="120" x2="214" y2="120"/>
      <line class="skl-axis" x1="24" y1="20" x2="24" y2="120"/>
      <line id="pnorm-one" class="skl-dash c-cal" x1="24" y1="34" x2="214" y2="34" style="opacity:0"/>
      <text id="pnorm-onelbl" x="200" y="30" fill="#4fd6c4" style="font:9px sans-serif;opacity:0">1.0</text>
      <path id="pnorm-curve" class="skl-draw c-mid" d="M24 118 L48 108 L70 92 L92 64 L114 50 L136 64 L160 96 L188 110 L214 116"/>
      <line id="pnorm-peak" class="skl-dash c-faint" x1="114" y1="50" x2="114" y2="120"/>
      <text id="pnorm-divlbl" x="120" y="86" fill="#f5b971" style="font:italic 10px serif">/ s</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const curve = svg.querySelector('#pnorm-curve');
    tl.set(curve, { transformOrigin: "24px 120px", scaleY: 1 });
    tl.to(svg.querySelector('#pnorm-divlbl'), { opacity: 1, scale: 1.2, transformOrigin: "128px 80px", duration: 0.4, yoyo: true, repeat: 1 }, 0.2)
      .to(curve, { scaleY: 1.886, duration: 1.0, ease: "power2.inOut" }, 0.6)
      .fromTo(svg.querySelector('#pnorm-one'), { opacity: 0 }, { opacity: 1, duration: 0.5 }, 1.2)
      .fromTo(svg.querySelector('#pnorm-onelbl'), { opacity: 0 }, { opacity: 1, duration: 0.5 }, 1.2)
      .to(svg.querySelector('#pnorm-peak'), { opacity: 0.5, duration: 0.4 }, 1.2);
    return tl;
  },
},
peakfind: {
  tip: "find_peaks scans the raw profile with no smoothing, keeping a maximum only when its topographic prominence reaches p_frac of the peak and it sits at least d_min bins from any rival.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="120" x2="214" y2="120"/>
      <path class="skl-draw c-meas" d="M24 118 L44 96 L60 40 L78 102 L96 110 L114 70 L132 104 L150 112 L168 58 L190 104 L214 116"/>
      <g id="peakfind-prom">
        <line class="skl-dash c-mid" x1="60" y1="40" x2="60" y2="84"/>
        <line class="skl-dash c-mid" x1="168" y1="58" x2="168" y2="92"/>
        <line class="skl-dash c-faint" x1="114" y1="70" x2="114" y2="80"/>
      </g>
      <circle id="peakfind-p1" cx="60" cy="40" r="4.5" class="skl-pop f-cal"/>
      <circle id="peakfind-p2" cx="168" cy="58" r="4.5" class="skl-pop f-cal"/>
      <circle id="peakfind-rej" cx="114" cy="70" r="4" class="skl-pop f-faint"/>
      <line id="peakfind-dmin" class="skl-draw c-faint" x1="60" y1="132" x2="80" y2="132" style="stroke-width:1.4"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    tl.set([svg.querySelector('#peakfind-p1'), svg.querySelector('#peakfind-p2'), svg.querySelector('#peakfind-rej')], { scale: 0 });
    tl.to(svg.querySelector('#peakfind-prom'), { opacity: 1, duration: 0.4 }, 0.1)
      .to(svg.querySelector('#peakfind-p1'), { scale: 1, duration: 0.4, ease: "back.out(2.2)", transformOrigin: "60px 40px" }, 0.4)
      .to(svg.querySelector('#peakfind-p2'), { scale: 1, duration: 0.4, ease: "back.out(2.2)", transformOrigin: "168px 58px" }, 0.7)
      .to(svg.querySelector('#peakfind-rej'), { scale: 1, duration: 0.3, transformOrigin: "114px 70px" }, 1.0)
      .to(svg.querySelector('#peakfind-rej'), { scale: 0.5, opacity: 0.2, duration: 0.5, transformOrigin: "114px 70px" }, 1.4)
      .fromTo(svg.querySelector('#peakfind-dmin'), { opacity: 0 }, { opacity: 1, duration: 0.4 }, 1.4);
    return tl;
  },
},
geometry: {
  tip: "The span-derived sigma_base sets the residual-suppression distance and seeds the initial width sigma_base over D_sigma, while Adam is clamped between one elevation bin and half the elevation span.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="120" x2="40" y2="24"/>
      <line id="geometry-hi" class="skl-dash c-cal" x1="34" y1="40" x2="120" y2="40"/>
      <line id="geometry-lo" class="skl-dash c-meas" x1="34" y1="110" x2="120" y2="110"/>
      <rect id="geometry-band" x="40" y="40" width="0" height="70" class="skl-pop f-mid" style="opacity:0.12"/>
      <g id="geometry-bell" transform="translate(170 0)">
        <path id="geometry-curve" class="skl-draw c-mid" d="M-44 116 q44 -78 88 0"/>
        <line id="geometry-arrowL" class="skl-draw c-mid" x1="0" y1="80" x2="-22" y2="80" style="stroke-width:1.4"/>
        <line id="geometry-arrowR" class="skl-draw c-mid" x1="0" y1="80" x2="22" y2="80" style="stroke-width:1.4"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const band = svg.querySelector('#geometry-band');
    const bell = svg.querySelector('#geometry-bell');
    tl.set(band, { attr: { width: 0 } });
    tl.fromTo([svg.querySelector('#geometry-hi'), svg.querySelector('#geometry-lo')], { attr: { x2: 34 }, opacity: 0 }, { attr: { x2: 120 }, opacity: 1, duration: 0.6, stagger: 0.2 }, 0.2)
      .to(band, { attr: { width: 80 }, duration: 0.7, ease: "power2.out" }, 0.8)
      .fromTo(bell, { scaleX: 0.3, transformOrigin: "170px 116px" }, { scaleX: 1, duration: 0.9, ease: "elastic.out(1,0.6)" }, 1.2);
    return tl;
  },
},
residfill: {
  tip: "When fewer than K peaks are found, a window of half-width d_min is zeroed around each detected peak and the empty slots are filled by repeated argmax of that masked residual.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <path id="residfill-prof" class="skl-draw c-meas" d="M24 116 L48 70 L66 100 L86 84 L108 50 L130 96 L150 78 L172 104 L196 66 L214 110" style="opacity:0.35"/>
      <circle cx="48" cy="70" r="4" class="skl-pop f-cal"/>
      <circle cx="108" cy="50" r="4" class="skl-pop f-cal"/>
      <rect id="residfill-mask1" x="32" y="20" width="32" height="98" class="skl-pop f-faint" style="opacity:0"/>
      <rect id="residfill-mask2" x="92" y="20" width="32" height="98" class="skl-pop f-faint" style="opacity:0"/>
      <circle id="residfill-new" cx="196" cy="66" r="4.5" class="skl-pop f-mid" style="opacity:0"/>
      <text id="residfill-arg" x="172" y="52" fill="#f5b971" style="font:italic 9px serif;opacity:0">argmax</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    tl.set([svg.querySelector('#residfill-mask1'), svg.querySelector('#residfill-mask2'), svg.querySelector('#residfill-new'), svg.querySelector('#residfill-arg')], { opacity: 0 });
    tl.to([svg.querySelector('#residfill-mask1'), svg.querySelector('#residfill-mask2')], { opacity: 0.5, duration: 0.5, stagger: 0.15 }, 0.3)
      .fromTo(svg.querySelector('#residfill-arg'), { opacity: 0, x: 12 }, { opacity: 1, x: 0, duration: 0.5 }, 1.0)
      .fromTo(svg.querySelector('#residfill-new'), { opacity: 0, scale: 0, transformOrigin: "196px 66px" }, { opacity: 1, scale: 1, duration: 0.5, ease: "back.out(2.2)" }, 1.3)
      .to(svg.querySelector('#residfill-new'), { scale: 1.4, yoyo: true, repeat: 1, duration: 0.25, transformOrigin: "196px 66px" }, 1.8);
    return tl;
  },
},
seed: {
  tip: "Amplitude and mean are read straight off the raw-profile peaks and frozen through Phase 2, collapsing the fit to a well-conditioned one-dimensional width search per component.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <path class="skl-draw c-meas" d="M24 116 L52 100 L78 44 L104 102 L130 110 L158 60 L190 104 L214 112" style="opacity:0.4"/>
      <g id="seed-lock1">
        <line class="skl-dash c-cal" x1="78" y1="44" x2="78" y2="118"/>
        <line class="skl-draw c-cal" x1="62" y1="44" x2="94" y2="44" style="stroke-width:1.6"/>
        <circle cx="78" cy="44" r="4" class="skl-pop f-cal"/>
      </g>
      <g id="seed-lock2">
        <line class="skl-dash c-cal" x1="158" y1="60" x2="158" y2="118"/>
        <line class="skl-draw c-cal" x1="142" y1="60" x2="174" y2="60" style="stroke-width:1.6"/>
        <circle cx="158" cy="60" r="4" class="skl-pop f-cal"/>
      </g>
      <text id="seed-mu" x="70" y="132" fill="#4fd6c4" style="font:italic 9px serif">mu</text>
      <text id="seed-a" x="98" y="40" fill="#4fd6c4" style="font:italic 9px serif">a</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    tl.fromTo(svg.querySelector('#seed-lock1'), { opacity: 0, scale: 0.7, transformOrigin: "78px 118px" }, { opacity: 1, scale: 1, duration: 0.5, ease: "back.out(2)" }, 0.3)
      .fromTo(svg.querySelector('#seed-lock2'), { opacity: 0, scale: 0.7, transformOrigin: "158px 118px" }, { opacity: 1, scale: 1, duration: 0.5, ease: "back.out(2)" }, 0.6)
      .fromTo([svg.querySelector('#seed-mu'), svg.querySelector('#seed-a')], { opacity: 0 }, { opacity: 1, duration: 0.5 }, 1.0)
      .to([svg.querySelector('#seed-lock1'), svg.querySelector('#seed-lock2')], { opacity: 0.55, duration: 0.5, yoyo: true, repeat: 1 }, 1.6);
    return tl;
  },
},
objective: {
  tip: "With amplitudes and means frozen, the loss is the mean-squared gap between the K-Gaussian sum and the normalised profile, with the exponent clipped and each sigma floored at 1e-6 before squaring.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="116" x2="214" y2="116"/>
      <path id="objective-target" class="skl-draw c-mid" d="M24 114 q40 -76 80 0 q40 -56 86 0 L214 114"/>
      <path id="objective-model" class="skl-draw c-cal" d="M24 114 q40 -40 80 0 q40 -30 86 0 L214 114"/>
      <g id="objective-resid">
        <line class="skl-draw c-fin" x1="64" y1="78" x2="64" y2="98" style="stroke-width:1.4"/>
        <line class="skl-draw c-fin" x1="150" y1="86" x2="150" y2="100" style="stroke-width:1.4"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const model = svg.querySelector('#objective-model');
    tl.set(svg.querySelector('#objective-resid'), { opacity: 0 });
    tl.fromTo(model, { attr: { d: "M24 114 q40 -40 80 0 q40 -30 86 0 L214 114" } }, { attr: { d: "M24 114 q40 -70 80 0 q40 -52 86 0 L214 114" }, duration: 1.1, ease: "power2.inOut" }, 0.3)
      .fromTo(svg.querySelector('#objective-resid'), { opacity: 0 }, { opacity: 1, duration: 0.4 }, 0.3)
      .to(svg.querySelector('#objective-resid'), { opacity: 0.25, duration: 0.6 }, 1.0);
    return tl;
  },
},
adam: {
  tip: "Bias-corrected Adam runs as a single lax.scan of T equals 3000 steps compiled into one XLA program, with the widths clamped to the sigma_lo and sigma_hi band before the loop and after every step.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="120" x2="214" y2="120"/>
      <line class="skl-axis" x1="28" y1="20" x2="28" y2="120"/>
      <line id="adam-hi" class="skl-dash c-faint" x1="28" y1="36" x2="214" y2="36"/>
      <line id="adam-lo" class="skl-dash c-faint" x1="28" y1="104" x2="214" y2="104"/>
      <path id="adam-curve" class="skl-draw c-cal" d="M28 100 C70 96 84 70 108 56 S160 44 214 44"/>
      <circle id="adam-ball" cx="28" cy="100" r="4.5" class="skl-pop f-cal"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const curve = svg.querySelector('#adam-curve');
    const ball = svg.querySelector('#adam-ball');
    const len = curve.getTotalLength();
    tl.set(curve, { attr: { "stroke-dasharray": len, "stroke-dashoffset": len } });
    tl.set(ball, { x: 0, y: 0 });
    tl.to(curve, { attr: { "stroke-dashoffset": 0 }, duration: 2.0, ease: "power2.out" }, 0)
      .to(ball, { x: 80, y: -44, duration: 0.9, ease: "power2.out" }, 0)
      .to(ball, { x: 186, y: -56, duration: 1.1, ease: "sine.out" }, 0.9)
      .to(ball, { scale: 1.5, transformOrigin: "center", yoyo: true, repeat: 1, duration: 0.25 }, 2.0)
      .to(svg.querySelector('#adam-hi'), { opacity: 0.7, duration: 0.3, yoyo: true, repeat: 1 }, 1.6);
    return tl;
  },
},
scoreK: {
  tip: "Each order K is scored as its normalised-profile MSE plus the complexity penalty lambda_K times K times the mean amplitude a-bar_K, so a slot is paid for only when a real peak fills it.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="32" y1="118" x2="214" y2="118"/>
      <g id="scoreK-mse">
        <rect x="44" y="58" width="20" height="60" class="skl-pop f-mid"/>
        <rect x="92" y="92" width="20" height="26" class="skl-pop f-mid"/>
        <rect x="140" y="86" width="20" height="32" class="skl-pop f-mid"/>
        <rect x="188" y="80" width="20" height="38" class="skl-pop f-mid"/>
      </g>
      <g id="scoreK-pen">
        <rect x="44" y="52" width="20" height="6" class="skl-pop f-fin"/>
        <rect x="92" y="80" width="20" height="12" class="skl-pop f-fin"/>
        <rect x="140" y="62" width="20" height="24" class="skl-pop f-fin"/>
        <rect x="188" y="44" width="20" height="36" class="skl-pop f-fin"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const pens = svg.querySelectorAll('#scoreK-pen rect');
    const mses = svg.querySelectorAll('#scoreK-mse rect');
    tl.from(mses, { attr: { height: 0, y: 118 }, duration: 0.6, stagger: 0.12, ease: "power2.out" }, 0);
    tl.from(pens, { scaleY: 0, transformOrigin: "center bottom", opacity: 0, duration: 0.5, stagger: 0.12, ease: "back.out(1.8)" }, 0.7);
    tl.fromTo(pens, { opacity: 0.5 }, { opacity: 1, duration: 0.4, stagger: 0.1 }, 0.7);
    return tl;
  },
},
selectK: {
  tip: "The penalised score is minimised over model order with ties broken toward the smaller K, so a tie between two and three components is always resolved in favour of the simpler fit.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="32" y1="116" x2="214" y2="116"/>
      <rect x="44" y="48" width="22" height="68" class="skl-pop f-mid"/>
      <rect id="selectK-win" x="92" y="84" width="22" height="32" class="skl-pop f-cal"/>
      <rect x="140" y="76" width="22" height="40" class="skl-pop f-mid"/>
      <rect x="188" y="62" width="22" height="54" class="skl-pop f-mid"/>
      <path id="selectK-arrow" class="skl-draw c-cal" d="M103 34 L103 62 M97 56 L103 62 L109 56" style="opacity:0"/>
      <text id="selectK-star" x="98" y="28" fill="#4fd6c4" style="font:bold 11px sans-serif;opacity:0">K*=2</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const bars = svg.querySelectorAll('rect');
    const win = svg.querySelector('#selectK-win');
    tl.set(win, { fill: "#f5b971" });
    tl.from(bars, { attr: { height: 0, y: 116 }, duration: 0.6, stagger: 0.1, ease: "power2.out" }, 0)
      .to(win, { fill: "#4fd6c4", duration: 0.4 }, 0.9)
      .to(win, { scaleY: 1.08, transformOrigin: "center bottom", yoyo: true, repeat: 1, duration: 0.3 }, 0.9)
      .fromTo(svg.querySelector('#selectK-arrow'), { opacity: 0, y: -8 }, { opacity: 1, y: 0, duration: 0.4 }, 1.2)
      .fromTo(svg.querySelector('#selectK-star'), { opacity: 0, scale: 0.6, transformOrigin: "103px 28px" }, { opacity: 1, scale: 1, duration: 0.4, ease: "back.out(2)" }, 1.4);
    return tl;
  },
},
rescale: {
  tip: "The winner's amplitudes are multiplied back by the per-pixel scale s to return to raw backscatter units, while its means and widths are written through unchanged.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <line id="rescale-one" class="skl-dash c-faint" x1="24" y1="50" x2="214" y2="50"/>
      <path id="rescale-curve" class="skl-draw c-cal" d="M40 116 q40 -66 80 0 q34 -46 70 0"/>
      <text id="rescale-mul" x="120" y="40" fill="#f5b971" style="font:italic 10px serif;opacity:0">x s</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const curve = svg.querySelector('#rescale-curve');
    tl.set(curve, { transformOrigin: "40px 116px", scaleY: 1, stroke: "#4fd6c4" });
    tl.fromTo(svg.querySelector('#rescale-mul'), { opacity: 0, y: 6 }, { opacity: 1, y: 0, duration: 0.4 }, 0.3)
      .to(curve, { scaleY: 1.45, duration: 0.9, ease: "power2.out" }, 0.7)
      .to(curve, { stroke: "#c4a3ff", duration: 0.6 }, 0.7);
    return tl;
  },
},
assemble: {
  tip: "Active components are sorted by ascending mean elevation, inactive slots keyed to infinity drop to the end, and the result is written into the interleaved 3K target with zeros beyond K*.",
  build(svg) {
    svg.innerHTML = `
      <g id="assemble-src">
        <rect id="assemble-c1" x="40" y="34" width="26" height="20" class="skl-pop f-cal"/>
        <rect id="assemble-c2" x="40" y="62" width="26" height="20" class="skl-pop f-cal"/>
        <rect id="assemble-c0" x="40" y="90" width="26" height="20" class="skl-pop f-faint"/>
      </g>
      <line class="skl-draw c-faint" x1="78" y1="72" x2="104" y2="72" style="stroke-width:1.4"/>
      <path class="skl-draw c-faint" d="M98 67 L104 72 L98 77" style="stroke-width:1.4"/>
      <g id="assemble-tgt">
        <rect x="118" y="58" width="16" height="28" class="skl-pop f-fin" style="opacity:0.25"/>
        <rect x="136" y="58" width="16" height="28" class="skl-pop f-fin" style="opacity:0.25"/>
        <rect x="154" y="58" width="16" height="28" class="skl-pop f-fin" style="opacity:0.25"/>
        <rect x="172" y="58" width="16" height="28" class="skl-pop f-fin" style="opacity:0.25"/>
        <rect x="190" y="58" width="16" height="28" class="skl-pop f-fin" style="opacity:0.25"/>
        <rect x="208" y="58" width="6" height="28" class="skl-pop f-faint" style="opacity:0.4"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const tgt = svg.querySelectorAll('#assemble-tgt rect');
    tl.set([svg.querySelector('#assemble-c1'), svg.querySelector('#assemble-c2')], { y: "+=0" });
    tl.to(svg.querySelector('#assemble-c2'), { y: 34, opacity: 0, duration: 0.5, ease: "power1.in" }, 0.2)
      .to(svg.querySelector('#assemble-c1'), { y: 48, opacity: 0, duration: 0.5, ease: "power1.in" }, 0.35)
      .to(svg.querySelector('#assemble-c0'), { opacity: 0.2, x: 8, duration: 0.5 }, 0.4)
      .to([tgt[0], tgt[1], tgt[2]], { opacity: 1, duration: 0.4, stagger: 0.1 }, 0.7)
      .to([tgt[3], tgt[4]], { opacity: 0.18, duration: 0.4 }, 1.2);
    return tl;
  },
},
quality: {
  tip: "The per-pixel R-squared compares the reconstructed mixture against the thresholded, truncated profile with a 1e-12 stabiliser on the total sum of squares, then paints a fit-quality map.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="116" x2="110" y2="116"/>
      <path class="skl-draw c-meas" d="M24 114 L36 100 L48 60 L62 96 L76 108 L90 72 L104 110" style="opacity:0.5"/>
      <path id="quality-fit" class="skl-draw c-cal" d="M24 114 q24 -60 40 -54 q20 6 20 40 q8 14 20 14"/>
      <text id="quality-r2" x="40" y="30" fill="#4fd6c4" style="font:italic 10px serif;opacity:0">R2</text>
      <g id="quality-map">
        <rect x="134" y="34" width="22" height="22" class="skl-pop f-cal" style="opacity:0"/>
        <rect x="158" y="34" width="22" height="22" class="skl-pop f-cal" style="opacity:0"/>
        <rect x="182" y="34" width="22" height="22" class="skl-pop f-mid" style="opacity:0"/>
        <rect x="134" y="58" width="22" height="22" class="skl-pop f-mid" style="opacity:0"/>
        <rect x="158" y="58" width="22" height="22" class="skl-pop f-cal" style="opacity:0"/>
        <rect x="182" y="58" width="22" height="22" class="skl-pop f-cal" style="opacity:0"/>
        <rect x="134" y="82" width="22" height="22" class="skl-pop f-cal" style="opacity:0"/>
        <rect x="158" y="82" width="22" height="22" class="skl-pop f-faint" style="opacity:0"/>
        <rect x="182" y="82" width="22" height="22" class="skl-pop f-cal" style="opacity:0"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const cells = svg.querySelectorAll('#quality-map rect');
    tl.set(cells, { opacity: 0, scale: 0.6, transformOrigin: "center" });
    tl.fromTo(svg.querySelector('#quality-fit'), { opacity: 0 }, { opacity: 1, duration: 0.7 }, 0.2)
      .fromTo(svg.querySelector('#quality-r2'), { opacity: 0, y: 6 }, { opacity: 1, y: 0, duration: 0.4 }, 0.7)
      .to(cells, { opacity: 0.85, scale: 1, duration: 0.4, stagger: { each: 0.08, from: "start" }, ease: "back.out(1.6)" }, 1.0);
    return tl;
  },
},
diagnostics: {
  tip: "Post-hoc only, the relative margin between the runner-up score L_2nd and L_K* flags ambiguous pixels, and the peak-to-floor contrast in dB uses the lowest-quartile bins as its noise floor.",
  build(svg) {
    svg.innerHTML = `
      <text x="30" y="24" fill="#c4a3ff" style="font:8px sans-serif">m_rel</text>
      <rect x="30" y="32" width="80" height="14" class="skl-pop f-faint" style="opacity:0.3"/>
      <rect id="diagnostics-mbar" x="30" y="32" width="0" height="14" class="skl-pop f-fin"/>
      <line id="diagnostics-flag" class="skl-dash c-mid" x1="70" y1="28" x2="70" y2="50"/>
      <line class="skl-axis" x1="128" y1="120" x2="214" y2="120"/>
      <line id="diagnostics-floor" class="skl-dash c-faint" x1="128" y1="104" x2="214" y2="104"/>
      <path class="skl-draw c-meas" d="M128 118 L144 110 L156 50 L170 112 L186 108 L200 116 L214 118"/>
      <line id="diagnostics-cdb" class="skl-draw c-cal" x1="156" y1="104" x2="156" y2="50" style="stroke-width:1.6;opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    tl.set(svg.querySelector('#diagnostics-mbar'), { attr: { width: 0 } });
    tl.to(svg.querySelector('#diagnostics-mbar'), { attr: { width: 56 }, duration: 0.9, ease: "power2.out" }, 0.2)
      .to(svg.querySelector('#diagnostics-flag'), { opacity: 1, x: 0, duration: 0.4 }, 0.2)
      .fromTo(svg.querySelector('#diagnostics-floor'), { opacity: 0 }, { opacity: 1, duration: 0.5 }, 1.0)
      .fromTo(svg.querySelector('#diagnostics-cdb'), { opacity: 0 }, { opacity: 1, duration: 0.6 }, 1.2);
    return tl;
  },
},
splitgeom: {
  tip: "The azimuth range splits 70/15/15 into contiguous train, validation and test bands that share the full range extent with no overlap.",
  build(svg) {
    svg.innerHTML = `
      <rect x="40" y="22" width="160" height="106" rx="2" class="skl-draw c-faint" style="fill:none;stroke-width:1.5"/>
      <rect id="splitgeom-tr" x="40" y="22" width="160" height="74" class="skl-pop f-mid" style="opacity:0.85"/>
      <rect id="splitgeom-va" x="40" y="96" width="160" height="16" class="skl-pop f-cal" style="opacity:0.85"/>
      <rect id="splitgeom-te" x="40" y="112" width="160" height="16" class="skl-pop f-fin" style="opacity:0.85"/>
      <line id="splitgeom-c1" class="skl-dash c-faint" x1="40" y1="96" x2="200" y2="96"/>
      <line id="splitgeom-c2" class="skl-dash c-faint" x1="40" y1="112" x2="200" y2="112"/>
      <text x="120" y="62" text-anchor="middle" style="fill:#cfd8e8;font-size:9px;font-family:sans-serif">train 70%</text>
      <text x="120" y="107" text-anchor="middle" style="fill:#cfd8e8;font-size:7px;font-family:sans-serif">val 15%</text>
      <text x="120" y="123" text-anchor="middle" style="fill:#cfd8e8;font-size:7px;font-family:sans-serif">test 15%</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const tr = svg.querySelector('#splitgeom-tr'), va = svg.querySelector('#splitgeom-va'), te = svg.querySelector('#splitgeom-te');
    tl.set([svg.querySelector('#splitgeom-c1'), svg.querySelector('#splitgeom-c2')], { opacity: 0 })
      .fromTo(tr, { attr: { y: 22, height: 0 }, opacity: 0 }, { attr: { y: 22, height: 74 }, opacity: 0.85, duration: 0.9, ease: "power2.out" }, 0.1)
      .to(svg.querySelector('#splitgeom-c1'), { opacity: 0.5, duration: 0.3 }, 0.9)
      .fromTo(va, { attr: { y: 96, height: 0 }, opacity: 0 }, { attr: { y: 96, height: 16 }, opacity: 0.85, duration: 0.6, ease: "power2.out" }, 1.0)
      .to(svg.querySelector('#splitgeom-c2'), { opacity: 0.5, duration: 0.3 }, 1.5)
      .fromTo(te, { attr: { y: 112, height: 0 }, opacity: 0 }, { attr: { y: 112, height: 16 }, opacity: 0.85, duration: 0.6, ease: "power2.out" }, 1.6);
    return tl;
  },
},
localslice: {
  tip: "Subtracting the global-crop origin az0_G converts each band's absolute azimuth bounds into zero-based slices into the memory-mapped arrays.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="48" x2="214" y2="48"/>
      <line class="skl-axis" x1="30" y1="104" x2="214" y2="104"/>
      <text x="30" y="38" style="fill:#7e8aa0;font-size:7px;font-family:sans-serif">absolute</text>
      <text x="30" y="124" style="fill:#7e8aa0;font-size:7px;font-family:sans-serif">local (0-based)</text>
      <line class="skl-draw c-faint" x1="30" y1="44" x2="30" y2="52" style="stroke-width:1.5"/>
      <text x="30" y="64" text-anchor="middle" style="fill:#7e8aa0;font-size:7px;font-family:sans-serif">az0_G</text>
      <rect id="localslice-abs" x="92" y="40" width="78" height="16" class="skl-pop f-mid" style="opacity:0.85"/>
      <line id="localslice-arrow" class="skl-draw c-cal" x1="131" y1="58" x2="131" y2="94" style="stroke-width:1.5"/>
      <rect id="localslice-loc" x="30" y="96" width="78" height="16" class="skl-pop f-cal" style="opacity:0.85"/>
      <text id="localslice-shift" x="170" y="78" text-anchor="middle" style="fill:#3fc7c0;font-size:8px;font-family:sans-serif;opacity:0">- az0_G</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const abs = svg.querySelector('#localslice-abs'), loc = svg.querySelector('#localslice-loc'), sh = svg.querySelector('#localslice-shift');
    tl.set(abs, { attr: { x: 92 }, opacity: 0.85 })
      .set(loc, { attr: { x: 92 }, opacity: 0 })
      .set(svg.querySelector('#localslice-arrow'), { opacity: 0 })
      .set(sh, { opacity: 0 })
      .to(sh, { opacity: 1, duration: 0.4 }, 0.4)
      .to(svg.querySelector('#localslice-arrow'), { opacity: 0.8, duration: 0.4 }, 0.6)
      .fromTo(loc, { attr: { x: 92 }, opacity: 0 }, { attr: { x: 30 }, opacity: 0.85, duration: 1.1, ease: "power2.inOut" }, 1.0)
      .to(sh, { opacity: 0, duration: 0.4 }, 2.2);
    return tl;
  },
},
secselect: {
  tip: "Flight-qualified labels in L_req map to positional indices pi, e.g. {3,5,7,25}, gathered identically from both the secondary SLCs and the interferograms.",
  build(svg) {
    svg.innerHTML = `
      <g id="secselect-row">
        <rect data-i="0" x="28" y="32" width="20" height="20" class="skl-pop f-faint" style="opacity:0.6"/>
        <rect data-i="1" x="52" y="32" width="20" height="20" class="skl-pop f-faint" style="opacity:0.6"/>
        <rect data-i="2" x="76" y="32" width="20" height="20" class="skl-pop f-faint" style="opacity:0.6"/>
        <rect data-i="3" x="100" y="32" width="20" height="20" class="skl-pop f-faint" style="opacity:0.6"/>
        <rect data-i="4" x="124" y="32" width="20" height="20" class="skl-pop f-faint" style="opacity:0.6"/>
        <rect data-i="5" x="148" y="32" width="20" height="20" class="skl-pop f-faint" style="opacity:0.6"/>
        <rect data-i="6" x="172" y="32" width="20" height="20" class="skl-pop f-faint" style="opacity:0.6"/>
      </g>
      <rect class="sel-mark" x="100" y="32" width="20" height="20" style="fill:none;stroke:#3fc7c0;stroke-width:0;opacity:0"/>
      <rect class="sel-mark" x="148" y="32" width="20" height="20" style="fill:none;stroke:#3fc7c0;stroke-width:0;opacity:0"/>
      <rect class="sel-mark" x="172" y="32" width="20" height="20" style="fill:none;stroke:#3fc7c0;stroke-width:0;opacity:0"/>
      <text x="120" y="92" text-anchor="middle" style="fill:#3fc7c0;font-size:7px;font-family:sans-serif">pi = {3, 5, 6}</text>
      <g id="secselect-out">
        <rect id="secselect-o0" x="100" y="104" width="20" height="20" class="skl-pop f-cal" style="opacity:0"/>
        <rect id="secselect-o1" x="124" y="104" width="20" height="20" class="skl-pop f-cal" style="opacity:0"/>
        <rect id="secselect-o2" x="148" y="104" width="20" height="20" class="skl-pop f-cal" style="opacity:0"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const marks = svg.querySelectorAll('.sel-mark');
    const outs = [svg.querySelector('#secselect-o0'), svg.querySelector('#secselect-o1'), svg.querySelector('#secselect-o2')];
    const src = [svg.querySelector('[data-i="3"]'), svg.querySelector('[data-i="5"]'), svg.querySelector('[data-i="6"]')];
    tl.set(marks, { attr: { 'stroke-width': 0 }, opacity: 0 })
      .set(outs, { opacity: 0, attr: { y: 104 } })
      .set(src, { opacity: 0.6 });
    marks.forEach((m, i) => {
      tl.to(m, { attr: { 'stroke-width': 2.5 }, opacity: 1, duration: 0.3 }, 0.3 + i * 0.35)
        .to(src[i], { opacity: 1, duration: 0.3 }, 0.3 + i * 0.35)
        .fromTo(outs[i], { opacity: 0, attr: { y: 88 } }, { opacity: 0.9, attr: { y: 104 }, duration: 0.5, ease: "power2.out" }, 0.45 + i * 0.35);
    });
    return tl;
  },
},
stack: {
  tip: "Primary lands at slot 0, the Ns secondaries fill X[1:1+Ns], and the Ni interferograms fill X[1+Ns:], all written by pass-index into one pre-allocated complex buffer.",
  build(svg) {
    svg.innerHTML = `
      <rect id="stack-s0" x="40" y="22" width="40" height="22" class="skl-pop f-meas" style="opacity:0"/>
      <rect class="stack-sec" x="92" y="22" width="40" height="22" class="skl-pop f-meas" style="opacity:0"/>
      <rect class="stack-sec" x="92" y="48" width="40" height="22" class="skl-pop f-meas" style="opacity:0"/>
      <rect class="stack-ifg" x="144" y="22" width="40" height="22" class="skl-pop f-meas" style="opacity:0"/>
      <rect class="stack-ifg" x="144" y="48" width="40" height="22" class="skl-pop f-meas" style="opacity:0"/>
      <g id="stack-buf">
        <rect x="86" y="84" width="68" height="44" rx="2" class="skl-draw c-cal" style="fill:none;stroke-width:1.5"/>
        <rect class="stack-slot" x="90" y="88" width="60" height="6" class="skl-pop f-faint" style="opacity:0.25"/>
        <rect class="stack-slot" x="90" y="96" width="60" height="6" class="skl-pop f-faint" style="opacity:0.25"/>
        <rect class="stack-slot" x="90" y="104" width="60" height="6" class="skl-pop f-faint" style="opacity:0.25"/>
        <rect class="stack-slot" x="90" y="112" width="60" height="6" class="skl-pop f-faint" style="opacity:0.25"/>
        <rect class="stack-slot" x="90" y="120" width="60" height="6" class="skl-pop f-faint" style="opacity:0.25"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const s0 = svg.querySelector('#stack-s0');
    const secs = svg.querySelectorAll('.stack-sec'), ifgs = svg.querySelectorAll('.stack-ifg');
    const slots = svg.querySelectorAll('.stack-slot');
    tl.set([s0, ...secs, ...ifgs], { opacity: 0 })
      .set(slots, { opacity: 0.25 });
    const fly = (src, slotIdx, t) => {
      const slot = slots[slotIdx];
      const dx = 90 - parseFloat(src.getAttribute('x'));
      const dy = parseFloat(slot.getAttribute('y')) - parseFloat(src.getAttribute('y'));
      tl.to(src, { opacity: 1, duration: 0.25 }, t)
        .to(src, { x: dx, y: dy, scaleX: 1, scaleY: 0.3, transformOrigin: "0px 0px", opacity: 0.4, duration: 0.55, ease: "power2.in" }, t + 0.25)
        .to(slot, { opacity: 0.9, duration: 0.3 }, t + 0.7);
    };
    fly(s0, 0, 0.2);
    fly(secs[0], 1, 0.7);
    fly(secs[1], 2, 1.1);
    fly(ifgs[0], 3, 1.5);
    fly(ifgs[1], 4, 1.9);
    return tl;
  },
},
patchgrid: {
  tip: "A strided P-by-P window tiles the region; ceil((Az-P)/s)+1 rows by ceil((Rg-P)/s)+1 columns guarantees the last patch still covers the far border.",
  build(svg) {
    svg.innerHTML = `
      <rect x="36" y="24" width="168" height="102" rx="2" class="skl-draw c-faint" style="fill:none;stroke-width:1.5"/>
      <g id="patchgrid-tiles"></g>
      <rect id="patchgrid-win" x="40" y="28" width="44" height="34" style="fill:none;stroke:#3fc7c0;stroke-width:2;opacity:0.9"/>`;
    const g = svg.querySelector('#patchgrid-tiles');
    for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) {
      const x = 40 + c * 41, y = 28 + r * 33;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("class", "patchgrid-cell");
      rect.setAttribute("x", x); rect.setAttribute("y", y);
      rect.setAttribute("width", 44); rect.setAttribute("height", 34);
      rect.setAttribute("data-x", x); rect.setAttribute("data-y", y);
      rect.setAttribute("style", "opacity:0.18;stroke:#e2a45a;stroke-width:0.8;fill:#e2a45a");
      g.appendChild(rect);
    }
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const cells = svg.querySelectorAll('.patchgrid-cell');
    const win = svg.querySelector('#patchgrid-win');
    tl.set(cells, { opacity: 0.08 });
    cells.forEach((cell, i) => {
      tl.to(win, { attr: { x: cell.getAttribute('data-x'), y: cell.getAttribute('data-y') }, duration: 0.22, ease: "power1.inOut" }, i * 0.22)
        .to(cell, { opacity: 0.4, duration: 0.18 }, i * 0.22 + 0.05);
    });
    return tl;
  },
},
padgeom: {
  tip: "The azimuth deficit pv splits as floor(pv/2) on top and pv-floor(pv/2) on the bottom, so an odd deficit puts the extra pixel at the bottom edge.",
  build(svg) {
    svg.innerHTML = `
      <rect x="78" y="40" width="84" height="70" rx="2" class="skl-pop f-mid" style="opacity:0.8"/>
      <text x="120" y="78" text-anchor="middle" style="fill:#cfd8e8;font-size:8px;font-family:sans-serif">region</text>
      <rect id="padgeom-top" x="78" y="40" width="84" height="0" class="skl-pop f-cal" style="opacity:0.45"/>
      <rect id="padgeom-bot" x="78" y="110" width="84" height="0" class="skl-pop f-cal" style="opacity:0.45"/>
      <text id="padgeom-tl" x="170" y="34" style="fill:#3fc7c0;font-size:7px;font-family:sans-serif;opacity:0">floor(pv/2)</text>
      <text id="padgeom-bl" x="170" y="120" style="fill:#3fc7c0;font-size:7px;font-family:sans-serif;opacity:0">pv-floor(pv/2)</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const top = svg.querySelector('#padgeom-top'), bot = svg.querySelector('#padgeom-bot');
    tl.set(top, { attr: { y: 40, height: 0 } })
      .set(bot, { attr: { y: 110, height: 0 } })
      .set([svg.querySelector('#padgeom-tl'), svg.querySelector('#padgeom-bl')], { opacity: 0 })
      .to(top, { attr: { y: 28, height: 12 }, duration: 0.8, ease: "power2.out" }, 0.3)
      .to(svg.querySelector('#padgeom-tl'), { opacity: 1, duration: 0.3 }, 0.9)
      .to(bot, { attr: { height: 16 }, duration: 0.8, ease: "power2.out" }, 1.3)
      .to(svg.querySelector('#padgeom-bl'), { opacity: 1, duration: 0.3 }, 1.9);
    return tl;
  },
},
extract: {
  tip: "The clipped read window is deep-copied so it never aliases the mmap, then reflect-padded in one pass by the same routine that serves the stack, the parameters and the DEM.",
  build(svg) {
    svg.innerHTML = `
      <rect x="30" y="30" width="80" height="80" rx="2" class="skl-draw c-faint" style="fill:none;stroke-width:1.2"/>
      <rect id="extract-read" x="62" y="62" width="48" height="48" class="skl-pop f-meas" style="opacity:0.75"/>
      <rect id="extract-copy" x="150" y="62" width="48" height="48" class="skl-pop f-mid" style="opacity:0"/>
      <g id="extract-pad" style="opacity:0">
        <rect x="150" y="42" width="48" height="20" class="skl-pop f-cal" style="opacity:0.4"/>
        <rect x="130" y="62" width="20" height="48" class="skl-pop f-cal" style="opacity:0.4"/>
      </g>
      <line id="extract-fl" class="skl-draw c-cal" x1="116" y1="86" x2="144" y2="86" style="stroke-width:1.5;opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const copy = svg.querySelector('#extract-copy'), pad = svg.querySelector('#extract-pad'), fl = svg.querySelector('#extract-fl');
    tl.set(copy, { opacity: 0, x: -88 })
      .set(pad, { opacity: 0 })
      .set(fl, { opacity: 0 })
      .to(svg.querySelector('#extract-read'), { opacity: 1, duration: 0.3 }, 0.2)
      .to(fl, { opacity: 0.9, duration: 0.3 }, 0.5)
      .to(copy, { opacity: 0.9, x: 0, duration: 0.8, ease: "power2.out" }, 0.7)
      .to(pad, { opacity: 1, duration: 0.6, ease: "power2.out" }, 1.6);
    return tl;
  },
},
represent: {
  tip: "By default an SLC pass keeps its magnitude |p| and an interferogram keeps its phase angle, while magnitude-normalised channels divide by m = max(|p|, 1) to guard zero magnitude.",
  build(svg) {
    svg.innerHTML = `
      <circle cx="64" cy="62" r="30" class="skl-axis" style="fill:none"/>
      <line class="skl-axis" x1="34" y1="62" x2="94" y2="62"/>
      <line class="skl-axis" x1="64" y1="32" x2="64" y2="92"/>
      <line id="represent-vec" class="skl-draw c-meas" x1="64" y1="62" x2="86" y2="42" style="stroke-width:2"/>
      <circle id="represent-tip" cx="86" cy="42" r="3" class="skl-pop f-meas"/>
      <path id="represent-arc" class="skl-draw c-mid" d="M84 62 A20 20 0 0 0 79 49" style="fill:none;stroke-width:1.5;opacity:0"/>
      <rect id="represent-mag" x="140" y="34" width="0" height="14" class="skl-pop f-cal" style="opacity:0.85"/>
      <text x="138" y="44" text-anchor="end" style="fill:#3fc7c0;font-size:7px;font-family:sans-serif">|p|</text>
      <rect id="represent-ph" x="140" y="56" width="0" height="14" class="skl-pop f-mid" style="opacity:0.85"/>
      <text x="138" y="66" text-anchor="end" style="fill:#e2a45a;font-size:7px;font-family:sans-serif">ang</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const vec = svg.querySelector('#represent-vec'), tip = svg.querySelector('#represent-tip');
    const arc = svg.querySelector('#represent-arc');
    const mag = svg.querySelector('#represent-mag'), ph = svg.querySelector('#represent-ph');
    tl.set(vec, { rotation: 0, transformOrigin: "64px 62px" })
      .set(arc, { opacity: 0 })
      .set([mag, ph], { attr: { width: 0 } })
      .to(vec, { rotation: -20, transformOrigin: "64px 62px", duration: 0.7, ease: "sine.inOut" }, 0.2)
      .to(tip, { x: 6, y: 8, duration: 0.7, ease: "sine.inOut" }, 0.2)
      .to(arc, { opacity: 0.9, duration: 0.3 }, 0.6)
      .to(mag, { attr: { width: 44 }, duration: 0.7, ease: "power2.out" }, 1.0)
      .to(ph, { attr: { width: 30 }, duration: 0.7, ease: "power2.out" }, 1.3);
    return tl;
  },
},
assemble_in: {
  tip: "Each source's real channels are concatenated along the channel axis in fixed order: primary, then secondaries r_S, then interferograms r_I, with the optional DEM channel last.",
  build(svg) {
    svg.innerHTML = `
      <rect class="asm-in" x="26" y="46" width="22" height="40" class="skl-pop f-meas" style="opacity:0.85"/>
      <rect class="asm-in" x="58" y="46" width="22" height="40" class="skl-pop f-mid" style="opacity:0.85"/>
      <rect class="asm-in" x="84" y="46" width="22" height="40" class="skl-pop f-mid" style="opacity:0.85"/>
      <rect class="asm-in" x="116" y="46" width="22" height="40" class="skl-pop f-cal" style="opacity:0.85"/>
      <rect class="asm-in" x="148" y="46" width="22" height="40" class="skl-pop f-faint" style="opacity:0.85"/>
      <rect id="assemble_in-out" x="40" y="100" width="0" height="26" rx="2" class="skl-draw c-cal" style="fill:none;stroke-width:1.6"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const ins = svg.querySelectorAll('.asm-in');
    const out = svg.querySelector('#assemble_in-out');
    tl.set(ins, { opacity: 0, y: -10 })
      .set(out, { attr: { x: 120, width: 0 } });
    let w = 0;
    ins.forEach((el, i) => {
      tl.to(el, { opacity: 0.85, y: 0, duration: 0.3, ease: "power2.out" }, 0.2 + i * 0.3)
        .to(out, { attr: { x: 120 - w - 14, width: w + 28 }, duration: 0.3, ease: "power2.out" }, 0.35 + i * 0.3);
      w += 28;
    });
    return tl;
  },
},
target: {
  tip: "Channels 3g+r are gathered from the interleaved (a, mu, sigma) ground-truth layout; enabling all three roles keeps every one of the n_g*3 parameter channels.",
  build(svg) {
    svg.innerHTML = `<g id="target-row"></g><g id="target-out"></g>`;
    const row = svg.querySelector('#target-row'), out = svg.querySelector('#target-out');
    const roles = ['a','mu','s'], cls = ['f-meas','f-mid','f-cal'];
    for (let g = 0; g < 3; g++) for (let r = 0; r < 3; r++) {
      const idx = 3 * g + r, x = 30 + idx * 20;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("class", "target-cell skl-pop " + cls[r]);
      rect.setAttribute("x", x); rect.setAttribute("y", 40);
      rect.setAttribute("width", 16); rect.setAttribute("height", 22);
      rect.setAttribute("style", "opacity:0.5");
      row.appendChild(rect);
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", x + 8); t.setAttribute("y", 55); t.setAttribute("text-anchor", "middle");
      t.setAttribute("style", "fill:#cfd8e8;font-size:6px;font-family:sans-serif;pointer-events:none");
      t.textContent = roles[r]; row.appendChild(t);
      const o = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      o.setAttribute("class", "target-pick");
      o.setAttribute("x", x); o.setAttribute("y", 92);
      o.setAttribute("width", 16); o.setAttribute("height", 22);
      o.setAttribute("style", "opacity:0;fill:#3fc7c0");
      out.appendChild(o);
    }
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const cells = svg.querySelectorAll('.target-cell');
    const picks = svg.querySelectorAll('.target-pick');
    tl.set(cells, { opacity: 0.5 })
      .set(picks, { opacity: 0, y: -8 });
    cells.forEach((c, i) => {
      tl.to(c, { opacity: 1, duration: 0.18 }, 0.4 + i * 0.16)
        .to(picks[i], { opacity: 0.85, y: 0, duration: 0.32, ease: "power2.out" }, 0.45 + i * 0.16);
    });
    return tl;
  },
},
augment_geo: {
  tip: "On the train split only, the sampled flip-or-rot90 transform T is applied identically to input x and target y, so a horizontal flip mirrors both and keeps every pixel aligned.",
  build(svg) {
    svg.innerHTML = `
      <g id="augment_geo-x">
        <rect x="42" y="34" width="52" height="52" rx="2" class="skl-draw c-meas" style="fill:none;stroke-width:1.5"/>
        <polygon points="50,78 50,42 74,42" style="fill:#5b9bd5;opacity:0.7"/>
        <circle cx="82" cy="48" r="4" style="fill:#5b9bd5;opacity:0.7"/>
      </g>
      <g id="augment_geo-y">
        <rect x="146" y="34" width="52" height="52" rx="2" class="skl-draw c-cal" style="fill:none;stroke-width:1.5"/>
        <polygon points="154,78 154,42 178,42" style="fill:#3fc7c0;opacity:0.7"/>
        <circle cx="186" cy="48" r="4" style="fill:#3fc7c0;opacity:0.7"/>
      </g>
      <text id="augment_geo-t" x="120" y="112" text-anchor="middle" style="fill:#e2a45a;font-size:8px;font-family:sans-serif">T = flip_H</text>
      <line id="augment_geo-axx" class="skl-dash c-faint" x1="68" y1="32" x2="68" y2="88" style="opacity:0"/>
      <line id="augment_geo-axy" class="skl-dash c-faint" x1="172" y1="32" x2="172" y2="88" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const gx = svg.querySelector('#augment_geo-x'), gy = svg.querySelector('#augment_geo-y');
    const axx = svg.querySelector('#augment_geo-axx'), axy = svg.querySelector('#augment_geo-axy');
    tl.set(gx, { scaleX: 1, transformOrigin: "68px 60px" })
      .set(gy, { scaleX: 1, transformOrigin: "172px 60px" })
      .set([axx, axy], { opacity: 0 })
      .to([axx, axy], { opacity: 0.6, duration: 0.3 }, 0.3)
      .to(gx, { scaleX: -1, transformOrigin: "68px 60px", duration: 1.0, ease: "power2.inOut" }, 0.6)
      .to(gy, { scaleX: -1, transformOrigin: "172px 60px", duration: 1.0, ease: "power2.inOut" }, 0.6)
      .to([axx, axy], { opacity: 0, duration: 0.3 }, 1.9);
    return tl;
  },
},
slotkeys: {
  tip: "The same strided layout that built the tensor labels each channel by family/slot, so pass/mag, ifg/phase and dem/elev each pick up their own normalisation strategy.",
  build(svg) {
    svg.innerHTML = `<g id="slotkeys-row"></g><g id="slotkeys-keys"></g>`;
    const labels = [['pass/mag','f-meas','#5b9bd5'],['pass/mag','f-meas','#5b9bd5'],['ifg/phase','f-mid','#e2a45a'],['ifg/phase','f-mid','#e2a45a'],['dem/elev','f-faint','#7e8aa0']];
    const row = svg.querySelector('#slotkeys-row'), keys = svg.querySelector('#slotkeys-keys');
    labels.forEach((l, i) => {
      const x = 32 + i * 36;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("class", "slotkeys-ch skl-pop " + l[1]);
      rect.setAttribute("x", x); rect.setAttribute("y", 36);
      rect.setAttribute("width", 28); rect.setAttribute("height", 28);
      rect.setAttribute("style", "opacity:0.8");
      row.appendChild(rect);
      const ln = document.createElementNS("http://www.w3.org/2000/svg", "line");
      ln.setAttribute("class", "slotkeys-link");
      ln.setAttribute("x1", x + 14); ln.setAttribute("y1", 64);
      ln.setAttribute("x2", x + 14); ln.setAttribute("y2", 96);
      ln.setAttribute("style", "stroke:" + l[2] + ";stroke-width:1.4;opacity:0");
      keys.appendChild(ln);
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("class", "slotkeys-tag");
      t.setAttribute("x", x + 14); t.setAttribute("y", 108); t.setAttribute("text-anchor", "middle");
      t.setAttribute("transform", "rotate(40 " + (x + 14) + " 108)");
      t.setAttribute("style", "fill:" + l[2] + ";font-size:6px;font-family:sans-serif;opacity:0");
      t.textContent = l[0]; keys.appendChild(t);
    });
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const chs = svg.querySelectorAll('.slotkeys-ch');
    const links = svg.querySelectorAll('.slotkeys-link');
    const tags = svg.querySelectorAll('.slotkeys-tag');
    tl.set(chs, { opacity: 0.3 })
      .set(links, { opacity: 0 })
      .set(tags, { opacity: 0 });
    chs.forEach((c, i) => {
      tl.to(c, { opacity: 0.85, duration: 0.2 }, 0.3 + i * 0.3)
        .to(links[i], { opacity: 0.8, duration: 0.3 }, 0.4 + i * 0.3)
        .to(tags[i], { opacity: 1, duration: 0.3 }, 0.55 + i * 0.3);
    });
    return tl;
  },
},
fitstats: {
  tip: "Fitted on the train split only in float64, each slot's z-score uses mean and std of f(x), with an optional log1p compression applied before fitting and the scale floored at 1e-8.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="112" x2="214" y2="112"/>
      <g id="fitstats-hist"></g>
      <line id="fitstats-mu" class="skl-dash c-cal" x1="150" y1="40" x2="150" y2="112" style="opacity:0"/>
      <text id="fitstats-mulbl" x="150" y="34" text-anchor="middle" style="fill:#3fc7c0;font-size:7px;font-family:sans-serif;opacity:0">mu_c</text>
      <path id="fitstats-arrow" class="skl-draw c-mid" d="M150 124 L100 124" style="fill:none;stroke-width:1.5;opacity:0"/>`;
    const g = svg.querySelector('#fitstats-hist');
    const hs = [14,26,40,52,46,32,20,10];
    hs.forEach((h, i) => {
      const x = 116 + i * 12;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("class", "fitstats-bar skl-pop f-meas");
      rect.setAttribute("x", x); rect.setAttribute("y", 112 - h);
      rect.setAttribute("width", 10); rect.setAttribute("height", h);
      rect.setAttribute("style", "opacity:0.8");
      g.appendChild(rect);
    });
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const bars = svg.querySelectorAll('.fitstats-bar');
    const mu = svg.querySelector('#fitstats-mu'), mul = svg.querySelector('#fitstats-mulbl');
    const arr = svg.querySelector('#fitstats-arrow');
    tl.set(bars, { scaleY: 0, transformOrigin: "center bottom", opacity: 0.8 })
      .set([mu, mul], { opacity: 0 })
      .set(arr, { opacity: 0 });
    bars.forEach((b, i) => {
      tl.to(b, { scaleY: 1, transformOrigin: "center bottom", duration: 0.4, ease: "power2.out" }, 0.2 + i * 0.07);
    });
    tl.to(mu, { opacity: 0.9, duration: 0.4 }, 1.0)
      .to(mul, { opacity: 1, duration: 0.4 }, 1.0)
      .to(arr, { opacity: 0.9, duration: 0.5 }, 1.5);
    return tl;
  },
},
normalise: {
  tip: "Subtracting mu_c and dividing by s_c shifts each slot's distribution to zero mean and unit scale, applied identically to every split to feed the network dimensionless tensors.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="100" x2="214" y2="100"/>
      <line id="normalise-zero" class="skl-dash c-faint" x1="120" y1="32" x2="120" y2="108"/>
      <path id="normalise-curve" class="skl-draw c-meas" d="M160 100 C168 62, 196 62, 204 100" style="fill:none;stroke-width:2"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const c = svg.querySelector('#normalise-curve');
    tl.set(c, { attr: { d: "M160 100 C168 62, 196 62, 204 100" }, stroke: "#5b9bd5" })
      .to(c, { attr: { d: "M88 100 C104 50, 136 50, 152 100" }, duration: 1.0, ease: "power2.inOut" }, 0.3)
      .to(c, { attr: { d: "M92 100 C106 44, 134 44, 148 100" }, stroke: "#bb86fc", duration: 0.8, ease: "power2.inOut" }, 1.3);
    return tl;
  },
},
noise: {
  tip: "On the train split only and with probability p_N, Gaussian noise of std 0.01 is added to the already-normalised input, jittering x-hat while the target stays untouched.",
  build(svg) {
    svg.innerHTML = `
      <path id="noise-clean" class="skl-draw c-faint" d="M30 78 Q70 40 110 78 T190 78" style="fill:none;stroke-width:1.5"/>
      <path id="noise-noisy" class="skl-draw c-fin" d="M30 78 Q70 40 110 78 T190 78" style="fill:none;stroke-width:1.8"/>
      <g id="noise-pts"></g>`;
    const g = svg.querySelector('#noise-pts');
    for (let i = 0; i < 9; i++) {
      const x = 30 + i * 20;
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("class", "noise-dot");
      dot.setAttribute("cx", x); dot.setAttribute("cy", 78);
      dot.setAttribute("r", 1.8);
      dot.setAttribute("style", "fill:#bb86fc;opacity:0");
      g.appendChild(dot);
    }
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const noisy = svg.querySelector('#noise-noisy');
    const dots = svg.querySelectorAll('.noise-dot');
    const base = "M30 78 Q70 40 110 78 T190 78";
    tl.set(noisy, { attr: { d: base }, opacity: 0 })
      .set(dots, { opacity: 0, y: 0 })
      .to(noisy, { opacity: 1, duration: 0.3 }, 0.2)
      .to(dots, { opacity: 1, duration: 0.3, stagger: 0.04 }, 0.3);
    tl.to(dots, { y: () => (Math.random() - 0.5) * 18, duration: 0.5, ease: "power1.inOut", stagger: 0.03 }, 0.7)
      .to(noisy, { attr: { d: "M30 80 Q70 36 110 82 T190 74" }, duration: 0.5, ease: "power1.inOut" }, 0.7)
      .to(dots, { y: () => (Math.random() - 0.5) * 18, duration: 0.5, ease: "power1.inOut", stagger: 0.03 }, 1.3)
      .to(noisy, { attr: { d: "M30 76 Q70 44 110 75 T190 81" }, duration: 0.5, ease: "power1.inOut" }, 1.3);
    return tl;
  },
},
denorm: {
  tip: "The inverse multiplies by s_c and adds mu_c, and for log1p slots takes expm1 of the result with the exponent argument clamped at 80 to prevent float32 overflow.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="100" x2="214" y2="100"/>
      <line id="denorm-zero" class="skl-dash c-faint" x1="78" y1="34" x2="78" y2="106"/>
      <path id="denorm-curve" class="skl-draw c-fin" d="M50 100 C60 56, 96 56, 106 100" style="fill:none;stroke-width:2"/>
      <line id="denorm-cap" class="skl-dash c-mid" x1="30" y1="46" x2="214" y2="46" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const c = svg.querySelector('#denorm-curve');
    const cap = svg.querySelector('#denorm-cap');
    tl.set(c, { attr: { d: "M50 100 C60 56, 96 56, 106 100" }, stroke: "#bb86fc" })
      .set(cap, { opacity: 0 })
      .to(c, { attr: { d: "M40 100 C70 30, 150 30, 196 100" }, stroke: "#3fc7c0", duration: 0.9, ease: "power2.in" }, 0.3)
      .to(cap, { opacity: 0.9, duration: 0.3 }, 1.0)
      .to(c, { attr: { d: "M40 100 C70 46, 150 46, 196 100" }, duration: 0.7, ease: "power2.out" }, 1.3);
    return tl;
  },
},
forward: {
  tip: "One autocast forward pass turns the normalised input patch into 3K interleaved Gaussian channels of amplitude, mean and sigma per pixel.",
  build(svg) {
    svg.innerHTML = `
      <rect id="forward-in" x="22" y="58" width="20" height="34" rx="2" class="skl-pop f-meas"/>
      <path class="skl-draw c-faint" d="M56 50 L96 62 L96 98 L56 110 Z" style="fill:none"/>
      <path class="skl-draw c-faint" d="M104 62 L132 72 L132 96 L104 106 Z" style="fill:none"/>
      <path class="skl-draw c-faint" d="M140 72 L160 80 L160 100 L140 108 Z" style="fill:none"/>
      <line id="forward-f1" class="skl-dash c-mid" x1="42" y1="75" x2="56" y2="75"/>
      <line id="forward-f2" class="skl-dash c-mid" x1="96" y1="80" x2="104" y2="84"/>
      <line id="forward-f3" class="skl-dash c-mid" x1="132" y1="84" x2="140" y2="90"/>
      <line id="forward-f4" class="skl-dash c-mid" x1="160" y1="90" x2="176" y2="90"/>
      <rect id="forward-oa" x="178" y="44" width="30" height="8" rx="2" class="skl-pop f-cal"/>
      <rect id="forward-ou" x="178" y="56" width="30" height="8" rx="2" class="skl-pop f-cal" style="opacity:0.7"/>
      <rect id="forward-os" x="178" y="68" width="30" height="8" rx="2" class="skl-pop f-cal" style="opacity:0.45"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const flows = ['#forward-f1','#forward-f2','#forward-f3','#forward-f4'].map(s => svg.querySelector(s));
    const outs = ['#forward-oa','#forward-ou','#forward-os'].map(s => svg.querySelector(s));
    tl.set(flows, { strokeDashoffset: 14, opacity: 0.2 })
      .set(outs, { scaleX: 0, transformOrigin: '178px 0px' })
      .set(svg.querySelector('#forward-in'), { opacity: 0.4 })
      .to(svg.querySelector('#forward-in'), { opacity: 1, duration: 0.4 }, 0)
      .to(flows, { strokeDashoffset: 0, opacity: 1, duration: 0.5, stagger: 0.25, ease: 'power2.out' }, 0.2)
      .to(outs, { scaleX: 1, duration: 0.4, stagger: 0.12, ease: 'back.out(2)' }, 1.2);
    return tl;
  },
},
tdenorm: {
  tip: "expm1 inverts the log1p amplitude and sigma channels, with the exponent argument clamped at 80 so an early blow-up cannot become a NaN.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="120" x2="214" y2="120"/>
      <line class="skl-axis" x1="30" y1="20" x2="30" y2="120"/>
      <line id="tdenorm-cap" class="skl-dash c-faint" x1="30" y1="38" x2="214" y2="38"/>
      <path id="tdenorm-lin" class="skl-dash c-meas" d="M30 118 L150 70"/>
      <path id="tdenorm-exp" class="skl-draw c-cal" d="M30 118 Q120 116 168 38 L168 38"/>
      <line id="tdenorm-clip" class="skl-draw c-cal" x1="168" y1="38" x2="214" y2="38"/>
      <circle id="tdenorm-dot" cx="30" cy="118" r="3.5" class="skl-pop f-mid"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const exp = svg.querySelector('#tdenorm-exp');
    const len = exp.getTotalLength();
    tl.set(exp, { strokeDasharray: len, strokeDashoffset: len })
      .set(svg.querySelector('#tdenorm-clip'), { opacity: 0 })
      .set(svg.querySelector('#tdenorm-cap'), { opacity: 0.3 })
      .fromTo(svg.querySelector('#tdenorm-dot'), { attr: { cx: 30, cy: 118 } }, { attr: { cx: 168, cy: 38 }, duration: 1.6, ease: 'power3.in' }, 0)
      .to(exp, { strokeDashoffset: 0, duration: 1.6, ease: 'power3.in' }, 0)
      .to(svg.querySelector('#tdenorm-cap'), { opacity: 0.85, duration: 0.3 }, 1.5)
      .to(svg.querySelector('#tdenorm-dot'), { attr: { cx: 214 }, duration: 0.5, ease: 'none' }, 1.6)
      .fromTo(svg.querySelector('#tdenorm-clip'), { opacity: 0 }, { opacity: 1, duration: 0.5, ease: 'none' }, 1.6);
    return tl;
  },
},
clamp: {
  tip: "Out-of-bounds amplitude and sigma are clipped to grid limits but kept a 0.01 leaky straight-through slope so the heads still pass a small gradient.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="120" x2="214" y2="120"/>
      <line class="skl-axis" x1="120" y1="18" x2="120" y2="122"/>
      <line id="clamp-lo" class="skl-dash c-faint" x1="58" y1="18" x2="58" y2="122"/>
      <line id="clamp-hi" class="skl-dash c-faint" x1="182" y1="18" x2="182" y2="122"/>
      <path id="clamp-curve" class="skl-draw c-cal" d="M30 110 L58 96 L182 30 L214 28.7"/>
      <path id="clamp-ideal" class="skl-dash c-faint" d="M30 118 L214 22" style="opacity:0.3"/>
      <circle id="clamp-dot" cx="58" cy="96" r="3.5" class="skl-pop f-mid"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.4 });
    const dot = svg.querySelector('#clamp-dot');
    tl.set(dot, { attr: { cx: 58, cy: 96 } })
      .to(dot, { attr: { cx: 182, cy: 30 }, duration: 1.1, ease: 'sine.inOut' }, 0)
      .to([svg.querySelector('#clamp-lo'), svg.querySelector('#clamp-hi')], { opacity: 0.7, duration: 0.3 }, 0)
      .to(dot, { attr: { cx: 214, cy: 28.7 }, duration: 0.6, ease: 'none' }, 1.1)
      .to(dot, { attr: { cx: 30, cy: 110 }, duration: 0.9, ease: 'sine.inOut' }, 1.9)
      .fromTo(svg.querySelector('#clamp-curve'), { opacity: 0.5 }, { opacity: 1, duration: 0.4 }, 1.1);
    return tl;
  },
},
renorm: {
  tip: "log1p minus the offset over the scale maps the clamped physical predictions back into the same normalised units as the labels.",
  build(svg) {
    svg.innerHTML = `
      <rect id="renorm-phys" x="26" y="50" width="42" height="50" rx="3" class="skl-pop f-cal"/>
      <text x="47" y="79" style="font-size:9px;fill:#0b0f14;text-anchor:middle">12.4</text>
      <path id="renorm-arrow" class="skl-draw c-mid" d="M74 75 L160 75" style="fill:none"/>
      <polygon id="renorm-head" points="160,70 170,75 160,80" class="skl-pop f-mid"/>
      <rect id="renorm-norm" x="176" y="50" width="42" height="50" rx="3" class="skl-pop f-faint" style="opacity:0.3"/>
      <text id="renorm-val" x="197" y="79" style="font-size:9px;fill:#0b0f14;text-anchor:middle;opacity:0">0.74</text>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const arr = svg.querySelector('#renorm-arrow');
    const len = arr.getTotalLength();
    tl.set(arr, { strokeDasharray: len, strokeDashoffset: len })
      .set(svg.querySelector('#renorm-head'), { opacity: 0, x: -20 })
      .set(svg.querySelector('#renorm-norm'), { opacity: 0.3 })
      .set(svg.querySelector('#renorm-val'), { opacity: 0 })
      .to(arr, { strokeDashoffset: 0, duration: 1.0, ease: 'power2.inOut' }, 0)
      .to(svg.querySelector('#renorm-head'), { opacity: 1, x: 0, duration: 0.8, ease: 'power2.inOut' }, 0.2)
      .to(svg.querySelector('#renorm-norm'), { opacity: 1, duration: 0.5 }, 1.0)
      .to(svg.querySelector('#renorm-val'), { opacity: 1, duration: 0.5 }, 1.1);
    return tl;
  },
},
reconstruct: {
  tip: "Predicted and GT parameters each sum K Gaussian bumps on the elevation axis into a curve; the GT curve is built once under no_grad.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="118" x2="216" y2="118"/>
      <path id="reconstruct-b1" class="skl-dash c-faint" d="M40 118 Q70 70 100 118"/>
      <path id="reconstruct-b2" class="skl-dash c-faint" d="M96 118 Q128 48 160 118"/>
      <path id="reconstruct-b3" class="skl-dash c-faint" d="M150 118 Q176 82 202 118"/>
      <path id="reconstruct-sum" class="skl-draw c-cal" d="M40 118 Q70 70 100 100 Q128 48 160 92 Q176 82 202 118" style="opacity:0"/>
      <path id="reconstruct-gt" class="skl-dash c-meas" d="M40 118 Q72 76 102 98 Q130 54 162 90 Q178 86 202 118" style="opacity:0.5"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const bumps = ['#reconstruct-b1','#reconstruct-b2','#reconstruct-b3'].map(s => svg.querySelector(s));
    const sum = svg.querySelector('#reconstruct-sum');
    tl.set(bumps, { opacity: 0, scaleY: 0, transformOrigin: '0px 118px' })
      .set(sum, { opacity: 0 })
      .set(svg.querySelector('#reconstruct-gt'), { opacity: 0 })
      .to(bumps, { opacity: 0.55, scaleY: 1, duration: 0.5, stagger: 0.3, ease: 'back.out(1.6)' }, 0)
      .to(sum, { opacity: 1, duration: 0.6, ease: 'power2.out' }, 1.3)
      .to(bumps, { opacity: 0.18, duration: 0.5 }, 1.3)
      .to(svg.querySelector('#reconstruct-gt'), { opacity: 0.55, duration: 0.5 }, 1.9);
    return tl;
  },
},
residual: {
  tip: "The single elementwise difference y-hat minus y becomes the residual bars shared by the MSE, L1, Huber and Charbonnier terms.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="80" x2="216" y2="80"/>
      <path id="residual-pred" class="skl-draw c-cal" d="M30 70 Q60 30 90 56 Q120 18 150 50 Q180 44 214 70"/>
      <path id="residual-gt" class="skl-dash c-meas" d="M30 74 Q60 40 90 60 Q120 30 150 56 Q180 52 214 74"/>
      <g id="residual-bars">
        <rect class="skl-pop f-mid" x="58" y="48" width="5" height="12"/>
        <rect class="skl-pop f-mid" x="90" y="56" width="5" height="6"/>
        <rect class="skl-pop f-mid" x="120" y="22" width="5" height="10"/>
        <rect class="skl-pop f-mid" x="150" y="50" width="5" height="8"/>
        <rect class="skl-pop f-mid" x="182" y="48" width="5" height="6"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const bars = svg.querySelectorAll('#residual-bars rect');
    tl.set(bars, { scaleY: 0, transformOrigin: '0px 60px', opacity: 0 })
      .fromTo(svg.querySelector('#residual-pred'), { opacity: 0 }, { opacity: 1, duration: 0.5 }, 0)
      .fromTo(svg.querySelector('#residual-gt'), { opacity: 0 }, { opacity: 0.7, duration: 0.5 }, 0.3)
      .to(bars, { scaleY: 1, opacity: 1, duration: 0.5, stagger: 0.12, ease: 'power2.out' }, 0.8)
      .to(bars, { opacity: 0.5, duration: 0.4 }, 1.9)
      .to(bars, { opacity: 1, duration: 0.4 }, 2.3);
    return tl;
  },
},
curvepoint: {
  tip: "Four pointwise reductions of the shared residual: MSE squares it, L1 takes magnitude, Huber bends at delta and Charbonnier smooths the L1 with epsilon.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="120" y1="118" x2="120" y2="22"/>
      <line class="skl-axis" x1="28" y1="100" x2="214" y2="100"/>
      <path id="curvepoint-mse" class="skl-draw c-cal" d="M40 30 Q120 134 200 30"/>
      <path id="curvepoint-l1" class="skl-dash c-mid" d="M40 38 L120 100 L200 38"/>
      <path id="curvepoint-hub" class="skl-draw c-faint" d="M40 40 Q86 54 100 86 L120 100 L140 86 Q154 54 200 40"/>
      <line id="curvepoint-delta1" class="skl-dash c-faint" x1="100" y1="22" x2="100" y2="118" style="opacity:0.25"/>
      <line id="curvepoint-delta2" class="skl-dash c-faint" x1="140" y1="22" x2="140" y2="118" style="opacity:0.25"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const paths = ['#curvepoint-mse','#curvepoint-l1','#curvepoint-hub'].map(s => svg.querySelector(s));
    paths.forEach(p => { const l = p.getTotalLength(); gsap.set(p, { strokeDasharray: l }); });
    tl.set(paths[0], { strokeDashoffset: paths[0].getTotalLength() })
      .set(paths[1], { strokeDashoffset: paths[1].getTotalLength() })
      .set(paths[2], { strokeDashoffset: paths[2].getTotalLength() })
      .set([svg.querySelector('#curvepoint-delta1'), svg.querySelector('#curvepoint-delta2')], { opacity: 0 })
      .to(paths[0], { strokeDashoffset: 0, duration: 0.7, ease: 'power2.out' }, 0.0)
      .to(paths[1], { strokeDashoffset: 0, duration: 0.7, ease: 'power2.out' }, 0.7)
      .to(paths[2], { strokeDashoffset: 0, duration: 0.7, ease: 'power2.out' }, 1.4)
      .to([svg.querySelector('#curvepoint-delta1'), svg.querySelector('#curvepoint-delta2')], { opacity: 0.5, duration: 0.4 }, 1.6);
    return tl;
  },
},
curveshape: {
  tip: "Three shape terms ignore magnitude: cosine angle over valid pixels, windowed spectral coherence, and per-slice SSIM on jointly normalised images.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="62" y1="118" x2="62" y2="30"/>
      <line class="skl-axis" x1="40" y1="96" x2="86" y2="96"/>
      <line id="curveshape-va" class="skl-draw c-cal" x1="62" y1="96" x2="84" y2="44"/>
      <line id="curveshape-vb" class="skl-draw c-meas" x1="62" y1="96" x2="80" y2="56"/>
      <rect x="108" y="36" width="40" height="40" rx="2" class="skl-pop f-faint" style="opacity:0.2"/>
      <path id="curveshape-coh" class="skl-draw c-cal" d="M110 70 Q118 40 128 56 Q138 42 146 64"/>
      <rect x="168" y="36" width="44" height="40" rx="2" class="skl-pop f-faint" style="opacity:0.15"/>
      <rect id="curveshape-ssim" x="168" y="36" width="44" height="40" rx="2" class="skl-draw c-cal" style="fill:none"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    tl.set(svg.querySelector('#curveshape-vb'), { rotation: -30, transformOrigin: '62px 96px' })
      .set(svg.querySelector('#curveshape-coh'), { opacity: 0 })
      .set(svg.querySelector('#curveshape-ssim'), { opacity: 0, scale: 0.6, transformOrigin: '190px 56px' })
      .to(svg.querySelector('#curveshape-vb'), { rotation: 0, duration: 1.0, ease: 'sine.inOut' }, 0)
      .to(svg.querySelector('#curveshape-coh'), { opacity: 1, duration: 0.6, ease: 'power2.out' }, 1.0)
      .to(svg.querySelector('#curveshape-ssim'), { opacity: 1, scale: 1, duration: 0.6, ease: 'back.out(1.8)' }, 1.7);
    return tl;
  },
},
physgeom: {
  tip: "The vertical wavenumber kz scales the master-relative perpendicular baseline by the monostatic 4-pi-over-lambda-r0 factor to build the steering phasors.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="120" x2="40" y2="24"/>
      <line class="skl-axis" x1="40" y1="120" x2="210" y2="120"/>
      <circle cx="40" cy="40" r="3" class="skl-pop f-faint"/>
      <line id="physgeom-bperp" class="skl-dash c-mid" x1="40" y1="40" x2="120" y2="64"/>
      <circle cx="120" cy="64" r="3" class="skl-pop f-meas"/>
      <g id="physgeom-phasors">
        <line class="skl-draw c-cal" x1="160" y1="100" x2="178" y2="100"/>
        <line class="skl-draw c-cal" x1="160" y1="100" x2="172" y2="86"/>
        <line class="skl-draw c-cal" x1="160" y1="100" x2="160" y2="82"/>
        <line class="skl-draw c-cal" x1="160" y1="100" x2="148" y2="86"/>
      </g>
      <circle cx="160" cy="100" r="18" class="skl-axis" style="fill:none;opacity:0.3"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const phasors = svg.querySelectorAll('#physgeom-phasors line');
    tl.set(svg.querySelector('#physgeom-bperp'), { attr: { x2: 40, y2: 40 } })
      .set(phasors, { rotation: 0, transformOrigin: '160px 100px', opacity: 0 })
      .to(svg.querySelector('#physgeom-bperp'), { attr: { x2: 120, y2: 64 }, duration: 0.9, ease: 'power2.out' }, 0)
      .to(phasors, { opacity: 1, duration: 0.3, stagger: 0.1 }, 0.9)
      .fromTo(phasors, { rotation: 0 }, { rotation: (i) => 40 + i * 38, duration: 1.2, stagger: 0.08, ease: 'power2.out' }, 1.0);
    return tl;
  },
},
physmoments: {
  tip: "Ratio terms compare relative integrated power plus the mass, centroid and spread moments, reduced only over GT-strong pixels.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="216" y2="118"/>
      <path id="physmoments-prof" class="skl-draw c-cal" d="M40 118 Q90 40 124 70 Q150 92 200 118" style="fill:rgba(127,216,200,0.12)"/>
      <line id="physmoments-cen" class="skl-dash c-mid" x1="112" y1="118" x2="112" y2="48"/>
      <circle id="physmoments-cdot" cx="112" cy="48" r="3.5" class="skl-pop f-mid"/>
      <line id="physmoments-spr" class="skl-dash c-faint" x1="84" y1="100" x2="140" y2="100"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    tl.set(svg.querySelector('#physmoments-prof'), { opacity: 0.4 })
      .set(svg.querySelector('#physmoments-cen'), { scaleY: 0, transformOrigin: '112px 118px' })
      .set(svg.querySelector('#physmoments-cdot'), { opacity: 0 })
      .set(svg.querySelector('#physmoments-spr'), { scaleX: 0, transformOrigin: '112px 100px' })
      .to(svg.querySelector('#physmoments-prof'), { opacity: 1, duration: 0.5 }, 0)
      .to(svg.querySelector('#physmoments-cen'), { scaleY: 1, duration: 0.5, ease: 'power2.out' }, 0.8)
      .to(svg.querySelector('#physmoments-cdot'), { opacity: 1, duration: 0.3 }, 1.3)
      .to(svg.querySelector('#physmoments-spr'), { scaleX: 1, duration: 0.6, ease: 'power2.out' }, 1.5);
    return tl;
  },
},
physcov: {
  tip: "Coherence compares the two normalised characteristic functions while covariance matching transforms only the profile difference R[P-T] thanks to R's linearity.",
  build(svg) {
    svg.innerHTML = `
      <circle cx="78" cy="58" r="26" class="skl-axis" style="fill:none;opacity:0.3"/>
      <line id="physcov-gp" class="skl-draw c-cal" x1="78" y1="58" x2="100" y2="44"/>
      <line id="physcov-gt" class="skl-draw c-meas" x1="78" y1="58" x2="96" y2="40"/>
      <g id="physcov-mat">
        <rect x="150" y="36" width="14" height="14" class="skl-pop f-mid"/>
        <rect x="166" y="36" width="14" height="14" class="skl-pop f-faint" style="opacity:0.3"/>
        <rect x="150" y="52" width="14" height="14" class="skl-pop f-faint" style="opacity:0.3"/>
        <rect x="166" y="52" width="14" height="14" class="skl-pop f-mid"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const cells = svg.querySelectorAll('#physcov-mat rect');
    tl.set(svg.querySelector('#physcov-gp'), { rotation: 35, transformOrigin: '78px 58px' })
      .set(cells, { opacity: 0, scale: 0, transformOrigin: '50% 50%' })
      .to(svg.querySelector('#physcov-gp'), { rotation: 0, duration: 1.0, ease: 'sine.inOut' }, 0)
      .to(cells, { opacity: (i) => (i === 0 || i === 3) ? 1 : 0.3, scale: 1, duration: 0.4, stagger: 0.1, ease: 'back.out(2)' }, 1.0);
    return tl;
  },
},
physcapon: {
  tip: "Capon synthesises R[P], adds signal-adaptive epsilon-sigma-bar diagonal loading, then solves once per pixel to form the spectrum compared mass-normalised.",
  build(svg) {
    svg.innerHTML = `
      <g id="physcapon-mat">
        <rect x="36" y="40" width="48" height="48" rx="2" class="skl-pop f-mid" style="opacity:0.5"/>
        <line id="physcapon-diag" class="skl-draw c-fin" x1="36" y1="40" x2="84" y2="88"/>
      </g>
      <path id="physcapon-solve" class="skl-draw c-mid" d="M92 64 L132 64" style="fill:none"/>
      <polygon points="132,59 142,64 132,69" class="skl-pop f-mid"/>
      <line class="skl-axis" x1="150" y1="100" x2="214" y2="100"/>
      <path id="physcapon-spec" class="skl-draw c-cal" d="M152 100 Q178 36 184 78 Q192 50 212 100" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const solve = svg.querySelector('#physcapon-solve');
    const sl = solve.getTotalLength();
    tl.set(svg.querySelector('#physcapon-diag'), { opacity: 0, strokeWidth: 1 })
      .set(solve, { strokeDasharray: sl, strokeDashoffset: sl })
      .set(svg.querySelector('#physcapon-spec'), { opacity: 0 })
      .to(svg.querySelector('#physcapon-diag'), { opacity: 1, strokeWidth: 3, duration: 0.6, ease: 'power2.out' }, 0)
      .to(solve, { strokeDashoffset: 0, duration: 0.7, ease: 'power2.inOut' }, 0.7)
      .to(svg.querySelector('#physcapon-spec'), { opacity: 1, duration: 0.7, ease: 'power2.out' }, 1.4);
    return tl;
  },
},
paramterms: {
  tip: "GT components are mu-sorted and empty slots mask mu and sigma to zero weight, so Param-L1/Huber act in normalised space while TV penalises spatial irregularity.",
  build(svg) {
    svg.innerHTML = `
      <g id="paramterms-slots">
        <rect x="34" y="40" width="22" height="20" rx="2" class="skl-pop f-cal"/>
        <rect x="34" y="64" width="22" height="20" rx="2" class="skl-pop f-cal"/>
        <rect x="34" y="88" width="22" height="20" rx="2" class="skl-pop f-faint" style="opacity:0.25"/>
      </g>
      <g id="paramterms-sorted">
        <rect x="98" y="40" width="22" height="20" rx="2" class="skl-draw c-cal" style="fill:none"/>
        <rect x="98" y="64" width="22" height="20" rx="2" class="skl-draw c-cal" style="fill:none"/>
        <rect id="paramterms-mask" x="98" y="88" width="22" height="20" rx="2" class="skl-dash c-faint" style="fill:none"/>
      </g>
      <rect x="150" y="40" width="62" height="62" rx="2" class="skl-axis" style="fill:none;opacity:0.3"/>
      <path id="paramterms-tv" class="skl-draw c-cal" d="M158 92 L170 60 L182 80 L194 52 L206 72"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const sorted = svg.querySelectorAll('#paramterms-sorted rect');
    const tv = svg.querySelector('#paramterms-tv');
    const tvl = tv.getTotalLength();
    tl.set(sorted, { opacity: 0, x: -36 })
      .set(svg.querySelector('#paramterms-mask'), { opacity: 1 })
      .set(tv, { strokeDasharray: tvl, strokeDashoffset: tvl })
      .to(sorted, { opacity: 1, x: 0, duration: 0.5, stagger: 0.15, ease: 'power2.out' }, 0.2)
      .to(svg.querySelector('#paramterms-mask'), { opacity: 0.3, duration: 0.4 }, 1.0)
      .to(tv, { strokeDashoffset: 0, duration: 0.9, ease: 'power1.inOut' }, 1.2);
    return tl;
  },
},
composite: {
  tip: "Each term's effective weight is the user weight times a fixed empirical normaliser; the weighted terms sum and divide by total weight into one loss.",
  build(svg) {
    svg.innerHTML = `
      <g id="composite-bars">
        <rect x="36" y="92" width="14" height="26" class="skl-pop f-cal"/>
        <rect x="56" y="74" width="14" height="44" class="skl-pop f-cal"/>
        <rect x="76" y="100" width="14" height="18" class="skl-pop f-cal"/>
        <rect x="96" y="58" width="14" height="60" class="skl-pop f-cal"/>
        <rect x="116" y="86" width="14" height="32" class="skl-pop f-cal"/>
      </g>
      <line class="skl-axis" x1="30" y1="118" x2="140" y2="118"/>
      <path id="composite-arrow" class="skl-draw c-mid" d="M146 75 L176 75" style="fill:none"/>
      <polygon points="176,70 186,75 176,80" class="skl-pop f-mid"/>
      <rect id="composite-total" x="192" y="48" width="24" height="56" rx="2" class="skl-pop f-fin"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const bars = svg.querySelectorAll('#composite-bars rect');
    const total = svg.querySelector('#composite-total');
    const arr = svg.querySelector('#composite-arrow');
    const al = arr.getTotalLength();
    tl.set(bars, { scaleY: 0, transformOrigin: '0px 118px', opacity: 0.4 })
      .set(arr, { strokeDasharray: al, strokeDashoffset: al })
      .set(total, { scaleY: 0, transformOrigin: '204px 104px' })
      .to(bars, { scaleY: 1, opacity: 1, duration: 0.4, stagger: 0.1, ease: 'power2.out' }, 0)
      .to(bars, { opacity: 0.6, duration: 0.3 }, 1.0)
      .to(arr, { strokeDashoffset: 0, duration: 0.6, ease: 'power2.inOut' }, 1.0)
      .to(total, { scaleY: 1, duration: 0.6, ease: 'back.out(1.5)' }, 1.5);
    return tl;
  },
},
gradclip: {
  tip: "When the global gradient norm exceeds the threshold tau, every gradient is rescaled by tau over the norm so the clipped vector lands exactly on the limit.",
  build(svg) {
    svg.innerHTML = `
      <circle cx="70" cy="80" r="44" class="skl-dash c-faint" style="fill:none;opacity:0.4"/>
      <line id="gradclip-vec" class="skl-draw c-mid" x1="70" y1="80" x2="118" y2="34"/>
      <polygon id="gradclip-head" points="118,34 110,38 116,44" class="skl-pop f-mid"/>
      <line id="gradclip-clipped" class="skl-draw c-cal" x1="70" y1="80" x2="101" y2="50" style="opacity:0"/>
      <circle cx="70" cy="80" r="3" class="skl-pop f-faint"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const vec = svg.querySelector('#gradclip-vec');
    const head = svg.querySelector('#gradclip-head');
    const clip = svg.querySelector('#gradclip-clipped');
    tl.set(vec, { attr: { x2: 90, y2: 60 } })
      .set(head, { x: -28, y: 20 })
      .set(clip, { opacity: 0 })
      .to(vec, { attr: { x2: 118, y2: 34 }, duration: 0.9, ease: 'power2.out' }, 0)
      .to(head, { x: 0, y: 0, duration: 0.9, ease: 'power2.out' }, 0)
      .to(clip, { opacity: 1, duration: 0.4 }, 1.0)
      .to([vec, head], { opacity: 0.3, duration: 0.4 }, 1.0)
      .fromTo(clip, { attr: { x2: 118, y2: 34 } }, { attr: { x2: 101, y2: 50 }, duration: 0.6, ease: 'power2.inOut' }, 1.0);
    return tl;
  },
},
adamw: {
  tip: "Bias-corrected adaptive moments with decoupled weight decay step the weights down a loss surface, driving the training loss lower each epoch.",
  build(svg) {
    svg.innerHTML = `
      <path id="adamw-bowl" class="skl-draw c-faint" d="M30 36 Q120 156 210 36" style="fill:none;opacity:0.5"/>
      <circle id="adamw-ball" cx="42" cy="60" r="5" class="skl-pop f-fin"/>
      <g id="adamw-steps">
        <circle cx="42" cy="60" r="2.5" class="skl-pop f-mid" style="opacity:0.4"/>
        <circle cx="70" cy="92" r="2.5" class="skl-pop f-mid" style="opacity:0.4"/>
        <circle cx="98" cy="114" r="2.5" class="skl-pop f-mid" style="opacity:0.4"/>
        <circle cx="120" cy="122" r="2.5" class="skl-pop f-mid" style="opacity:0.4"/>
      </g>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const ball = svg.querySelector('#adamw-ball');
    const trail = svg.querySelectorAll('#adamw-steps circle');
    tl.set(ball, { attr: { cx: 42, cy: 60 } })
      .set(trail, { opacity: 0 })
      .to(ball, { attr: { cx: 70, cy: 92 }, duration: 0.6, ease: 'power1.in' }, 0)
      .to(trail[0], { opacity: 0.5, duration: 0.2 }, 0.4)
      .to(ball, { attr: { cx: 98, cy: 114 }, duration: 0.6, ease: 'power1.inOut' }, 0.6)
      .to(trail[1], { opacity: 0.5, duration: 0.2 }, 1.0)
      .to(ball, { attr: { cx: 120, cy: 122 }, duration: 0.6, ease: 'power1.out' }, 1.2)
      .to(trail[2], { opacity: 0.5, duration: 0.2 }, 1.6)
      .to(trail[3], { opacity: 0.5, duration: 0.2 }, 1.9);
    return tl;
  },
},
schedule: {
  tip: "The effective LR is the base rate times a per-epoch cosine decay times a linear per-step warmup, with the loss curriculum swapping objectives at the swap epoch.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="118" x2="216" y2="118"/>
      <line class="skl-axis" x1="28" y1="20" x2="28" y2="118"/>
      <path id="schedule-warm" class="skl-draw c-mid" d="M28 110 L58 40"/>
      <path id="schedule-cos" class="skl-draw c-cal" d="M58 40 Q120 44 150 78 Q180 108 210 114"/>
      <line id="schedule-swap" class="skl-dash c-fin" x1="138" y1="20" x2="138" y2="118"/>
      <circle id="schedule-dot" cx="28" cy="110" r="3.5" class="skl-pop f-cal"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const warm = svg.querySelector('#schedule-warm');
    const cos = svg.querySelector('#schedule-cos');
    const wl = warm.getTotalLength();
    const cl = cos.getTotalLength();
    const dot = svg.querySelector('#schedule-dot');
    tl.set(warm, { strokeDasharray: wl, strokeDashoffset: wl })
      .set(cos, { strokeDasharray: cl, strokeDashoffset: cl })
      .set(svg.querySelector('#schedule-swap'), { opacity: 0 })
      .set(dot, { attr: { cx: 28, cy: 110 } })
      .to(warm, { strokeDashoffset: 0, duration: 0.6, ease: 'none' }, 0)
      .to(dot, { attr: { cx: 58, cy: 40 }, duration: 0.6, ease: 'none' }, 0)
      .to(cos, { strokeDashoffset: 0, duration: 1.4, ease: 'none' }, 0.6)
      .to(dot, { attr: { cx: 210, cy: 114 }, duration: 1.4, ease: 'none' }, 0.6)
      .to(svg.querySelector('#schedule-swap'), { opacity: 0.8, duration: 0.3 }, 1.2);
    return tl;
  },
},
checkpoint: {
  tip: "Validation runs every few epochs; a strict improvement checkpoints the best epoch and early stopping reverts to it after patience evaluations without a new minimum.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="118" x2="216" y2="118"/>
      <path id="checkpoint-val" class="skl-draw c-cal" d="M30 50 L62 72 L94 60 L120 90 L150 78 L180 96 L210 88"/>
      <circle id="checkpoint-best" cx="120" cy="90" r="5" class="skl-pop f-fin"/>
      <g id="checkpoint-patience">
        <circle cx="150" cy="78" r="2.5" class="skl-pop f-faint"/>
        <circle cx="180" cy="96" r="2.5" class="skl-pop f-faint"/>
        <circle cx="210" cy="88" r="2.5" class="skl-pop f-faint"/>
      </g>
      <path id="checkpoint-revert" class="skl-dash c-fin" d="M210 88 Q165 30 122 86" style="fill:none;opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const val = svg.querySelector('#checkpoint-val');
    const vl = val.getTotalLength();
    const best = svg.querySelector('#checkpoint-best');
    const pat = svg.querySelectorAll('#checkpoint-patience circle');
    const rev = svg.querySelector('#checkpoint-revert');
    tl.set(val, { strokeDasharray: vl, strokeDashoffset: vl })
      .set(best, { scale: 0, transformOrigin: '120px 90px' })
      .set(pat, { opacity: 0 })
      .set(rev, { opacity: 0 })
      .to(val, { strokeDashoffset: 0, duration: 1.3, ease: 'none' }, 0)
      .to(best, { scale: 1, duration: 0.4, ease: 'back.out(2)' }, 0.7)
      .to(pat, { opacity: 1, duration: 0.3, stagger: 0.2 }, 1.0)
      .to(rev, { opacity: 1, duration: 0.7, ease: 'power2.inOut' }, 1.8)
      .to(best, { scale: 1.4, duration: 0.3, yoyo: true, repeat: 1 }, 2.4);
    return tl;
  },
},
load: {
  tip: "The architecture is rebuilt verbatim from the saved config before the best-epoch theta-star tensor is loaded with no EMA, refusing any input that is not one contiguous region.",
  build(svg) {
    svg.innerHTML = `
      <rect id="load-shell" x="38" y="40" width="64" height="70" rx="4" class="skl-draw c-faint" style="fill:#1b242f"/>
      <line x1="48" y1="56" x2="92" y2="56" class="skl-axis"/>
      <line x1="48" y1="68" x2="92" y2="68" class="skl-axis"/>
      <line x1="48" y1="80" x2="92" y2="80" class="skl-axis"/>
      <line x1="48" y1="92" x2="92" y2="92" class="skl-axis"/>
      <rect id="load-w0" x="48" y="52" width="44" height="6" rx="2" class="skl-pop f-meas" style="opacity:0"/>
      <rect id="load-w1" x="48" y="64" width="44" height="6" rx="2" class="skl-pop f-meas" style="opacity:0"/>
      <rect id="load-w2" x="48" y="76" width="44" height="6" rx="2" class="skl-pop f-meas" style="opacity:0"/>
      <rect id="load-w3" x="48" y="88" width="44" height="6" rx="2" class="skl-pop f-meas" style="opacity:0"/>
      <path id="load-arrow" class="skl-draw c-cal" d="M108 75 L150 75 M142 69 L150 75 L142 81"/>
      <rect id="load-region" x="160" y="48" width="50" height="50" rx="3" class="skl-draw c-cal" style="fill:#15302d"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const ws = ['#load-w0','#load-w1','#load-w2','#load-w3'].map(s=>svg.querySelector(s));
    tl.set(ws, { opacity: 0, scaleX: 0, transformOrigin: "48px 50%" })
      .set(svg.querySelector('#load-region'), { opacity: 0.25, scale: 0.9, transformOrigin: "185px 73px" })
      .set(svg.querySelector('#load-arrow'), { opacity: 0.3 });
    ws.forEach((w,i)=> tl.to(w, { opacity: 1, scaleX: 1, duration: 0.4, ease: "power2.out" }, 0.15*i));
    tl.to(svg.querySelector('#load-arrow'), { opacity: 1, duration: 0.4 }, 0.9)
      .to(svg.querySelector('#load-region'), { opacity: 1, scale: 1, duration: 0.6, ease: "back.out(1.6)" }, 1.1);
    return tl;
  },
},
predict: {
  tip: "A single window slides across the patch grid in deterministic raster order, and the frozen model emits the raw normalised z-hat at each stop so the cube ends with no holes.",
  build(svg) {
    let cells = "";
    for (let r=0;r<3;r++) for (let c=0;c<4;c++){ const x=44+c*34, y=40+r*28; cells += `<rect x="${x}" y="${y}" width="30" height="24" rx="2" style="fill:#1b242f;stroke:#303d4c;stroke-width:1"/>`; }
    svg.innerHTML = `
      ${cells}
      <rect id="predict-win" x="44" y="40" width="30" height="24" rx="2" class="skl-draw c-meas" style="fill:#1d3350;opacity:0.85"/>
      <rect id="predict-out" x="178" y="40" width="30" height="24" rx="2" class="skl-pop f-mid" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.4 });
    const win = svg.querySelector('#predict-win');
    const out = svg.querySelector('#predict-out');
    const pos = [];
    for (let r=0;r<3;r++) for (let c=0;c<4;c++) pos.push({x:44+c*34, y:40+r*28});
    tl.set(win, { x: 0, y: 0 });
    pos.forEach((p,i)=>{
      tl.to(win, { x: p.x-44, y: p.y-40, duration: 0.22, ease: "power1.inOut" }, i*0.26)
        .fromTo(out, { opacity: 0, scale: 0.6, transformOrigin:"193px 52px" }, { opacity: 1, scale: 1, duration: 0.16 }, i*0.26+0.1)
        .to(out, { opacity: 0, duration: 0.12 }, i*0.26+0.24);
    });
    return tl;
  },
},
idenorm: {
  tip: "Raw outputs are denormalised then hard-clamped with a flat saturating transfer curve, pinning amplitude into [0, a_max] with zero leaky slope outside the bounds.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="110" x2="210" y2="110"/>
      <line class="skl-axis" x1="40" y1="110" x2="40" y2="28"/>
      <line class="skl-dash c-faint" x1="40" y1="44" x2="210" y2="44"/>
      <path id="idenorm-clamp" class="skl-draw c-cal" d="M40 110 L40 110 L100 110 L155 44 L210 44"/>
      <path id="idenorm-lin" class="skl-dash c-mid" d="M30 130 L220 14"/>
      <circle id="idenorm-dot" r="4" cx="40" cy="110" class="skl-pop f-cal"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const dot = svg.querySelector('#idenorm-dot');
    tl.set(dot, { attr:{ cx:40, cy:110 } })
      .to(svg.querySelector('#idenorm-lin'), { opacity: 0.3, duration: 0.3 }, 0)
      .to(dot, { attr:{ cx:100, cy:110 }, duration: 0.7, ease: "none" }, 0.3)
      .to(dot, { attr:{ cx:155, cy:44 }, duration: 0.7, ease: "none" }, 1.0)
      .to(dot, { attr:{ cx:210, cy:44 }, duration: 0.7, ease: "none" }, 1.7)
      .to(dot, { attr:{ r:6 }, duration: 0.2, yoyo:true, repeat:1 }, 1.9);
    return tl;
  },
},
align: {
  tip: "GT slots are stably sorted by mu with inactive slots (a below 1e-3) pushed to the end, while the prediction keeps its raw slot order so the metric scores the network's own arrangement.",
  build(svg) {
    svg.innerHTML = `
      <rect id="align-g0" x="44" y="44" width="32" height="16" rx="2" class="skl-pop f-meas"/>
      <rect id="align-g1" x="44" y="64" width="32" height="16" rx="2" class="skl-pop f-faint"/>
      <rect id="align-g2" x="44" y="84" width="32" height="16" rx="2" class="skl-pop f-meas"/>
      <rect id="align-g3" x="44" y="104" width="32" height="16" rx="2" class="skl-pop f-meas"/>
      <line id="align-l0" class="skl-dash c-faint" x1="78" y1="52" x2="162" y2="52"/>
      <line id="align-l1" class="skl-dash c-faint" x1="78" y1="72" x2="162" y2="112"/>
      <line id="align-l2" class="skl-dash c-faint" x1="78" y1="92" x2="162" y2="72"/>
      <line id="align-l3" class="skl-dash c-faint" x1="78" y1="112" x2="162" y2="92"/>
      <rect x="164" y="44" width="32" height="16" rx="2" class="skl-draw c-mid" style="fill:#2a2418"/>
      <rect x="164" y="64" width="32" height="16" rx="2" class="skl-draw c-mid" style="fill:#2a2418"/>
      <rect x="164" y="84" width="32" height="16" rx="2" class="skl-draw c-mid" style="fill:#2a2418"/>
      <rect x="164" y="104" width="32" height="16" rx="2" class="skl-dash c-faint" style="fill:#181818"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const g0=svg.querySelector('#align-g0'), g1=svg.querySelector('#align-g1'), g2=svg.querySelector('#align-g2'), g3=svg.querySelector('#align-g3');
    const ls=['#align-l0','#align-l1','#align-l2','#align-l3'].map(s=>svg.querySelector(s));
    tl.set([g0,g1,g2,g3], { x:0, y:0 })
      .set(ls, { opacity: 0.12 })
      .to(ls, { opacity: 0.5, duration: 0.4 }, 0.2)
      .to(g0, { x: 120, duration: 0.8, ease:"power2.inOut" }, 0.6)
      .to(g2, { x: 120, y: -20, duration: 0.8, ease:"power2.inOut" }, 0.6)
      .to(g3, { x: 120, y: -20, duration: 0.8, ease:"power2.inOut" }, 0.6)
      .to(g1, { x: 120, y: 40, duration: 0.8, ease:"power2.inOut" }, 0.6)
      .to(g1, { opacity: 0.35, duration: 0.4 }, 1.4);
    return tl;
  },
},
recon: {
  tip: "Each patch's clamped Gaussians are evaluated on the elevation axis with amplitudes rectified at zero and a 2-sigma-squared-plus-1e-8 kernel denominator, summing into one spectrum.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="34" y1="116" x2="214" y2="116"/>
      <path id="recon-k0" class="skl-dash c-faint" d="M34 116 C70 116 78 64 96 64 C114 64 122 116 158 116"/>
      <path id="recon-k1" class="skl-dash c-faint" d="M110 116 C140 116 148 80 162 80 C176 80 184 116 214 116"/>
      <path id="recon-sum" class="skl-draw c-mid" d="M34 116 C70 116 78 64 96 64 C108 64 116 92 134 92 C150 92 156 80 162 80 C176 80 184 116 214 116" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const k0=svg.querySelector('#recon-k0'), k1=svg.querySelector('#recon-k1'), sum=svg.querySelector('#recon-sum');
    tl.set([k0,k1], { opacity: 0.15, scaleY: 0, transformOrigin:"50% 116px" })
      .set(sum, { opacity: 0 })
      .to(k0, { opacity: 0.45, scaleY: 1, duration: 0.6, ease:"back.out(1.5)" }, 0.1)
      .to(k1, { opacity: 0.45, scaleY: 1, duration: 0.6, ease:"back.out(1.5)" }, 0.5)
      .to([k0,k1], { opacity: 0.2, duration: 0.4 }, 1.2)
      .to(sum, { opacity: 1, duration: 0.7, ease:"power2.out" }, 1.2);
    return tl;
  },
},
window: {
  tip: "A separable Hann taper has each 1D axis factor floored at 1e-3 before the outer product, so every covered position carries a strictly positive overlap-add weight.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="118" x2="120" y2="118"/>
      <path id="window-bell" class="skl-draw c-mid" d="M40 118 C58 118 64 50 80 50 C96 50 102 118 120 118"/>
      <line id="window-floor" class="skl-dash c-faint" x1="40" y1="116" x2="120" y2="116"/>
      <rect id="window-grid" x="150" y="40" width="60" height="60" class="skl-axis" style="fill:none"/>
      <radialGradient id="window-rg" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#f5b971" stop-opacity="0.9"/><stop offset="100%" stop-color="#f5b971" stop-opacity="0.05"/></radialGradient>
      <rect id="window-2d" x="150" y="40" width="60" height="60" fill="url(#window-rg)" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const bell=svg.querySelector('#window-bell'), w2=svg.querySelector('#window-2d'), fl=svg.querySelector('#window-floor');
    tl.set(bell, { scaleY: 0, transformOrigin:"80px 118px" })
      .set(w2, { opacity: 0, scale: 0.7, transformOrigin:"180px 70px" })
      .to(bell, { scaleY: 1, duration: 0.8, ease:"power2.out" }, 0.1)
      .fromTo(fl, { opacity: 0 }, { opacity: 0.6, duration: 0.3 }, 0.6)
      .to(w2, { opacity: 1, scale: 1, duration: 0.7, ease:"back.out(1.4)" }, 0.9);
    return tl;
  },
},
ola: {
  tip: "Each windowed patch is scattered additively into the value accumulator A at its grid origin while the bare window is added to the parallel weight buffer W, making the running sum order-independent.",
  build(svg) {
    svg.innerHTML = `
      <rect x="46" y="44" width="64" height="64" class="skl-axis" style="fill:#15302d"/>
      <rect x="140" y="44" width="64" height="64" class="skl-axis" style="fill:#2a2418"/>
      <rect id="ola-tA0" x="46" y="44" width="36" height="36" class="skl-pop f-cal" style="opacity:0"/>
      <rect id="ola-tA1" x="74" y="72" width="36" height="36" class="skl-pop f-cal" style="opacity:0"/>
      <rect id="ola-tW0" x="140" y="44" width="36" height="36" class="skl-pop f-mid" style="opacity:0"/>
      <rect id="ola-tW1" x="168" y="72" width="36" height="36" class="skl-pop f-mid" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const a0=svg.querySelector('#ola-tA0'), a1=svg.querySelector('#ola-tA1'), w0=svg.querySelector('#ola-tW0'), w1=svg.querySelector('#ola-tW1');
    tl.set([a0,a1,w0,w1], { opacity: 0 })
      .fromTo(a0, { opacity: 0, x: -20 }, { opacity: 0.5, x: 0, duration: 0.5, ease:"power2.out" }, 0.1)
      .fromTo(w0, { opacity: 0, x: -20 }, { opacity: 0.5, x: 0, duration: 0.5, ease:"power2.out" }, 0.1)
      .fromTo(a1, { opacity: 0, x: 20 }, { opacity: 0.5, x: 0, duration: 0.5, ease:"power2.out" }, 0.9)
      .fromTo(w1, { opacity: 0, x: 20 }, { opacity: 0.5, x: 0, duration: 0.5, ease:"power2.out" }, 0.9)
      .to([a0,a1], { opacity: 0.75, duration: 0.4 }, 1.5)
      .to([w0,w1], { opacity: 0.75, duration: 0.4 }, 1.5);
    return tl;
  },
},
finalise: {
  tip: "The value accumulator is divided elementwise by max(W,1) so uncovered positions divide by one and read zero, then the grid padding is trimmed to the scene extent.",
  build(svg) {
    svg.innerHTML = `
      <rect id="finalise-A" x="40" y="52" width="44" height="44" class="skl-pop f-cal" style="opacity:0.7"/>
      <text x="100" y="80" text-anchor="middle" style="fill:#9fb0c0;font-size:16px">&#247;</text>
      <rect id="finalise-W" x="116" y="52" width="44" height="44" class="skl-pop f-mid" style="opacity:0.7"/>
      <text x="176" y="80" text-anchor="middle" style="fill:#9fb0c0;font-size:16px">=</text>
      <rect id="finalise-C" x="190" y="58" width="32" height="32" class="skl-draw c-fin" style="fill:#241d33;opacity:0"/>
      <rect id="finalise-trim" x="190" y="58" width="32" height="32" class="skl-dash c-faint" style="fill:none;opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const C=svg.querySelector('#finalise-C'), trim=svg.querySelector('#finalise-trim');
    tl.set(C, { opacity: 0, scale: 1.3, transformOrigin:"206px 74px" })
      .set(trim, { opacity: 0, scale: 1.3, transformOrigin:"206px 74px" })
      .to(svg.querySelector('#finalise-A'), { opacity: 1, duration: 0.4 }, 0)
      .to(svg.querySelector('#finalise-W'), { opacity: 1, duration: 0.4 }, 0)
      .to([C,trim], { opacity: 1, scale: 1, duration: 0.7, ease:"back.out(1.5)" }, 0.7)
      .to(trim, { attr:{ width: 26, height: 26, x: 193, y: 61 }, duration: 0.5, ease:"power2.inOut" }, 1.5)
      .to(C, { attr:{ width: 26, height: 26, x: 193, y: 61 }, duration: 0.5, ease:"power2.inOut" }, 1.5);
    return tl;
  },
},
pixelmaps: {
  tip: "Five per-pixel maps reduce over the N elevation bins at each azimuth-range cell: MSE, MAE, R-squared, cosine similarity, and peak-bin index error.",
  build(svg) {
    let cells="";
    const vals=[0.9,0.7,0.95,0.6,0.8,0.4,0.85,0.5,0.92,0.65,0.75,0.55];
    let i=0;
    for(let r=0;r<3;r++)for(let c=0;c<4;c++){const x=120+c*24,y=44+r*24;cells+=`<rect class="pixelmaps-cell" x="${x}" y="${y}" width="22" height="22" style="fill:#4fd6c4;opacity:${0.15+vals[i]*0.6}"/>`;i++;}
    svg.innerHTML = `
      <g id="pixelmaps-stack">
        <rect x="40" y="60" width="40" height="40" class="skl-draw c-faint" style="fill:#1b242f"/>
        <rect x="46" y="54" width="40" height="40" class="skl-draw c-faint" style="fill:#1b242f"/>
        <rect x="52" y="48" width="40" height="40" class="skl-draw c-meas" style="fill:#1d3350"/>
      </g>
      <path id="pixelmaps-arrow" class="skl-draw c-cal" d="M98 74 L112 74 M106 69 L112 74 L106 79"/>
      ${cells}`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const cells=svg.querySelectorAll('.pixelmaps-cell');
    tl.set(cells, { opacity: 0, scale: 0.4, transformOrigin:"50% 50%" })
      .to(svg.querySelector('#pixelmaps-stack'), { y: -4, duration: 0.4, yoyo:true, repeat:1, ease:"sine.inOut" }, 0)
      .to(cells, { opacity: 1, scale: 1, duration: 0.5, stagger: { each: 0.05, from: "start" }, ease:"back.out(1.6)" }, 0.5);
    return tl;
  },
},
globalcurve: {
  tip: "Cube-wide scalars at physical scale settle out: MSE, RMSE, an overall R-squared, and a PSNR whose peak signal is the GT-only dynamic range C_max minus C_min.",
  build(svg) {
    svg.innerHTML = `
      <rect x="40" y="40" width="160" height="70" rx="5" class="skl-axis" style="fill:#1b242f"/>
      <text x="60" y="62" style="fill:#9fb0c0;font-size:11px">R2</text>
      <text id="globalcurve-r2" x="190" y="62" text-anchor="end" style="fill:#c4a3ff;font-size:13px;font-weight:600">0.00</text>
      <line x1="56" y1="72" x2="184" y2="72" class="skl-axis"/>
      <text x="60" y="92" style="fill:#9fb0c0;font-size:11px">PSNR</text>
      <text id="globalcurve-psnr" x="190" y="92" text-anchor="end" style="fill:#4fd6c4;font-size:13px;font-weight:600">0.0 dB</text>
      <rect id="globalcurve-bar" x="56" y="98" width="0" height="5" rx="2" class="skl-pop f-fin"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const r2=svg.querySelector('#globalcurve-r2'), ps=svg.querySelector('#globalcurve-psnr'), bar=svg.querySelector('#globalcurve-bar');
    const o={r:0, p:0};
    tl.set(bar, { attr:{ width: 0 } })
      .to(o, { r: 0.94, duration: 1.4, ease:"power2.out", onUpdate:()=>{ r2.textContent=o.r.toFixed(2); } }, 0)
      .to(o, { p: 28.4, duration: 1.4, ease:"power2.out", onUpdate:()=>{ ps.textContent=o.p.toFixed(1)+" dB"; } }, 0.3)
      .to(bar, { attr:{ width: 128 }, duration: 1.4, ease:"power2.out" }, 0.3)
      .to(r2, { scale: 1.15, transformOrigin:"190px 58px", duration: 0.25, yoyo:true, repeat:1 }, 1.5);
    return tl;
  },
},
elevssim: {
  tip: "Per elevation bin the MAE, RMSE, R-squared and a cross-entropy between column-normalised distributions are accumulated, alongside a mean SSIM swept over the elevation slices.",
  build(svg) {
    let bars="";
    const h=[20,34,46,40,28,18];
    for(let i=0;i<6;i++){const x=44+i*16;bars+=`<rect class="elevssim-bar skl-pop f-cal" x="${x}" y="${108-h[i]}" width="11" height="${h[i]}" rx="1"/>`;}
    svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="108" x2="148" y2="108"/>
      ${bars}
      <rect x="162" y="44" width="52" height="52" class="skl-axis" style="fill:#1b242f"/>
      <rect id="elevssim-slice" x="162" y="44" width="52" height="9" class="skl-pop f-fin" style="opacity:0.5"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const bars=svg.querySelectorAll('.elevssim-bar');
    const slice=svg.querySelector('#elevssim-slice');
    tl.set(bars, { scaleY: 0, transformOrigin:"50% 108px" })
      .set(slice, { y: 0 })
      .to(bars, { scaleY: 1, duration: 0.5, stagger: 0.08, ease:"power2.out" }, 0.1)
      .to(slice, { y: 43, duration: 1.2, ease:"none", yoyo:true, repeat:1 }, 0.8);
    return tl;
  },
},
paramslot: {
  tip: "On active pixels the per-Gaussian mu and sigma MAE/RMSE, placeholder F1 and mu-ordering rate are pooled, with a permutation consensus voted from per-pixel mu-distance assignment.",
  build(svg) {
    svg.innerHTML = `
      <circle cx="60" cy="56" r="5" class="skl-pop f-cal"/>
      <circle cx="60" cy="56" r="9" class="skl-dash c-meas" style="fill:none"/>
      <line id="paramslot-d" class="skl-dash c-mid" x1="60" y1="56" x2="60" y2="47"/>
      <rect id="paramslot-b0" x="110" y="56" width="14" height="0" class="skl-pop f-faint"/>
      <rect id="paramslot-b1" x="132" y="56" width="14" height="0" class="skl-pop f-fin"/>
      <rect id="paramslot-b2" x="154" y="56" width="14" height="0" class="skl-pop f-faint"/>
      <rect id="paramslot-b3" x="176" y="56" width="14" height="0" class="skl-pop f-faint"/>
      <line class="skl-axis" x1="104" y1="100" x2="196" y2="100"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const d=svg.querySelector('#paramslot-d');
    const bs=['#paramslot-b0','#paramslot-b1','#paramslot-b2','#paramslot-b3'].map(s=>svg.querySelector(s));
    const heights=[14,40,10,16];
    tl.set(bs, { attr:{ height: 0, y: 100 } })
      .fromTo(d, { attr:{ y2: 56 } }, { attr:{ y2: 40 }, duration: 0.4, yoyo:true, repeat:1, ease:"sine.inOut" }, 0);
    bs.forEach((b,i)=> tl.to(b, { attr:{ height: heights[i], y: 100-heights[i] }, duration: 0.5, ease:"power2.out" }, 0.5+i*0.12));
    tl.to(bs[1], { opacity: 1, scale: 1.05, transformOrigin:"139px 80px", duration: 0.3, yoyo:true, repeat:1 }, 1.3);
    return tl;
  },
},
reduced: {
  tip: "For a strict secondary subset a reduced Capon tomogram is re-synthesised, then GT, prediction and reduced cubes are unit-area normalised and the network's per-pixel MSE gain over the reduced baseline is mapped.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="100" x2="150" y2="100"/>
      <path id="reduced-gt" class="skl-draw c-meas" d="M40 100 C66 100 72 48 95 48 C118 48 124 100 150 100"/>
      <path id="reduced-pred" class="skl-draw c-fin" d="M40 100 C68 100 74 56 95 56 C116 56 122 100 150 100" style="opacity:0"/>
      <path id="reduced-red" class="skl-draw c-mid" d="M40 100 C62 100 70 74 88 70 C108 66 118 100 150 100" style="opacity:0"/>
      <rect x="166" y="46" width="48" height="48" class="skl-axis" style="fill:#1b242f"/>
      <rect id="reduced-imp" x="166" y="46" width="48" height="48" class="skl-pop f-fin" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.7 });
    const pred=svg.querySelector('#reduced-pred'), red=svg.querySelector('#reduced-red'), imp=svg.querySelector('#reduced-imp');
    tl.set([pred,red], { opacity: 0 })
      .set(imp, { opacity: 0, scale: 0.5, transformOrigin:"190px 70px" })
      .to(red, { opacity: 0.85, duration: 0.6, ease:"power2.out" }, 0.2)
      .to(pred, { opacity: 0.95, duration: 0.6, ease:"power2.out" }, 0.8)
      .to(imp, { opacity: 0.8, scale: 1, duration: 0.7, ease:"back.out(1.5)" }, 1.5);
    return tl;
  },
},
spacelr: {
  tip: "Encoder, bottleneck, decoder and head learning rates are each drawn from log-uniform on the decade span 1e-5 to 1e-2, with dropout drawn linearly across 0 to 0.5.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <line class="skl-dash c-faint" x1="78" y1="114" x2="78" y2="122"/>
      <line class="skl-dash c-faint" x1="126" y1="114" x2="126" y2="122"/>
      <line class="skl-dash c-faint" x1="174" y1="114" x2="174" y2="122"/>
      <circle id="spacelr-e" class="skl-pop f-mid" cx="58" cy="118" r="4"/>
      <circle id="spacelr-b" class="skl-pop f-mid" cx="112" cy="118" r="4"/>
      <circle id="spacelr-d" class="skl-pop f-mid" cx="150" cy="118" r="4"/>
      <circle id="spacelr-h" class="skl-pop f-mid" cx="192" cy="118" r="4"/>
      <line class="skl-axis" x1="30" y1="40" x2="210" y2="40"/>
      <circle id="spacelr-p" class="skl-pop f-cal" cx="96" cy="40" r="4"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const ids = ['#spacelr-e','#spacelr-b','#spacelr-d','#spacelr-h'].map(s => svg.querySelector(s));
    const tx = [58,112,150,192];
    tl.set(ids, { attr: { cy: 118 }, opacity: 0.3 })
      .set(svg.querySelector('#spacelr-p'), { attr: { cx: 36 }, opacity: 0.3 });
    ids.forEach((g, i) => {
      tl.fromTo(g, { attr: { cx: 30 }, opacity: 0.3 }, { attr: { cx: tx[i] }, opacity: 1, duration: 0.7, ease: "power2.out" }, 0.2 + i * 0.25);
    });
    tl.fromTo(svg.querySelector('#spacelr-p'), { attr: { cx: 36 }, opacity: 0.3 }, { attr: { cx: 96 }, opacity: 1, duration: 0.8, ease: "sine.inOut" }, 1.3)
      .to([...ids, svg.querySelector('#spacelr-p')], { opacity: 0.3, duration: 0.4 }, 2.6);
    return tl;
  },
},
spacearch: {
  tip: "Five categorical knobs are sampled, with the features list stored as index k in {0,1,2,3} and decoded through the lookup table C back to a channel list like [64,128,256,512].",
  build(svg) {
    svg.innerHTML = `
      <rect class="spacearch-k skl-pop f-faint" x="28" y="22" width="26" height="16" rx="3"/>
      <rect class="spacearch-k skl-pop f-faint" x="58" y="22" width="26" height="16" rx="3"/>
      <rect id="spacearch-k2" class="spacearch-k skl-pop f-mid" x="88" y="22" width="26" height="16" rx="3"/>
      <rect class="spacearch-k skl-pop f-faint" x="118" y="22" width="26" height="16" rx="3"/>
      <path id="spacearch-arrow" class="skl-draw c-mid" d="M101 40 q24 18 60 18" style="opacity:0;fill:none"/>
      <rect id="spacearch-list" class="skl-pop f-cal" x="150" y="48" width="62" height="18" rx="3" style="opacity:0"/>
      <text id="spacearch-listtx" x="155" y="61" style="font-size:8px;fill:#0b0f14;opacity:0">[64,128,256,512]</text>
      <rect class="skl-pop f-faint" x="28" y="86" width="40" height="14" rx="3"/>
      <rect class="skl-pop f-faint" x="74" y="86" width="30" height="14" rx="3"/>
      <rect class="skl-pop f-faint" x="110" y="86" width="34" height="14" rx="3"/>
      <rect class="skl-pop f-faint" x="150" y="86" width="32" height="14" rx="3"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const chips = svg.querySelectorAll('.spacearch-k');
    tl.set([svg.querySelector('#spacearch-list'), svg.querySelector('#spacearch-listtx'), svg.querySelector('#spacearch-arrow')], { opacity: 0 });
    chips.forEach((c, i) => {
      tl.to(c, { scale: 1.18, transformOrigin: "center", duration: 0.18, yoyo: true, repeat: 1 }, 0.15 * i);
    });
    tl.fromTo(svg.querySelector('#spacearch-k2'), { scale: 1 }, { scale: 1.25, transformOrigin: "center", duration: 0.3, yoyo: true, repeat: 1 }, 0.9)
      .fromTo(svg.querySelector('#spacearch-arrow'), { opacity: 0, strokeDasharray: 90, strokeDashoffset: 90 }, { opacity: 1, strokeDashoffset: 0, duration: 0.6, ease: "power2.out" }, 1.5)
      .fromTo([svg.querySelector('#spacearch-list'), svg.querySelector('#spacearch-listtx')], { opacity: 0, x: -10 }, { opacity: 1, x: 0, duration: 0.5, ease: "power2.out" }, 2.1);
    return tl;
  },
},
merge: {
  tip: "The 9-dim learning block and the 5-dim architecture block are stacked into one 14-dimensional joint space that multivariate TPE samples jointly.",
  build(svg) {
    svg.innerHTML = `
      <rect id="merge-lr" class="skl-pop f-mid" x="30" y="38" width="50" height="74" rx="4" style="opacity:.85"/>
      <rect id="merge-ar" class="skl-pop f-cal" x="160" y="38" width="50" height="74" rx="4" style="opacity:.85"/>
      <rect id="merge-join" class="skl-pop f-meas" x="95" y="46" width="50" height="58" rx="4" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    tl.set(svg.querySelector('#merge-lr'), { x: 30, opacity: 0.85 })
      .set(svg.querySelector('#merge-ar'), { x: 160, opacity: 0.85 })
      .set(svg.querySelector('#merge-join'), { opacity: 0 })
      .to(svg.querySelector('#merge-lr'), { x: 60, duration: 1.0, ease: "power2.inOut" }, 0.3)
      .to(svg.querySelector('#merge-ar'), { x: 110, duration: 1.0, ease: "power2.inOut" }, 0.3)
      .to([svg.querySelector('#merge-lr'), svg.querySelector('#merge-ar')], { opacity: 0, duration: 0.4 }, 1.3)
      .fromTo(svg.querySelector('#merge-join'), { opacity: 0, scale: 0.85, transformOrigin: "120px 75px" }, { opacity: 1, scale: 1, duration: 0.5, ease: "back.out(1.6)" }, 1.3);
    return tl;
  },
},
tpesplit: {
  tip: "Trials are split at the gamma quantile y_gamma of their loss into a good set below and a bad set above, each fitted with its own KDE: l(theta) and g(theta).",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <line id="tpesplit-cut" class="skl-dash c-faint" x1="120" y1="24" x2="120" y2="124"/>
      <path id="tpesplit-good" class="skl-draw c-cal" d="M34 118 C70 118 86 40 110 40 C116 40 118 118 120 118 Z" style="fill:rgba(45,212,191,.18)"/>
      <path id="tpesplit-bad" class="skl-draw c-mid" d="M120 118 C122 118 132 70 158 70 C190 70 196 118 206 118 Z" style="fill:rgba(251,146,60,.16)"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const good = svg.querySelector('#tpesplit-good');
    const bad = svg.querySelector('#tpesplit-bad');
    const flat = "M34 118 C70 118 86 118 110 118 C116 118 118 118 120 118 Z";
    const flatB = "M120 118 C122 118 132 118 158 118 C190 118 196 118 206 118 Z";
    tl.set(good, { attr: { d: flat } })
      .set(bad, { attr: { d: flatB } })
      .fromTo(svg.querySelector('#tpesplit-cut'), { attr: { y1: 124 }, opacity: 0 }, { attr: { y1: 24 }, opacity: 1, duration: 0.6, ease: "power2.out" }, 0.2)
      .to(good, { attr: { d: "M34 118 C70 118 86 40 110 40 C116 40 118 118 120 118 Z" }, duration: 0.9, ease: "power2.out" }, 0.7)
      .to(bad, { attr: { d: "M120 118 C122 118 132 70 158 70 C190 70 196 118 206 118 Z" }, duration: 0.9, ease: "power2.out" }, 0.9);
    return tl;
  },
},
tpeacq: {
  tip: "TPE proposes the theta maximising the density ratio l(theta) over g(theta), darting a marker to its argmax, except during the first n0 = 8 startup trials which sample uniformly.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <path class="skl-draw c-cal" d="M34 110 C70 110 86 48 108 48 C120 48 124 110 130 110" style="opacity:.5"/>
      <path class="skl-draw c-mid" d="M40 110 C90 110 110 80 150 80 C188 80 196 110 206 110" style="opacity:.5"/>
      <path id="tpeacq-ratio" class="skl-draw c-fin" d="M34 116 C66 116 80 36 104 30 C124 26 132 116 150 116 C170 116 190 116 206 116"/>
      <line id="tpeacq-vline" class="skl-dash c-fin" x1="106" y1="30" x2="106" y2="118" style="opacity:0"/>
      <circle id="tpeacq-mark" class="skl-pop f-fin" cx="40" cy="116" r="5"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const path = svg.querySelector('#tpeacq-ratio');
    const mark = svg.querySelector('#tpeacq-mark');
    const len = path.getTotalLength();
    const peak = path.getPointAtLength(len * 0.28);
    const samples = [0.08, 0.55, 0.78, 0.4, 0.2];
    tl.set(mark, { attr: { cx: 40, cy: 116 } })
      .set(svg.querySelector('#tpeacq-vline'), { opacity: 0 });
    samples.forEach((s, i) => {
      const p = path.getPointAtLength(len * s);
      tl.to(mark, { attr: { cx: p.x, cy: p.y }, duration: 0.35, ease: "power1.inOut" }, 0.2 + i * 0.38);
    });
    tl.to(mark, { attr: { cx: peak.x, cy: peak.y }, duration: 0.6, ease: "power3.out" }, 2.2)
      .fromTo(mark, { scale: 1 }, { scale: 1.6, transformOrigin: "center", duration: 0.3, yoyo: true, repeat: 1 }, 2.8)
      .fromTo(svg.querySelector('#tpeacq-vline'), { opacity: 0 }, { opacity: 1, duration: 0.4 }, 2.8);
    return tl;
  },
},
liar: {
  tip: "Each pending trial across the parallel GPU workers is temporarily handed the worst completed objective max f as a phantom value so the workers do not all crowd the same proposed point.",
  build(svg) {
    svg.innerHTML = `
      <rect class="skl-pop f-faint" x="96" y="14" width="48" height="18" rx="4"/>
      <rect class="skl-pop f-meas" x="28" y="74" width="36" height="22" rx="4"/>
      <rect class="skl-pop f-meas" x="84" y="74" width="36" height="22" rx="4"/>
      <rect class="skl-pop f-meas" x="140" y="74" width="36" height="22" rx="4"/>
      <line class="skl-axis" x1="120" y1="32" x2="46" y2="74" style="opacity:.4"/>
      <line class="skl-axis" x1="120" y1="32" x2="102" y2="74" style="opacity:.4"/>
      <line class="skl-axis" x1="120" y1="32" x2="158" y2="74" style="opacity:.4"/>
      <circle id="liar-p0" class="skl-pop f-mid" cx="46" cy="116" r="6" style="opacity:0"/>
      <circle id="liar-p1" class="skl-pop f-mid" cx="102" cy="116" r="6" style="opacity:0"/>
      <circle id="liar-p2" class="skl-pop f-mid" cx="158" cy="116" r="6" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const phantoms = ['#liar-p0','#liar-p1','#liar-p2'].map(s => svg.querySelector(s));
    tl.set(phantoms, { opacity: 0, attr: { cy: 100 } });
    phantoms.forEach((p, i) => {
      tl.fromTo(p, { opacity: 0, attr: { cy: 100 } }, { opacity: 1, attr: { cy: 116 }, duration: 0.5, ease: "back.out(2)" }, 0.4 + i * 0.3);
    });
    tl.to(phantoms, { attr: { r: 7 }, duration: 0.3, yoyo: true, repeat: 1 }, 1.8);
    return tl;
  },
},
trialsetup: {
  tip: "The base config is deep-copied per trial and overridden with E = 30 epochs, patience = 8, and seed = 42 + trial.number before any training begins.",
  build(svg) {
    svg.innerHTML = `
      <rect id="trialsetup-base" class="skl-pop f-faint" x="30" y="50" width="56" height="50" rx="5"/>
      <path id="trialsetup-arr" class="skl-draw c-faint" d="M88 75 L128 75" style="fill:none"/>
      <rect id="trialsetup-clone" class="skl-pop f-mid" x="134" y="46" width="60" height="58" rx="5" style="opacity:0"/>
      <line id="trialsetup-l0" class="skl-draw c-cal" x1="142" y1="60" x2="186" y2="60" style="opacity:0"/>
      <line id="trialsetup-l1" class="skl-draw c-cal" x1="142" y1="72" x2="186" y2="72" style="opacity:0"/>
      <line id="trialsetup-l2" class="skl-draw c-cal" x1="142" y1="84" x2="186" y2="84" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const arr = svg.querySelector('#trialsetup-arr');
    tl.set(svg.querySelector('#trialsetup-clone'), { opacity: 0, x: -40 })
      .set([svg.querySelector('#trialsetup-l0'), svg.querySelector('#trialsetup-l1'), svg.querySelector('#trialsetup-l2')], { opacity: 0, attr: { x2: 142 } })
      .fromTo(arr, { strokeDasharray: 44, strokeDashoffset: 44 }, { strokeDashoffset: 0, duration: 0.5, ease: "power2.out" }, 0.2)
      .fromTo(svg.querySelector('#trialsetup-clone'), { opacity: 0, x: -40 }, { opacity: 1, x: 0, duration: 0.6, ease: "power2.out" }, 0.6)
      .to(svg.querySelector('#trialsetup-l0'), { opacity: 1, attr: { x2: 186 }, duration: 0.35 }, 1.3)
      .to(svg.querySelector('#trialsetup-l1'), { opacity: 1, attr: { x2: 186 }, duration: 0.35 }, 1.6)
      .to(svg.querySelector('#trialsetup-l2'), { opacity: 1, attr: { x2: 186 }, duration: 0.35 }, 1.9);
    return tl;
  },
},
trial: {
  tip: "A trial trains a full model on the fixed canonical split for up to 30 epochs and returns f(theta), the minimum validation loss reached along that curve.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="120" x2="210" y2="120"/>
      <line class="skl-axis" x1="30" y1="24" x2="30" y2="120"/>
      <path id="trial-curve" class="skl-draw c-cal" d="M30 36 C60 70 78 92 104 100 C140 110 170 104 206 102"/>
      <line id="trial-minline" class="skl-dash c-fin" x1="30" y1="102" x2="206" y2="102" style="opacity:0"/>
      <circle id="trial-min" class="skl-pop f-cal" cx="158" cy="102" r="4" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 });
    const c = svg.querySelector('#trial-curve');
    const len = c.getTotalLength();
    tl.set(c, { strokeDasharray: len, strokeDashoffset: len })
      .set([svg.querySelector('#trial-min'), svg.querySelector('#trial-minline')], { opacity: 0 })
      .to(c, { strokeDashoffset: 0, duration: 1.6, ease: "power1.inOut" }, 0.2)
      .fromTo(svg.querySelector('#trial-min'), { opacity: 0, scale: 0.5, transformOrigin: "center" }, { opacity: 1, scale: 1, duration: 0.4, ease: "back.out(2)" }, 1.7)
      .fromTo(svg.querySelector('#trial-minline'), { opacity: 0 }, { opacity: 0.8, duration: 0.4 }, 1.8);
    return tl;
  },
},
prune: {
  tip: "At step t a trial whose validation loss rises above the running median m of completed trials is pruned, but only once n_done and t both clear the 8-step warmup gate.",
  build(svg) {
    svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="120" x2="210" y2="120"/>
      <line id="prune-med" class="skl-dash c-mid" x1="30" y1="72" x2="210" y2="72"/>
      <line id="prune-gate" class="skl-dash c-faint" x1="92" y1="24" x2="92" y2="120"/>
      <path id="prune-curve" class="skl-draw c-cal" d="M30 40 C50 56 70 64 92 66 C104 67 110 66 116 64"/>
      <line id="prune-x1" class="skl-draw c-fin" x1="118" y1="58" x2="130" y2="70" style="opacity:0"/>
      <line id="prune-x2" class="skl-draw c-fin" x1="130" y1="58" x2="118" y2="70" style="opacity:0"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const c = svg.querySelector('#prune-curve');
    const len = c.getTotalLength();
    const xs = [svg.querySelector('#prune-x1'), svg.querySelector('#prune-x2')];
    tl.set(c, { strokeDasharray: len, strokeDashoffset: len })
      .set(svg.querySelector('#prune-med'), { attr: { x2: 30 } })
      .set(xs, { opacity: 0 })
      .to(svg.querySelector('#prune-med'), { attr: { x2: 210 }, duration: 0.6, ease: "power2.out" }, 0.1)
      .to(c, { strokeDashoffset: 0, duration: 1.3, ease: "power1.inOut" }, 0.5)
      .fromTo(xs, { opacity: 0, scale: 0.5, transformOrigin: "124px 64px" }, { opacity: 1, scale: 1, duration: 0.3, ease: "back.out(2)" }, 1.9);
    return tl;
  },
},
best: {
  tip: "The study dispatches only the remaining n_rem = max(0, 100 - n_done) trials in GPU chunks, and after each completion rewrites theta-star as the argmin of f, decoding the features index back to its channel list.",
  build(svg) {
    svg.innerHTML = `
      <rect class="skl-pop f-faint" x="28" y="26" width="14" height="18" rx="2"/>
      <rect class="skl-pop f-faint" x="46" y="26" width="14" height="18" rx="2"/>
      <rect id="best-rem" class="skl-pop f-mid" x="64" y="26" width="14" height="18" rx="2"/>
      <rect class="skl-pop f-mid" x="82" y="26" width="14" height="18" rx="2"/>
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <circle class="skl-pop f-cal" cx="52" cy="86" r="3" style="opacity:.5"/>
      <circle class="skl-pop f-cal" cx="86" cy="70" r="3" style="opacity:.5"/>
      <circle class="skl-pop f-cal" cx="118" cy="98" r="3" style="opacity:.5"/>
      <circle id="best-star" class="skl-pop f-fin" cx="150" cy="110" r="5"/>
      <circle class="skl-pop f-cal" cx="184" cy="82" r="3" style="opacity:.5"/>`;
  },
  anim(svg, gsap) {
    if (!gsap) return null;
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.6 });
    const star = svg.querySelector('#best-star');
    const candidates = [[52,86],[86,70],[118,98],[150,110],[184,82]];
    tl.set(star, { attr: { cx: 52, cy: 86 }, scale: 1, transformOrigin: "center" })
      .fromTo(svg.querySelector('#best-rem'), { opacity: 0.3, scale: 0.8, transformOrigin: "center" }, { opacity: 1, scale: 1, duration: 0.4 }, 0.2);
    candidates.forEach((p, i) => {
      tl.to(star, { attr: { cx: p[0], cy: p[1] }, duration: 0.3, ease: "power1.inOut" }, 0.4 + i * 0.32);
    });
    tl.to(star, { attr: { cx: 150, cy: 110 }, duration: 0.45, ease: "power3.out" }, 2.0)
      .fromTo(star, { scale: 1 }, { scale: 1.6, transformOrigin: "center", duration: 0.35, yoyo: true, repeat: 1 }, 2.45);
    return tl;
  },
},
};
