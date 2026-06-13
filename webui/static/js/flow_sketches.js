"use strict";

const pulse = (svg, gsap) => {
  if (!gsap) return null;
  const els = svg.querySelectorAll(".sk-live");
  if (!els.length) return null;
  return gsap.to(els, { opacity: 0.35, duration: 1.25, ease: "sine.inOut", repeat: -1, yoyo: true, stagger: { each: 0.16, from: "start" } });
};

const grid = (cells, build) => {
  let s = "";
  for (let r = 0; r < cells; r++) for (let c = 0; c < cells; c++) s += build(r, c);
  return s;
};

window.FLOW_SKETCHES = {

  slc_load: {
    tip: "The master loads as an RGI-SLC range-Doppler image while each secondary arrives as an INF-SLC already co-registered and carrying its own DEM-predicted phase.",
    build(svg) { svg.innerHTML = `
      <rect x="34" y="36" width="50" height="80" rx="3" class="skl-pop f-meas"/>
      <text x="59" y="128" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">master</text>
      <rect x="150" y="28" width="46" height="74" rx="3" class="skl-pop f-meas" style="opacity:.4"/>
      <rect x="142" y="36" width="46" height="74" rx="3" class="skl-pop f-meas" style="opacity:.6"/>
      <rect x="134" y="44" width="46" height="74" rx="3" class="skl-pop f-meas" style="opacity:.85"/>
      <text x="160" y="128" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">secondaries</text>
      <path class="skl-dash c-mid" d="M86 76 C106 64 116 62 132 74" style="fill:none"/>`; },
    anim: null,
  },

  baselines: {
    tip: "Horizontal and vertical track positions are averaged over the azimuth window and re-expressed relative to track 0, aborting if any per-track position std exceeds 5 metres.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="116" x2="206" y2="116"/>
      <line class="skl-axis" x1="40" y1="116" x2="40" y2="28"/>
      <circle cx="40" cy="100" r="4" class="skl-pop f-faint"/>
      <text x="40" y="132" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">ref</text>
      <line class="skl-dash c-cal" x1="40" y1="100" x2="92" y2="68"/>
      <line class="skl-dash c-cal" x1="40" y1="100" x2="138" y2="90"/>
      <line class="skl-dash c-cal" x1="40" y1="100" x2="186" y2="46"/>
      <circle class="sk-live skl-pop f-cal" cx="92" cy="68" r="4"/>
      <circle class="sk-live skl-pop f-cal" cx="138" cy="90" r="4"/>
      <circle class="sk-live skl-pop f-cal" cx="186" cy="46" r="4"/>`; },
    anim: pulse,
  },

  deramp: {
    tip: "Multiplying each secondary by exp(j phi_DEM) cancels the DEM-predicted terrain ramp under the later conjugation, leaving only sub-resolution elevation structure.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="120" x2="214" y2="120"/>
      <line class="skl-dash c-faint" x1="28" y1="112" x2="150" y2="46"/>
      <path class="skl-draw c-meas" d="M28 100 q12 -12 24 0 t24 0 t24 0 t24 0 t24 0" style="fill:none"/>
      <text x="76" y="36" style="fill:#7e8aa0;font-size:7px">DEM ramp</text>
      <path class="skl-draw c-cal" d="M28 88 q14 -12 28 0 t28 0 t28 0 t28 0 t28 0 t14 0" transform="translate(0 24)" style="fill:none"/>
      <text x="150" y="118" style="fill:#4fd6c4;font-size:7px">flat residual</text>`; },
    anim: null,
  },

  crossprod: {
    tip: "Conjugating the deramped secondary against the master subtracts its phase, and because conjugation flips the DEM sign the term phi_DEM is effectively removed from arg(c_i).",
    build(svg) { svg.innerHTML = `
      <circle cx="80" cy="74" r="36" class="skl-axis" style="fill:none"/>
      <line class="skl-axis" x1="44" y1="74" x2="116" y2="74"/>
      <line class="skl-axis" x1="80" y1="38" x2="80" y2="110"/>
      <line class="skl-draw c-meas" x1="80" y1="74" x2="108" y2="52"/>
      <text x="110" y="50" style="fill:#6ea8ff;font-size:8px">s0</text>
      <line class="skl-draw c-mid" x1="80" y1="74" x2="100" y2="98"/>
      <text x="100" y="110" style="fill:#f5b971;font-size:8px">s_i</text>
      <text x="150" y="78" style="fill:#7e8aa0;font-size:14px">=</text>
      <circle cx="190" cy="74" r="20" class="skl-axis" style="fill:none;opacity:.4"/>
      <line class="skl-draw c-cal" x1="190" y1="74" x2="210" y2="58"/>
      <text x="198" y="104" style="fill:#4fd6c4;font-size:8px">c_i</text>`; },
    anim: null,
  },

  phasor: {
    tip: "Dividing by |c_i| floored at 1e-30 collapses every cross-product onto the unit circle, equalising inter-pass amplitude while null pixels go to zero instead of NaN.",
    build(svg) { svg.innerHTML = `
      <circle class="sk-live skl-pop f-faint" cx="120" cy="76" r="42" style="fill:none;stroke:#4a5a6b;stroke-width:1.4;stroke-dasharray:3 4;opacity:1"/>
      <line class="skl-axis" x1="70" y1="76" x2="170" y2="76"/>
      <line class="skl-axis" x1="120" y1="30" x2="120" y2="122"/>
      <line class="skl-draw c-mid" x1="120" y1="76" x2="150" y2="46"/>
      <line class="skl-draw c-mid" x1="120" y1="76" x2="84" y2="98"/>
      <line class="skl-draw c-mid" x1="120" y1="76" x2="156" y2="98"/>
      <circle cx="150" cy="46" r="3.5" class="skl-pop f-cal"/>
      <circle cx="84" cy="98" r="3.5" class="skl-pop f-cal"/>
      <circle cx="156" cy="98" r="3.5" class="skl-pop f-cal"/>`; },
    anim: pulse,
  },

  clip: {
    tip: "The secondary amplitude is capped at c_max = 1.25 so a single bright corner reflector or artefact cannot dominate the per-pass weight.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="214" y2="118"/>
      <line class="sk-live skl-pop" x1="30" y1="56" x2="214" y2="56" style="stroke:#c4a3ff;stroke-width:1.4;stroke-dasharray:4 4;opacity:1"/>
      <text x="210" y="50" text-anchor="end" style="fill:#c4a3ff;font-size:8px">c_max</text>
      <rect x="46" y="90" width="18" height="28" class="skl-pop f-mid"/>
      <rect x="74" y="76" width="18" height="42" class="skl-pop f-mid"/>
      <rect x="102" y="56" width="18" height="62" class="skl-pop f-mid"/>
      <rect x="130" y="98" width="18" height="20" class="skl-pop f-mid"/>
      <rect x="158" y="56" width="18" height="62" class="skl-pop f-mid"/>
      <rect x="186" y="82" width="18" height="36" class="skl-pop f-mid"/>`; },
    anim: pulse,
  },

  interf: {
    tip: "The clipped amplitude A_i is re-attached as the modulus of the unit phasor, producing an interferogram whose argument is the residual elevation phase and whose magnitude is a bounded SNR proxy.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="74" y1="76" x2="166" y2="76"/>
      <line class="skl-axis" x1="120" y1="32" x2="120" y2="120"/>
      <circle cx="120" cy="76" r="20" class="skl-axis" style="fill:none;opacity:.4"/>
      <circle cx="120" cy="76" r="40" class="skl-dash c-faint" style="fill:none;opacity:.4"/>
      <line class="skl-draw c-cal" x1="120" y1="76" x2="152" y2="54"/>
      <circle cx="152" cy="54" r="3.5" class="skl-pop f-cal"/>
      <text x="148" y="118" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">A_i p_i</text>`; },
    anim: null,
  },

  subdivide: {
    tip: "A crop above W_max = 1000 lines is split into M non-overlapping azimuth subsections processed by a worker plan resolved from the core budget B = floor(C f_effort).",
    build(svg) { svg.innerHTML = `
      <rect x="40" y="30" width="38" height="104" rx="2" class="skl-pop f-meas" style="opacity:.8"/>
      <text x="108" y="80" text-anchor="middle" style="fill:#7e8aa0;font-size:12px">&#8594;</text>
      <rect x="134" y="30" width="76" height="22" rx="2" class="skl-pop f-mid"/>
      <rect x="134" y="56" width="76" height="22" rx="2" class="skl-pop f-mid"/>
      <rect x="134" y="82" width="76" height="22" rx="2" class="skl-pop f-mid"/>
      <rect x="134" y="108" width="76" height="22" rx="2" class="skl-pop f-mid"/>
      <circle class="sk-live skl-pop f-cal" cx="146" cy="41" r="3"/>
      <circle class="sk-live skl-pop f-cal" cx="146" cy="67" r="3"/>
      <circle class="sk-live skl-pop f-cal" cx="146" cy="93" r="3"/>
      <circle class="sk-live skl-pop f-cal" cx="146" cy="119" r="3"/>`; },
    anim: pulse,
  },

  covariance: {
    tip: "A 20x10 px Boxcar window averages the interferometric stack into the per-pixel sample covariance R-hat that the Capon estimator later inverts; its diagonal holds per-pass power.",
    build(svg) {
      const m = grid(4, (r, c) => {
        const x = 92 + c * 26, y = 34 + r * 26;
        const cl = r === c ? "sk-live skl-pop f-cal" : (Math.abs(r - c) === 1 ? "skl-pop f-mid" : "skl-pop f-faint");
        const op = r === c ? 1 : (Math.abs(r - c) === 1 ? 0.5 : 0.28);
        return `<rect class="${cl}" x="${x}" y="${y}" width="22" height="22" rx="2" style="opacity:${op}"/>`;
      });
      svg.innerHTML = `
        <rect x="36" y="50" width="34" height="20" rx="2" class="skl-draw c-mid" style="fill:none"/>
        <text x="53" y="84" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">boxcar</text>
        ${m}
        <text x="143" y="126" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">R-hat</text>`;
    },
    anim: pulse,
  },

  capon: {
    tip: "Over a uniform elevation grid spanning [x_min, x_max] the minimum-variance estimator evaluates 1/(a^H R^-1 a), peaking sharply at the true scatterer height.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="36" y1="120" x2="212" y2="120"/>
      <path class="skl-draw c-cal" d="M36 116 L74 113 L104 108 L124 56 L144 108 L176 114 L212 117" style="fill:none"/>
      <line class="skl-dash c-faint" x1="124" y1="56" x2="124" y2="120"/>
      <circle class="sk-live skl-pop f-fin" cx="124" cy="56" r="4"/>`; },
    anim: pulse,
  },

  concat: {
    tip: "Each worker's HDF5 subsection is reassembled along azimuth, stacking the DEM on axis 0 and the tomogram on axis 1 into the full-stack outputs.",
    build(svg) { svg.innerHTML = `
      <rect x="44" y="34" width="58" height="20" rx="2" class="skl-pop f-cal"/>
      <rect x="44" y="58" width="58" height="20" rx="2" class="skl-pop f-cal"/>
      <rect x="44" y="82" width="58" height="20" rx="2" class="skl-pop f-cal"/>
      <rect x="44" y="106" width="58" height="20" rx="2" class="skl-pop f-cal"/>
      <text x="120" y="84" text-anchor="middle" style="fill:#7e8aa0;font-size:12px">&#8594;</text>
      <rect x="150" y="34" width="58" height="92" rx="3" class="skl-draw c-fin" style="fill:rgba(196,163,255,0.1)"/>
      <text x="179" y="138" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">T_comb</text>`; },
    anim: null,
  },

  threshold: {
    tip: "Magnitude samples below the relative floor t_f times the profile peak are zeroed and everything past index H_tr is dropped before the loss or R-squared see the profile.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="120" x2="214" y2="120"/>
      <line class="skl-dash c-mid" x1="24" y1="96" x2="180" y2="96"/>
      <text x="26" y="92" style="fill:#f5b971;font-size:7px">floor</text>
      <line class="skl-dash c-faint" x1="180" y1="22" x2="180" y2="122"/>
      <text x="184" y="32" style="fill:#7e8aa0;font-size:7px">H_tr</text>
      <path class="skl-draw c-meas" d="M24 118 L52 70 L64 104 L98 44 L114 100 L150 78 L168 106 L186 90 L204 86" style="fill:none;opacity:.28"/>
      <path class="skl-draw c-cal" d="M24 120 L52 70 L64 104 L82 120 L98 44 L114 100 L132 120 L150 78 L168 120 L180 120" style="fill:none"/>`; },
    anim: null,
  },

  activity: {
    tip: "A pixel is fitted only when its profile maximum clears the activity threshold tau_a of 1e-3; otherwise it is skipped with parameters left at zero and scale fixed to one.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <line class="skl-dash c-mid" x1="24" y1="58" x2="214" y2="58"/>
      <text x="26" y="52" style="fill:#f5b971;font-size:7px">tau_a</text>
      <path class="skl-draw c-cal" d="M40 118 L52 110 L66 44 L80 112 L92 118" style="fill:none"/>
      <circle cx="66" cy="44" r="4" class="skl-pop f-cal"/>
      <path class="skl-draw c-faint" d="M150 118 L162 112 L176 88 L190 113 L202 118" style="fill:none;opacity:.5"/>
      <circle cx="176" cy="88" r="4" class="skl-pop f-faint"/>
      <text x="176" y="78" text-anchor="middle" style="fill:#4a5a6b;font-size:8px">skip</text>`; },
    anim: null,
  },

  pnorm: {
    tip: "Dividing every bin by the per-pixel maximum lifts the tallest peak to exactly one, so the MSE and complexity penalty are comparable across pixels of any backscatter level.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="120" x2="214" y2="120"/>
      <line class="skl-axis" x1="24" y1="22" x2="24" y2="120"/>
      <line class="skl-dash c-cal" x1="24" y1="40" x2="214" y2="40"/>
      <text x="208" y="36" text-anchor="end" style="fill:#4fd6c4;font-size:7px">1.0</text>
      <path class="skl-draw c-faint" d="M24 116 L60 96 L96 70 L120 56 L150 92 L188 110 L214 114" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-cal" d="M24 112 L60 78 L96 40 L120 40 L150 84 L188 110 L214 116" style="fill:none"/>`; },
    anim: null,
  },

  peakfind: {
    tip: "find_peaks scans the raw profile with no smoothing, keeping a maximum only when its topographic prominence reaches p_frac of the peak and it sits at least d_min bins from any rival.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="120" x2="214" y2="120"/>
      <path class="skl-draw c-meas" d="M24 118 L44 96 L60 40 L78 102 L96 110 L114 70 L132 104 L150 112 L168 58 L190 104 L214 116" style="fill:none"/>
      <circle class="sk-live skl-pop f-cal" cx="60" cy="40" r="4.5"/>
      <circle class="sk-live skl-pop f-cal" cx="168" cy="58" r="4.5"/>
      <circle cx="114" cy="70" r="4" class="skl-pop f-faint"/>`; },
    anim: pulse,
  },

  geometry: {
    tip: "The span-derived sigma_base seeds the initial width sigma_base over D_sigma, while Adam is later clamped between one elevation bin and half the elevation span.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="120" x2="40" y2="24"/>
      <line class="skl-dash c-cal" x1="34" y1="40" x2="200" y2="40"/>
      <text x="46" y="36" style="fill:#4fd6c4;font-size:7px">sigma_hi = span/2</text>
      <line class="skl-dash c-meas" x1="34" y1="108" x2="200" y2="108"/>
      <text x="46" y="120" style="fill:#6ea8ff;font-size:7px">sigma_lo = 1 bin</text>
      <path class="skl-draw c-mid" d="M126 116 q44 -76 88 0" style="fill:none"/>
      <line class="skl-draw c-mid" x1="148" y1="80" x2="192" y2="80" style="stroke-width:1.4"/>
      <text x="158" y="74" style="fill:#f5b971;font-size:8px">sigma0</text>`; },
    anim: null,
  },

  residfill: {
    tip: "When fewer than K peaks are found, a window of half-width d_min is zeroed around each detected peak and the empty slots are filled by repeated argmax of that masked residual.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <rect x="32" y="22" width="32" height="96" class="skl-pop f-faint" style="opacity:.4"/>
      <rect x="92" y="22" width="32" height="96" class="skl-pop f-faint" style="opacity:.4"/>
      <path class="skl-draw c-meas" d="M24 116 L48 70 L66 100 L86 84 L108 50 L130 96 L150 78 L172 104 L196 66 L214 110" style="fill:none;opacity:.45"/>
      <circle cx="48" cy="70" r="4" class="skl-pop f-cal"/>
      <circle cx="108" cy="50" r="4" class="skl-pop f-cal"/>
      <circle cx="196" cy="66" r="4.5" class="skl-pop f-mid"/>
      <text x="196" y="56" text-anchor="middle" style="fill:#f5b971;font-size:7px">argmax</text>`; },
    anim: null,
  },

  seed: {
    tip: "Amplitude and mean are read straight off the raw-profile peaks and frozen through Phase 2, collapsing the fit to a well-conditioned one-dimensional width search per component.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <path class="skl-draw c-meas" d="M24 116 L52 100 L78 44 L104 102 L130 110 L158 60 L190 104 L214 112" style="fill:none;opacity:.45"/>
      <line class="skl-dash c-cal" x1="78" y1="44" x2="78" y2="118"/>
      <line class="skl-draw c-cal" x1="62" y1="44" x2="94" y2="44" style="stroke-width:1.6"/>
      <circle cx="78" cy="44" r="4" class="skl-pop f-cal"/>
      <line class="skl-dash c-cal" x1="158" y1="60" x2="158" y2="118"/>
      <line class="skl-draw c-cal" x1="142" y1="60" x2="174" y2="60" style="stroke-width:1.6"/>
      <circle cx="158" cy="60" r="4" class="skl-pop f-cal"/>
      <text x="118" y="132" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">a, mu frozen</text>`; },
    anim: null,
  },

  objective: {
    tip: "With amplitudes and means frozen, the loss is the mean-squared gap between the K-Gaussian sum and the normalised profile, with each sigma floored at 1e-6 before squaring.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="116" x2="214" y2="116"/>
      <path class="skl-draw c-mid" d="M24 114 q40 -74 80 0 q40 -54 86 0 L214 114" style="fill:none"/>
      <path class="skl-draw c-cal" d="M24 114 q40 -54 80 0 q40 -40 86 0 L214 114" style="fill:none"/>
      <line class="skl-draw c-fin" x1="64" y1="74" x2="64" y2="92" style="stroke-width:1.6"/>
      <line class="skl-draw c-fin" x1="150" y1="82" x2="150" y2="98" style="stroke-width:1.6"/>
      <text x="120" y="22" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">L = mean(residual^2)</text>`; },
    anim: null,
  },

  adam: {
    tip: "Bias-corrected Adam runs as a single lax.scan of T = 3000 steps compiled into one XLA program, with the widths clamped to the sigma_lo and sigma_hi band on every step.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="120" x2="214" y2="120"/>
      <line class="skl-axis" x1="28" y1="22" x2="28" y2="120"/>
      <line class="skl-dash c-faint" x1="28" y1="40" x2="214" y2="40"/>
      <line class="skl-dash c-faint" x1="28" y1="104" x2="214" y2="104"/>
      <path class="skl-draw c-cal" d="M28 100 C70 96 84 70 108 56 S170 44 214 44" style="fill:none"/>
      <circle class="sk-live skl-pop f-cal" cx="214" cy="44" r="4"/>`; },
    anim: pulse,
  },

  scoreK: {
    tip: "Each order K is scored as its normalised MSE plus the penalty lambda_K times K times the mean amplitude, so a slot is paid for only when a real peak fills it.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="32" y1="118" x2="214" y2="118"/>
      <rect x="44" y="58" width="20" height="60" class="skl-pop f-mid"/>
      <rect x="44" y="52" width="20" height="6" class="skl-pop f-fin"/>
      <rect x="92" y="92" width="20" height="26" class="skl-pop f-mid"/>
      <rect x="92" y="80" width="20" height="12" class="skl-pop f-fin"/>
      <rect x="140" y="86" width="20" height="32" class="skl-pop f-mid"/>
      <rect x="140" y="62" width="20" height="24" class="skl-pop f-fin"/>
      <rect x="188" y="80" width="20" height="38" class="skl-pop f-mid"/>
      <rect x="188" y="44" width="20" height="36" class="skl-pop f-fin"/>
      <text x="54" y="130" style="fill:#4a5a6b;font-size:7px">1</text>
      <text x="146" y="130" style="fill:#4a5a6b;font-size:7px">3</text>`; },
    anim: null,
  },

  selectK: {
    tip: "The penalised score is minimised over model order with ties broken toward the smaller K, so a tie between two and three components resolves in favour of the simpler fit.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="32" y1="116" x2="214" y2="116"/>
      <rect x="44" y="48" width="22" height="68" class="skl-pop f-mid"/>
      <rect class="sk-live skl-pop f-cal" x="92" y="84" width="22" height="32"/>
      <rect x="140" y="76" width="22" height="40" class="skl-pop f-mid"/>
      <rect x="188" y="62" width="22" height="54" class="skl-pop f-mid"/>
      <path class="skl-draw c-cal" d="M103 36 L103 62 M97 56 L103 62 L109 56" style="fill:none"/>
      <text x="103" y="30" text-anchor="middle" style="fill:#4fd6c4;font-size:9px;font-weight:600">K*=2</text>`; },
    anim: pulse,
  },

  rescale: {
    tip: "The winner's amplitudes are multiplied back by the per-pixel scale s to return to raw backscatter units, while its means and widths are written through unchanged.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="24" y1="118" x2="214" y2="118"/>
      <line class="skl-dash c-faint" x1="24" y1="56" x2="214" y2="56"/>
      <text x="28" y="52" style="fill:#7e8aa0;font-size:7px">1.0 (norm)</text>
      <path class="skl-draw c-faint" d="M40 116 q40 -56 80 0 q34 -40 70 0" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-fin" d="M40 116 q40 -84 80 0 q34 -60 70 0" style="fill:none"/>
      <text x="150" y="40" style="fill:#c4a3ff;font-size:8px">x s</text>`; },
    anim: null,
  },

  assemble: {
    tip: "Active components are sorted by ascending mean elevation, inactive slots keyed to infinity drop to the end, and the result is written into the interleaved 3K target with zeros beyond K*.",
    build(svg) { svg.innerHTML = `
      <rect x="40" y="40" width="26" height="20" class="skl-pop f-cal"/>
      <rect x="40" y="66" width="26" height="20" class="skl-pop f-cal"/>
      <rect x="40" y="92" width="26" height="20" class="skl-pop f-faint"/>
      <text x="86" y="80" style="fill:#7e8aa0;font-size:12px">&#8594;</text>
      <rect x="118" y="58" width="16" height="28" class="skl-pop f-fin"/>
      <rect x="136" y="58" width="16" height="28" class="skl-pop f-fin"/>
      <rect x="154" y="58" width="16" height="28" class="skl-pop f-fin"/>
      <rect x="172" y="58" width="16" height="28" class="skl-pop f-faint" style="opacity:.4"/>
      <rect x="190" y="58" width="16" height="28" class="skl-pop f-faint" style="opacity:.4"/>
      <text x="118" y="100" style="fill:#c4a3ff;font-size:7px">a mu sig a mu sig</text>`; },
    anim: null,
  },

  quality: {
    tip: "The per-pixel R-squared compares the reconstructed mixture against the thresholded profile with a 1e-12 stabiliser on the total sum of squares, then paints a fit-quality map.",
    build(svg) {
      const cells = [[0.95,"f-cal"],[0.9,"f-cal"],[0.6,"f-mid"],[0.7,"f-mid"],[0.92,"f-cal"],[0.88,"f-cal"],[0.85,"f-cal"],[0.4,"f-faint"],[0.9,"f-cal"]];
      let m = "";
      cells.forEach((cl, i) => { const x = 132 + (i % 3) * 26, y = 36 + Math.floor(i / 3) * 26; m += `<rect class="sk-live skl-pop ${cl[1]}" x="${x}" y="${y}" width="22" height="22" rx="2" style="opacity:${cl[0]}"/>`; });
      svg.innerHTML = `
        <line class="skl-axis" x1="24" y1="116" x2="108" y2="116"/>
        <path class="skl-draw c-meas" d="M24 114 L48 60 L72 96 L96 110" style="fill:none;opacity:.5"/>
        <path class="skl-draw c-cal" d="M24 114 q24 -58 44 -54 q22 4 22 40 q8 14 18 14" style="fill:none"/>
        <text x="40" y="34" style="fill:#4fd6c4;font-size:9px">R2</text>
        ${m}`;
    },
    anim: pulse,
  },

  diagnostics: {
    tip: "Post-hoc only, the relative margin between the runner-up score and L_K* flags ambiguous pixels, and the peak-to-floor contrast uses the lowest-quartile bins as its noise floor.",
    build(svg) { svg.innerHTML = `
      <text x="30" y="26" style="fill:#c4a3ff;font-size:8px">m_rel</text>
      <rect x="30" y="32" width="80" height="14" class="skl-pop f-faint" style="opacity:.3"/>
      <rect x="30" y="32" width="46" height="14" class="skl-pop f-fin"/>
      <line class="skl-axis" x1="128" y1="120" x2="214" y2="120"/>
      <line class="skl-dash c-faint" x1="128" y1="104" x2="214" y2="104"/>
      <text x="128" y="100" style="fill:#7e8aa0;font-size:7px">floor (Q1)</text>
      <path class="skl-draw c-meas" d="M128 118 L156 50 L172 112 L200 116 L214 118" style="fill:none"/>
      <line class="skl-draw c-cal" x1="156" y1="104" x2="156" y2="50" style="stroke-width:1.6"/>
      <text x="160" y="74" style="fill:#4fd6c4;font-size:7px">C_dB</text>`; },
    anim: null,
  },

  splitgeom: {
    tip: "The azimuth range splits 70/15/15 into contiguous train, validation and test bands that share the full range extent with no overlap.",
    build(svg) { svg.innerHTML = `
      <rect x="40" y="22" width="160" height="74" class="skl-pop f-mid" style="opacity:.85"/>
      <rect x="40" y="96" width="160" height="16" class="skl-pop f-cal" style="opacity:.85"/>
      <rect x="40" y="112" width="160" height="16" class="skl-pop f-fin" style="opacity:.85"/>
      <text x="120" y="62" text-anchor="middle" style="fill:#0b1014;font-size:9px">train 70%</text>
      <text x="120" y="107" text-anchor="middle" style="fill:#0b1014;font-size:7px">val 15%</text>
      <text x="120" y="123" text-anchor="middle" style="fill:#0b1014;font-size:7px">test 15%</text>`; },
    anim: null,
  },

  localslice: {
    tip: "Subtracting the global-crop origin az0 converts each band's absolute azimuth bounds into zero-based slices into the memory-mapped arrays.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="50" x2="214" y2="50"/>
      <text x="30" y="40" style="fill:#7e8aa0;font-size:7px">absolute</text>
      <rect x="92" y="42" width="78" height="16" class="skl-pop f-mid"/>
      <line class="skl-axis" x1="30" y1="104" x2="214" y2="104"/>
      <text x="30" y="124" style="fill:#7e8aa0;font-size:7px">local (0-based)</text>
      <rect x="30" y="96" width="78" height="16" class="skl-pop f-cal"/>
      <path class="skl-draw c-cal" d="M131 60 L60 94 M66 90 L60 94 L66 98" style="fill:none"/>
      <text x="138" y="80" style="fill:#4fd6c4;font-size:8px">- az0</text>`; },
    anim: null,
  },

  secselect: {
    tip: "Flight-qualified labels in L_req map to positional indices, e.g. {3,5,7,25}, gathered identically from both the secondary SLCs and the interferograms.",
    build(svg) {
      let row = "";
      for (let i = 0; i < 7; i++) { const sel = (i === 3 || i === 5 || i === 6); row += `<rect class="skl-pop ${sel ? "f-cal" : "f-faint"}" x="${28 + i * 24}" y="32" width="20" height="20" style="opacity:${sel ? 1 : 0.5}"/>`; if (sel) row += `<rect class="sk-live" x="${28 + i * 24}" y="32" width="20" height="20" style="fill:none;stroke:#4fd6c4;stroke-width:2.5"/>`; }
      svg.innerHTML = `
        ${row}
        <text x="120" y="80" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">pi = {3, 5, 6}</text>
        <rect x="100" y="96" width="20" height="20" class="skl-pop f-cal"/>
        <rect x="124" y="96" width="20" height="20" class="skl-pop f-cal"/>
        <rect x="148" y="96" width="20" height="20" class="skl-pop f-cal"/>`;
    },
    anim: pulse,
  },

  stack: {
    tip: "Primary lands at slot 0, the Ns secondaries fill X[1:1+Ns], and the Ni interferograms fill X[1+Ns:], all written by pass-index into one pre-allocated complex buffer.",
    build(svg) { svg.innerHTML = `
      <rect x="38" y="30" width="44" height="18" class="skl-pop f-meas"/>
      <text x="60" y="43" text-anchor="middle" style="fill:#0b1014;font-size:8px">s0</text>
      <rect x="38" y="52" width="44" height="18" class="skl-pop f-mid"/>
      <rect x="38" y="74" width="44" height="18" class="skl-pop f-mid"/>
      <rect x="38" y="96" width="44" height="18" class="skl-pop f-cal"/>
      <rect x="38" y="118" width="44" height="18" class="skl-pop f-cal"/>
      <text x="100" y="86" style="fill:#7e8aa0;font-size:12px">&#8594;</text>
      <rect x="134" y="30" width="72" height="106" rx="3" class="skl-draw c-fin" style="fill:rgba(196,163,255,0.08)"/>
      <text x="170" y="92" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">X buffer</text>`; },
    anim: null,
  },

  patchgrid: {
    tip: "A strided P-by-P window tiles the region; ceil((Az-P)/s)+1 rows by ceil((Rg-P)/s)+1 columns guarantees the last patch still covers the far border.",
    build(svg) {
      let g = "";
      for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) g += `<rect x="${40 + c * 41}" y="${30 + r * 32}" width="40" height="30" class="skl-pop f-mid" style="opacity:.16;stroke:#f5b971;stroke-width:.8"/>`;
      svg.innerHTML = `${g}<rect class="sk-live" x="81" y="62" width="40" height="30" style="fill:none;stroke:#4fd6c4;stroke-width:2.2"/>`;
    },
    anim: pulse,
  },

  padgeom: {
    tip: "The azimuth deficit pv splits as floor(pv/2) on top and pv-floor(pv/2) on the bottom, so an odd deficit puts the extra pixel at the bottom edge.",
    build(svg) { svg.innerHTML = `
      <rect x="78" y="44" width="84" height="62" class="skl-pop f-mid" style="opacity:.8"/>
      <text x="120" y="80" text-anchor="middle" style="fill:#0b1014;font-size:8px">region</text>
      <rect x="78" y="28" width="84" height="14" class="skl-pop f-cal" style="opacity:.5"/>
      <text x="170" y="38" style="fill:#4fd6c4;font-size:7px">floor(pv/2)</text>
      <rect x="78" y="106" width="84" height="18" class="skl-pop f-cal" style="opacity:.5"/>
      <text x="170" y="120" style="fill:#4fd6c4;font-size:7px">pv-floor(pv/2)</text>`; },
    anim: null,
  },

  extract: {
    tip: "The clipped read window is deep-copied so it never aliases the mmap, then reflect-padded in one pass by the same routine that serves the stack, the parameters and the DEM.",
    build(svg) { svg.innerHTML = `
      <rect x="28" y="32" width="74" height="74" class="skl-draw c-faint" style="fill:none"/>
      <rect x="58" y="62" width="44" height="44" class="skl-pop f-meas" style="opacity:.75"/>
      <text x="110" y="78" style="fill:#7e8aa0;font-size:12px">&#8594;</text>
      <rect x="138" y="42" width="64" height="64" class="skl-pop f-cal" style="opacity:.3"/>
      <rect x="148" y="52" width="44" height="44" class="skl-pop f-mid"/>
      <text x="170" y="124" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">pad(copy)</text>`; },
    anim: null,
  },

  represent: {
    tip: "By default an SLC pass keeps its magnitude |p| and an interferogram keeps its phase angle, while magnitude-normalised channels divide by m = max(|p|, 1) to guard zero magnitude.",
    build(svg) { svg.innerHTML = `
      <circle cx="62" cy="64" r="30" class="skl-axis" style="fill:none"/>
      <line class="skl-axis" x1="32" y1="64" x2="92" y2="64"/>
      <line class="skl-axis" x1="62" y1="34" x2="62" y2="94"/>
      <line class="skl-draw c-meas" x1="62" y1="64" x2="86" y2="44" style="stroke-width:2"/>
      <path class="skl-draw c-mid" d="M82 64 A20 20 0 0 0 77 50" style="fill:none"/>
      <text x="62" y="108" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">complex p</text>
      <rect x="140" y="40" width="44" height="14" class="skl-pop f-cal"/>
      <text x="190" y="51" style="fill:#4fd6c4;font-size:7px">|p|</text>
      <rect x="140" y="62" width="30" height="14" class="skl-pop f-mid"/>
      <text x="176" y="73" style="fill:#f5b971;font-size:7px">ang</text>`; },
    anim: null,
  },

  assemble_in: {
    tip: "Each source's real channels are concatenated along the channel axis in fixed order: primary, then secondaries, then interferograms, with the optional DEM channel last.",
    build(svg) { svg.innerHTML = `
      <rect x="28" y="44" width="22" height="40" class="skl-pop f-meas"/>
      <rect x="54" y="44" width="22" height="40" class="skl-pop f-mid"/>
      <rect x="80" y="44" width="22" height="40" class="skl-pop f-mid"/>
      <rect x="106" y="44" width="22" height="40" class="skl-pop f-cal"/>
      <rect x="132" y="44" width="22" height="40" class="skl-pop f-faint"/>
      <text x="91" y="100" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">r0 | rS | rI | D</text>
      <rect x="40" y="110" width="130" height="22" rx="2" class="skl-draw c-cal" style="fill:rgba(79,214,196,0.12)"/>
      <text x="105" y="125" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">x : C_in</text>`; },
    anim: null,
  },

  target: {
    tip: "Channels 3g+r are gathered from the interleaved (a, mu, sigma) ground-truth layout; enabling all three roles keeps every one of the n_g*3 parameter channels.",
    build(svg) {
      const cls = ["f-meas", "f-mid", "f-cal"], roles = ["a", "mu", "s"];
      let row = "", out = "";
      for (let i = 0; i < 9; i++) { const x = 30 + i * 20, r = i % 3; row += `<rect class="skl-pop ${cls[r]}" x="${x}" y="40" width="16" height="22"/><text x="${x + 8}" y="55" text-anchor="middle" style="fill:#0b1014;font-size:7px">${roles[r]}</text>`; out += `<rect class="sk-live skl-pop f-cal" x="${x}" y="92" width="16" height="22"/>`; }
      svg.innerHTML = `<text x="120" y="30" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">interleaved theta</text>${row}<text x="120" y="80" text-anchor="middle" style="fill:#7e8aa0;font-size:12px">&#8595;</text>${out}`;
    },
    anim: pulse,
  },

  augment_geo: {
    tip: "On the train split only, the sampled flip-or-rot90 transform is applied identically to input x and target y, so a horizontal flip mirrors both and keeps every pixel aligned.",
    build(svg) { svg.innerHTML = `
      <rect x="42" y="36" width="52" height="52" class="skl-draw c-meas" style="fill:none"/>
      <polygon points="50,80 50,44 74,44" style="fill:#6ea8ff;opacity:.7"/>
      <text x="68" y="100" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">x</text>
      <text x="120" y="66" text-anchor="middle" style="fill:#f5b971;font-size:8px">flip</text>
      <rect x="146" y="36" width="52" height="52" class="skl-draw c-cal" style="fill:none"/>
      <polygon points="190,80 190,44 166,44" style="fill:#4fd6c4;opacity:.7"/>
      <text x="172" y="100" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">y</text>`; },
    anim: null,
  },

  slotkeys: {
    tip: "The same strided layout that built the tensor labels each channel by family/slot, so pass/mag, ifg/phase and dem/elev each pick up their own normalisation strategy.",
    build(svg) {
      const L = [["pass/mag", "f-meas", "#6ea8ff"], ["pass/mag", "f-meas", "#6ea8ff"], ["ifg/phase", "f-mid", "#f5b971"], ["ifg/phase", "f-mid", "#f5b971"], ["dem/elev", "f-faint", "#7e8aa0"]];
      let s = "";
      L.forEach((l, i) => { const x = 34 + i * 36; s += `<rect class="skl-pop ${l[1]}" x="${x}" y="34" width="28" height="28"/><line x1="${x + 14}" y1="64" x2="${x + 14}" y2="92" style="stroke:${l[2]};stroke-width:1.3"/><text x="${x + 14}" y="106" text-anchor="middle" transform="rotate(38 ${x + 14} 106)" style="fill:${l[2]};font-size:6px">${l[0]}</text>`; });
      svg.innerHTML = s;
    },
    anim: null,
  },

  fitstats: {
    tip: "Fitted on the train split only in float64, each slot's z-score uses mean and std of f(x), with an optional log1p compression applied before fitting and the scale floored at 1e-8.",
    build(svg) {
      const hs = [14, 26, 40, 52, 46, 32, 20, 10];
      let b = "";
      hs.forEach((h, i) => { const x = 60 + i * 14; b += `<rect class="skl-pop f-meas" x="${x}" y="${112 - h}" width="12" height="${h}" style="opacity:.8"/>`; });
      svg.innerHTML = `<line class="skl-axis" x1="40" y1="112" x2="200" y2="112"/>${b}<line class="skl-dash c-cal" x1="116" y1="40" x2="116" y2="112"/><text x="116" y="34" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">mu_c</text>`;
    },
    anim: null,
  },

  normalise: {
    tip: "Subtracting mu_c and dividing by s_c shifts each slot's distribution to zero mean and unit scale, applied identically to every split to feed the network dimensionless tensors.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="104" x2="214" y2="104"/>
      <line class="skl-dash c-faint" x1="120" y1="34" x2="120" y2="110"/>
      <text x="120" y="122" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">0</text>
      <path class="skl-draw c-faint" d="M156 104 C166 60 198 60 208 104" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-cal" d="M92 104 C106 48 134 48 148 104" style="fill:none"/>
      <path class="skl-draw c-mid" d="M168 104 L150 104 M156 100 L150 104 L156 108" style="fill:none"/>`; },
    anim: null,
  },

  noise: {
    tip: "On the train split only and with probability p_N, Gaussian noise of std 0.01 is added to the already-normalised input, jittering x-hat while the target stays untouched.",
    build(svg) { svg.innerHTML = `
      <path class="skl-draw c-faint" d="M30 80 Q70 42 110 80 T190 80" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-fin" d="M30 82 Q50 60 70 64 Q90 36 110 78 Q130 92 150 70 Q170 50 190 76" style="fill:none"/>
      <text x="120" y="116" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">x-hat + N(0, 0.01^2)</text>`; },
    anim: null,
  },

  denorm: {
    tip: "The inverse multiplies by s_c and adds mu_c, and for log1p slots takes expm1 of the result with the exponent argument clamped at 80 to prevent float32 overflow.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="104" x2="214" y2="104"/>
      <line class="skl-dash c-mid" x1="30" y1="48" x2="214" y2="48"/>
      <text x="208" y="44" text-anchor="end" style="fill:#f5b971;font-size:7px">cap 80</text>
      <path class="skl-draw c-cal" d="M40 104 C72 50 150 48 200 48" style="fill:none"/>
      <text x="120" y="124" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">expm1(min(x s + mu, 80))</text>`; },
    anim: null,
  },

  forward: {
    tip: "One autocast forward pass turns the normalised input patch into 3K interleaved Gaussian channels of amplitude, mean and sigma per pixel.",
    build(svg) { svg.innerHTML = `
      <rect x="24" y="56" width="20" height="40" rx="2" class="skl-pop f-meas"/>
      <path class="skl-draw c-faint" d="M56 48 L100 64 L100 96 L56 112 Z" style="fill:none"/>
      <path class="skl-draw c-faint" d="M108 64 L140 74 L140 94 L108 104 Z" style="fill:none"/>
      <path class="skl-draw c-faint" d="M148 74 L168 82 L168 100 L148 108 Z" style="fill:none"/>
      <rect x="178" y="46" width="34" height="9" rx="2" class="skl-pop f-cal"/>
      <rect x="178" y="58" width="34" height="9" rx="2" class="skl-pop f-cal" style="opacity:.7"/>
      <rect x="178" y="70" width="34" height="9" rx="2" class="skl-pop f-cal" style="opacity:.5"/>
      <text x="195" y="96" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">a mu s</text>`; },
    anim: null,
  },

  tdenorm: {
    tip: "expm1 inverts the log1p amplitude and sigma channels, with the exponent argument clamped at 80 so an early blow-up cannot become a NaN.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="214" y2="118"/>
      <line class="skl-axis" x1="30" y1="22" x2="30" y2="118"/>
      <line class="skl-dash c-mid" x1="30" y1="40" x2="214" y2="40"/>
      <text x="206" y="36" text-anchor="end" style="fill:#f5b971;font-size:7px">cap 80</text>
      <path class="skl-draw c-cal" d="M30 116 Q120 114 166 40 L214 40" style="fill:none"/>`; },
    anim: null,
  },

  clamp: {
    tip: "Out-of-bounds amplitude and sigma are clipped to grid limits but keep a 0.01 leaky straight-through slope so the heads still pass a small gradient.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="214" y2="118"/>
      <line class="skl-dash c-faint" x1="62" y1="20" x2="62" y2="120"/>
      <line class="skl-dash c-faint" x1="182" y1="20" x2="182" y2="120"/>
      <path class="skl-dash c-faint" d="M30 124 L214 18" style="opacity:.3"/>
      <path class="skl-draw c-cal" d="M30 108 L62 96 L182 34 L214 32" style="fill:none"/>
      <text x="48" y="132" style="fill:#7e8aa0;font-size:7px">min</text>
      <text x="170" y="132" style="fill:#7e8aa0;font-size:7px">max</text>`; },
    anim: null,
  },

  renorm: {
    tip: "log1p minus the offset over the scale maps the clamped physical predictions back into the same normalised units as the labels.",
    build(svg) { svg.innerHTML = `
      <rect x="26" y="50" width="44" height="50" rx="3" class="skl-pop f-cal"/>
      <text x="48" y="79" text-anchor="middle" style="fill:#0b1014;font-size:9px">12.4</text>
      <text x="48" y="116" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">phys</text>
      <path class="skl-draw c-mid" d="M76 75 L162 75 M154 69 L162 75 L154 81" style="fill:none"/>
      <text x="119" y="68" text-anchor="middle" style="fill:#f5b971;font-size:7px">(log1p-l)/s</text>
      <rect x="170" y="50" width="44" height="50" rx="3" class="skl-pop f-meas"/>
      <text x="192" y="79" text-anchor="middle" style="fill:#0b1014;font-size:9px">0.74</text>
      <text x="192" y="116" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">norm</text>`; },
    anim: null,
  },

  reconstruct: {
    tip: "Predicted and GT parameters each sum K Gaussian bumps on the elevation axis into a curve; the GT curve is built once under no_grad.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="118" x2="216" y2="118"/>
      <path class="skl-draw c-faint" d="M40 118 Q70 70 100 118" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-faint" d="M96 118 Q128 48 160 118" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-faint" d="M150 118 Q176 82 202 118" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-cal" d="M40 118 Q70 70 100 100 Q128 48 160 92 Q176 82 202 118" style="fill:none"/>
      <path class="skl-dash c-meas" d="M40 118 Q72 76 102 98 Q130 54 162 90 Q178 86 202 118" style="fill:none;opacity:.6"/>`; },
    anim: null,
  },

  residual: {
    tip: "The single elementwise difference y-hat minus y becomes the residual bars shared by the MSE, L1, Huber and Charbonnier terms.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="80" x2="216" y2="80"/>
      <path class="skl-draw c-cal" d="M30 70 Q60 30 90 56 Q120 18 150 50 Q180 44 214 70" style="fill:none"/>
      <path class="skl-dash c-meas" d="M30 74 Q60 40 90 60 Q120 30 150 56 Q180 52 214 74" style="fill:none"/>
      <line class="sk-live skl-pop" x1="60" y1="48" x2="60" y2="60" style="stroke:#f5b971;stroke-width:3;opacity:1"/>
      <line class="sk-live skl-pop" x1="120" y1="22" x2="120" y2="32" style="stroke:#f5b971;stroke-width:3;opacity:1"/>
      <line class="sk-live skl-pop" x1="150" y1="50" x2="150" y2="58" style="stroke:#f5b971;stroke-width:3;opacity:1"/>`; },
    anim: pulse,
  },

  curvepoint: {
    tip: "Four pointwise reductions of the shared residual: MSE squares it, L1 takes magnitude, Huber bends at delta and Charbonnier smooths the L1 with epsilon.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="120" y1="118" x2="120" y2="22"/>
      <line class="skl-axis" x1="28" y1="100" x2="214" y2="100"/>
      <path class="skl-draw c-cal" d="M40 30 Q120 134 200 30" style="fill:none"/>
      <path class="skl-dash c-mid" d="M40 38 L120 100 L200 38" style="fill:none"/>
      <path class="skl-draw c-faint" d="M40 40 Q86 54 100 86 L120 100 L140 86 Q154 54 200 40" style="fill:none"/>
      <text x="150" y="40" style="fill:#4fd6c4;font-size:7px">MSE</text>
      <text x="44" y="44" style="fill:#f5b971;font-size:7px">L1</text>`; },
    anim: null,
  },

  curveshape: {
    tip: "Three shape terms ignore magnitude: cosine angle over valid pixels, windowed spectral coherence, and per-slice SSIM on jointly normalised images.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="62" y1="116" x2="62" y2="32"/>
      <line class="skl-axis" x1="40" y1="96" x2="86" y2="96"/>
      <line class="skl-draw c-cal" x1="62" y1="96" x2="84" y2="44"/>
      <line class="skl-draw c-meas" x1="62" y1="96" x2="78" y2="58"/>
      <text x="50" y="40" style="fill:#7e8aa0;font-size:7px">cos</text>
      <rect x="108" y="40" width="40" height="40" class="skl-pop f-faint" style="opacity:.2"/>
      <path class="skl-draw c-cal" d="M110 72 Q118 42 128 58 Q138 44 146 66" style="fill:none"/>
      <text x="110" y="92" style="fill:#7e8aa0;font-size:7px">coh</text>
      <rect x="168" y="40" width="44" height="40" class="skl-draw c-cal" style="fill:rgba(79,214,196,0.1)"/>
      <text x="172" y="92" style="fill:#7e8aa0;font-size:7px">SSIM</text>`; },
    anim: null,
  },

  physgeom: {
    tip: "The vertical wavenumber kz scales the master-relative perpendicular baseline by the monostatic 4-pi-over-lambda-r0 factor to build the steering phasors.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="120" x2="40" y2="24"/>
      <line class="skl-axis" x1="40" y1="120" x2="210" y2="120"/>
      <circle cx="40" cy="40" r="3" class="skl-pop f-faint"/>
      <line class="skl-dash c-mid" x1="40" y1="40" x2="120" y2="64"/>
      <circle cx="120" cy="64" r="3" class="skl-pop f-meas"/>
      <text x="78" y="46" style="fill:#f5b971;font-size:7px">b_perp</text>
      <circle cx="166" cy="98" r="18" class="skl-axis" style="fill:none;opacity:.3"/>
      <line class="skl-draw c-cal" x1="166" y1="98" x2="184" y2="98"/>
      <line class="skl-draw c-cal" x1="166" y1="98" x2="178" y2="84"/>
      <line class="skl-draw c-cal" x1="166" y1="98" x2="166" y2="80"/>
      <line class="skl-draw c-cal" x1="166" y1="98" x2="154" y2="84"/>
      <text x="150" y="128" style="fill:#4fd6c4;font-size:7px">exp(j kz xi)</text>`; },
    anim: null,
  },

  physmoments: {
    tip: "Ratio terms compare relative integrated power plus the mass, centroid and spread moments, reduced only over GT-strong pixels.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="216" y2="118"/>
      <path class="skl-draw c-cal" d="M40 118 Q90 40 124 70 Q150 92 200 118" style="fill:rgba(79,214,196,0.12)"/>
      <line class="skl-dash c-mid" x1="112" y1="118" x2="112" y2="50"/>
      <circle cx="112" cy="50" r="3.5" class="skl-pop f-mid"/>
      <text x="116" y="48" style="fill:#f5b971;font-size:7px">z-bar</text>
      <line class="skl-dash c-faint" x1="84" y1="100" x2="140" y2="100"/>
      <text x="92" y="114" style="fill:#7e8aa0;font-size:7px">sigma_z</text>`; },
    anim: null,
  },

  physcov: {
    tip: "Coherence compares the two normalised characteristic functions while covariance matching transforms only the profile difference R[P-T] thanks to R's linearity.",
    build(svg) {
      const m = grid(3, (r, c) => { const x = 132 + c * 26, y = 36 + r * 26; const cl = r === c ? "sk-live skl-pop f-mid" : "skl-pop f-faint"; const op = r === c ? 1 : 0.3; return `<rect class="${cl}" x="${x}" y="${y}" width="22" height="22" rx="2" style="opacity:${op}"/>`; });
      svg.innerHTML = `
        <circle cx="70" cy="62" r="26" class="skl-axis" style="fill:none;opacity:.4"/>
        <line class="skl-draw c-cal" x1="70" y1="62" x2="92" y2="48"/>
        <line class="skl-draw c-meas" x1="70" y1="62" x2="88" y2="44"/>
        <text x="50" y="102" style="fill:#7e8aa0;font-size:7px">gamma_P vs T</text>
        ${m}
        <text x="171" y="128" text-anchor="middle" style="fill:#f5b971;font-size:7px">R[P-T]</text>`;
    },
    anim: pulse,
  },

  physcapon: {
    tip: "Capon synthesises R[P], adds signal-adaptive diagonal loading, then solves once per pixel to form the spectrum compared mass-normalised.",
    build(svg) { svg.innerHTML = `
      <rect x="36" y="40" width="48" height="48" rx="2" class="skl-pop f-mid" style="opacity:.5"/>
      <line class="skl-draw c-fin" x1="36" y1="40" x2="84" y2="88" style="stroke-width:3"/>
      <text x="40" y="104" style="fill:#7e8aa0;font-size:7px">R + eI</text>
      <path class="skl-draw c-mid" d="M92 64 L132 64 M124 59 L132 64 L124 69" style="fill:none"/>
      <line class="skl-axis" x1="150" y1="100" x2="214" y2="100"/>
      <path class="skl-draw c-cal" d="M152 100 Q178 40 184 78 Q192 52 212 100" style="fill:none"/>`; },
    anim: null,
  },

  paramterms: {
    tip: "GT components are mu-sorted and empty slots mask mu and sigma to zero weight, so Param-L1/Huber act in normalised space while TV penalises spatial irregularity.",
    build(svg) { svg.innerHTML = `
      <rect x="34" y="42" width="22" height="20" rx="2" class="skl-pop f-cal"/>
      <rect x="34" y="66" width="22" height="20" rx="2" class="skl-pop f-cal"/>
      <rect x="34" y="90" width="22" height="20" rx="2" class="skl-dash c-faint" style="fill:none"/>
      <text x="45" y="124" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">mask</text>
      <text x="86" y="78" style="fill:#7e8aa0;font-size:12px">&#8594;</text>
      <rect x="150" y="40" width="62" height="62" class="skl-axis" style="fill:none;opacity:.3"/>
      <path class="skl-draw c-cal" d="M158 92 L170 60 L182 80 L194 52 L206 72" style="fill:none"/>
      <text x="170" y="116" style="fill:#4fd6c4;font-size:7px">TV</text>`; },
    anim: null,
  },

  composite: {
    tip: "Each term's effective weight is the user weight times a fixed empirical normaliser; the weighted terms sum and divide by total weight into one loss.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="140" y2="118"/>
      <rect x="36" y="92" width="14" height="26" class="skl-pop f-cal"/>
      <rect x="56" y="74" width="14" height="44" class="skl-pop f-cal"/>
      <rect x="76" y="100" width="14" height="18" class="skl-pop f-cal"/>
      <rect x="96" y="58" width="14" height="60" class="skl-pop f-cal"/>
      <rect x="116" y="86" width="14" height="32" class="skl-pop f-cal"/>
      <path class="skl-draw c-mid" d="M146 75 L176 75 M168 70 L176 75 L168 80" style="fill:none"/>
      <text x="160" y="68" text-anchor="middle" style="fill:#f5b971;font-size:7px">/ sum</text>
      <rect class="sk-live skl-pop f-fin" x="192" y="48" width="24" height="70"/>
      <text x="204" y="132" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">L</text>`; },
    anim: pulse,
  },

  gradclip: {
    tip: "When the global gradient norm exceeds the threshold tau, every gradient is rescaled by tau over the norm so the clipped vector lands exactly on the limit.",
    build(svg) { svg.innerHTML = `
      <circle cx="76" cy="80" r="44" class="skl-dash c-faint" style="fill:none;opacity:.5"/>
      <text x="40" y="36" style="fill:#7e8aa0;font-size:7px">tau</text>
      <line class="skl-dash c-faint" x1="76" y1="80" x2="124" y2="34" style="opacity:.5"/>
      <line class="skl-draw c-cal" x1="76" y1="80" x2="107" y2="50"/>
      <circle cx="76" cy="80" r="3" class="skl-pop f-faint"/>
      <circle cx="107" cy="50" r="3.5" class="skl-pop f-cal"/>
      <text x="130" y="80" style="fill:#4fd6c4;font-size:7px">g min(1, tau/|g|)</text>`; },
    anim: null,
  },

  adamw: {
    tip: "Bias-corrected adaptive moments with decoupled weight decay step the weights down a loss surface, driving the training loss lower each epoch.",
    build(svg) { svg.innerHTML = `
      <path class="skl-draw c-faint" d="M30 36 Q120 156 210 36" style="fill:none;opacity:.5"/>
      <circle cx="42" cy="60" r="2.5" class="skl-pop f-mid" style="opacity:.4"/>
      <circle cx="70" cy="92" r="2.5" class="skl-pop f-mid" style="opacity:.4"/>
      <circle cx="98" cy="114" r="2.5" class="skl-pop f-mid" style="opacity:.4"/>
      <circle class="sk-live skl-pop f-fin" cx="120" cy="122" r="5"/>`; },
    anim: pulse,
  },

  schedule: {
    tip: "The effective LR is the base rate times a per-epoch cosine decay times a linear per-step warmup, with the loss curriculum swapping objectives at the swap epoch.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="118" x2="216" y2="118"/>
      <line class="skl-axis" x1="28" y1="22" x2="28" y2="118"/>
      <path class="skl-draw c-mid" d="M28 110 L58 40" style="fill:none"/>
      <path class="skl-draw c-cal" d="M58 40 Q120 44 150 78 Q180 108 210 114" style="fill:none"/>
      <line class="skl-dash c-fin" x1="138" y1="22" x2="138" y2="118"/>
      <text x="142" y="32" style="fill:#c4a3ff;font-size:7px">swap</text>`; },
    anim: null,
  },

  checkpoint: {
    tip: "Validation runs every few epochs; a strict improvement checkpoints the best epoch and early stopping reverts to it after patience evaluations without a new minimum.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="28" y1="118" x2="216" y2="118"/>
      <path class="skl-draw c-cal" d="M30 50 L62 72 L94 60 L120 90 L150 78 L180 96 L210 88" style="fill:none"/>
      <circle class="sk-live skl-pop f-fin" cx="120" cy="90" r="5"/>
      <text x="120" y="108" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">best</text>`; },
    anim: pulse,
  },

  load: {
    tip: "The architecture is rebuilt verbatim from the saved config before the best-epoch theta-star tensor is loaded with no EMA, refusing any input that is not one contiguous region.",
    build(svg) { svg.innerHTML = `
      <rect x="38" y="40" width="64" height="70" rx="4" class="skl-draw c-faint" style="fill:#1b242f"/>
      <rect x="48" y="52" width="44" height="6" rx="2" class="skl-pop f-meas"/>
      <rect x="48" y="64" width="44" height="6" rx="2" class="skl-pop f-meas"/>
      <rect x="48" y="76" width="44" height="6" rx="2" class="skl-pop f-meas"/>
      <rect x="48" y="88" width="44" height="6" rx="2" class="skl-pop f-meas"/>
      <text x="70" y="34" text-anchor="middle" style="fill:#6ea8ff;font-size:9px">theta*</text>
      <path class="skl-draw c-cal" d="M108 75 L150 75 M142 69 L150 75 L142 81" style="fill:none"/>
      <rect x="160" y="48" width="50" height="50" rx="3" class="skl-draw c-cal" style="fill:rgba(79,214,196,0.12)"/>
      <text x="185" y="118" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">1 region</text>`; },
    anim: null,
  },

  predict: {
    tip: "For every patch on the sliding-window grid the frozen model emits the raw normalised z-hat in deterministic raster order, so the cube ends with no holes.",
    build(svg) {
      let g = "";
      for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) g += `<rect x="${44 + c * 34}" y="${40 + r * 28}" width="30" height="24" rx="2" style="fill:#1b242f;stroke:#303d4c"/>`;
      svg.innerHTML = `${g}<rect class="sk-live" x="44" y="40" width="30" height="24" rx="2" style="fill:rgba(110,168,255,0.25);stroke:#6ea8ff;stroke-width:2"/><text x="120" y="138" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">sliding window</text>`;
    },
    anim: pulse,
  },

  idenorm: {
    tip: "Predictions are denormalised then hard-clamped (no leaky slope) with a flat saturating transfer curve, pinning amplitude into [0, a_max].",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="110" x2="210" y2="110"/>
      <line class="skl-axis" x1="40" y1="110" x2="40" y2="28"/>
      <line class="skl-dash c-faint" x1="40" y1="44" x2="210" y2="44"/>
      <text x="34" y="48" text-anchor="end" style="fill:#7e8aa0;font-size:7px">a_max</text>
      <path class="skl-dash c-mid" d="M30 124 L210 18" style="opacity:.3"/>
      <path class="skl-draw c-cal" d="M40 110 L100 110 L155 44 L210 44" style="fill:none"/>`; },
    anim: null,
  },

  align: {
    tip: "GT slots are stably sorted by mu with inactive slots pushed to the end, while the prediction keeps its raw order so the metric scores the network's own arrangement.",
    build(svg) { svg.innerHTML = `
      <text x="60" y="34" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">GT</text>
      <text x="180" y="34" text-anchor="middle" style="fill:#f5b971;font-size:8px">mu-sorted</text>
      <rect x="44" y="44" width="32" height="16" rx="2" class="skl-pop f-meas"/>
      <rect x="44" y="64" width="32" height="16" rx="2" class="skl-pop f-faint"/>
      <rect x="44" y="84" width="32" height="16" rx="2" class="skl-pop f-meas"/>
      <rect x="44" y="104" width="32" height="16" rx="2" class="skl-pop f-meas"/>
      <line class="skl-dash c-faint" x1="78" y1="52" x2="162" y2="52"/>
      <line class="skl-dash c-faint" x1="78" y1="72" x2="162" y2="112"/>
      <line class="skl-dash c-faint" x1="78" y1="92" x2="162" y2="72"/>
      <line class="skl-dash c-faint" x1="78" y1="112" x2="162" y2="92"/>
      <rect x="164" y="44" width="32" height="16" rx="2" class="skl-pop f-mid"/>
      <rect x="164" y="64" width="32" height="16" rx="2" class="skl-pop f-mid"/>
      <rect x="164" y="84" width="32" height="16" rx="2" class="skl-pop f-mid"/>
      <rect x="164" y="104" width="32" height="16" rx="2" class="skl-pop f-faint" style="opacity:.4"/>`; },
    anim: null,
  },

  recon: {
    tip: "Each patch's clamped Gaussians are evaluated on the elevation axis with amplitudes rectified at zero and a 2-sigma-squared-plus-1e-8 kernel denominator, summing into one spectrum.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="34" y1="116" x2="214" y2="116"/>
      <path class="skl-draw c-faint" d="M34 116 C70 116 78 64 96 64 C114 64 122 116 158 116" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-faint" d="M110 116 C140 116 148 80 162 80 C176 80 184 116 214 116" style="fill:none;opacity:.4"/>
      <path class="skl-draw c-mid" d="M34 116 C70 116 78 64 96 64 C108 64 116 92 134 92 C150 92 156 80 162 80 C176 80 184 116 214 116" style="fill:none"/>`; },
    anim: null,
  },

  window: {
    tip: "A separable Hann taper has each 1D axis factor floored at 1e-3 before the outer product, so every covered position carries a strictly positive overlap-add weight.",
    build(svg) {
      let cells = "";
      for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) { const d = Math.hypot(r - 2, c - 2) / 2.83; const op = (0.95 - d * 0.85).toFixed(2); cells += `<rect x="${150 + c * 12}" y="${40 + r * 12}" width="11" height="11" style="fill:#f5b971;opacity:${op}"/>`; }
      svg.innerHTML = `
        <line class="skl-axis" x1="40" y1="118" x2="120" y2="118"/>
        <path class="skl-draw c-mid" d="M40 118 C58 118 64 50 80 50 C96 50 102 118 120 118" style="fill:none"/>
        <text x="80" y="134" text-anchor="middle" style="fill:#f5b971;font-size:7px">Hann</text>
        ${cells}`;
    },
    anim: null,
  },

  ola: {
    tip: "Each windowed patch is scattered additively into the value accumulator A at its grid origin while the bare window is added to the parallel weight buffer W.",
    build(svg) { svg.innerHTML = `
      <rect x="46" y="44" width="64" height="64" class="skl-axis" style="fill:#15302d"/>
      <text x="78" y="36" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">A += p w</text>
      <rect x="46" y="44" width="40" height="40" class="skl-pop f-cal" style="opacity:.5"/>
      <rect x="70" y="68" width="40" height="40" class="skl-pop f-cal" style="opacity:.5"/>
      <rect x="140" y="44" width="64" height="64" class="skl-axis" style="fill:#2a2418"/>
      <text x="172" y="36" text-anchor="middle" style="fill:#f5b971;font-size:8px">W += w</text>
      <rect x="140" y="44" width="40" height="40" class="skl-pop f-mid" style="opacity:.5"/>
      <rect x="164" y="68" width="40" height="40" class="skl-pop f-mid" style="opacity:.5"/>`; },
    anim: null,
  },

  finalise: {
    tip: "The value accumulator is divided elementwise by max(W,1) so uncovered positions divide by one and read zero, then the grid padding is trimmed to the scene extent.",
    build(svg) { svg.innerHTML = `
      <rect x="38" y="52" width="44" height="44" class="skl-pop f-cal" style="opacity:.7"/>
      <text x="60" y="78" text-anchor="middle" style="fill:#0b1014;font-size:11px">A</text>
      <text x="98" y="80" text-anchor="middle" style="fill:#9fb0c0;font-size:16px">&#247;</text>
      <rect x="114" y="52" width="44" height="44" class="skl-pop f-mid" style="opacity:.7"/>
      <text x="136" y="78" text-anchor="middle" style="fill:#0b1014;font-size:9px">max(W,1)</text>
      <text x="174" y="80" text-anchor="middle" style="fill:#9fb0c0;font-size:16px">=</text>
      <rect x="190" y="56" width="34" height="34" class="skl-draw c-fin" style="fill:rgba(196,163,255,0.15)"/>
      <text x="207" y="106" text-anchor="middle" style="fill:#c4a3ff;font-size:8px">cube</text>`; },
    anim: null,
  },

  pixelmaps: {
    tip: "Five per-pixel maps reduce over the N elevation bins at each azimuth-range cell: MSE, MAE, R-squared, cosine similarity, and peak-bin index error.",
    build(svg) {
      const vals = [0.9, 0.7, 0.95, 0.6, 0.8, 0.4, 0.85, 0.5, 0.92, 0.65, 0.75, 0.55];
      let cells = "";
      vals.forEach((v, i) => { const x = 120 + (i % 4) * 24, y = 44 + Math.floor(i / 4) * 24; cells += `<rect class="sk-live" x="${x}" y="${y}" width="22" height="22" style="fill:#4fd6c4;opacity:${0.2 + v * 0.6}"/>`; });
      svg.innerHTML = `
        <rect x="40" y="60" width="40" height="40" class="skl-draw c-faint" style="fill:#1b242f"/>
        <rect x="46" y="54" width="40" height="40" class="skl-draw c-faint" style="fill:#1b242f"/>
        <rect x="52" y="48" width="40" height="40" class="skl-draw c-meas" style="fill:#1d3350"/>
        <path class="skl-draw c-cal" d="M98 74 L112 74 M106 69 L112 74 L106 79" style="fill:none"/>
        ${cells}`;
    },
    anim: pulse,
  },

  globalcurve: {
    tip: "Cube-wide scalars at physical scale: MSE, RMSE, an overall R-squared, and a PSNR whose peak signal is the GT-only dynamic range C_max minus C_min.",
    build(svg) { svg.innerHTML = `
      <rect x="40" y="40" width="160" height="70" rx="5" class="skl-axis" style="fill:#1b242f"/>
      <text x="58" y="64" style="fill:#9fb0c0;font-size:11px">R2</text>
      <text x="184" y="64" text-anchor="end" style="fill:#c4a3ff;font-size:13px;font-weight:600">0.94</text>
      <line x1="56" y1="74" x2="184" y2="74" class="skl-axis"/>
      <text x="58" y="96" style="fill:#9fb0c0;font-size:11px">PSNR</text>
      <text x="184" y="96" text-anchor="end" style="fill:#4fd6c4;font-size:13px;font-weight:600">28.4 dB</text>`; },
    anim: null,
  },

  elevssim: {
    tip: "Per elevation bin the MAE, RMSE, R-squared and a cross-entropy between column-normalised distributions are accumulated, alongside a mean SSIM over the elevation slices.",
    build(svg) {
      const h = [20, 34, 46, 40, 28, 18];
      let b = "";
      h.forEach((v, i) => { b += `<rect class="skl-pop f-cal" x="${44 + i * 16}" y="${108 - v}" width="11" height="${v}" rx="1"/>`; });
      svg.innerHTML = `<line class="skl-axis" x1="40" y1="108" x2="148" y2="108"/>${b}<text x="92" y="124" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">R2 per bin</text><rect x="162" y="44" width="52" height="52" class="skl-axis" style="fill:#1b242f"/><rect x="162" y="62" width="52" height="9" class="skl-pop f-fin"/><text x="188" y="110" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">SSIM</text>`;
    },
    anim: null,
  },

  paramslot: {
    tip: "On active pixels the per-Gaussian mu and sigma MAE/RMSE, placeholder F1 and mu-ordering rate are pooled, with a permutation consensus voted from per-pixel mu-distance assignment.",
    build(svg) { svg.innerHTML = `
      <circle cx="60" cy="58" r="9" class="skl-dash c-meas" style="fill:none"/>
      <circle cx="60" cy="58" r="4" class="skl-pop f-cal"/>
      <text x="60" y="84" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">mu err</text>
      <line class="skl-axis" x1="104" y1="100" x2="196" y2="100"/>
      <rect x="110" y="86" width="14" height="14" class="skl-pop f-faint"/>
      <rect x="132" y="60" width="14" height="40" class="skl-pop f-fin"/>
      <rect x="154" y="90" width="14" height="10" class="skl-pop f-faint"/>
      <rect x="176" y="84" width="14" height="16" class="skl-pop f-faint"/>
      <text x="150" y="50" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">consensus</text>`; },
    anim: null,
  },

  reduced: {
    tip: "For a strict secondary subset a reduced Capon tomogram is re-synthesised, then GT, prediction and reduced cubes are unit-area normalised and the network's per-pixel MSE gain is mapped.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="40" y1="100" x2="150" y2="100"/>
      <path class="skl-draw c-meas" d="M40 100 C66 100 72 48 95 48 C118 48 124 100 150 100" style="fill:none"/>
      <path class="skl-draw c-fin" d="M40 100 C68 100 74 56 95 56 C116 56 122 100 150 100" style="fill:none"/>
      <path class="skl-draw c-mid" d="M40 100 C62 100 70 74 88 70 C108 66 118 100 150 100" style="fill:none"/>
      <text x="40" y="34" style="fill:#6ea8ff;font-size:7px">GT</text>
      <text x="74" y="34" style="fill:#c4a3ff;font-size:7px">pred</text>
      <text x="112" y="34" style="fill:#f5b971;font-size:7px">reduced</text>
      <rect x="166" y="46" width="48" height="48" class="skl-pop f-fin" style="opacity:.6"/>
      <text x="190" y="108" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">delta MSE</text>`; },
    anim: null,
  },

  spacelr: {
    tip: "Encoder, bottleneck, decoder and head learning rates are each drawn from log-uniform on the decade span 1e-5 to 1e-2, with dropout drawn linearly across 0 to 0.5.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <text x="30" y="132" style="fill:#7e8aa0;font-size:7px">1e-5</text>
      <text x="190" y="132" text-anchor="end" style="fill:#7e8aa0;font-size:7px">1e-2</text>
      <circle class="sk-live skl-pop f-mid" cx="58" cy="118" r="4"/>
      <circle class="sk-live skl-pop f-mid" cx="112" cy="118" r="4"/>
      <circle class="sk-live skl-pop f-mid" cx="150" cy="118" r="4"/>
      <circle class="sk-live skl-pop f-mid" cx="192" cy="118" r="4"/>
      <line class="skl-axis" x1="30" y1="44" x2="210" y2="44"/>
      <text x="30" y="38" style="fill:#7e8aa0;font-size:7px">p_drop</text>
      <circle class="sk-live skl-pop f-cal" cx="96" cy="44" r="4"/>`; },
    anim: pulse,
  },

  spacearch: {
    tip: "Five categorical knobs are sampled, with the features list stored as index k in {0,1,2,3} and decoded through the lookup table back to a channel list like [64,128,256,512].",
    build(svg) { svg.innerHTML = `
      <rect x="28" y="26" width="26" height="16" rx="3" class="skl-pop f-faint"/>
      <rect x="58" y="26" width="26" height="16" rx="3" class="skl-pop f-faint"/>
      <rect x="88" y="26" width="26" height="16" rx="3" class="skl-pop f-mid"/>
      <rect x="118" y="26" width="26" height="16" rx="3" class="skl-pop f-faint"/>
      <path class="skl-draw c-mid" d="M101 44 q24 16 58 16" style="fill:none"/>
      <rect x="150" y="50" width="62" height="18" rx="3" class="skl-pop f-cal"/>
      <text x="181" y="63" text-anchor="middle" style="fill:#0b1014;font-size:7px">[64,128,256,512]</text>
      <rect x="28" y="88" width="40" height="14" rx="3" class="skl-pop f-faint"/>
      <rect x="74" y="88" width="30" height="14" rx="3" class="skl-pop f-faint"/>
      <rect x="110" y="88" width="34" height="14" rx="3" class="skl-pop f-faint"/>
      <rect x="150" y="88" width="32" height="14" rx="3" class="skl-pop f-faint"/>`; },
    anim: null,
  },

  merge: {
    tip: "The 9-dim learning block and the 5-dim architecture block are stacked into one 14-dimensional joint space that multivariate TPE samples jointly.",
    build(svg) { svg.innerHTML = `
      <rect x="34" y="42" width="46" height="66" rx="4" class="skl-pop f-mid" style="opacity:.85"/>
      <text x="57" y="34" text-anchor="middle" style="fill:#f5b971;font-size:8px">lr (9)</text>
      <text x="100" y="80" text-anchor="middle" style="fill:#7e8aa0;font-size:14px">+</text>
      <rect x="118" y="42" width="46" height="66" rx="4" class="skl-pop f-cal" style="opacity:.85"/>
      <text x="141" y="34" text-anchor="middle" style="fill:#4fd6c4;font-size:8px">arch (5)</text>
      <text x="178" y="80" text-anchor="middle" style="fill:#7e8aa0;font-size:14px">=</text>
      <rect x="190" y="48" width="28" height="54" rx="4" class="skl-pop f-meas"/>
      <text x="204" y="116" text-anchor="middle" style="fill:#6ea8ff;font-size:8px">d=14</text>`; },
    anim: null,
  },

  tpesplit: {
    tip: "Trials are split at the gamma quantile of their loss into a good set below and a bad set above, each fitted with its own KDE: l(theta) and g(theta).",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <line class="skl-dash c-faint" x1="120" y1="26" x2="120" y2="122"/>
      <text x="110" y="22" style="fill:#7e8aa0;font-size:7px">y_g</text>
      <path class="skl-draw c-cal" d="M34 118 C70 118 86 40 110 40 C116 40 118 118 120 118 Z" style="fill:rgba(79,214,196,.18)"/>
      <path class="skl-draw c-mid" d="M120 118 C122 118 132 70 158 70 C190 70 196 118 206 118 Z" style="fill:rgba(245,185,113,.16)"/>
      <text x="64" y="60" style="fill:#4fd6c4;font-size:7px">l</text>
      <text x="160" y="92" style="fill:#f5b971;font-size:7px">g</text>`; },
    anim: null,
  },

  tpeacq: {
    tip: "TPE proposes the theta maximising the density ratio l(theta) over g(theta), at the argmax marker, except during the first n0 = 8 startup trials which sample uniformly.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <path class="skl-draw c-cal" d="M34 110 C70 110 86 48 108 48 C120 48 124 110 130 110" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-mid" d="M40 110 C90 110 110 80 150 80 C188 80 196 110 206 110" style="fill:none;opacity:.5"/>
      <path class="skl-draw c-fin" d="M34 116 C66 116 80 36 104 30 C124 26 132 116 150 116 C170 116 206 116 206 116" style="fill:none"/>
      <line class="skl-dash c-fin" x1="104" y1="30" x2="104" y2="118"/>
      <circle class="sk-live skl-pop f-fin" cx="104" cy="30" r="5"/>
      <text x="60" y="26" style="fill:#c4a3ff;font-size:7px">l/g</text>`; },
    anim: pulse,
  },

  liar: {
    tip: "Each pending trial across the parallel GPU workers is temporarily handed the worst completed objective max f as a phantom value so the workers do not all crowd the same point.",
    build(svg) { svg.innerHTML = `
      <rect x="96" y="14" width="48" height="18" rx="4" class="skl-pop f-faint"/>
      <text x="120" y="27" text-anchor="middle" style="fill:#cfd8e8;font-size:7px">study</text>
      <rect x="28" y="74" width="36" height="22" rx="4" class="skl-pop f-meas"/>
      <rect x="84" y="74" width="36" height="22" rx="4" class="skl-pop f-meas"/>
      <rect x="140" y="74" width="36" height="22" rx="4" class="skl-pop f-meas"/>
      <line class="skl-axis" x1="120" y1="32" x2="46" y2="74" style="opacity:.4"/>
      <line class="skl-axis" x1="120" y1="32" x2="102" y2="74" style="opacity:.4"/>
      <line class="skl-axis" x1="120" y1="32" x2="158" y2="74" style="opacity:.4"/>
      <circle cx="46" cy="116" r="6" class="skl-pop f-mid"/>
      <circle cx="102" cy="116" r="6" class="skl-pop f-mid"/>
      <circle cx="158" cy="116" r="6" class="skl-pop f-mid"/>
      <text x="120" y="138" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">phantom max f</text>`; },
    anim: null,
  },

  trialsetup: {
    tip: "The base config is deep-copied per trial and overridden with E = 30 epochs, patience = 8, and seed = 42 + trial.number before any training begins.",
    build(svg) { svg.innerHTML = `
      <rect x="30" y="50" width="56" height="50" rx="5" class="skl-pop f-faint"/>
      <text x="58" y="116" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">base cfg</text>
      <path class="skl-draw c-faint" d="M90 75 L128 75 M120 70 L128 75 L120 80" style="fill:none"/>
      <rect x="134" y="46" width="60" height="58" rx="5" class="skl-pop f-mid"/>
      <line class="skl-draw c-cal" x1="142" y1="60" x2="186" y2="60"/>
      <line class="skl-draw c-cal" x1="142" y1="72" x2="186" y2="72"/>
      <line class="skl-draw c-cal" x1="142" y1="84" x2="186" y2="84"/>
      <text x="164" y="118" text-anchor="middle" style="fill:#7e8aa0;font-size:7px">E=30 pat=8</text>`; },
    anim: null,
  },

  trial: {
    tip: "A trial trains a full model on the fixed canonical split for up to 30 epochs and returns f(theta), the minimum validation loss reached along that curve.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="120" x2="210" y2="120"/>
      <line class="skl-axis" x1="30" y1="24" x2="30" y2="120"/>
      <path class="skl-draw c-cal" d="M30 36 C60 70 78 92 104 100 C140 110 170 104 206 102" style="fill:none"/>
      <line class="skl-dash c-fin" x1="30" y1="102" x2="206" y2="102"/>
      <circle class="sk-live skl-pop f-cal" cx="158" cy="102" r="4"/>
      <text x="158" y="96" text-anchor="middle" style="fill:#4fd6c4;font-size:7px">min L</text>`; },
    anim: pulse,
  },

  prune: {
    tip: "At step t a trial whose validation loss rises above the running median m is pruned, but only once n_done and t both clear the 8-step warmup gate.",
    build(svg) { svg.innerHTML = `
      <line class="skl-axis" x1="30" y1="120" x2="210" y2="120"/>
      <line class="skl-dash c-mid" x1="30" y1="72" x2="210" y2="72"/>
      <text x="34" y="66" style="fill:#f5b971;font-size:7px">median m</text>
      <line class="skl-dash c-faint" x1="92" y1="24" x2="92" y2="120"/>
      <text x="68" y="20" style="fill:#7e8aa0;font-size:7px">t>=8</text>
      <path class="skl-draw c-cal" d="M30 40 C50 56 70 64 92 66 C104 67 110 66 116 64" style="fill:none"/>
      <line class="skl-draw c-fin" x1="118" y1="58" x2="130" y2="70" style="stroke-width:2"/>
      <line class="skl-draw c-fin" x1="130" y1="58" x2="118" y2="70" style="stroke-width:2"/>`; },
    anim: null,
  },

  best: {
    tip: "The study dispatches only the remaining trials in GPU chunks, and after each completion rewrites theta-star as the argmin of f, decoding the features index back to its channel list.",
    build(svg) { svg.innerHTML = `
      <rect x="28" y="26" width="14" height="18" rx="2" class="skl-pop f-faint"/>
      <rect x="46" y="26" width="14" height="18" rx="2" class="skl-pop f-faint"/>
      <rect x="64" y="26" width="14" height="18" rx="2" class="skl-pop f-mid"/>
      <rect x="82" y="26" width="14" height="18" rx="2" class="skl-pop f-mid"/>
      <text x="120" y="40" style="fill:#7e8aa0;font-size:7px">n_rem of 100</text>
      <line class="skl-axis" x1="30" y1="118" x2="210" y2="118"/>
      <circle cx="52" cy="86" r="3" class="skl-pop f-cal" style="opacity:.5"/>
      <circle cx="86" cy="70" r="3" class="skl-pop f-cal" style="opacity:.5"/>
      <circle cx="118" cy="98" r="3" class="skl-pop f-cal" style="opacity:.5"/>
      <circle class="sk-live skl-pop f-fin" cx="150" cy="110" r="5"/>
      <circle cx="184" cy="82" r="3" class="skl-pop f-cal" style="opacity:.5"/>
      <text x="150" y="132" text-anchor="middle" style="fill:#c4a3ff;font-size:7px">argmin f</text>`; },
    anim: pulse,
  },

};
