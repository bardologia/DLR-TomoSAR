# Build

Compile the full deck with tectonic:

    tectonic full_project_story.tex

Produces `full_project_story.pdf` (146 rendered pages as of 2026-07-30 — the K=2/K=5 table grouping replaced twelve per-K frames with six merged ones, see design log; earlier chain: 109 content frames + 8 act dividers + 1 sub-divider + 3 backup, 16:9; 124 rendered pages incl. title and overlays — re-measured 2026-07-16 after the head-2x2 + presence-lever five-seed refreshes (each got the standing three-table treatment; the 12-run lever grid became the three presence frames, see the design log). 120 predated that (the input-ablation refresh recut the two amplitude-only frames into four). 118 predated that (the augmentation five-seed refresh added its by-scatterer-count + GT-statistics pair 20b2/20b3). 116 predated that (the normalization five-seed refresh added the same pair as 12c/12d). 114 predated that (the price-of-reach + operator-split pair went into the context section). 112 predated that (the exact ladder argument + its rung table frame), 109 predated that (the context-section theory chain had just been deleted — six frames + their three backup frames; it rested on a degenerate homogeneity premise). 109 also predated that chain going in; 107 predated that (manipulation-per-line expansion, eight frames became thirteen), 102 predated that (raw-derivation expansion, eight frames), 100 predated that (argument-line recut, six frames), 97 predated that (three-frame L1-vs-MSE section), 94 predated that (step-8 mechanism frame), 93 predated that (summed-up frame), 92 predated that (reduced-stack detection frame), 91 predated that, 90 predated the step-8/9 split, 89 predated the K-scatterer frame, 88 predated the convergence/gradient continuation frame, 87 predated the joint-error-landscape frame).
Pure Beamer + TikZ, no external fonts, so it builds offline once the CTAN cache is warm.

# Structure (since 2026-07-12)

The slides live in shared section files; the full deck and the sub-presentations both read them, so an edit to a section propagates to both.

- `preamble.tex` — theme, colours, packages, tikz styles, macros (`\act`, `\subact`, `\pending`, table tints). Input by the full deck and every sub-deck; never duplicated.
- `sections/NN_theme.tex` — the frames, cut at *content-theme* boundaries (finer than the acts): 00_front (title/outline, full-only), 01_overview, 02_signal_model, 03_dataset_labels, 04_representation, 05_stability, 06_set_prediction, 07_augmentation, 08_context_information, 09_error_theory (the error-theory chain, seven frames ending at the recall paradox — de-stepped 2026-07-20; the mechanism/summed-up/gate tail and the reduced-stack verdict frame were deleted the same day), 10_benchmark_board, 11_loss_design (composition ambiguity), 11b_l1_vs_mse (why param-L1 beats param-MSE: label laws, mean/median optima, tail share + the Huber/gate checks + the limits frame, nine frames — de-stepped and trimmed 2026-07-20), 12_winner, 12b_dual (the dual-trunk block: idea + routing grid + K=2/K=5 routing results + arm-ratio intro, the K=2 trio and the K=5 quartet, closing on four qualitative 90-10 cube-slice frames; moved here from 08_context_information on 2026-07-26 because the dual experiments postdate the benchmark), 13_physics_losses, 14_unrolled, 15_jepa, 16_close, 17_backup. **Edit slides here**, never in the wrapper files.
- `full_project_story.tex` — documentclass + `\input{preamble}` + title block, then the `\act`/`\subact` divider lines inline and the 19 section inputs in the original narrative order. Nothing else.
- `sub_NN_theme.tex` — ten standalone thematic sub-presentations, each = own title page + `\input` of its section file(s). Themes regroup content across act boundaries: 01_problem (overview + signal model), 02_dataset_labels, 03_representation, 04_stability (stability + augmentation), 05_set_prediction (small-K through the 12-run imbalance grid), 06_context_information, 07_loss_theory (error theory + loss design + L1 vs MSE), 08_benchmark (board + winner + dual-trunk block), 09_physics (loss terms + unrolled solver), 10_jepa. Build the same way, e.g. `tectonic sub_07_loss_theory.tex`.

All builds run inside this directory (images resolve via `../../results/...`).

# Design log

- K=2/K=5 TABLES GROUPED SIDE BY SIDE 2026-07-30 (user: "usually the results
  are in tables, it shows for K=2 and then everything again for K=5, but I
  think the best way is to group them. I think there is enough horizontal
  space for that"). The two four-arm result blocks in 06_set_prediction now
  carry ONE table per topic with $K{=}2$ and $K{=}5$ as top-level column
  groups: label + 8 mean+-std columns, tabcolsep 2.5pt, 2pt intra-group / 8pt
  inter-group gaps, caption parbox widened to 0.94\textwidth -- fits the
  140 mm text width with zero overfull hboxes at the deck-standard scriptsize.
  Head 2x2 block: 15b+15e -> "Set prediction, controlled --- the head leads at
  both K; the matching matters only at K=5" (seedtag mgA, stretch 1.05);
  15c(count rows)+15f -> "The head by scatterer count" (mgB); 15c(component
  rows)+15g -> "The head components --- layover pays at K=2, the crowd pays at
  K=5" (mgC). Presence-lever block: 18+18d -> "The two levers --- A stays
  free; B costs recall, then the curve" (mgD, stretch 0.74; title cut to one
  line, the Hungarian qualifier lives in the runs caption); 18b(count)+18e
  (mgE); 18b(components)+18f (mgF). Cells were copied verbatim by a
  parse-and-zip script (scratchpad merge_k_tables.py), and 700 non-dash cells
  were machine-verified against the pre-change snapshot in docs/presentations:
  every merged 4-cell quad equals one complete source row, so no cell moved
  columns. Bold / \hlval marks are PRESERVED per K group and the captions
  restate the rule as "best in row within its K group", plus a flag that the
  two groups are different label sets (K_2 vs K_max=5 test data) so
  cross-group reading is on the reader. Ranks k>=3 / pred k>=3 have no K=2
  reference (two-slot labels) -> "---" cells; the merged lever headline unions
  the seed-spread rows (K=2 exported curve MAE / SSIM range / profile cos med,
  K=5 exported count exact / count over; the absent side shows "---"). The
  four stats-vs-GT frames stay UNMERGED: their row structures differ (slots
  0-1 incl. conditional mu/sigma at K=2 vs slots 0-4 activity/leakage at K=5)
  and each K needs its own GT reference column; the K=2 pair is retitled "The
  K=2 head / levers, predicted statistics ..." so the pairing reads. The dual
  ROUTING (7 arms per K) and dual RATIO (6/5 arms) blocks cannot group
  horizontally -- 14/11 mean+-std columns exceed the text width at scriptsize
  -- and keep their split frames. Deck 152 -> 146 pages, sub_05 24 -> 20;
  warning set unchanged (only pre-existing overfulls remain).

- FIVE-PERCENT BOLD GATE + FILLED COLOUR CELLS 2026-07-27 (user: "in all tables
  of values, highlight only the lines in which the best and worst values vary
  more than 5%", then, after reading the render: "the green and red numbers are
  hard too, color the hole area of the number with the color itself"). Two
  passes over every value table in both decks.
  (1) BOLD GATE. Bold still marks the best cell, but a row keeps it only when
  its best and worst values differ by more than 5%, measured on the DISPLAYED
  (rounded) numbers as |best - worst| / min(|best|, |worst|). A row under the
  threshold loses every bold, so the only lines carrying emphasis are the ones
  where the winner is real. Exact ties fall out at 0% (the four tied bolds on
  the levers peak-median row). Directional rows read best/worst off the metric
  arrow; closest-to-GT rows take best = smallest |mean - GT| and worst =
  largest, so the compared pair is the winning arm against the losing arm, not
  the row range. Pools exclude the label column, the GT reference column and
  every Delta column. 317 bold cells removed over 37 tables in 04/06/07/08/10/
  11b/12/12b/17 and project_status. Column-bolded boards (10 at lines 45 and
  192, 17_backup, the project_status R2 table) run the same test down the
  column, since that is the axis their bold runs on; only the PSNR, SSIM, F1
  and recall columns fell. The closest-to-GT tables lost nothing, their arm
  spreads all sit far past 5%. NOT TOUCHED, because bold there is not a
  best-in-row marker: 08 line 184 (error decomposition, bold is the symbol 0),
  10 line 249 (bold marks the Delta-total column), 10 line 296 (bold marks the
  exactly-zero deltas), and every text table. All 43 caption bold definitions
  gained ", marked only where best and worst differ by more than 5%"
  (two-arm tables say "where the two differ"); the two bolded tables that
  carried no definition (11b line 275, 17_backup) got one. The gate also went
  into gen_dual_round2_tables.py as a SpreadGate class injected into
  TableEmitter, so regeneration reproduces it. CROSS-CHECK: the eleven
  regenerated fragments were diffed against the hand-edited 12b_dual tables and
  the bold placement is identical in every row, colour markup aside.
  (2) FILLED COLOUR CELLS. New \hlval{colour}{value} in preamble.tex (colorbox
  at fboxsep 1.2pt, white text) replaces \textcolor on NUMERIC cell content
  only: 216 cells across seven section files. Coloured words ("robust IQR
  (log1p)", "$\infty$ / NaN", "bounded") stay coloured text, and the
  \textcolor{soft} deltas stay grey.
  HEIGHT RECLAIMED. Three frames went over after the two passes: kdD
  arraystretch 0.85 -> 0.82 plus caption vspace 1pt -> 0pt, krD 0.81 -> 0.74
  plus vspace 1pt -> 0pt, drB 0.98 -> 0.92. Arraystretch still bites at these
  values, but weakly (krD needed 0.07 to buy 3.8pt).
  VERIFY: full deck, all ten sub-decks and project_status rebuilt; deck stays
  147 pages; the only surviving vbox overfull is the pre-existing
  11_loss_design one (17.3528pt, in a file this pass never opened). Pages read
  back off the render: sub_08 9/15/26, sub_06 6, sub_04 6.

- 90-10 CUBE-SLICE FRAMES ADDED 2026-07-27 (user: "after the dual ratio
  experiment results, add some images of the slices from the 90-10 k=2, they
  are inside the dir of the experiment"): FOUR qualitative frames appended to
  12b_dual after the K=5 ratio quartet, closing the dual block (markers
  25v-25y, full-deck pages 120-123; deck 143 -> 147, sub_08 29 -> 33). Images
  = results/K2/dual_k2_ratio/cube_slices-90-10/cube_slices/<az>_<rg>/ (six
  saved pixel locations x {range, azimuth} x {full, gt, pred}, all
  *_normalized, i.e. the webui cube explorer's per-column normalised space).
  Frame plan, all four identical in layout: full-tomogram | GT (Gaussian) |
  90-10 prediction, three minipages at 0.345\textwidth inside a
  \makebox[\textwidth] (the pairs do not fit two-up on 16:9 at a readable
  size, and three 1.62-aspect panels leave the vertical slack the caption
  uses). 25v = rg=370, the frame that introduces the chain; 25w = rg=1219
  canopy-against-bare-ground; 25x = rg=3230 two-layer; 25y = rg=3466 dense
  canopy with a bare corridor. RANGE CUTS ONLY (user, same day: "use only the
  range slices") -- 25y was first built on the az=1234 azimuth cut through all
  ~3500 range bins and was swapped for the rg=3466 range cut, the densest of
  the six saved locations; the azimuth_*.png files in the slice dir are now
  unused by the deck. TOMOGRAM IN EVERY FRAME (user, same day: "keep the
  mesuarments in all 3, and dont called it messurement, called it
  full-tomogram") -- 25w-25y started as GT | pred pairs at 0.485\textwidth and
  were rebuilt on the 25v three-panel layout; the panel label is
  "full-tomogram", never "measurement", and the GT panel label is "GT
  (Gaussian)", never "label". AXIS CAVEAT stated on 25v: the
  "full" source carries x_axis = np.arange(n_bins) (cube_explorer._load_all),
  so its vertical axis is elevation BIN INDEX on the full inversion grid,
  while gt/pred/reduced share the physical curve axis z in [-20,80] m -- the
  three panels are NOT on a common vertical scale and the caption says so.
  Seed unknown: the slice PNGs were copied in from the server (Downloads,
  2026-07-27 07:00) without their run directory, so the captions say "one
  seed" and never name one. Captions are read off the rendered panels, so
  every claim is checkable against the image beside it: reproduced features
  (ground line, canopy ramps, layer separation, the raised patch at az
  1130-1220 on 25x, both canopy blocks and the sharp wall at az~580 on 25y)
  and misses (the 70 m spike at az~700 on 25v, the single-column GT streak at
  az~480 on 25w, the tower at az~790 on 25x, the canopy reopening late at
  az~980 vs the label's ~870 and the 60 m point return at az~1145 on 25y).
  One caption claim was corrected after reading the render: 25x said the
  second scatterer was "populated over most of the swath" when it runs out at
  az~850. The retired azimuth-cut caption also had two errors worth recording
  in case that cut is ever brought back: the ground tilt is about -3 -> -6 m,
  not -2 -> -7, and the quieter bare stretches cannot be attributed to the
  second slot staying shut, since a re-synthesised profile image does not show
  slot occupancy. The 25y corridor caption was also retuned once the tomogram
  went in: the canopy reopening was first written as az~980 (pred) vs ~870
  (label), but a 2x zoom of the az 680-1210 strip across all three panels puts
  the label's first elevated return at ~955 and the prediction's at ~985, so
  the frame now says 985 vs 955. Same zoom confirmed the ~60 m point return at
  az~1145 is in the tomogram as well as the label, so the miss is a real
  return dropped, not a fit artefact the model was right to skip. Verified:
  tectonic full (147pp) + sub_08 (33pp), zero new warnings (only the two
  pre-existing 12b_dual overfulls at lines 76/189), pages 120-123 rendered
  with pdftoppm and inspected.
- K=5 ARM-RATIO FRAMES ADDED 2026-07-26 (user: "I have the ratio results for
  K=5 too, same thing about the model, it is using the skip_unet"): FOUR
  frames appended to 12b_dual after the K=2 ratio trio (markers 25r-25u, tags
  krA-krD, K=5 four-frame pattern, full-deck pages 116-119; deck 139 -> 143,
  sub_08 25 -> 29). Data = results/K5/dual_ratio_k5 (five dr splits x five
  seeds). Anchor verified: the K=5 dr-50-50 arm is per-seed BIT-IDENTICAL to
  the K=5 input experiment's di-full-full reference (overall_r2_gt +
  matched_recall), so routing (full stack -> both trunks) and unet_skip
  trunks carry over. Parity verified by local builds at K=5/unet_skip/9-9
  channels: totals 31,181,225-31,202,405 (shares exact to 0.01) -- the
  ~31.19M caption claim holds. VERDICT ON-SLIDE: nothing moves at K=5 --
  curve rows flat, detection edges at most at the rim of combined noise
  (80-20 recall 0.683 vs 0.666), under-count 0.31-0.33 at every split, slot-0
  fires 0.76-0.78 everywhere ("a 3.1M existence arm under-fires exactly like
  a 15.6M one"), and unlike K=2 the freed capacity buys NO localisation
  (krC title "buys nothing the noise can see"). No colours on any of the
  four frames -- no delta clears noise; captions carry the flatness claims.
  The ratio intro footer now reads "5 splits x 5 seeds, at K=2 and K=5 = 50
  runs". Generator refactored: the four K=5 row specs are shared methods
  (k5_headline/detection/component/stats_rows) used by both k5_frames and
  the new ratio_k5_frames; kdA fragment byte-identical after the refactor.
  FIT: krD needed arraystretch 0.81 (0.85 left a 2pt vbox; 24-line stats
  table + two-line title). Warning set: only the two long-standing 25e/25f
  hboxes.

- DUAL BLOCK MOVED AFTER THE BENCHMARK 2026-07-26 (user: "all of the dual
  experiments (input and ratio) were done after the benchmark ones, move them
  after them"): the thirteen dual frames (25e-25q) were cut from the end of
  08_context_information into NEW sections/12b_dual.tex, input directly after
  12_winner at the end of Act V; the Act V divider subtitle gained "dual
  trunks". sub_08_benchmark now inputs the section (12 -> 25 pages, subtitle
  extended); sub_06_context_information drops back to 15 pages. Full deck
  stays 139 pages with the dual block at pp. 103-115, right after the winner
  microscope (p. 102) and before the Act VI divider (p. 116). No references
  broke: 25e's "the input ablation split the labour" bullet and the ratio
  captions' benchmark-single citation both point backward after the move. The
  two long-standing 25e/25f hbox warnings travelled with the block (now
  12b_dual lines 76/189); no other warnings.

- DUAL K=5 INPUT + K=2 ARM-RATIO BLOCKS ADDED 2026-07-26 (user: "the dual input
  experiment for k=5 and the dual ratio experiment for k=2, add them to the
  slides, add a introduction slide with a visualization to the ratio experiment
  too"). EIGHT frames appended to 08_context_information after the K=2 routing
  trio (markers 25j-25q, full-deck pages 75-82, deck 131 -> 139 pages, sub_06
  20 -> 28). Data: results/K5/dual_input_k5 (seven di-<params>-<gate> routings,
  five seeds, dual_resunet-set_pred-hungarian-K_5-hv-A-param_l1, parity twin
  trunks [48,96,184,352] = 31.19M verified by a local model build at K=5) and
  results/K2/dual_k2_ratio (five dr-<params>-<existence> budget splits of the
  31.19M parity total at fixed routing, five seeds). AS-RUN ROUTING GOTCHA
  (user correction mid-session): the ratio runs fed the FULL STACK TO BOTH
  TRUNKS, not the registered params-full/gate-phi entry default in vault note
  dual-arm-ratio-experiment-2026-07-21 -- proven locally by per-seed
  bit-identity of dr-50-50 with the routing block's di-full-full reference arm
  (overall_r2_gt, matched_recall, slot_1_active_pred_frac, all five seeds).
  All three dr captions + the intro bullets say "full stack -> both trunks"
  and call the 50-50 anchor the all-channels REFERENCE arm; never relabel it
  the phi-gate default.
  (a) 25j-25m = the K=5 four-frame pattern (headline / detection-by-count /
  components-by-count / stats-vs-GT; tags kdA-kdD, dlA/dlC geometries).
  On-slide verdicts: the parameter trunk still sets the tier (R2 0.68 full /
  0.62-0.64 phi / 0.46-0.48 |A|, accent on the |A| cells) and the gate input
  stays inside seed noise, but the dual gate turns CONSERVATIVE at K=5 --
  under-count 0.25-0.33 vs over <= 0.07 (the K=5 single-trunk arms over-fired),
  slot-0 fires on only 0.76-0.85 of pixels (kdD title), k=1 recall no longer
  saturates (0.76-0.86, accent on the k=1 seed spreads +-.09-.14 of the three
  weakest arms); position pays double without phase (mu k1 1.02/1.13, k2
  2.76/2.69, accent). Cross-experiment caveat kept honest: the single-trunk
  K=5 contrast is param-MSE vs param-L1 here, so captions state the contrast
  without attributing it.
  (b) 25n = the requested ratio INTRO frame "Splitting the budget --- how much
  of the model does detection need?": left = tikz budget-bar scene (five
  constant-length bars split at 50/60/70/80/90%, params arm argA / existence
  arm good with per-arm M counts, top brace "one budget ~31.19M, totals within
  0.06%", dashed parity line "every dual run so far", sqrt-share footer with
  the 24-48-84-156 vs 48-96-184-352 ladder example); right = footnotesize
  intro + five bold-led bullets (one budget five splits at full stack -> both
  trunks, arm = trunk + its heads / 50-50 = the all-channels reference arm,
  H1 detection cheap, H2 parity load-bearing, H3 mu/sigma-before-count = gate
  borrows capacity), hypotheses taken from the registered prediction in vault
  note dual-arm-ratio-experiment-2026-07-21.
  FIT: the first render overfulled 19.4pt BYTE-IDENTICALLY after shrinking the
  tikz (scale 0.84) -- the vbox was the RIGHT bullet column (small ->
  footnotesize + itemsep 3pt cleared it; the tell strikes again); transform
  shape was then dropped from the scale so fonts stay at deck size.
  (c) 25o-25q = the ratio standing trio (tags drA-drC, 5 columns under one
  "parameter : existence split" group header). VERDICT = H1: drA "detection
  survives on a tenth of the budget" (90-10 takes the displayed best on every
  curve row, each edge inside combined seed noise; detection rows level, no
  bold-sweep colouring claimed); drB "the freed capacity lands in
  localisation" (green: mu MAE 1.63 -> 1.59 at 80-20/90-10 clears ~2x combined
  seed noise, mu k1 0.31 / k2 3.62, a MAE 0.177 / a k2 0.371 just clear it);
  drC "calibration is split-blind" (every activity/stats row inside noise,
  caption "a 3.1M existence arm calibrates like a 15.6M one"). H3's telling
  failure order did NOT occur; count never collapsed.
  Rows emitted by gen_dual_round2_tables.py -- NOW PERSISTENT in the deck dir
  (class-based, Dune env python, parses overview.md + metrics_comparison.md,
  displayed-equality bolding, closest-to-GT mode, writes table_fragments/
  <tag>.tex), unlike the dead scratchpad generators; colours are injected at
  splice time, not by the generator. K=5 caption capacity claim verified by
  building DualResUNet locally: parity twins = 31,189,858 params at K=2
  (bit-identical to the vault-note anchor) and 31,193,689 at K=5.
  Warning set: only the two pre-existing 25e/25f hboxes; all eight new pages
  pdftoppm-verified.
  TRUNKS CORRECTION (user, same day, supersedes every "dual ResUNet" wording in
  the 07-21/07-24 entries below): ALL dual experiment generations (K=2 routing
  07-23, K=5 input 07-24, K=2 ratio 07-25) trained with UNET_SKIP trunks
  selected in the frontend -- invisible in run names because they compose from
  the wrapper model_name dual_resunet. Deck fixed: the ten dual run-recipe
  captions now open "dual UNet-skip trunks", the 25e trunk boxes and encoder
  bullet say UNet-skip, and the ratio intro brace says totals within 0.1%
  (as-run unet_skip spread 0.061%). Capacity claims survive the swap: unet_skip
  twins/ladders weigh 31.18-31.20M at the same widths (shares still exact --
  both arms scale together, which is also why the ratio budget guard never
  fired), so every ~31.2M and per-arm M figure on the slides is unchanged.
  Vault note dual-arm-ratio-experiment-2026-07-21 carries the full as-run
  correction (inputs AND trunks).
  SINGLE-TRUNK COLUMN ADDED TO THE RATIO TRIO (user, same day): drA/drB/drC
  gained a sixth "single" column = the benchmark board's
  unet_skip-set_pred-hungarian-K_2-hv-A__param_l1 arm (like-for-like backbone,
  head, matching, loss, seeds, patch), separated by an hspace-7pt spacer with
  its own header cell outside the "parameter : existence split" group.
  Capacity stated in captions: 31,352,584 params ([64,128,248,504]), the
  benchmark bisection to the registry-UNet target 31,041,414 REPRODUCED
  LOCALLY via SizeMatcher with GaussianConfig.from_dataset mocked to K=2 (the
  server dataset meta is unreachable locally). Generator change:
  gen_dual_round2_tables.py merges results/K2/Benchmark/{metrics_comparison,
  overview}.md into the ratio values (run key = last path segment) and the
  single competes in the bold pool. On-slide findings: the single sits on the
  dual curve tier throughout, its count-exact edge (0.917 vs 0.914) is inside
  seed noise, it wins amplitude (0.175/0.055/0.363) and marginally recall, but
  pays on localisation past combined noise (mu 1.72 vs 1.59-1.63, sigma 0.735
  vs 0.686-0.698, accent-red) -- "splitting the encoder buys localisation"
  replicates on unet_skip at every split; on drC the single keeps the widest
  predicted spreads (closest to GT on every width row). NOTE the 07-24
  "single column removed, compare later" decision is deliberately reversed
  here FOR THE RATIO TRIO ONLY (user asked); the routing trios stay
  single-free.

- DUAL K=2 PARITY DATA REFRESH 2026-07-24 (user: "new data on the Dual experiment
  for K=2 in the results dir, update the presentation"). The three dual routing
  tables (dlA/dlB/dlC, frames 25g/25h/25i, full deck pages 72-74) were re-emitted
  from the fresh parity generation results/K2/Dual (seven di-<params>-<gate>
  routings, five seeds, dual_resunet-set_pred-hungarian-K_2-hv-A-param_l1;
  overview.md + metrics_comparison.md + seven per-arm seed reports, generated
  2026-07-24, inference runs 20260723_*). This is the PARITY rerun the earlier
  half-width generation was waiting on: fresh logdir runs/results/K2/dual trained
  AFTER commit acd75c5 (2026-07-21) which set DualInputTrialsConfig defaults to the
  parity ladder [48,96,184,352] = 31.19M total (params arm 15.60M + existence arm
  15.59M), within 0.5% of the single ResUNet 31.36M. So the on-slide capacity
  caveat FLIPPED: "≈16.2M per arm / ≈2× a dual arm / capacity not matched" became
  "≈31.2M per arm / parameter-matched" in the 25f intro and the three captions.
  The parity numbers barely move from the half-width generation (curve R2 full
  0.730-0.741 vs 0.72-0.73, phi 0.685-0.690 vs 0.67-0.68, |A| 0.489-0.493 vs
  0.48-0.49) -- doubling capacity is near-free on held-out R2, consistent with the
  context/prior-saturation theme of Act IV. New headline shift: at matched
  capacity a dual full arm (full-pass) now TAKES curve R2 0.741 and PSNR 52.5,
  where the single previously led; the full-parameter dual arms still win every
  mu/sigma localisation row.
  The scratchpad gen_dual_tables.py from 07-21 was gone, so the generator was
  rebuilt in the job tmp (parse each seed-comparison report's "mean +- std" rows,
  displayed-tie bold, leading-zero-stripped errors, $-x$ math for negatives,
  closest-to-GT bold for dlC) and the three tabular bodies spliced atomically
  (assert-unique). SAME SESSION, user follow-ups: (a) "remove the column talking
  about the single backbone, I will compare it later" -- the eighth "single"
  column was dropped from all three tables (specs back to 7 dual arms:
  ccc/cc/cc; group headers multicolumn 9->8 for dlA/dlB, 10->9 for dlC; cmidrules
  and sub-headers trimmed), row-best bold recomputed over the seven dual arms
  only, the dlC single-leakage accent-red removed, and every "single = ..." caption
  clause + the 25f single-trunk-reference bullet clause deleted; dlB green legend
  reworded "the localisation the split encoder buys" -> "the sharpest localisation,
  both on a full-parameter trunk". (b) "increase the distance between columns" --
  with only seven columns there was slack, so tabcolsep went 0.3pt->4pt (dlA/dlB)
  and 0.05pt->2.5pt (dlC), and the parameter-group spacers 1.5pt->9pt (dlA/dlB) /
  0.4pt->7pt (dlC). Pages 72-74 pdftoppm-verified: no overflow, groups clearly
  separated, dlC white GT seams intact. (c) "increase for all other tables as
  well" -- swept tabcolsep up on ~28 results tables deck-wide (04 156/196/238,
  05 120, all 06 K=2+K=5 lever/head tables, 07 85/131/172, 08 228/438 + input-
  ablation 831/871/915, 10 15/43/88/248, 12 winner, roughly +2pt or tight->2.5pt).
  Tabular overflow is a detectable overfull hbox (unlike silent tikz), so the
  sweep was verified by diffing the overfull set against baseline: three ALREADY-
  full-width tables overflowed and were reverted -- the two "l l c l l" text tables
  (04:78 normalization migration, 14:127 gamma-Net) and the dense 18-column board
  (10:190 "all 35 configurations"); those are at the width limit and cannot widen.
  Final overfull set == pre-existing baseline exactly (no new). Full deck + six
  section previews (sub_03/04/05/06/08/09) rebuilt tectonic-clean (the two
  pre-existing 08 hboxes at the 25e/25f frames untouched).
  results/Dual is now results/K2/Dual; the single reference lives in
  results/K2/Benchmark/resunet-set_pred-hungarian-K_2-hv-A__param_l1.md for the
  deferred comparison.

- RESULTS FRAMES RESTYLED MINIMAL 2026-07-22 (user: "delete the bullet points,
  centralize the table, colour only the important values"): all 35 side-by-side
  results frames (bullets column beside a painted table) across
  04_representation / 06_set_prediction / 07_augmentation /
  08_context_information / 10_benchmark_board / 11_loss_design / 12_winner
  dropped their bullet columns; each is now title + centred table +
  \parbox{0.82\textwidth} runs caption. All \rowcolor / \cellcolor / \redlast
  tints stripped; instead only the decisive values are coloured
  (\textcolor{good} = gain, \textcolor{accent} = cost), chosen per frame from
  the old bullet arguments and the deltas that clear combined seed noise (on
  spread findings the +- std is coloured, not the mean); caption legends
  rewritten to say in one clause what green/red mark (old keyed-bullet legends
  deleted). Saturation-verdict frame keeps its heatmap at 0.56\linewidth with
  the registered/observed table centred beneath, drift note folded into the
  caption. NOT touched: frames whose tables were already centred with
  standalone bullets (run-health tooling, Huber checks, failure modes, backup
  detail) and the dense 35-config appendix (soft row shading kept). Removed
  bullet text + old paint map archived in sections.bak-20260722/. Full deck +
  six affected sub-decks tectonic-clean; warning set SHRANK (the four
  bullet-column overfulls on the dual/Hungarian frames are gone). Deck 131
  pages, frame count unchanged. Mechanics: frame-scoped python restyle (drop
  itemize column, unwrap columns env, strip paint, wrap caption) + per-file
  value-paint scripts with assert-unique replacements.

- PRESENCE LEVERS RERUN AT K=5 ADDED 2026-07-22 (user: "I added results for the
  presence results for the K5, add them too"): FOUR frames appended at the end of
  06_set_prediction, directly after the K=2 presence trio (markers 18d-18g, tags
  kpA-kpD, full-deck pages 46-49, deck 127 -> 131 pages, sub_05 20 -> 24). Data =
  results/K5/Presence_K5 (none/A/B/AB on unet-set_pred-hungarian-K_5-hv-<presence>-
  param_mse_1, five seeds; generated 2026-07-22 12:43). The none arm IS the K=5 heads
  block's set-pred-Hungarian arm (identical training runs, val loss 0.046367 --
  cross-consistency stated in every caption). Same four-frame split as the heads K=5
  block. Frames: 18d "The levers at K=5 --- A stays free, B now pays in the curve"
  (setD geometry incl. a 3-row seed-spread block: curve R2 / count exact / count
  over): A still a free no-op (nominal lean, nothing past combined noise); B's price
  MOVED from k=2 recall (K=2) to the curve (R2 0.566 -> 0.473, PSNR -0.8 dB, peak p95
  5.8 -> 7.9) and precision (0.573 -> 0.483, F1 0.696 -> 0.622), ~2x combined noise,
  recall flat; B destabilizes seeds (count exact +-.06 -> +-.20, over +-.07 -> +-.21,
  R2 +-.016 -> +-.040). 18e by-count: precision drops at EVERY k under B; the k=5
  tail payoff B was kept for does not appear (0.327 -> 0.336/0.344, inside noise) and
  k=3 recall goes the wrong way (0.763 -> 0.733). 18f components: B pays mu
  (1.94 -> 2.14) and sigma (0.77 -> 0.85 AB) but BUYS amplitude at every split
  (a MAE 0.552 -> 0.469 AB, k=5 0.915 -> 0.627, 3-5x noise) -- honesty bullet: the
  four arms' recalls are near-identical so the matched subsets are comparable, the a
  gain is not a selection artifact. 18g stats-vs-GT: B re-brightens the dimmed tail
  (slots 2-4 active amplitude 0.18-0.57 -> 0.53-0.76 vs GT 0.66-1.41; slot-0 mu width
  2.39 -> 2.84 vs GT 3.02 -- presence balance DOES fight the shrinkage it was built
  against) but the re-brightened slots over-fire (tail activity 0.33-0.43 vs GT
  0.085/0.035/0.015) and the suppression never reaches the tail empties (slot-1 leak
  trimmed 0.39 -> 0.25, tail leaks stay 0.38-0.54). Verdict bullet: the standard
  (+A on, B off) survives K=5. Frame 18's Adopted bullet updated: "(untested here)"
  -> "at K=5 it pays in the curve, no tail payoff (three frames ahead)" -- that
  edit initially overfulled frame 18 by 10.3pt (longer bullet), cleared by shortening
  + itemsep 4->3pt. 18g's 3.6pt vbox was the TABLE column (byte-identical after
  bullet itemsep change -- the tell works in both directions): cleared by caption
  shortening + par-vspace 0. Leaderboard caveat: the overview's min-max composite
  makes B look like total collapse (grouped score 0.000); the raw deltas are ~2x
  noise, real but not apocalyptic -- slides quote raw numbers, never the composite.
  Val losses (none 0.046 / A 0.085 / B 0.157 / AB 0.247) kept off-slide: the levers
  change the loss itself. Rows emitted by scratchpad gen_k5_presence.py (same parser
  as gen_k5_heads.py, presence column layout).

- HEADS 2x2 RERUN AT K=5 ADDED 2026-07-22 (user: "new results of the heads/matching
  experiment, using K=5 now. Add the new results, in sequence of the K=2 ones"): FOUR
  new frames inserted in 06_set_prediction directly after the K=2 heads trio, before
  "The imbalance failure (high K)" (markers 15e-15h, full-deck pages 36-39, deck
  123 -> 127 pages, sub_05 16 -> 20). Data = results/K5/Heads_K5 (head x matching 2x2
  on unet-*-K_5-hv-none-param_mse_1, five seeds each, patch 64x32; overview.md +
  metrics_comparison.md generated 2026-07-22 10:10). The standing 3-table pattern needed
  a fourth frame here: with five k-splits the by-count table (35 seedpm rows) cannot fit
  one frame, so it was split into detection-by-count (15f: precision / recall /
  count-acc-pred-k) and components-by-count (15g: mu / sigma / a per k). Frames:
  15e "Set prediction at K=5 --- the head still leads, and the matching now matters"
  (headline table, tag khA, setA geometry): gate detection lift holds (recall 0.65 ->
  0.87-0.89, under 0.32 -> 0.03) at decade-tighter seed spreads; sorted-GT now BREAKS
  the conv head (recall 0.648 vs 0.802 Hungarian, peak err 4.8 vs 2.6, and conv-Hung
  seed spreads balloon +-.09-.13) -- the K=2 "free insurance" claim is confirmed
  load-bearing; over-count climbs across the board (0.25 -> 0.39, worst sp-Hung);
  conv-Hung is the best curve arm (R2 0.587, every reconstruction row). NOTE: best val
  losses (sorted 0.064 vs Hungarian 0.046) were deliberately NOT put on-slide --
  Hungarian's loss is a min over assignments, mechanically <= sorted's, not comparable.
  15f (tag khB): gate saturates k=1 (0.99-1.00), sp-Hung wins every rung k>=2, every
  arm under 0.53 recall past k=3; claimed counts >= 2 unreliable (pred-2 0.17-0.32 vs
  K=2's 0.63-0.73, pred >= 3 at most 0.12); precision block painted soft (its bullet
  is the rewards-under-prediction caveat). 15g (tag khC): mu MAE decays 0.6-0.8 (k=1)
  -> 9.2-9.6 (k=5); sorted arms keep sigma (k2-5 conv-sort) and a (overall 0.442
  sp-sort); soft bullet = matched-pairs selection caveat (low-recall arms graded on an
  easier subset). 15h (tag khD, setC geometry + GT white seams): five-slot activity
  ladder (gate calibrates slot 0 at 0.98-0.99 vs conv-sort 0.70; every arm over-fires
  the tail, sp-Hung slot 4 at 0.35 vs GT 0.015 = the headline over-count), a-active
  undershoot deepens with slot depth, leakage block slots 1-4, width block trimmed to
  slots 0-1 (documented deviation, caption points to the report for tail slots; 24
  table lines only fit at arraystretch 0.85 + itemsep 6pt -- a 0.19pt vbox overfull
  was the BULLET column, cleared by itemsep, confirming the byte-identical-warning
  tell). All table rows emitted by scratchpad gen_k5_heads.py (parses
  metrics_comparison.md, mean-only displayed-equality bolding, closest-to-GT mode for
  khD) -- regenerate from it on any refresh. ALSO: the user's results/ reorganization
  into K2/ and K5/ subtrees broke three deck figure paths -- repointed
  results/augmentation_experiment -> results/K2/augmentation_experiment (07_augmentation)
  and results/Patch -> results/K2/Patch (08_context_information), and the three dual
  caption provenance cites to results/K2/Benchmark; sub_04 / sub_05 / sub_06 rebuilt.
  New-frame warning set: zero; pre-existing set untouched (A/B-verified the dual-frame
  hboxes predate the caption edit).

- DUAL-TRUNK ROUTING BLOCK ADDED 2026-07-21 (user: "results on the dual backbone
  architecture experiment ... add the idea, the visualizations and the results table
  with the discussion"): FIVE new frames appended to 08_context_information after the
  input-ablation trio (markers 25e-25i, full-deck pages 62-66, deck 118 -> 123 pages,
  sub_06 15 -> 20). Data = results/Dual (seven trunk-input routings di-<params>-<gate>
  over full/ifg/pass, five seeds, dual_resunet-set_pred-hungarian-K_2-hv-A-param_l1,
  symmetric twin trunks [32,64,128,256] ~16.2M params, patch 64x32; overview.md +
  metrics_comparison.md, generated 2026-07-21 12:28).
  (a) 25e "The dual model --- two trunks, one head": idea frame mirroring 15a's head
  diagram — two input cards -> parameter/existence trunk boxes -> Gaussian heads /
  existence head -> pill + g_k=sigma(e_k), dashed argA "gates the blend" arrow; gate
  blend formula line reused verbatim from 15a; accent bullet = default routing (full
  -> params, phi -> gate, verified in models/dual/dual_resunet.py + DualEntryConfig).
  (b) 25f "Routing the trunks": 3x3 grid scene (rows = params input, cols = gate
  input) with per-cell two-row glyph pairs (top = params, bottom = gate; amp grid /
  phase rings via tikz pics), phi/phi + |A|/|A| dashed "not run", ref./default corner
  tags; hypotheses H1 (params trunk needs phase) / H2 (gate runs on phase alone) /
  H3 (reversed default pays on the curve).
  (c) 25g/25h/25i = the standing three-table pattern (seedtags dlA/dlB/dlC; rows
  emitted by scratchpad gen_dual_tables.py — parses the two md reports, mean-only
  displayed-equality bolding, closest-to-GT for dlC). Column layout for SEVEN arms:
  groups by params input (3/2/2 sub-columns = gate input) with hspace-2pt spacers
  (dlA 0.68 col, tabcolsep 0.5pt, stretch 0.98; dlB 0.69, 0.5pt, 1.02; dlC 0.73,
  0.1pt, 0.90 + 0.6pt white GT seams, hspace-0.4pt spacers). FIT LESSON: seven seedpm
  columns do NOT fit at 4-decimal means — the eqparbox M-box is table-wide, so ONE
  wide row pads every cell; dlA had to drop to 3 decimals (PSNR 1) which produces
  triple displayed-tie bolds across the params-full tier — kept deliberately, the
  ties ARE the frame's message (gate input inside seed noise). First render
  overflowed the right edge on all three tables WITH ZERO overfull warnings
  (tikz/minipage silent-overflow gotcha re-confirmed) — pdftoppm render-verify
  caught it. count-acc rows restructured into a "count acc | pred k" group header +
  two \quad rows to shorten the label column (no abbreviations rule kept).
  VERDICTS on-slide: params trunk sets the tier (R2 0.72-0.73 full / 0.67-0.68 phi /
  0.48-0.49 |A|; H1 holds, H3 holds); gate input moves headline rows less than
  combined seed noise (H2 stronger than registered — even |A|-only gating matches;
  full-pass zeroes the median peak error); recall rides the params trunk (phi arms
  0.893/0.890); mu k=1 0.32-0.37 with phase vs 0.79-0.86 without; a MAE routing-blind
  0.183-0.194; slot-0 calibration follows params trunk (phi arms 0.986/0.981),
  slot-1 activity lands on GT only with full params (0.265 vs 0.260), |A| arms
  over-fire slot 1 but leak dimmest (0.52 vs 0.72-0.91). All warnings pre-existing
  (10, none on pages 62-66); both decks tectonic-clean, pages 62-66 pdftoppm-verified.
  SAME DAY, SINGLE-TRUNK REFERENCE COLUMN (user: "compare it as well (add to the
  tables) the regular single backbone resunet"): all three tables gained an eighth
  column "single" = resunet-set_pred-hungarian-K_2-hv-A__param_l1 from
  results/Benchmark (same head, loss, seeds, patch). CAPACITY CHECK 2026-07-21
  (user believed the dual trials were parameter-matched to the single arm — they
  are NOT): the benchmark arms are SIZE-MATCHED by pipelines/benchmark/sizing.py to
  the registry-UNet budget (31.05M at in_channels 9); reproducing the deterministic
  bisection gives the benchmark resunet features [64,128,248,504] = 31.36M (+1.0%;
  confirmable against the run's pipeline/size_match.json on the server). The dual
  trial arms are 16.23M (counted live): half-width twin trunks quarter each trunk
  (params scale with width squared), so two halves = ~52% of one full-width net,
  not parity. Parity twins would be [48,96,176,352] = 30.87M (counted) — the
  config for a capacity-matched rerun. Captions now say ~31.4M vs ~16.2M. ALSO
  CORRECTED same pass: the captions/card labels claimed 29 |A| + 28 phi — WRONG;
  the runs use the DEFAULT secondaries (PS04/06/08/26 -> 5 tracks), so full = 5
  |A| + 4 phi = 9 channels, phi = 4, |A| = 5 (InputConfig.full_stack() selects
  modality GROUPS, not track count — do not conflate). 25e cards + bullet + all
  three captions fixed. Fit changes: dlA/dlB bullets 0.26 / tables 0.72, tabcolsep
  0.3pt, spacers 1.5pt; dlB row labels DROPPED the "matched " prefix (caption line
  "all precision/recall/mu/sigma/a rows are matched" carries it) and dlA means went
  3-decimal (PSNR 1) — the resulting triple displayed-tie bolds across the
  params-full tier are deliberate (they visualize the gate-is-free claim). dlC could
  NOT fit an eighth seedpm column (GT column + negative means bind the eqparbox
  boxes): its single column shows SEED MEANS ONLY — documented deviation per the
  29g2 precedent, caption points at the benchmark reports for spreads — plus \quad
  -> \; label indents, header "prediction mean" -> "mean", bullets 0.22 / table
  0.76, tabcolsep 0.05pt. FINDINGS vs the single trunk (on-slide): it lands on the
  dual full tier (R2 0.734, val 0.190), matches recall (0.894) and takes count
  exact (0.918) + amplitude outright (a 0.178, best k=2 0.367); the dual
  full-parameter arms beat it on EVERY mu and sigma row (mu 1.59-1.62 vs 1.75,
  sigma 0.69 vs 0.76) — "splitting the encoder buys localisation"; it keeps the
  widest predicted spreads (mu slot-1 9.8 vs dual <= 9.2, GT 16.2) and the closest
  slot-1 active a/mu, but leaks the most into empty slots (0.92). ALSO SAME DAY,
  25e/25f REBUILT BIGGER (user: "make the visualization bigger ... improve it, its
  not good at all"): 25e diagram now scale 0.92 with stacked-sheet input cards
  (deck card grammar, 5 |A| / 4 phi labels), 21mm trunk boxes, "embed." midway
  labels deleted (they collided with node edges); 25f grid rebuilt at scale 0.92
  with 2.05x1.5 cells, per-cell par/gate labels + soft divider line in EVERY cell,
  spelled-out "reference"/"default" corner tags in a clear top strip, and the
  glyphs enlarged to 0.42. Table rows still emitted by scratchpad
  gen_dual_tables.py (now parses results/Benchmark too; mean_only={"single"} for
  dlC).

- BENCHMARK TRIO COMPLETED 2026-07-20 (user follow-up: "the standard is 3 tables,
  look other results"): the benchmark section now carries the standing three-table
  pattern like every other five-seed experiment section. Two NEW frames between the
  board (29g) and the dense board (29g2), both on the 6-arm column set = the five
  objectives on the WINNER BACKBONE (UNet-skip: L1 / Charbonnier / param-L1 / MSE /
  param-MSE) + the reference cell UNet · param-MSE, grouped cmidrule header
  "UNet-skip --- winner backbone | UNet" with a hspace-3pt spacer before the
  reference column (geometry copied from the input-ablation ctxB/ctxC frames).
  (a) 29gB "The board by scatterer count --- curve-MSE drops the single scatterer"
  (seedtag bbB, scriptsize, stretch 1.00, tabcolsep 1.0pt — 1.1pt left the tabular
  0.87pt hbox-overfull in the 0.66 column): precision/recall/count-acc/count-exact +
  mu/sigma/a MAE each with k=1/k=2 splits; argC paints recall+count rows (detection
  belongs to the head: k=1 recall 0.98-1.00 everywhere EXCEPT curve-MSE 0.81+-.12 —
  the collapse is a k=1 disease, k=2 recall barely moves; param-L1 = count
  calibrator, count-acc|pred-k2 0.847 and count exact 0.917), argA paints the
  component rows (param-L1 best a at both counts 0.055/0.363 + best k=1 mu/sigma;
  curve-L1 pair holds layover position, k=2 mu 3.32 vs 3.87), soft bullet = k=2 hard
  case + the reference cell's trade (board-best recall 0.894, worst k=2 mu 4.38).
  (b) 29gC "The board, predicted statistics vs the GT --- the gate sets activity,
  param-L1 calibrates amplitude" (seedtag bbC, stretch 0.90, tabcolsep 0.35pt, GT
  column good!10 with white vrule seams): slot-0/slot-1 blocks + width block, bold =
  closest-to-GT; argC paints the two active-frac rows (L1-family arms fire slot 0 at
  0.97 and slot 1 within 0.02 of GT 0.26; curve-MSE slot-0 firing drops to 0.82 —
  the same k=1 disease from the activity side; param-MSE over-fires slot 1
  0.38-0.39), argB paints slot-1 a-active/a-inact/mu-active + slot-0 a-active
  (param-L1 amplitudes sit ON the GT 0.74-vs-0.77 / 1.42-vs-1.43 and param arms
  carry slot-1 mu closest 8.3-8.8 vs 11.3 — priced by the LARGEST leakage 0.88 vs
  0), soft paints width a-slot-1 (reference cell narrowest 0.83 vs GT 1.45; every
  width below GT). addlinespace[2pt] separators at every argC->argB junction.
  FIT: the first cut of 29gC was 60pt vbox-overfull — the 0.28 BULLET column again,
  not the table; bullets trimmed to ctxC terseness cleared it. Row emitters:
  scratchpad trio_emit.py (paints + splices applied by the splice script; GT-column
  bold computed on display-rounded |mean-GT| with the 2-way-tie rule; a-inact GT
  hardcoded 0.00, mu/sigma-inact GT = "---" unbolded). Deck 116 -> 118 pages,
  sub_08 10 -> 12; warning set = only the pre-existing 29h 3.22pt vbox.

- BENCHMARK BOARD REBUILT ON THE FIVE-SEED RUN 2026-07-20 (user: "new run of
  benchmarking, smaller, but with 5 seeds each ... the head now is fixed on the set
  pred"): data = results/Benchmark — 35 arms = 7 backbones {unet, unet_skip, resunet,
  attention_unet, deeplabv3plus, segformer, nafnet} x 5 objectives {MSE/L1/Charbonnier
  curve, param MSE/L1}, EVERY cell set_pred · hungarian · K_2 · hv · A, five seeds,
  patch 64x32; ranking = the overview's Grouped score (equal-weight mean of the five
  metric groups, min-max over the 35 arms — the deck's five-group composite; the
  vs-Capon column is empty for all arms). Frames rebuilt in 10_benchmark_board:
  (1) tooling frame axes table -> 7 backbones / 5 objectives / 1 head fixed ("the Act
  III verdict, baked into every cell") / 5 seeds (35 x 5 = 175 runs); recipe bullet ->
  set-pred · Hungarian · presence A · hv flips · K=2, reference cell UNet + param-MSE.
  (2) 29g -> "The board at five seeds --- the L1 family owns the top nine":
  bullets-left footnotesize (argB podium / argA 3x3 / argC residual-block lever / soft
  MSE), seedpm table right (tag bbA, arraystretch 0.95, tabcolsep 2.6): ranks 1-12 +
  exhibit rows 16 (deeplab MSE-curve, first MSE arm), 20 (plain unet L1-curve, the
  block-lever comparison) and 21 (unet_skip param-MSE), a vdots row and the caption
  stating exactly which ranks are omitted; tints amber=unet_skip podium / blue=3x3 /
  purple=plain-unet arm / grey=MSE, addlinespace at every distinct-tint junction.
  (3) 29g2 -> "Every headline metric, all 35 configurations": tiny 35x18 board (#,
  model, loss, R2, PSNR, cosine, SSIM x3, F1, recall, 2-scatterer recall, precision,
  mu/sigma/a MAE, peak, score), MEANS ONLY — the caption states the spreads live on
  29g and in the per-arm reports (deliberate deviation from the every-number-seedpm
  rule: 18 seedpm columns cannot fit); a-MAE column added (new reports export it),
  head/matching columns dropped (axis gone), grey rows = MSE family, per-column-best
  bold with the 2-way-tie rule. 12_winner: "Winner vs baseline" rebuilt bullets-left +
  seedpm table right (tag bbW): winner unet_skip · set_pred · L1-curve vs reference
  unet · param-MSE, three painted blocks (argA reconstruction / soft detection / argC
  localisation), Delta column coloured good/accent only past combined seed noise
  sqrt(s_w^2+s_b^2) — every row clears it, and the recall row is honestly RED: the
  param-MSE baseline holds the best matched recall on the whole board (0.894 vs
  0.886). Microscope pending boxes + failure-modes parenthetical now cite the board
  winner's inference run 20260720_193330; 16_close take-home winner -> UNet-skip
  set-pred + L1-curve (Charbonnier and param-L1 on the same backbone close behind).
  ON-SLIDE FINDINGS: unet_skip arms sweep ranks 1-3 trading the group wins (L1-curve:
  curve error + variance + SSIM; param-L1: cosine shape; Charbonnier: peak); the top
  nine is exactly {unet_skip, resunet, deeplabv3plus} x {L1, Charbonnier, param-L1}
  with unet_skip's weakest arm (0.963) above the rest's best (0.946); the plain-UNet
  chain isolates the architecture levers (unet 0.760 -> unet_skip 0.983 = residual
  blocks, unet_skip -> resunet 0.930 = strided-conv downsampling, per the UNet Skip
  model note); the two MSE families fail differently (curve-MSE: detector goes
  seed-unstable, recall means to 0.66 with spreads to +-0.11 — DeepLabV3+ is the one
  stable exception; param-MSE: keeps the gate's recall ~0.88 but R2 <= 0.65).
  KEPT UNTOUCHED on purpose: 29h/29i (head-vs-matching decomposition + Hungarian-at-
  K2 — the old-grid 2x2 attribution the fixed head rests on) and the old-generation
  paired analyses in 11_loss_design / 11b (ambiguity + checks tables); flag to the
  user whether 29h/29i should move to backup now that Act III carries the five-seed
  controlled 2x2. FIT LESSON RECONFIRMED: the board frame's 41.8pt overfull was the
  BULLET column — caption cuts moved nothing until the bullets were shortened.
  Warning set after the rebuild = only the pre-existing 29h 3.22pt vbox. Full deck
  116 pages (frame-for-frame replacement), sub_08 10 pages. Tables emitted by
  scratchpad parse_benchmark.py + emit_tables.py (auto per-column bolding).

- L1-VS-MSE ADAM + FLOORS FRAMES DELETED 2026-07-20 (user, continuing the
  trim): "one update under Adam: drift and kick" (29n) and "MSE floors at
  the noise, L1 at the step" (29n2 floors) are gone. The section is nine
  frames; the gradient story now ends at the tail-share frame, then the
  checks. Residual wording: the limits frame's weight-axis bullet still
  mentions the noise floor and eta (|r| = eta/2w) --- the derivation behind
  it lives only in this log's history now, the claim itself still reads.
  Full deck 118 -> 116 pages; no warnings from this section.

- L1-VS-MSE FOUR FRAMES DELETED 2026-07-20 (user asks, same session as the
  de-step/trim passes): "position: the benchmark prices the two targets"
  (29l4), "the matcher pins the tails in place" (29m), "differentiate each
  loss" (29m2) and "the composite objective: MSE fades" (29n2 fade). The
  section is now eleven frames: label laws, MSE mean, L1 median, amplitude
  mean, amplitude median, position mean, tail share, Adam update, the two
  floors, Huber checks, limits. Dangling-reference fixes: the tail-share
  lead "MSE --- sum the slopes of the previous frame." became "MSE --- sum
  the slopes." (the differentiate frame it pointed at is gone); the
  tail/core populations are defined inline on the tail-share frame, so the
  matcher frame's deletion leaves no undefined term except R, which reads
  from context ("each 2|r_i| ~ 2R"). The limits frame's "floor and fade" and
  "the optimum half prices" wordings still read standalone. Full deck 122 ->
  118 pages; both decks build with no warnings from this section.

- L1-VS-MSE POST-EQUATION PROSE REMOVED 2026-07-20 (user: "remove all of
  this huge text after the equations sequences", quoting the CDF frame's
  flat-median note + closer). Thirty scriptsize soft paragraphs deleted
  across frames 1-13 of sections/11b_l1_vs_mse.tex: every per-column note
  under a display (symbol glossaries included, e.g. the pi/mask block on
  the label-laws frame) and every frame-bottom closer; orphaned
  vfill/smallskip lines cleaned so beamer recentres the remaining content.
  KEPT: the per-line \text{scriptsize soft} annotations inside the aligned
  displays (52 of them), the Huber-checks frame's post-table verdict
  paragraph (follows a table, and it carries the findings), and the limits
  frame's closer (no equations there). Frames now end at their banded
  result rows. If the removed arguments are ever needed again they are in
  this log's history and the vault note, not in the deck. Builds clean,
  renders verified on frames 14/18/20/24/26 of sub_07.

- L1-VS-MSE SECTION DE-STEPPED, BULLETED, BANDED 2026-07-20 (user: "the same
  problem of the colors/steps/bullet point, colored lines, need to be solved
  on [the L1-vs-MSE section], fix it all"). One pass over all fifteen frames
  of sections/11b_l1_vs_mse.tex, applying the error-theory treatment:
  (1) TITLES: every ", step N ---" prefix dropped; all frames now read "Why
  param-L1 beats param-MSE --- <descriptive>"; % markers kept their 29j-29p
  codes minus the step wording. (2) LEADS: the sixteen coloured argB/argC
  bold block leads became plain bold single-item itemize bullets ("Step 1
  --- the amplitude law..." -> "The amplitude law is spike-and-slab by
  construction." etc); the plain "The residual populations." lead was
  bulleted too, and the two checks-frames leads nest their existing
  scriptsize itemize as sub-bullets. (3) STEP REFERENCES: every in-prose
  "step N" pointer was rewritten descriptively or relatively ("as in step 6"
  -> "read off the CDF", "sum the slopes of step 10" -> "of the previous
  frame", "(steps 5--6)" -> "(the amplitude frames)", "steps 3--8 price" ->
  "the optimum half prices", "(step 10)" slope tags -> "(the MSE slope)" /
  "(the L1 slope)"); the only remaining "step" words are the optimiser's
  step size. (4) BANDS: the final-result row of every derivation chain
  carries an \hleq accent!12 band --- the mixture laws, c*(MSE)=E[theta|x],
  c*(L1)=med, the dimmed product, the L1 support row, the mean-distance
  identity, med-distance 0 + the false-alarm/miss vs always-matched pricing
  pair, the target-jump row, the pinned-populations row, both slope
  punchlines, S(MSE) + the S(MSE)->1 vs S(L1)=eps limit, drift + kick, the
  two floors, and the fade/constant pair. (5) The fourteen accent-coloured
  scriptsize closers became soft grey like every other deck caption ---
  emphasis now lives in the bands. Builds clean: 11b contributes zero
  warnings; renders verified on frames 14/17/21/25/28 of sub_07. FOLLOW-UP
  same day (user: "some slides are missing the bold bullet points"): the seven
  frames with no bold bullet --- the six single-column derivation frames
  (decision-theory instrument, absolute loss, amplitude mean, amplitude
  median, position mean, Adam update) and the Huber-checks frame --- got their
  intro sentence converted to a bold-led single-item bullet, so every frame
  in the section now opens with at least one bold bullet.

- REDUCED-STACK FRAME DELETED 2026-07-20 (user: "delete this slide: Why the
  reduced stack loses detections"). The section's closing tables frame
  (expected-vs-observed per input regime + reduced-vs-phi-only metric table +
  the collapse-not-calibration closing note, marker 25f-a2) is gone;
  09_error_theory now ends at "The recall paradox --- MSE turns doubt into
  dimming" (seven frames). No other frame referenced it. Its content
  (phase-only vs reduced-stack detection numbers) still lives in the
  stability-act amplitude-only verdict frame. Full deck 123 -> 122 pages;
  builds clean --- the error-theory section carries zero warnings.

- RECALL-PARADOX FRAME BULLETED 2026-07-20 (user: "needs some bold bullet
  points to separate its parts"). Five bold single-item-itemize leads now
  structure the frame: left column "Squared error is minimised by the
  conditional mean ---", "Split the mean over the two cases." (sentence order
  swapped so the bold imperative leads), "Bayes' rule, with pi = Pr(K=2)...";
  right column split into two displays headed "|A| runs:" and "phi-only:
  every pixel lands at an extreme:" --- the old in-math first-column labels
  (|A| runs: / phi-only:) moved out of the aligned env into the bullets, the
  [10pt] block gap became smallskip-between-displays, rows and hleq bands
  unchanged. SIDE EFFECT: the frame's long-standing 2.08pt overfull hbox
  (09_error_theory:322/337/343 in earlier logs) is GONE --- it was the
  in-math side labels widening the display. Builds clean; the only remaining
  deck warnings are outside this section.

- FULL-PROFILE FRAME: LEFT-COLUMN BULLET GAP 2026-07-20 (user: "create a gap
  between the bullet points on the left column"). On "The full profile ---
  separated scatterers decouple" the \medskip between the "Write the error as
  residuals" block and the "Square the sum" block was too tight --- replaced
  with \vspace{16pt}. Page 67 rendered and checked; sub_07 rebuilt.

- BOWL-FRAME PLOTS ALIGNED AT SOURCE + PARADOX INTRO CUT 2026-07-20. (a) The
  two plots on "The bowl and the training pull" were vertically misaligned
  because render_gradient() in joint_error_heatmap.py gave its figure a
  set_title("-nabla (descent)") while render_convergence() has none --- with
  bbox_inches=tight the gradient PNG came out taller, dropping its axes. Fix
  at the generator, not in LaTeX: the label moved INSIDE the axes as an
  annotate at (0.025, 0.205) with a white alpha-0.75 backing; regenerated via
  the Dune env python --- both PNGs are now exactly 1377x798 and the axes
  align at equal includegraphics width. (b) Per user ask, the right-column
  intro sentence on "The recall paradox" ("Amplitudes see a second scatterer
  only as a small dent in |rho| ... a clean, direct signature:") was deleted
  together with its vspace --- the column now opens directly with the
  |A|-runs / phi-only derivation. Renders verified; warning set unchanged. FOLLOW-UP: the level curves
  (contour levels 0.02/0.05/0.1) were removed from the gradient plot per
  user ask --- the right panel is now heatmap + descent quiver only; the
  frame caption never mentioned the right panel's contours, so no text
  change.

- ERROR-THEORY RESULT LINES HIGHLIGHTED 2026-07-20 (user: "in each slide of
  this section, color the most important lines of equations (the final
  result)"; follow-up: "not change the color of the font, but color the line
  area"). Implementation: NEW preamble macro \hleq{...} = colorbox accent!12
  with fboxsep 1.5pt around $\displaystyle ...$ --- a background band behind
  the row's result expression, ink text unchanged; the leading Longrightarrow
  arrow cells stay unbanded (banding them overfulled the annotated One-chain
  row and reads busier). A first pass that accent-coloured the FONT of the
  same 13 rows was reverted the same session. Follow-up ask banded four more
  rows on Two exact invariances: the amplitude carries (E[A_n^2] -> scale a,
  cov -> sigma/Delta-mu) and the phase carries (arg rho -> line of slope mu,
  |rho| -> phase spread) --- 17 banded rows total.
  Thirteen concluding equation rows across sections/09_error_theory.tex are
  now accent-coloured ({\color{accent}...} per aligned cell, arrow cell
  included): One chain -> the closing Capon(R-hat) -> fit -> a,mu,sigma row;
  Two exact invariances -> both "unchanged" invariance rows; Putting the two
  invariances together -> the shift-error asymptotics row, the 141% scale row
  and the two derivative conclusions (flattens / never flattens); Shift and
  scale together -> the joint-error formula row and the dP/de = 0 at
  1+e = exp(-d^2/4s^2) < 1 dimming row; The full profile -> the decoupling
  row (<r_j,r_k> ~ 0 => sum of ||r_k||^2); The recall paradox -> the
  split-mean identity row plus the "dimmed" and "undimmed" outcome rows. The
  bowl frame (images only) and the reduced-stack frame (tables only) carry no
  equation chains --- nothing coloured there. Builds clean, warning set
  unchanged.

- ERROR-THEORY BLOCK TITLES -> BULLETS 2026-07-20 (user: "the sections have
  their title with different colors, just make them bullet points instead").
  The four coloured bold block lead-ins in sections/09_error_theory.tex ---
  argB/argC \textbf{\textcolor{...}} on "Amplitudes are shift-blind." /
  "Phases are scale-blind." (Two exact invariances) and "Write the error as
  residuals." / "Assume the scatterers are separated." (The full profile) ---
  are now plain \textbf items in single-item itemize blocks (theme circle
  bullet, ink text; the argB/argC colouring is gone from this section). The
  uncoloured sibling lead "Square the sum --- bilinearity, no approximation"
  in the same column was bulleted too so the column reads as one list of
  moves. SAME DAY follow-up (user: "put bullet points here too"): the three
  plain bold lead-ins on "Putting the two invariances together" ("The shift
  error.", "The scale error.", "Take the derivative of each error:") got the
  same single-item itemize treatment --- and, per two further user asks, the two
  lead-ins on "Shift and scale together --- the joint error" ("Both mistakes
  at once", "Take the partial derivatives:") likewise --- every block lead in
  the section is now a theme bullet (remaining \textbf in the file are table
  best-value bolds only). Content unchanged; both frames re-rendered clean; warning set
  unchanged (09_error_theory overfull is the recall-paradox frame, line
  shifted 322 -> 332 by the itemize lines).

- ERROR-THEORY SECTION DE-STEPPED 2026-07-20 (user: "i dont like that this
  section is structured in steps, change that, keep the content, change the
  titles of things"). All eight frame titles in sections/09_error_theory.tex
  lost their "Step N ---" prefixes; content untouched. Old -> new:
  "Steps 1--2 --- one chain from the label to the channels, and back" ->
  "One chain --- from the label to the channels, and back"; "Steps 3--4 ---
  two exact invariances" -> "Two exact invariances" (its two in-frame column
  leads "Step 3 --- amplitudes are shift-blind." / "Step 4 --- phases are
  scale-blind." -> "Amplitudes are shift-blind." / "Phases are scale-blind.");
  "Step 5 --- put the two invariances together" -> "Putting the two
  invariances together"; "Step 6 --- shift and scale together: the joint
  error" -> "Shift and scale together --- the joint error"; "Step 6,
  continued --- the bowl and the training pull" -> "The bowl and the training
  pull"; "Step 7 --- the full profile: separated scatterers decouple" ->
  "The full profile --- separated scatterers decouple"; "Step 8 --- the
  recall paradox: MSE turns doubt into dimming" -> "The recall paradox ---
  MSE turns doubt into dimming"; "Step 8, continued --- why the reduced stack
  loses detections" -> "Why the reduced stack loses detections". The one
  in-prose cross-step pointer "dim drift feeds the recall paradox (step 8)"
  became "(two frames ahead)"; the % comment markers kept their 25c-25f codes
  but dropped the step wording; zero "step" occurrences remain in the file
  (11b_l1_vs_mse keeps its own independent step numbering by design). Builds
  clean, warning set unchanged.

- ERROR-THEORY TAIL FRAMES DELETED 2026-07-20 (user: "delete this slide Step
  8, the mechanism --- how training falls past the mean and the 2 next").
  Removed from sections/09_error_theory.tex: "Step 8, the mechanism --- how
  training falls past the mean" (25f-a2b), "Step 8, summed up --- the paradox
  side by side" (25f-a3) and "Step 9 --- the fix: the set-prediction gate"
  (25f-b) --- the section now ends at "Step 8, continued --- why the reduced
  stack loses detections", whose closing pointer "The gate (step 9) splits
  the two questions" was reworded to "The set-prediction gate splits the two
  questions" (the fix content lives in Act III set_prediction, which precedes
  this section; no other frame referenced the deleted three --- the "step 9"
  hits in 11b_l1_vs_mse are that section's own step numbering). The
  error-theory chain now runs steps 1-8. Full deck 126 -> 123 pages;
  sub_07 + full builds clean, warning set unchanged (09_error_theory:322 is
  the surviving recall-paradox frame's known overfull).

- ONE-GRID-PER-SCALE: BOLD PARAGRAPHS REMOVED, PAIR FIGURE ENLARGED
  2026-07-20 (user: remove the "U-Net --- one grid per scale" and "Local CNN
  --- scale 0 only" paragraphs too, and grow the left representation to fill
  the space). The left column is now lead sentence + pair equation + the
  pair->fringes->grid figure ONLY, back at \small; the figure was rebuilt
  ~1.6x: row pitch 0.65 -> 1.30, strips 1.8x0.38 -> 2.2x0.70, glyphs
  0.8x0.5 -> 1.2x0.8 (steps 0.05/0.10/0.40 = 24x16 / 12x8 / 3x2 cells, all
  exact), ladder span P->S_1 3.5, all label fonts \tiny -> \scriptsize, and
  the scale 0/1/3 labels moved from right of the glyphs to UNDER them
  (keeps the picture inside the 0.50 column). The architecture naming now
  lives only in the right column's chain headers; the frame title carries
  the claim. No overfull; sub_06 p11 rendered and checked.

- ONE-GRID-PER-SCALE LEFT COLUMN: INTRO PROSE SWAPPED FOR PAIR PHYSICS
  2026-07-20 (user: swap some text for equations/visualizations about track
  pairs contributing different resolutions that the U-Net matches in its
  layers). The five-line multi-scale intro paragraph is now: one bold lead
  ("Multi-scale by construction --- each track pair (P, S_n) sees the same
  relief z(x) at its own fringe rate"), the pair equation
  varphi_n(x) = (k_z^(n) - k_z^(P)) z(x) = k_n z(x), k_n prop b_perp,n
  (notation matches 02_signal_model's k_z^(i) and 09_error_theory's
  k_n prop b_perp,n), and a three-row tikz: baseline ladder (accent P at the
  bottom, S_1..S_3 dots at growing separation, |-| bracket labelled b_perp)
  -> fringe strip (10 / 5 / 1.25 periods, ratio 8:4:1) -> matching grid
  glyph (steps 0.05/0.10/0.40, labels scale 0/1/3), tiny column headers
  "track pair / its fringes / matching grid". S_n rows align top = longest
  baseline = finest fringes = scale 0, echoing the right chain's top panel.
  Column dropped to \footnotesize to fit; U-Net/local-CNN paragraphs trimmed
  (the "4x fewer cells per scale" clause folded away --- the div-4 arrows
  carry it). No overfull; sub_06 p11. FOLLOW-UP same day (user): the
  scriptsize representation-caveat footer ("A statement about
  representation, not about the error floor ... Pending: the
  capacity-matched U-Net vs local-CNN run") DELETED from this frame; the
  pending run itself is still described on the price-of-reach frame's
  footer and stays tracked there.

- GT COLUMN ISOLATED BY WHITE SEAMS 2026-07-20 (user: "different colors are
  touching in the tables that have a column GT"). In the five GT-reference
  tables (setC/setF/augC/ctxC stats tables + the 04 normalization stats table
  repC) the good!10 GT cells sat directly against the row/cell tints. Fix:
  white vertical rules in the column spec --- !{\color{white}\vrule width 2pt}
  on BOTH sides of the GT column in the four rowcolor stats tables (rowcolor
  paints the label cell, so the left side needs a seam too), and on the RIGHT
  side only in repC (its label column is unpainted; its blue/red cell groups
  were already separated by the existing !{\hspace} group gaps). Width
  reclaim where the added seams overflowed: ctxC seams 1.2pt + tabcolsep
  0.4->0.3pt + group gap 2->1pt (was 3.7pt over); repC seam 1.6pt + tabcolsep
  1.0->0.9pt (was 1.1pt over). Warning set back to baseline; pages
  21/35/41/45/61 rendered and checked; sub_03..sub_06 rebuilt. RULE: any new
  GT-column table gets these seams from the start.

- ONE-GRID-PER-SCALE FLIPPED TO TWO-COLUMN VERTICAL CHAINS 2026-07-20 (user:
  both visualizations vertical instead of horizontal, side by side in a right
  column, all text left). Was a full-width frame with two horizontal 4-panel
  chains interleaved between the prose blocks. Now columns 0.50/0.46: LEFT =
  intro + U-Net paragraph + local-CNN paragraph + representation caveat;
  RIGHT = one tikzpicture (scale 0.85), two vertical chains side by side
  under scriptsize headers "U-Net" / "local CNN", row pitch 1.7 (panel 1.2 +
  0.5 arrow gap). U-Net chain: grids coarsen 0.05/0.10/0.20/0.40, downward
  \div4 arrows, row labels anchored east of the chain (scale 0 + "a value per
  pixel", scale 1, scale 2, scale 3 + "two cells" as two-line align=right
  tiny nodes). CNN chain: four 0.05 grids, \times1 arrows, x pixel in the
  bottom panel, "scale 0 --- unchanged" under it, m-channel stack moved to
  the RIGHT of the bottom panel (horizontal arrow) with a two-line "m
  channels / at x" label under the stack. Panel content (argA!30 window,
  accent dashed B(x), soft border) unchanged. No overfull; page 57.

- ROW-TINT CONTRAST + NO-TOUCHING RULE 2026-07-20 (user: "I dont want distinct
  colors touching, and the colors are a bit too similar, choose colors with
  more contrast"). Two deck-wide changes to the painted results tables:
  (1) all \rowcolor tints unified and strengthened --- argA/argB/argC/accent
  at !22 (was a mix of !10/!12/!14/!16/!18), soft at !18 (was !15); hues
  unchanged (palette stable), cellcolor conventions (good!10 GT column, the
  04 trio's blue/red cells) untouched. (2) STANDING RULE: two adjacent table
  rows must never carry different tints --- every distinct-color junction now
  gets a thin white separator \addlinespace[2pt]. 15 junctions fixed across
  06 (head-stats slot blocks, levers-by-count recall), 08 (reach mini-table,
  input-by-count mu block), 10 (top-twenty rows 2/3, gain-table curve/param
  block boundaries x3, Hungarian-table p-L1 rows x3), 11 (ambiguity mini
  tables x2). Adjacency checked by script (0 remaining); warning set
  unchanged (pre-existing vboxes shifted line numbers only); pages 34/35/39/
  55/61/75/77/78/82 rendered and checked; sub_04..sub_08 rebuilt.

- "The price of reach" --- SECOND PASS 2026-07-20 (user: "keep each
  visualization in its own column, add a arrow for each line of comparison";
  mid-task: "this forced alignment with the = sign is not good, make the
  columns horizontally centralized instead"). SUPERSEDES the same-day merged
  single-tikzpicture entry below. Final form: the 0.59 column holds [t]
  minipages 0.43 trunk / 0.075 arrow strip / 0.485 pyramid; each grid figure
  went BACK into its own side minipage under its equations, given equal
  vertical bounding boxes (trunk useasboundingbox y +-1.128 = the pyramid's
  scale-0.88 extent) so both figures and their tiny captions sit on the same
  rows; a trunk-side vphantom of the pyramid fraction equalizes the third
  equation row and \par\vspace{0.06cm} before the trunk picture absorbs the
  residual (grid centers verified 305.0 vs 305.5 px at 110 dpi by measuring
  the render with PIL --- tune with that loop, not by eye). The arrow strip
  is ONE tikzpicture, baseline=(current bounding box.north), useasboundingbox
  (0,0)-(0.55,-4.85) so the drawn arrows never widen the strip: four flow
  arrows, one per comparison line --- y -0.55 (r row), -1.66 (P / m_ell row),
  -2.82 (m / cost row), and -4.77 spanning (-0.50)-(1.05) between the grids
  (THIRD PASS same day: user "make the last arrow a bit smaller" --- was
  (-0.85)-(1.40)). FOURTH PASS same day: the frame-top intro line ("One
  block --- two kxk convolutions ... two places to hang it") DELETED and the
  scriptsize symbol legend (r/B/m/P/station ell/D) MOVED from the frame top
  into the right column, raggedright ABOVE the reach plot (wraps to three
  lines in the 0.375 column); the comparison columns now start at the frame
  top. Equation blocks switched
  aligned -> gathered (every line + its soft annotation centred in its
  column; user rejected =-sign alignment for side-by-side comparisons).
  Rendered sub_06 p10; sub + full deck clean, no warning from this frame.

- BULLET-TO-ROW KEYING COMPLETED DECK-WIDE 2026-07-20 (user: "each bullet point
  ... coloring the numbers on the table it talks about"). Audit of all 26
  table+bullets frames. Already keyed and left alone: the 04 normalization trio
  (own blue-input/red-output cellcolor semantics), "Set prediction, controlled",
  "Augmentation, measured", "Ladder result", "every channel pays", the three
  benchmark-board frames. Left unkeyed with reason: Step 8 summed up, run-health,
  Winner vs baseline, Failure modes, backup detail --- their bullets are
  conceptual / cite numbers not in the table; painting would break the keying
  grammar. ELEVEN frames gained keying: rowcolor tints matched to the existing
  bullet squares on the four "by scatterer count" and four "predicted statistics
  vs GT" frames (rows painted = exactly the rows each bullet cites; accent!12
  for accent bullets, argX!14 otherwise, soft!15 for soft bullets that cite
  ties); "The two levers, re-measured" gained accent rows (curve R2 / PSNR /
  recall); the reach verdict gained bullet squares (argA azimuth / argC range /
  soft plateau) + painted mini-table rows; the ambiguity frame gained argB
  (sorted) / argA (set-pred) squares + row tints on both mini tables. In the
  stats tables \rowcolor precedes the row and the GT column's
  \cellcolor{good!10} correctly overrides it. Warning set identical to baseline
  (the old 08:448 8.6pt hbox even resolved); all 11 pages rendered and checked;
  sub_04..sub_07 rebuilt.

- "The price of reach --- one grid against a pyramid" --- ARROWS + ALIGNED
  FIGURES 2026-07-20 (user: "add a arrow from one to another, and the
  visualization on the bottom: make the pyramid one a bit smaller, and make
  both of them vertically aligned and add a arrow in them too"). The two
  0.295 trunk/pyramid columns were merged into ONE 0.59 column: the equation
  blocks now sit in [t] minipages (0.43 trunk / 0.075 arrow strip / 0.485
  pyramid --- the pyramid display is the wide one; 0.45/0.09/0.45 overfulled
  7.8pt) with a flow arrow between them (vspace* 1.55cm centres it on the
  equation stack), and the two grid figures moved into a SINGLE tikzpicture
  below both blocks: trunk grid unchanged at the origin, flow arrow
  (0.82,0)--(1.46,0), pyramid grid in a shift={(2.72,0)} scope at scale=0.88
  (uniform shrink keeps the station-doubling geometry exact; grid step and
  B(x) box scale with it), both centred on the shared y=0 axis so they are
  vertically aligned; the tiny captions became nodes under each figure
  (trunk y=-0.98, pyramid y=-1.42). Right plot column (0.375) untouched.
  Rendered sub_06 p10 clean; sub + full deck carry no warning from this
  frame (the 08_context_information:448 overfull visible earlier today was
  fixed by a concurrent session, not this edit).

- READING-THE-REACH GRID SCENE REBUILT 2026-07-20 (user: right-hand
  visualization "a bit strange" --- it drew x' adjacent to x with a large
  shared block, W off-centre, and a floating "one last shared pixel" outside
  the actual overlap). New geometry, cell = 1 px, boxcar 7x3 cells, patch
  14x6: B(x) (accent dashed) and W = 2w_a x 2w_r (ink dashed) both centred on
  the accent pixel x; x' (good pixel) is the far-corner pixel of W with its
  own centred B(x') (good dashed); B(x) and B(x') overlap in EXACTLY one cell
  (accent!65), callout "one shared look". W is aligned one cell tight on the
  x'-side so the corner pixel sits at offset (w_a-1, w_r-1) --- the last
  offset with a nonzero shared count --- rather than the zero-overlap w
  (even-width 2w windows cannot centre exactly; the tight side carries the
  message). Caption now reads "x' sits at the far corner of the patch: B(x')
  keeps a single shared look with B(x) --- one step further and the windows
  are disjoint". Dashed window borders frame the shared cell themselves (no
  ink border on it). Page 54; no overfull from 08_context_information.

- BULLET-COLUMN FONT SIZE UNIFIED ACROSS RESULTS FRAMES 2026-07-20 (user: "the
  font size is changing in the bullet points from slide to slide" on the
  table+bullets slides --- verified true). The two-column results family
  (bullets one column, painted table the other) mixed \small (6 frames) with
  \footnotesize (15 frames, including the standing three-table trio pattern).
  Standard is now \footnotesize for the bullet column of every two-column
  results frame. Converted: "Set prediction, controlled" (06), "Augmentation,
  measured" (07), "The verdict --- saturation at the reach" and "The verdict
  --- every channel pays" (08), "Step 8, summed up" (09, whole left column
  incl. prose/equations), "Hungarian at K=2" (10, incl. intro line).
  Full-width frames (centred table, bullets underneath: Winner vs baseline,
  Failure modes, run-health, ambiguity-survives, backup) stay \small ---
  a separate, internally consistent layout family. Warning set after rebuild
  is byte-identical to the baseline (no new overfull); six touched pages
  (33/43/55/59/71/78) rendered and checked; the five sub-decks sharing the
  touched sections (sub_04..sub_08) rebuilt.

- "Set-prediction lineage: DETR" --- FIGURE SWAPPED TO THE ORIGINAL MODEL
  2026-07-20 (user: "swap the visualization for the one used for the original
  model. We already have the adaptation in the next slide"). The right-column
  schematic no longer shows our gated head (backbone -> a,mu,sigma slot pills ->
  gate boxes -> Hungarian band), which duplicated frame 15a "Inside the
  set-prediction head"; it now shows the original DETR pipeline: CNN backbone
  box over a transformer enc.--dec. box (stacked left, flow down), N learned
  object queries pill feeding the decoder from below, three per-query
  prediction pills (c-hat_k, b-hat_k) with a tiny "class + box per query"
  header, query 3 annotated "c-hat_3 = empty-set: no object" in place of the
  old "gate closed", and the same crossing dashed-match grammar inside the
  argA Hungarian-assignment band (good output dots for matched queries,
  black!35 for the no-object one, accent GT 1 / GT 2 dots, empty-set target
  mid-band). Caption swapped from the gate blend formula to the DETR
  mechanism: absence is the explicit class empty-set, unmatched queries train
  onto it and leave the box loss. Bullets untouched (they already carry the
  lineage mapping: gate replaces the classification softmax). SUPERSEDES the
  2026-07-10 gate-glyph passes on this frame (entry retained below for
  history); the gated-head visualization survives on frame 15a. Local
  gateon/gateoff tikz styles dropped from the frame (unused). Rendered
  sub_05 p6 clean; full deck + sub_05 warning set unchanged (the
  06_set_prediction:492 1.4pt overfull belongs to frame 15a's sigmoid plot,
  pre-existing).

- GAMMA-NET ADAPTATION FRAMES ADDED 2026-07-20 (user: "add some slides talking
  about the adaptations needed"). Two new frames in sections/14_unrolled.tex
  between 39b (solver becomes the network) and 39c (training), markers
  39b2/39b3, grounded in the 2026-07-20 paper-fidelity audit of the
  implementation against GammaNet_Qian2022_2112.04211.pdf. 39b2 "Adapting
  gamma-Net --- a different unknown, a per-pixel operator": the section's
  split-background two-panel grammar (accent top = gamma-Net: single-look
  stack g in C^T -> fixed steering matrix R scene-wide -> learned coupled
  layer weights W^l -> complex sparse gamma-hat; good bottom = this project:
  coherences y + kz(x) -> exact per-pixel operator A(x) -> learned scalars
  alpha_l, theta_l + prox P_l -> nonnegative profile), white divider label "a
  fixed matrix can be learned --- an operator that changes at every pixel
  cannot", bottom line = the two inverted models g = R gamma + eps vs
  y = A(x) s with all symbols glossed in the scriptsize caption (first-order
  vs second-order data is the root cause: coherence averaging cancels the
  speckle phase, so the unknown becomes the real nonnegative power profile).
  39b3 "Adapting gamma-Net --- mechanism by mechanism": migration-style table
  (channel-table grammar from the per-channel normalization frame: rows =
  mechanisms, old column soft grey, = / -> arrow column, new column bold,
  why column) with a kept block (iteration skeleton, simulate-then-train) and
  an adapted block (unknown, measurements, operator, learned weights,
  shrinkage, model order); gamma-Net's support selection + piecewise-linear
  thresholding and BIC model-order selection are named and defined in the
  tiny caption (with the 12-stage vs default L=8 constant), colour legend
  green = exact physics / red = learned / grey = kept; accent punchline "the
  physics stays exact inside the network --- only the prior is learned".
  scriptsize, arraystretch 2.05, tabcolsep 3pt (4pt overfulled by 3.7pt).
  Deck 124 -> 126 pages, sub_09 11 -> 13; tectonic clean, no new overfull;
  pages 11-12 of sub_09 pdftoppm-verified.

- STATISTICAL-TIE BOLDING REVERTED 2026-07-20 (user: "revert this
  statistically tied part, its crazy now, use only the average"). Bold is
  back to MEAN-ONLY: the single best average per row, plus any column whose
  DISPLAYED mean exactly equals it (the original 2026-07-19 rule). The
  combined-seed-noise tie test is gone. Recomputed from scratch per row
  (scratchpad revert_ties.py) rather than un-bolding selectively: best-in-row
  tables use the row's arrow ($\uparrow$ = max, $\downarrow$ = min, sub-rows
  inherit their metric's arrow); closest-to-GT tables (repC/setC/setF/augC/
  ctxC) use min |mean - GT| against the \cellcolor{good!10} value, GT ---
  rows stay unbolded; plain-number spread rows (setD/augA, not \seedpm) were
  never touched by any pass and keep their single bold. Bold-cell counts
  dropped to ~1 per row (e.g. setE 60->21, setD 52->21). Every caption's
  bold definition had "or tied within the combined seed noise" stripped
  (15 captions across 04/06/07/08). Font stays \tiny and definitional, patch
  line stays (previous two entries unaffected). VERIFY: full deck + sub_03/
  04/05/06 rebuilt, warning set byte-identical to the pre-change baseline;
  pages 19/39/43/61 rendered and every row hand-checked against mean-only
  (incl. surviving exact ties SSIM-elev 0.958/0.958, levers peak-median 0.67
  x4, aug p95 15.44/15.44).
- SEEDPM CAPTIONS REWRITTEN 2026-07-19 (user, third pass on the seedpm tables:
  "the text under each table, some are discussing results thats wrong, and
  they are big so make a smaller font, and add which patch size was used").
  All 15 seedpm-table captions rebuilt from shared fragments (scratchpad
  captions_v2.py): (1) \scriptsize -> \tiny; (2) RESULT SENTENCES REMOVED —
  captions now carry ONLY the runs recipe, cell definition, bold/Delta
  definitions, colour legends and metric/unit definitions; deleted claims of
  the form "only the B/+AB recall drops clear it", "every Delta is grey ---
  none clears...", "only the two sigma width deltas clear..."; the rep-table
  colour keys became neutral ("blue = input-rung gains; red = output-rung
  gains"), the Hungarian-2x2 key became block-based; the "--- below-GT =
  regression toward the mean" and "--- the reliability of the claimed count"
  interpretation tails and setA's stale mu/sigma units line were trimmed;
  (3) every runs recipe gained "patch $64{\times}32$" (az x rg) — verified
  from the repo, not assumed: configuration/training/general/run.py
  patch_size (64,32) stride (32,16) since commit 4408ce1 (07-14 20:15); all
  five experiment batches (Normalization, Heads, Input, Augmentation,
  New_presence in results/) ran 07-16/07-17 and none of their trial planners
  override patch_size (only Reach and PatchSize planners do, experiments.py).
  The smaller captions returned the height the tie-clause had cost, so the
  compensations were rolled back: arraystretch repC 1.02, setD 0.86,
  augA 0.78; vspace repC/setC/augB 3pt, augA 1pt (setD stays 0pt). VERIFY:
  full deck + sub_03/04/05/06 rebuilt, warning set byte-identical to the
  pre-change baseline, pages 19/39/43/61 rendered and checked. OPEN
  INCONSISTENCY flagged to the user, NOT edited: the Act II I/O frame
  ("Input and output representation", 04 lines ~13/40) still says
  "$64\times 64$ patches / stride 32" — stale square-patch generation,
  contradicts the (64,32)/(32,16) config the runs used.
- SEEDPM TABLE POLISH 2026-07-19 (user: "bold every column tying at the best;
  align the +- symbols vertically; leave the no-+- tables alone"). (1) \seedpm
  redefined in preamble.tex on eqparbox (new package): mean right-aligned in
  \eqmakebox[\seedtag M], std left-aligned in \eqmakebox[\seedtag S], so every
  +- in a table sits on one vertical line regardless of negative means
  ($-4.52$) or magnitude changes across rows (0.99 vs 10.90). Each of the 14
  seedpm tables sets its own group via \renewcommand{\seedtag}{...} directly
  before the tabular (tags repA-C / setA-F / augA-C / ctxA-C) — a NEW seedpm
  table must add its own tag line or it joins group "g" and inherits foreign
  widths. (2) Tie-bolding, STATISTICAL (user follow-up same day: "the best
  number does not need to be exactly the same — given the std they should be
  statistically tied"): in every seedpm row, bold marks the best cell AND
  every cell tied with it within the combined seed noise, tie test
  |mean_i - mean_best| <= sqrt(std_i^2 + std_best^2) on the printed values;
  closest-to-GT tables compare distances to the GT instead
  (||m_i - GT| - |m_best - GT|| <= same bound); rows with no reference
  (GT ---) stay unbolded. Applied by script (first pass bolded the 20
  exact-display ties, the statistical pass added ~190 more cells across
  04/06/07/08). Consequence accepted by design: tables whose slide message
  is "nothing clears seed noise" (augmentation pair, levers +A arms) are now
  bold almost everywhere, and rows where only B/+AB clear the noise keep
  exactly the none/+A pair bold — the bolding literally encodes the
  captions' clears-the-noise claims. Every seedpm caption's bold definition
  gained "or tied within the combined seed noise". (3) The box alignment
  widens stats-vs-GT-style tables whose widest mean and widest std live in
  different rows — absorbed via tabcolsep: repA 2.6->2.4pt, repC 2.2->1.0pt,
  ctxA/ctxB 1.2->1.1pt, ctxC 1.5->0.4pt. (4) The longer captions cost one
  wrapped line on five full frames — reclaimed via caption vspace 3->0pt
  (repC, setC, augB), arraystretch repC 1.02->1.01 / setD 0.86->0.83 / augA
  0.78->0.74 (below ~0.9 the content height floors the row, diminishing
  returns), and two caption trims: setD dropped the "(the median ties at one
  bin in all four arms)" parenthetical (the all-bold row now shows it), augA
  second "clears the combined seed noise" -> "clears it". VERIFY: full deck +
  sub_03/04/05/06 rebuilt; warning set byte-identical to the pre-change
  baseline; pages 19/21/33/39/40/43/61 rendered and eyeballed. Tables without
  +- (ladder, three-rungs, migration, verdict) untouched — old generation,
  slated for replacement.
- STANDARD CONFIG MOVED TO HUNGARIAN, B REVERTED 2026-07-17 (user decision after
  the Hungarian presence refresh below: "flip PARAM_MATCH_FULL to hungarian and
  revert the B default"). Code (commit d7bd07e): AblationCatalog.PARAM_MATCH_FULL
  SORTED_GT -> HUNGARIAN and default_curriculum presence_balance True -> False in
  both stages (active normalization stays True) — standard is now set-pred ·
  Hungarian · +A. Also updated: equation_library.py matching-note default
  attribution, benchmark run-name test literals sorted_gt -> hungarian. Tests:
  targeted suites green (configuration/backbone_training/pipelines_shared/
  loss_metrics/tuning/benchmark/webui/entrypoints/models_backbone/models_baseline);
  two pre-existing failures NOT from this change: reach-scheduler size match at
  head set_pred (known "set_pred breaks the match", fails on clean HEAD) and two
  cube-explorer DEM tests owned by a concurrent session's uncommitted work. Deck:
  frame 18's accent bullet is now "Adopted --- +A is the standard, B reverted"
  (B stays in the toolbox for the K=5 tail, untested); honesty guards kept.
  Rebuilt sub_05 + full deck, warning set unchanged, page 14 re-rastered.

- PRESENCE LEVERS RE-MEASURED UNDER HUNGARIAN MATCHING 2026-07-17 (user: "new
  results, the old ones were using set-pred sort, but I moved to set-pred
  hungarian. Update the tables and the bullet points"). Deck stays 124 pages —
  the three presence frames 18/18b/18c updated in place (tables, bullets,
  captions, recipe lines sorted-GT -> Hungarian); the lever mechanism frames
  17a/17b untouched.
  - DATA: results/New_presence (generated 2026-07-17 09:26; the sorted-GT
    results/Presence dir is gone): none/A/B/AB on unet-set_pred-hungarian-K_2-hv,
    five seeds per arm, all cells mean ± seed std from metrics_comparison.md
    (cells regenerated by script; bold = best-per-row / closest-to-GT recomputed
    on the raw means, not the displayed roundings).
  - STORY CHANGES — the 07-16 sorted-GT verdict does NOT carry over: (a) A stays
    a free no-op (every gap <= 0.7x combined seed noise); (b) B is NO LONGER
    FREE — matched recall 0.893 -> 0.879 (B) / 0.882 (+AB) at 1.6-1.75x the
    combined seed noise, k=2 recall 0.744 -> 0.719/0.721 (1.6-2.1x), k=2
    precision down (1.2-1.5x), and B leans the whole curve block down (R2
    0.578 -> 0.553, PSNR -0.25 dB, ~1.2x; +AB just inside); (c) the "B halves
    the seed spread" stability gain is GONE — spread rows mixed (B tightest on
    curve R2 .011, none tightest on cos med .033); titles now 18 "A stays free,
    B now costs recall", 18b "B's cost lives in the second scatterer", 18c "the
    head sets the distribution, B trims the empties"; (d) mechanism on 18c: B's
    only footprint past noise is the inactive side — slot-1 leakage 0.39 -> 0.34
    (+AB), inactive sigma down, tail slot firing less — the same suppression
    that costs k=2 recall on 18b.
  - UNCHANGED: both honesty guards kept (K=5-tail-untested scope bullet — now
    phrased "priced insurance" at ~1 pt matched recall; no val rows, the levers
    change the loss). The "+AB stays the standard configuration" bullet kept
    because configuration/training/backbone.py still defaults both levers True —
    but the new data argues that default deserves a revisit, and
    AblationCatalog.PARAM_MATCH_FULL (general/ablation.py:12) is STILL SORTED_GT,
    so the code-side standard has not actually moved to Hungarian; both flagged
    to the user.
  - FIT: frame 18 bullets to footnotesize / itemsep 4pt (four bullets now);
    arraystretch 0.93 -> 0.86 (18), 1.12 -> 1.11 (18b), 0.98 -> 0.95 (18c);
    frame-18 caption vspace 3pt -> 0pt.
  - VERIFY: tectonic sub_05 + full deck — warning set = the same pre-existing
    five (04:48, 04:149, 05:260, 06:495, 08:448) + the untouched 09/10/11 ones;
    sub_05 pages 14-16 rastered and inspected.

- PATCH-SWEEP VERDICT REBUILT ON THE TWO-AXIS FIVE-SEED SWEEP 2026-07-17 (user:
  "updated results of the patch size sweep... sweeps 2 axis now, and does not vary
  the number of passes... maybe a heatmap of loss"). Deck stays 124 pages — frame
  24c replaced in place, everything else in the patch region (24b correlation law,
  24b2 reading-the-reach, 24d/24e) untouched and grep-checked free of old-sweep
  numbers (24d line ~648 references the reach concept only).
  - DATA: results/Patch (added by the user 2026-07-16): 5x3 per-axis grid, az
    {16,32,48,64,80} x rg {16,32,48}, run names ...-pAAAxRRR on the 17sartom-traun
    scene with the 26x12 boxcar, FIVE seeds per cell, input stack FIXED (the old
    square-W x track-count n=2..15 single-seed sweep is superseded). Cell metric =
    best val loss mean ± seed std from overview.md. Grid: best 64x48 (0.14156 ±
    .0006); statistical ties inside combined seed noise = 48x48 (0.14168), 80x48
    (0.14165), 64x32 (0.14209); worst 16x16 (0.15336). Az marginal at rg=48:
    16->48 = -0.0107, 48->64 = -0.0001, 64->80 = +0.0001 (knee brackets 2w_a=52).
    Rg marginal: 16->32 ~ -0.003, 32->48 <= 0.002 (clears seed noise at az=48/80,
    inside it at 64) — an honest small drift past 2w_r=24, called out in the soft
    footer.
  - HEATMAP: new generator presentations/full_project_story/patch_sweep_heatmap.py
    (same class/Logger/deck-palette pattern as joint_error_heatmap.py; runs in the
    Dune env, parses overview.md by run-name pAAAxRRR regex) ->
    results/Patch/patch_sweep_loss_heatmap.png. Axes in PIXELS (cells 16 wide) so
    the cells sit at true pixel positions; rose ramp dark=high loss, per-cell annotations
    (adaptive white/ink): loss mean, ±std, "% of reach" = share of the centred
    2w_a x 2w_r box (52x24) the patch covers, and "% outside" = share of the
    patch beyond that box (both added 07-17 on user request; the loss plateau is
    exactly the two 100%-of-reach columns), good-green outline on the best cell
    + dashed green on the tie set (computed as mean-best <= hypot(std, best_std)). Re-run the script
    UPDATE 07-17: the 2w_a/2w_r reach lines were REMOVED from the heatmap at user
    request (values remain in the frame's bullets + registered/observed table).
    if results/Patch is regenerated.
  - FRAME 24c: retitled "The verdict --- saturation at the reach, axis by axis"
    (was "...for every track count"). GONE: the six n-curves tikz plot, the
    knee-flat-in-n table row, the pooling-pyramid multiples-of-8 aside (its
    admissibility story belonged to the old square grid; the new grid is
    multiples of 16), and the single-seed ±0.005 caveat (five seeds now, std <=
    .0013). NEW: heatmap left (0.60) + runs-recipe caption (kept deliberately
    minimal — patch = cell, scene + boxcar from the dataset name; the model
    recipe is NOT in the run names, do not guess it); right = three bullets
    (azimuth pays then stops / range clears early / plateau corner = reach
    corner) + registered-vs-observed table (az knee 52 -> observed 48-64; rg
    knee 24 -> observed 32) + the range-drift honesty footer. Boxcar numbers on
    the frame remain the standing four (26x12, 52, 24) — nothing new invented.
  - VERIFY: tectonic sub_06 + full deck — warning set unchanged (only the
    pre-existing five + 09/10/11); sub_06 page 9 rastered and inspected.

- FOUR NEW METRIC FAMILIES ACROSS ALL FIVE RESULTS SETS 2026-07-16 (user: audit of
  "am I using all the important inference data" surfaced four unused families;
  "add 1,2,3 and 4, in all 5 experiments using the results dir data"). No new
  frames (deck stays 124 pages); every experiment section's standing three tables
  gained rows, all values from results/{Normalization,Augmentation,Heads,Presence,
  Input} per-trial seed reports (mean ± seed std, ddof=1). The additions are now
  PART OF THE STANDING 3-TABLE PATTERN — carry them in any future results refresh:
  - TABLE 1 (verdict) +3 rows: "peak err median/p95" (pixel_peak_err_units_
    median/p95_gt) as sub-rows of peak err mean (25b, which never had a mean row,
    got them as full rows), and "profile cos med" (pixel_cosine_gt_median). WHY:
    the peak-err mean is tail-dominated — median tells the typical pixel, p95 the
    tail. Killer numbers this exposed: normalization base rungs have cos med
    EXACTLY 0 and peak median 13.7 (the typical pixel misses entirely before the
    output rungs; median snaps to 0.9 bins / cosine 0.85 at the last rung —
    12b's accent bullet now says so); the set-pred gate collapses the peak p95
    tail 15.4 -> 6.0 (15b bullet extended); input: the phi arms own cos med
    (0.93/0.91 vs full 0.89) and the p95 tail (11.7-14.0), so 25b's bullet 1 was
    reworded "every curve metric" -> "every curve-error metric" (cosine is now a
    documented exception owned by phi — grep found no other frame quoting the old
    absolute claim).
  - TABLE 2 (by scatterer count) +2 rows: "count acc | pred k" (count_acc_pred1/2
    from metrics.py = share of pixels predicting exactly k active slots whose GT
    count is k — the PREDICTION-side reliability the GT-side reading guide's
    count_acc_gt{k} caveat does not cover). Caption sentence defining it added on
    every set. Stories: reliability of a claimed single is 0.95-0.98 everywhere;
    a claimed second scatterer is right 0.63-0.65 under the gate vs 0.73 conv
    (15c bullet), 0.69-0.72 at every normalization rung (12c bullet), 0.77 with
    phi vs 0.65 |A|-only (25c bullet).
  - TABLE 3 (predicted stats vs GT) +7 lines: "distribution width — std over
    GT-active pixels" block, rows a/mu/sigma x slot 0/1 (slot_{s}_{amp,mu,sig}_
    active_pred_std vs the gt_std in the green GT column; bold = closest to GT,
    same rule as the means). WHY: regression to the mean is invisible from the
    means — every arm in every set predicts widths at ~30-60% of the GT (slot-1
    mu spread 7.1-10.9 vs 16.2). Set stories: output rungs widen mu but narrow
    amplitude (12d, new 4th bullet); Hungarian arms marginally wider (15d, new
    4th bullet); levers identical here too (18c); widths grow with the input
    stack, phi-all/full widest (25d). Augmentation 20b3: the two sigma-width
    deltas are the ONLY deltas in the whole aug set that clear combined seed
    noise (flips narrow sigma, slightly AWAY from GT) — the "every Delta is
    grey" footer is now "every mean Delta is grey", colored accent on those two.
  - Presence extras: frame 18's argC seed-spread block gained a "profile cos med"
    row (±.034 -> ±.018, B/AB halve it — added to the stability bullet); peak
    median ties at exactly one bin (0.671141 ± 0) in all four arms — noted in the
    caption, no bold on a 4-way exact tie. Two-way exact ties (set-pred peak
    median in 15b, phi arms in 25b) carry bold on BOTH cells.
  - LAYOUT: paid for the new rows with arraystretch (12b 1.22->1.08, 12c ->1.00,
    12d ->1.02, 20b 0.92->0.78 + addlinespace[1pt] + caption tightened, 20b2
    1.12->1.08, 20b3 ->1.00, 15b unchanged 1.1, 15c 1.12->1.05, 15d ->0.95,
    18 1.12->0.93, 18b unchanged, 18c ->0.98, 25b 1.1->0.98 + tabcolsep 1.2pt,
    25c 1.04 + 1.2pt, 25d ->0.90 + addlinespace[1.5pt]). GOTCHA: below stretch
    ~0.9 the \seedpm rows stop shrinking (the ± strut binds) — reclaim from
    captions/bullets instead. GOTCHA: the 25b/25d vbox overfulls were the BULLET
    column, not the table (stretch changes left the overfull byte-identical) —
    tightened bullet wording + itemsep, not the tables. Verify: all four
    sub-decks + full deck rebuilt, warning set = the same pre-existing five
    (04:48, 04:149, 05:260, 06:495, 08:448) + the untouched 09/10/11 ones; pages
    5-7 (sub_03), 6-8 (sub_04), 8-10+14-16 (sub_05), 13-15 (sub_06) rastered and
    inspected.

- +AB ADOPTED AS THE STANDARD CONFIG 2026-07-16 (user decision after the presence
  refresh: "make it AB the standard config"). Code: BackboneEntryConfig
  backbone_head conv -> set_pred and default_curriculum presence_balance -> True
  in configuration/training/backbone.py (active normalization already True,
  matching already sorted via PARAM_MATCH_FULL) — standard is now set-pred ·
  sorted-GT · +AB (commit 4947275, tests green incl. state_dict baseline; the
  patch_sweep / cross_validation / jepa harness entries keep their own conv
  defaults, flagged to the user). Deck: presence frame 18's scope bullet became
  the accent "Adopted — +AB is the standard configuration" bullet (free at K=2,
  levers armed for the K=5 tail); title and data verdict unchanged — the DATA
  says all arms tie and B halves the spread, the ADOPTION is a decision, keep
  the two distinguishable on the slide. Rebuilt sub_05 + full deck, page 14
  raster inspected; warning set unchanged.

- HEAD 2x2 + PRESENCE LEVERS FIVE-SEED REFRESH 2026-07-16 (user: "keep going" after
  input; the campaign's last two results sets). Deck 120 -> 124 pages, sub_05
  11 -> 16 (frames 12 -> 15). Data = results/Heads (head x matching 2x2, UNet,
  hv flips, presence none, param-MSE) and results/Presence (none/A/B/AB on
  set_pred-sorted_gt). Cross-arm identities: hm-conv-sorted_gt == aug-on ==
  in-amp-redsec-ifg (0.14202); hm-set_pred-sorted_gt == pr-none (0.14533).
  - HEADS, frame 15b rebuilt (title kept "the head is the lever, not the
    matching"): detection lift confirmed and stronger on five seeds (recall
    0.64 -> 0.89, k=1 recall -> 1.00, count-exact 0.59 -> 0.85, under 0.33 -> 0.02,
    peak err 4.7 -> 1.7). STORY CHANGES vs the old single-seed frame: (a) the
    "conv+Hungarian breaks without the gate" grey column is GONE — conv-Hung ==
    conv-sort on every detection metric (0.642 vs 0.641 recall) and conv-Hung is
    now the BEST curve arm (R2 0.603, MAE 0.0471, PSNR 50.7, SSIM elev 0.9601);
    (b) within BOTH heads Hungarian leans better on the curve (+0.03-0.04 R2);
    (c) the gate's price is explicit: over-count roughly doubles (0.08 ->
    0.13-0.14) and k=2 components worsen (mu 4.05 -> 4.5, sigma 1.51 -> 1.70)
    while a improves (0.37 -> 0.31; k=1 0.16 -> 0.11). New companions 15c
    (by-count: gate saturates k=1, layover pays) and 15d (stats vs GT: gate
    calibrates activity 0.68 -> 0.99 + cuts leakage 0.42 -> 0.30 but over-fires
    slot 1 (0.38-0.39 vs GT 0.26) and undershoots active amplitude on both slots
    (slot 0: 0.58-0.61 vs GT 0.77 where conv overshoots 0.87-0.90) — "the gate
    calibrates activity, pays in amplitude").
  - PRESENCE: the old "two levers, full 12-run grid — A helps, B over-corrects"
    frame (old single-seed generation, head x matching x lever) is REPLACED by
    three frames on the new 4-arm data: 18 "re-measured on the gated head —
    accuracy no-ops, B buys stability" (all four arms tie on every held-out
    metric; the one real effect: presence balance B halves the reconstruction
    seed spread, R2 ±.026 -> ±.010, SSIM range ±.004 -> ±.001 — spread rows
    argC-painted like the augmentation frame), 18b by-count (no split moves;
    gate shape lever-independent), 18c stats vs GT (four identical
    distributions; the gated head's signature — slot-0 amp undershoot, slot-1
    over-firing, low leakage — is unmoved by any lever). HONESTY GUARDS on 18,
    do not remove: (1) scope bullet — K=2 imbalance is mild, the K=5 tail slots
    where collapse lives are UNTESTED here, the levers stay in the toolbox (the
    lever mechanism frames 17a/17b stay as-is); (2) val losses are NOT
    comparable across lever arms (the levers change the loss itself) — no val
    row on these tables. The old grid's conv+AB collapse claim is untested in
    the new arms (no conv rows) and is gone with the frame.
  - VERIFY: tectonic sub_05 + full deck — only the pre-existing 1.39pt overfull
    at 06:495 (gate-blend-match frame, untouched) in the section; full-deck
    warning set unchanged at 20; pdftoppm rasters of sub_05 pages 8-10 + 14-16
    inspected (headers, tints, bolds, GT columns as designed).

- INPUT ABLATION REFRESH, 4->6 ARMS, TWO FRAMES BECAME FOUR 2026-07-16 (user: "now
  the input results, in this one, i added 2 more experiments, so not only the table
  needs to be updated"). Deck 118 -> 120 pages, sub_06 13 -> 15. Data =
  results/Input, six arms = a modality x track-scope grid: {|A| only, phi only,
  |A|+phi} x {reduced 5 tracks, all 29 tracks}; the two NEW arms are
  noamp-allsec-ifg (28 phi) and amp-allsec-ifg (full 29|A|+28phi stack). Config
  truth: configuration/training/backbone.py _default_input_trials;
  "all" = every secondary in the baselines table.
  - TRACK-COUNT CORRECTION: the old frames said "30 |A|" / "all 30 track
    amplitudes" — the baselines table (test_data/meta/baselines.json, the
    production twin) holds 29 tracks total (primary + 28 secondaries), matching
    the patch-note N=29. All labels now read 29 |A| / 28 phi / full 57-channel
    stack; 09_error_theory's "30 of them are as blind as 5" fixed to 29. If the
    user knows a reason the old 30 was right, revisit.
  - 25 SETUP recut: the four-card figure + hypothesis axis became a 3x2 card grid
    (rows |A| only / phi only / |A|+phi, columns reduced/all; card grammar kept —
    grid square = |A| left, ring square = phi right, accent X = removed modality,
    accent border = reference) + hypotheses H1/H2 kept, H3 (tracks pay everywhere;
    can diversity substitute?) added. The old on-axis hollow-marker strip is GONE
    (results are one frame later).
  - 25b VERDICT retitled "every channel pays on the curve, phase still places the
    scatterers": every curve metric peaks on the FULL stack (R2 0.667, PSNR 51.4,
    MAE 0.043, val 0.122 vs reference 0.569/50.3/0.050/0.142) -> H2+H3, the
    9-channel reference underuses the stack; 4-phi still tops recall/F1/count
    (0.82/0.82/0.80) and amplitude still dilutes detection; 28-phi alone
    out-trains the reference on val loss (0.137 vs 0.142). Diversity does NOT
    substitute for modality. Factorial two-line header with cmidrule groups
    (|A| only / phi only / |A|+phi over red./all) — reuse this header for any
    6-arm table. Old component-error rows moved out to 25c.
  - 25c NEW (by scatterer count, standing pattern): phi owns detection at every k
    (4-phi takes k=1 recall 0.88, 28-phi takes k=2 0.745 + precision — the two phi
    arms trade k=1 against k=2); mu/sigma best in phi/full arms but the amplitude
    a needs |A| channels (phi-only a MAE 0.45-0.46 vs ~0.35) — each parameter
    reads its own modality; full stack = best k=2 localiser (mu 3.74, sigma 1.38,
    a 0.52).
  - 25d NEW (predicted stats vs GT): phi-only pushes slot-0 activity closest to
    the every-pixel truth (0.90/0.82) + closest slot-0 a and mu; the FULL stack is
    closest on every slot-1 active stat (frac 0.289, a 1.15, mu 10.79, sigma 3.18)
    — phase fires the slots, the full stack calibrates layover; leakage lowest in
    the reduced arms (0.42-0.44).
  - LAYOUT: three tables started 19-22pt overfull; fixed by widening the table
    columns (0.58->0.62 / 0.63->0.66 / 0.66->0.70), narrowing bullets, tabcolsep
    1.7/1.7/1.5pt. GOTCHA: sequential python replacements collided — resizing
    25c's table header made it byte-identical to 25d's; order edits so patterns
    stay unique. Only the pre-existing 448-line overfull remains in section 08;
    full-deck warning set unchanged at 20.
  - PENDING, IMPORTANT: 09_error_theory (step-8 collapse frames + the L1-vs-MSE
    limits frame region) still quotes the OLD single-seed tracks-experiment
    generation: reduced-vs-phi-only recall 0.66/0.87, precision 0.85/0.82 (the
    new five-seed data FLIPS precision: 0.821 vs 0.830, a noise-level tie),
    active slots 1.01/1.30, under-count 0.31/0.05, a-MAE 0.420, and the derived
    "activity-only recall cap ~0.75/~0.96" whose recomputation method is not in
    the results files. Update needs a deliberate pass (new gap is 18 recall
    points, 0.64 vs 0.82; the qualitative argument survives). 00_front outline
    now says "input ablation" instead of "amplitude-only".
  - VERIFY: tectonic sub_06 + full deck; pdftoppm rasters of sub_06 pages 12-15
    inspected (card grid, factorial headers, bolds, GT column).

- AUGMENTATION FIVE-SEED REFRESH + TWO NEW FRAMES 2026-07-16 (user: "lets keep this
  3 tables pattern its very good. Now lets move on to update the augmentation
  results"). Deck 116 -> 118 pages, sub_04 7 -> 9. Data = results/Augmentation
  (aug-on = hv flips vs aug-off = noaug; five seeds each; NOTE aug-off IS the
  normalization endpoint run nrm-4-out_sigma — identical numbers, keeps 12b/20b
  cross-consistent at val 0.144 / recall 0.656 / count-exact 0.611).
  - STORY CHANGED by seed averaging — the old single-seed claims do NOT survive:
    "every curve metric rises" and the recall/precision trade (0.679->0.652 etc.)
    are now inside the five-seed spread. What survives: (a) the training effect —
    val-loss knee epoch 26.2 -> 43.8 (1.7x, was "24 -> 49, 2x") and best val loss
    0.1443 -> 0.1420 (~2x combined seed noise); (b) NEW headline: flips roughly
    HALVE the seed-to-seed std (recall ±.056->±.029, count-exact ±.066->±.034,
    peak ±.86->±.47, pixel R² ±.050->±.031; k=1 recall ±.088->±.046; slot-1
    leakage level ±.038->±.007). Frame 20b retitled "flips delay the overfit and
    halve the seed spread"; groups now training (argA) / reconstruction + detection
    ties (soft) / seed-to-seed spread (argC, the stds themselves as rows, Δ in %).
  - Δ-COLOR RULE for two-arm seed-swept tables: Δ = flips − no aug is colored
    good/accent ONLY where |Δ| clears the combined seed noise sqrt(s1²+s2²),
    grey otherwise; footer says so. On these three frames only the two training
    deltas clear it.
  - RECIPE LINE CHANGED: the new runs are presence NONE on the full norm ladder
    (run names ...-hv-none-... / ...-noaug-none-...); the old caption's
    "active-norm" was dropped — do not reintroduce it.
  - 20b2 NEW (by scatterer count, the standing 3-table pattern): all deltas grey;
    directions only (flips lean precision + k=2 recall/μ; no-aug leans overall
    recall + σ/a); spread halves at every detection split. 20b3 NEW (predicted
    stats vs GT, 12d layout + Δ column): calibration unchanged (no-aug marginally
    closer on slot 0, flips on slot-1 frac/μ/leakage, all inside noise); flips pin
    the distribution down. No blue/red rung tints here — two arms, not a ladder;
    rows painted by argument only on 20b (deck results style).
  - 20c (loss curves) LEFT AS-IS: its two pngs are the OLD single-seed generation's
    training curves (results/augmentation_experiment, restored from Trash). Their
    qualitative story (val flattens early vs keeps descending) matches the new
    best-epoch data (26 vs 44), but regenerate from a new-run tensorboard export
    when available.
  - VERIFY: tectonic sub_04 + full deck — only pre-existing warnings remain
    (05_stability:260 8.2pt is pre-existing; full-deck set unchanged at 20); a
    20.4pt overfull vbox on 20b was cleared via arraystretch 1.05->0.92 + caption
    skip trim; pdftoppm rasters of sub_04 pages 6-8 inspected (groups, tints, Δ
    greys as designed). No other section quotes the old aug numbers (grep
    0.135/0.605/48.85/epoch 24/epoch 49 clean).

- NORMALIZATION FIVE-SEED REFRESH + TWO NEW FRAMES 2026-07-16 (user: "new data of
  the experiments... each trial was averaged on 5 seeds, so the results are a
  average and a std. Update the normalization part first"; follow-ups: "always
  bring precision and recall, segmented by slot, the component errors too", "keep
  the pattern of blue and red — improvement caused by input norm blue, by output
  red", "add a third table, with the comparison of the prediction averages for
  active/inactive of each param vs the gt stats"). Deck 114 -> 116 pages, sub_03
  5 -> 7.
  - DATA SOURCE: results/ was REPLACED wholesale by markdown-only five-seed
    comparison trees (Augmentation/Heads/Input/Normalization/Presence; per-trial
    seed-comparison reports + metrics_comparison.md + overview.md, generated
    2026-07-16). Every number on 12b/12c/12d is the across-seed mean of five
    seeded replicas from results/Normalization/metrics_comparison.md (val loss +
    best epoch from overview.md); the ± term is the seed sample std (ddof=1).
  - \seedpm{mean}{std} macro added to preamble.tex (mean + tiny soft ± std). USE
    IT for every seed-swept number in the coming section refreshes.
  - 12b (ladder) renumbered: the five-seed base is stronger than the old
    single-seed run (curve R² 0.51 vs 0.44), so the input-rung lifts are smaller
    but consistent; the output rungs remain the movers (recall 0.35->0.66,
    count-exact 0.25->0.61, peak 10.7->4.5, val loss -25% to 0.144, best epoch
    49->26). Curve-MAE best-in-row moved to +out σ (0.0501 vs 0.0504, inside seed
    noise — bold follows the source file's unrounded best); that row shows 4
    decimals so the bold is legible. tabcolsep 6 -> 2.6pt paid for the ± terms
    (2.5pt overfull at 3pt).
  - 12c NEW (by scatterer count): precision/recall + matched μ/σ/a errors,
    overall and split k=1/k=2. STANDING CONVENTION (user): every experiment
    section brings precision + recall segmented by count and the component
    errors. Story: the gains concentrate on k=1 pixels (recall 0.18->0.65, μ
    1.25->0.63); k=2 barely moves and its amp error even drifts up (0.55->0.59).
    Blue/red tints follow 12b semantics exactly (user insisted): blue = rows the
    input rungs improve (precision, μ, σ), red = cells the output rungs improve
    — deliberately ABSENT on the μ/a k=2 rows where the output rungs worsen the
    metric. Footer carries the precision-rewards-under-prediction caveat from the
    results reading guide.
  - 12d NEW (predicted stats vs GT): per-slot prediction means over GT-active /
    GT-inactive pixels against the GT statistics (green GT column, good!10).
    Conditioning is the slot's GT activity mask (slot-0 global==active in the
    source confirms it). Story: slot 0 calibrates (active frac 0.34->0.71 vs GT
    every-pixel, amp overshoot 2.4x -> 1.2x, μ/σ close on GT); slot 1 halves its
    leakage (inactive a 0.74->0.45 vs ~0) but active a/σ drift below GT — the
    k=2 softness of 12c seen from the distribution side. bold = closest to the
    GT column; inactive GT slots define no μ/σ reference (—). Width paid by
    label trim, tabcolsep 2.2pt, output-group sep 2pt (14.2pt overfull before).
  - IMAGE RESTORE: the results replacement had deleted every deck-referenced
    figure (13 files) — no deck could build. Restored by COPY from
    ~/.local/share/Trash/files into results/ (preprocessing_inference/
    channel_distributions, error_landscape, param_extraction incl.
    label_distributions, augmentation_experiment pngs). Consider moving deck
    figures into the presentation tree if results/ keeps being regenerated.
  - PENDING: 07_augmentation still quotes the old single-seed normalization
    endpoint (val 0.138, recall 0.679, count exact 0.615) — refresh it from
    results/Augmentation in that section's pass.
  - VERIFY: tectonic sub_03 + full deck clean of new warnings (only the two
    pre-existing section-04 overfulls at lines 48/149; full-deck warning set
    unchanged at 20); pdftoppm rasters of sub_03 pages 5-7 inspected — 12b ±
    terms legible, 12c/12d tints and bolds as designed.

- WHY-THE-U-NET PAIR ADDED 2026-07-16 (user: "add some slides, showing with a math
  model and tomosar theory, why the UNET outperforms a simple local CNN"). Two frames
  between 24c (sweep verdict) and 25 (amplitude): 24d "The price of reach — one grid
  against a pyramid" and 24e "What the field must carry — the label splits into cells
  and a rim". Deck 112 -> 114 pages, sub_06 11 -> 13. This is the "CNN -> enc-dec
  step" the user reserved when the enc-dec was dropped from the ladder (entry below):
  it adds MECHANISM, deliberately NOT floors — 24e's footer says so on the slide
  ("a mechanism, not a floor: the ladder's argument ended at containment"). It stays
  clear of the deleted (H) chain AND of the reach/exchange knee model (the user
  already declined that when offered): nothing here models the knee, prices
  freshness, or assumes anything about the scene.
  24d, THE ECONOMICS — both families hang the SAME block (ConvBlock = two kxk convs,
  k=3, models/blocks.py:140; LocalCNN and the U-Net's Encoder/Decoder both build from
  it), so the frame compares only WHERE blocks sit:
  - trunk law r = 1 + 2B(k-1) (the ladder entry's RF = 4B+1 in symbols), and at
    pinned capacity P ∝ B k^2 m^2 the shared width falls, m ∝ sqrt(P/B) — scaling
    only, and the ladder's own widths sit on it (1072 @ 2 blocks / 515 @ 7).
  - pyramid law: a block at grid step 2^l moves the full-resolution field 2^l times
    as far; one block per station, D halvings: r = 1 + 2(k-1)(2^(D+1)-1). Station
    fields 5/13/29/61: THREE stations hold the ladder's whole 29x29 (six convs vs
    fourteen — the "same 29" dotted tie in the drawing), the FOURTH leaps past
    2w_a. Channel doubling is the reference U-Net's real schedule (features
    [64,128,256,512] + bottleneck_factor 2, configuration/architectures/
    backbone.py:14-15), so m_l = 2^l m_0 exactly and per-station compute
    (HW/4^l) k^2 m_l^2 = HW k^2 m_0^2 is constant; Encoder pools after every block
    (MaxPool2d(2), models/blocks.py Encoder).
  - the drawing (right column) is a LAW plot in 24c's plot idiom — trunk line,
    pyramid staircase, dashed 2w_a, rung dots colored per frame 22 (good 9x9,
    accent 29x29). The dashed line is DRAWN at the value 24c already prints (52)
    but LABELLED symbolically 2w_a; "13" (blocks for the trunk to clear it) and
    "4 stations" are law arithmetic from that same on-deck value, not new facts.
    NO new measured or config numbers anywhere on either frame.
  24e, THE SHAPE MATCH — the label is a per-pixel readout of a wide average, and
  the window sum splits EXACTLY into interior-cell sums + a rim:
  w_a w_r Rhat(x) = sum_{cells C ⊆ B(x)} (sum_{p in C} g g^H) + sum_{p in rim} g g^H
  — a pure partition, any window, any l, no scene assumption, type-safe (every sum
  over pixels of look outer products). An interior cell enters through its sum
  only, the SAME object for every window containing the cell — the correlation
  law's sharing, made mechanical. The U-Net has one site per cell at every l with
  width doubled where cells widen (a native HOME for the aggregates), skips +
  decoder carry rim and placement, a 1x1 head (conv_head_kernel_size = 1) reads
  y(x) off per pixel; the trunk must ride every aggregate on full-resolution
  channels of width m — the m that 24d already taxed for reach. TWO HONESTY
  GUARDS, do not remove: (1) "while cells stay small against the window" — the rim
  is ∝ 2^l (w_a + w_r) and eventually eats the area, so the interior-dominates
  reading holds for cells small vs the window; (2) the frames claim only
  REPRESENTATIONAL geometry ("a place to hold the aggregate"), never that pooling
  computes the average — the real Encoder uses MAX pooling, so do not "fix" the
  frames to say the pyramid averages.
  THE PENDING STRIP on 24e pre-registers the separating run — the reach experiment
  (trials_mode=reach: 33x33 local_cnn vs default unet, both 31.0M, shared square
  32x32 patch, so BOTH fields cover the patch whole and reach is off the table):
  a tie says reach was the whole story (24d carries the section alone); a standing
  U-Net win says the split pays on its own (24e earns its keep). WHEN IT LANDS,
  replace the strip with the verdict. Layout: 24d overfull (14.5pt) fixed by
  folding the "geometric in depth" qualifier out of the display into prose; both
  decks now build with zero warnings from section 08; pdftoppm render of sub_06
  pages 10-11 inspected clean.
  SECOND PASS same day (user: "reduce the text, make it more equation focous,
  and add more visualizations"): both frames recut equation-first.
  24d is now THREE columns (0.295 / 0.295 / 0.375): trunk and pyramid each get
  a heading, a six-row aligned block (equation row / soft scriptstyle gloss row,
  x3 — glosses BELOW the equations, not beside), and a nested-field sketch
  beneath; the r-vs-blocks plot keeps the third column at scale 0.93. The two
  sketches share ONE px scale (0.04 cm/px) and both overlay the same dashed
  B(x) (26x12 px -> 1.04x0.48 cm): trunk nest r = 1/5/9/13/17 (four equal +4
  steps, still short of w_a), pyramid nest r = 1/5/13/29/61 (steps doubling,
  swallowing the window) — the linear-vs-geometric contrast is the drawing.
  All prose sentences deleted; the only text left is headings, glosses, sketch
  captions, and the plot caption (3 lines).
  24e: left column lost its closing paragraph to a one-line exactness gloss +
  a NEW cells/rim diagram (scale 1.30, 0.09 cm/px, cell = 4 px): B(x) accent
  dashed with accent!25 rim fill, interior cells argA!25, a good-dashed
  NEIGHBOUR window B(x') whose shared cells are argA!45 (darker) — the
  "same object for every window" claim drawn, with x, x', a 2^l bracket and
  colour-keyed pointers; caption is colour-keyed too. Right column lost the
  two body paragraphs to a U-SCHEMATIC (7 boxes e0-e1-e2/bottleneck/d2-d1-d0,
  fills accent!6/15/26/40 deepening with width, pixel grids on the full-res
  pair, argA site squares in e1/e2/b labelled "one site per cell — the
  aggregates' home", dashed soft skip arrows labelled "skips: rim + placement",
  flow arrow to a 1x1 head and y(x)) + a TRUNK STRIP (four equal full-res
  boxes, label "every aggregate at width m, every pixel"); the soft honesty
  footer and the \pending strip survive unchanged in content.
  FIT LESSONS, so nobody re-fights them: (a) side-by-side minipages inside a
  0.56 column CANNOT hold an aligned block with &&-glosses next to a sketch —
  the display overflowed 53.8pt into the pyramid sketch; glosses-below-in-
  narrow-columns is the working form; (b) the remaining 8.23pt overfull was
  the pyramid aligned block itself (aligned width = max lhs + max rhs across
  ALL rows, so the k^2 m_l^2 / 4^l lhs added onto the long r-row rhs);
  \tfrac{k^2 m_l^2}{4^l} shrank the lhs and zeroed it. Verified: zero
  warnings from section 08 in both decks, sub_06 still 13 pages / full deck
  114, pages 10-11 re-rendered and inspected.
  THIRD PASS same day (user: "there are a lot of variables that just come
  without any explanation"): every architecture symbol is now defined where it
  first appears. 24d gained a one-line soft scriptsize SYMBOL STRIP under the
  header ("r: field side / B: blocks stacked / m: width in channels / P:
  parameters / station l: the block l halvings down, width m_l / D: deepest
  station" with \cdot separators); k was already defined in the header
  sentence. 24e's lead lost its "cut the scene into cells" clause and the
  exactness gloss became the DEFINITIONS gloss: "cells C: the scene cut into
  2^l x 2^l tiles, l halvings coarse; rim(x): the pixels of B(x) in partly
  covered cells" + the partition/exactness sentence. y, Phi, g, B(x), Rhat,
  w_a w_r intentionally NOT re-defined — frames 22/22b/22c own them. FIT: the
  strip + longer gloss made 24e's vbox overfull by 6.7pt; recovered via
  U-schematic \\[6pt]->[2pt], \vspace{-2pt} before \pending, dropping the
  \smallskip above the cells diagram, and displayskips 4->3pt — the last one
  is a TRAP: \setlength{\abovedisplayskip} BEFORE \small is a NO-OP (\small
  resets display skips), so the frame now sets \small FIRST then the skips;
  22b/22c carry the same dead pattern (setlength then \small) and their skips
  silently never applied. Zero warnings again from section 08, both decks,
  after the swap.
  FOURTH PASS same day (user: "the slide after that one, its not very clear
  what is the argument there" — read as 24e, the slide after 24d): 24e recast
  as an EXPLICIT NUMBERED ARGUMENT in 22b's step grammar, title now states the
  claim ("The label splits by scale — and the U-Net is shaped as the split").
  The four steps: 1. the label is an average, then a readout (y(x) =
  Phi(g_B(x)) = (fit o Capon)(Rhat(x)), inline); 2. the mean splits by scale —
  an exact partition (the cells + rim equation, defs gloss); 3. whole cells
  enter as one object each — the same object for every window holding the
  cell, rim + readout are all the fine grid owes (the cells/rim diagram
  illustrates, with a THIRD keyed pointer "darker: shared with B(x')" added
  inside the diagram so the sharing claim needs no caption line); 4. the U-Net
  is the split; the trunk is not (U-schematic + trunk strip + footer, footer
  now also carries the computed-once/correlation-law tie). The under-diagram
  colour-key caption was DELETED (steps 3 and the in-diagram labels carry it).
  FIT: the recast overflowed 18pt; recovered by diagram scale 1.10 -> 1.00,
  tighter step-1/3 wording, 2-line defs gloss, dropped micro-skips — NOTE the
  right column's header/footer trims changed NOTHING (left column is the
  vbox driver; trim there). Zero warnings both decks, page counts unchanged
  (sub_06 13 / full 114), page 11 re-rendered and inspected.
  FIFTH PASS same day (user: "its still not good, reformulate the full slide,
  focous on equations more"): 24e rebuilt from scratch as a PURE-EQUATION
  2x2 GRID in 22b's exact tabular skeleton (p{0.437}|p{0.437}, \hline,
  struts) --- ALL TikZ deleted from the frame (cells/rim diagram,
  U-schematic, trunk strip all gone; 24d now carries the section's visual
  weight alone). Title: "The label factorizes by scale --- the U-Net is the
  factorization". The four quadrants, one display each (Q3 two):
  Q1 y(x) = Phi(g_B(x)) = (fit o Capon)(Rhat(x)); Q2 the partition, now via
  a NAMED CELL AGGREGATE S(C) := sum_{p in C} g g^H (defined in the intro,
  this is what makes the rest one-liners); Q3 the two cheapness properties
  as equations --- composition S(C_{l+1}) = sum_{i=1..4} S(C_l^i) (one
  halving builds the next level) and sharing C subset B(x) cap B(x') =>
  one S(C) for both; Q4 the correspondence as a \mapsto display (station l
  |-> S(C_l) one site per cell / one halving |-> the composition step /
  skips |-> rim(x) + the read-off at x / 1x1 head |-> fit o Capon), then the
  trunk line (every S(C) at width m, full resolution --- the width reach
  taxed) and the honesty gloss. THE \pending STRIP IS GONE --- its content
  moved into Q4's closing gloss as "PENDING, the separating run: ..." to buy
  the vertical room; when the reach runs land, replace THAT gloss sentence
  with the verdict. New symbols all defined on-frame: S(C) (Q2 intro), C_l
  and C_l^i (Q3 gloss), rim(x) (Q2 gloss). FIT: first cut was 46pt too tall
  + 9pt too wide; fixed by folding the strip into Q4, \Rightarrow for
  \Longrightarrow and "one S(C) for both" (the wide box was Q3's
  implication), \mapsto for \longmapsto in Q4, displayskips 2pt (after
  \small!), struts 11pt/-4pt, and the trunk line to two lines. Zero
  warnings both decks, page counts unchanged, page 11 inspected.
  SIXTH PASS same day (user: "it is still not clear at all"): diagnosis ---
  every earlier form (steps, quadrants, mapsto table) asserted a
  CORRESPONDENCE and stopped; none derived a quantity the two architectures
  could be COMPARED on, so the reader had to finish the argument alone. 24e
  is now ONE full-width derivation chain in 09_error_theory's
  one-manipulation-per-row grammar (equation left, soft scriptsize
  annotation right), with the question stated up front ("every architecture
  must realise the label's window mean at every pixel --- what must it add
  up?") and the verdict stated at the end. The eight rows: (1) the job,
  w_a w_r Rhat(x) = sum over B(x) of g g^H --- w_a w_r look-products per
  window; (2) S(C) := cell aggregate [definition]; (3) exact partition into
  interior cell sums + rim; (4) #{C subset B(x)} <= w_a w_r / 4^l --- the
  count falls 4x per halving; (5) composition S(C_{l+1}) = sum of 4 quarters;
  (6) sharing C subset B(x) cap B(x') => one S(C) for both; (7) pyramid:
  #terms(x) ~ w_a w_r/4^l + |rim(x)| [station l holds each S(C_l) at one
  site; skips return the rim]; (8) one grid: #terms(x) = w_a w_r [nothing
  coarser than a pixel to store --- the previous frame's bill]. Closing
  line: "the pyramid assembles every window from a few shared pieces; the
  grid re-assembles all w_a w_r looks at every pixel." Honesty gloss +
  PENDING sentence unchanged. Title now states the verdict: "Assembling the
  label's mean --- the pyramid counts cells, the grid counts looks". The
  2x2 tabular is gone. HONESTY GUARDS in the counts: row 4 says "while
  cells fit" (the 4^l fall stops near 2^l ~ w_r); row 7 uses ~ (approx),
  not =; row 8 counts LEVEL-0 pieces --- the depth/width cost of
  assembling them on one grid is 24d's law, referenced as "the previous
  frame's bill", NOT re-derived (a conv trunk can build running sums in
  depth ~ w, so never claim the grid pays w_a*w_r ADDS per pixel --- the
  honest per-grid statement is piece COUNT at the owned level). Zero
  warnings both decks, page counts unchanged, page 11 inspected clean.
  SEVENTH PASS same day (user: "the languague is super vage, this needs to
  be scientific, be more objetive"): the deck's literary voice ("the job",
  "the previous frame's bill", "counts cells / counts looks", "stored once,
  read everywhere") stripped from 24e and replaced with technical register.
  Title now "Forming Rhat(x) at every pixel --- the term count with and
  without multi-scale storage". Precision upgrades, keep them: rim is now
  SCALE-INDEXED rim_l(x) with a set-builder definition (B(x) minus the
  union of contained cells); the area bound is written with floor functions
  (#{C} <= floor(w_a/2^l) floor(w_r/2^l) <= w_a w_r/4^l) and its validity
  condition stated (interior tiling empty once 2^l > w_r); the count is a
  DEFINED quantity T_l(x) := #{C subset B(x)} + |rim_l(x)|, and the local
  CNN is the exact l = 0 case (T_0(x) = w_a w_r --- cells are single
  pixels), so both architectures are instances of ONE formula; sharing is
  stated as x-independence of S(C); the closing paragraph names which
  formula applies to which architecture and the 4^l interior factor; the
  footer states the scope objectively (bounds the computation to represent,
  NOT the error floor, which depends on the receptive field alone) and
  describes the pending run concretely (capacity-matched U-Net vs local
  CNN on a patch both fields cover --- does the advantage persist at
  matched reach). Nine annotated rows total, one manipulation each. NOTE:
  24d still carries the deck's literary glosses ("the far field sells at
  the near field's price", "a bite of everyone's width") --- the user's
  register complaint was raised on 24e; if it extends to 24d, apply the
  same treatment there. Zero warnings both decks, page counts unchanged,
  page 11 re-rendered and inspected.
  EIGHTH PASS same day (user: "not good still, try avoiding equations and
  focous only on represenation on that slide"): 24e inverted --- NO display
  equations at all; the slide now SHOWS the representation story. Title
  "Representing the window --- per-cell values against per-pixel packing".
  Two labelled blocks, each one bold lead sentence + one drawing:
  (1) U-NET, ONE GRID PER SCALE: four panels showing the SAME window B(x)
  (26x12 px at 0.06 cm/px, drawn dashed accent, identical placement) on
  grids of scale 0/1/2/3 (grid steps 0.06/0.12/0.24/0.48), interior cells
  argA!30, rim accent!30, "\div 4" arrows between panels, legend at right
  (interior: stored cells / rim: kept at scale 0 via the skips). Panel
  captions: "scale 0 --- a value per pixel" ... "scale 3 --- cells outgrow
  the window" (the scale-3 panel HONESTLY shows 2 interior cells + a large
  rim: coarse cells stop paying near the window's short side). The window
  placement [0.24,1.80]x[0.24,0.96] is 0.24-aligned, so scale-0/1 have
  empty rim and scale-2/3 rims match the real 26x12 arithmetic --- NO
  counts are printed (78/18/2 would be NEW numbers; the boxcar note allows
  only 26x12/52/24/312 without asking). (2) LOCAL CNN, SCALE 0 ONLY: the
  same window + pixel x + arrow to a channel-stack drawing, lead "each
  pixel must pack a summary of its whole window into its own m channels,
  and the packing repeats at every pixel". Footer keeps the objective
  scope + pending sentences from the seventh pass verbatim ("a statement
  about representation, not about the error floor..."). Language remains
  the technical register of the seventh pass; symbols on the frame are
  only names (B(x), m, x, scale numbers) --- no formula manipulation.
  Zero warnings both decks, page counts unchanged (sub_06 13 / full 114),
  page 11 re-rendered and inspected.
  NINTH PASS same day (user: "make the grid size divisible, to avoid regions
  cutting pixels in half"): 24e's panels redrawn on FULLY DIVISIBLE
  geometry. The old panels (1.92x1.20, window 26x12 px at 0.06 cm/px) had
  three violations: panel height 1.20 is not a multiple of the scale-3 cell
  0.48 (half-height cell row at the top), and the pixel-true window/rim
  fills ended mid-cell at scales 2-3 (0.24/1.80 are not 0.24- or
  0.48-multiples). New geometry: 0.05 cm/px; panels 1.60x1.20 (whole cells
  at every step 0.05/0.10/0.20/0.40); window SCHEMATIC 16x8 px = 0.80x0.40
  at [0.40,1.20]x[0.40,0.80] --- placed on the coarsest-cell lattice and
  sized divisible-by-8 in both axes, so it is a whole number of cells at
  EVERY scale (8x4 -> 4x2 -> 2x1 cells; captions "scale 0 --- a value per
  pixel" ... "scale 3 --- two cells"). CONSEQUENCE, deliberate: no rim
  exists in the drawings any more (the schematic window is exactly
  cell-aligned), so the red rim fills + legend are GONE and the remainder
  story lives in the U-Net lead sentence only ("skip connections keep the
  full-resolution remainder --- window edges and the position of x"). The
  16x8 window is a SCHEMATIC committed only to the 2:1 aspect (like the
  section's other drawings) --- the real 26x12 boxcar does NOT align with
  any dyadic tiling, which is exactly why the drawn-to-pixel version had
  half-covered cells; do not "fix" the schematic back to 26x12 without
  accepting partial cells again. Local-CNN panel updated to the same
  geometry, pixel x on the 0.05 lattice at the window centre. Zero
  warnings both decks, page counts unchanged, page 11 re-rendered and
  inspected clean.
  TENTH PASS same day (user: "to show the effect of the skip conections you
  need to show the downsampelling and upsampelling branch as well"): row A
  redrawn as the FULL U laid out horizontally --- five panels, scales
  0 -> 1 -> 2 -> 1 -> 0, "\div 4" arrows on the down branch, "\times 4" on
  the up branch, TWO dashed skip arcs joining equal scales (down-0 to up-0
  over the top, down-1 to up-1 inside), one label ("skips --- edges and the
  position of x, at each scale"), and an output arrow to y(x) after the
  final panel. THE SKIP EFFECT IS CARRIED BY THE x MARKER: scale-0 panels
  mark x as its exact 1-px cell (accent), scale-1 as its 2x2 cell
  (accent!50), scale-2 as its 4x4 cell (accent!40) --- so the position
  degrades down the branch, stays cell-coarse on the way up, and is exact
  again in the last panel only because the skip re-injects it (captions:
  "x at its pixel" / "x only to a cell" / "x exact again"). All new fills
  are lattice-aligned (x-cells at 0.80..0.90 / 0.80..1.00), so the ninth
  pass's divisibility holds. Lead rewritten to name both branches ("down,
  up, and across"). Geometry: panels at x-shifts 0/2.10/4.20/6.30/8.40
  (1.60 wide, 0.50 gaps), arcs peak ~1.7, y(x) at 10.44 --- total ~11 cm,
  fits. Zero warnings both decks, page counts unchanged, page 11
  re-rendered and inspected.
  ELEVENTH PASS same day (user: "go back to they way it was. And just
  arguee that the scale matches the diferent scales the ifgs have"): the
  tenth pass's down/up/skip U drawing is REVERTED to the ninth pass's
  four-panel sequence (scales 0 -> 1 -> 2 -> 3, \div 4 arrows, divisible
  geometry, captions ".. a value per pixel" / "scale 3 --- two cells"),
  and the slide's ARGUMENT is now SCALE MATCHING TO THE INPUT: title "One
  grid per scale --- matching the scales of the interferograms"; intro
  states the physics (interferogram n measures at vertical wavenumber
  k_n proportional to b_perp,n --- the signal-model relation section 09
  also uses --- so one relief appears as fringes at several spatial rates,
  coarse on short baselines, fine on long ones); U-Net lead: each fringe
  rate meets a grid whose sampling matches it, slow fringes fully
  described on a coarse grid (one value per cell, window 4x fewer cells
  per scale), fast fringes keep the fine grids; local-CNN lead gains the
  mirrored clause (every rate, however slow, carried at full resolution,
  packed into the m channels per pixel). The skip-connection argument is
  GONE from the slide (both the U drawing and the lead sentence) on the
  user's direction --- do not re-add skips here. k_n and b_perp,n are
  defined inline in the intro since sub_06 readers have not seen section
  02/09. Physics check for the record: phi_n = k_n z => grad phi_n =
  k_n grad z, so fringe spatial rate is proportional to k_n for the same
  relief --- the claim is exact in symbols. Zero warnings both decks,
  page counts unchanged, page 11 re-rendered and inspected.
  TWELFTH PASS same day (user: "in the bottom path of the normal cnn,
  expand and show the scale not changing"): the local-CNN row expanded
  from one panel to a FOUR-PANEL PATH mirroring the U-Net row exactly ---
  same panel geometry and x-positions (shifts 0/2.30/4.60/6.90), same
  argA!30 window fill (was !12), but ALL FOUR at scale 0 with "\times 1"
  arrow labels where the top row has "\div 4"; captions "scale 0" x3 +
  "scale 0 --- unchanged"; the x pixel and its label moved to the LAST
  panel (the readout site), followed by the arrow into the channel stack,
  whose label is shortened to "the m channels at x" (the lead sentence
  carries the packing claim; lead gained "layer after layer the grid
  stays at full resolution"). The two rows now contrast column-for-column:
  grids coarsening above, identical below. Zero warnings both decks, page
  counts unchanged, page 11 re-rendered and inspected.

- CONTEXT EXPERIMENT REDESIGNED AS A RECEPTIVE-FIELD SWEEP 2026-07-16, AND FRAME 23
  IS PENDING. The old ladder (pixel MLP -> local_cnn 13x13 -> enc-dec) was retired
  because its last rung moved TWO things at once — receptive field AND architecture
  family — so it could isolate neither (user: "the last part does not make sense, it
  change the receptive field and it changes the architecture, 2 things at once").
  The new ladder is THREE rungs of ONE trunk family, no downsampling, capacity pinned
  at ~31M so the field is the only variable: mlp 1x1 / cnn09 9x9 / cnn29 29x29. Both
  the enc-dec and a fourth rung (cnn41 41x41) were dropped on the user's call ("i
  will create one just for him latter"; "remove the big CNN, becuase I will discuss
  latter the CNN -> enc-decor step and them show that the over reach helps") — the
  over-reach story is told later against the enc-dec's own results, so this ladder
  stops exactly where its argument stops. WHY THESE FIELDS, against B(x) = w_a x w_r:
  9x9 sits strictly INSIDE the window (floor > 0, misses w_a w_r - r^2); 29x29 is the
  SMALLEST reachable field that holds it (floor = 0). So the ladder now tells ONE
  clean story — the floor falling to zero — and 22d's footer states the consequence
  rather than a hypothesis: past 29x29 this argument predicts NOTHING more to gain,
  so every later gain (the enc-dec's included) is something it cannot name. That is
  the handoff. The "Hypothesis, stated before the result" block was DELETED from
  frame 22 (user) — the prediction now lives in 22d's footer, not in a block of its
  own. Frame 22 shows the three rungs on ONE horizontal line to the right of a
  to-scale nested-field figure (9x9 inside B(x), 29x29 swallowing it); an earlier
  2x2-snake version and a `ghost` enc-dec box were both cut by the user.
  #############################################################################
  ## FRAME 23 SHOWS PLACEHOLDER NUMBERS AND NO LONGER SAYS SO. DO NOT PRESENT ##
  ## OR CITE IT UNTIL THE ctx-* RUNS LAND AND THE VALUES ARE REPLACED.        ##
  #############################################################################
  As of 2026-07-16 the three ctx-* runs DO NOT EXIST. Frame 23's columns read
  "pixel MLP / CNN small (9x9) / CNN big (29x29)" — the ladder as configured — but
  EVERY NUMBER under them was measured on the RETIRED tiers: a 13x13 local CNN and a
  full-patch encoder-decoder, neither of which is in this experiment. The mismatch is
  invisible on the slide. Sequence, so nobody re-derives it: the frame was first
  reduced to a \pending manifest with no numbers; the user then asked for the old
  numbers back as a placeholder ("use those old mlp, cnn, enc-dec, ones as a
  placeholder for now, use their numbers"), then for the columns to be renamed to
  small/big CNN ("just keep the numbers"), then for the PLACEHOLDER banner to be
  deleted. Each step was deliberate and each is fine for a working draft; the end
  state is a slide that attributes real measurements to runs that never produced them.
  WHEN THE RUNS LAND: replace all values wholesale — do NOT map old columns onto new
  rungs, since 13x13 -> 9x9 and enc-dec -> 29x29 are not the same models. The only
  figure already valid under the new geometry is "looks seen": with w_a x w_r = 26x12,
  9x9 meets the window in 81 of 312 and 29x29 in all 312.
  Field arithmetic (see the code entry below):
  RF = 1 + 4*blocks, always odd, so 9/29 = 2/7 blocks; widths 1072/515 hold ~31M at
  each depth. (A 41x41 rung would be 10 blocks at width 426 — same ~31M — if the
  fourth rung is ever wanted back.)

- SECTION 08 MADE PER-AXIS (w_a x w_r) 2026-07-16, AND NUMBER-FREE. The boxcar is
  NOT square and the deck had claimed "w x w, w = 20" throughout. Every w in the
  section is now w_a (azimuth) x w_r (range): frame 22 header + Rhat = 1/(w_a w_r)
  sum; 22c look counts (1/(w_a w_r), (w_a w_r)^2, w_a w_r - 1); 24b correlation law
  now SEPARABLE, rho(tau_a, tau_r) = (1 - tau_a/w_a)_+ (1 - tau_r/w_r)_+, vanishing
  once tau_a = w_a OR tau_r = w_r; 24b2 W = 2w -> W_a = 2w_a, W_r = 2w_r; 24c knee
  registered at 2w_a. 24c's sweep is over SQUARE patches W, so a new bullet states
  why azimuth binds: a square patch clears the range reach 2w_r early, leaving 2w_a
  as the reach to clear — that is what the dashed line marks.
  ** NO CONCRETE VALUES ANYWHERE IN THE SECTION — DELIBERATE. ** The repo says the
  boxcar is win = [20, 10] (configuration/sar/processing_config.py:22 and :153, and
  test_data/meta/config_state.json, i.e. the metadata of the dataset that actually
  produced the training data), while patch_size = (64, 32)
  (configuration/training/unrolled.py:35). The user stated the boxcar is 64x32 and
  rejected 20x10 outright ("where did I say 20x10? STOP INVENTING NUMBERS"), then
  chose symbols-only to settle it. DO NOT put a number back without asking. Note
  64x32 is exactly the patch_size default, and if B(x) really were 64x32 with a
  64x32 patch then B(x) subset F holds only at the exact centre pixel and 22b's
  "And it vanishes" step is false for every other pixel — so the two quantities
  must differ for that frame to stand. The az/rg assignment (first number =
  azimuth) is an INFERENCE from ps_az ~ 0.39 vs ps_rg ~ 0.60 (test_data/meta/
  track_parameters.json): [20,10] as (az,rg) gives a near-square ground cell,
  reversed it gives 3.9m x 17m. Unconfirmed — PyRAT is server-only (stetools) so
  its filargs convention could not be read. The drawings commit only to a 2:1
  az:rg aspect, which both candidate readings share.
  LAYOUT, per user: 24b's left column previously held a scene diagram on top and an
  overlap/triangle construction below; the scene diagram is DELETED and the
  construction is now shown TWICE — once sliding along azimuth, once across range —
  each with its own triangle dying at its own w. 24b2's W_a bar was dropped to
  y = -0.88 to clear the W_r label, and its "shared looks" callout moved to the
  right of the patch diagram.

- EXACT LADDER ARGUMENT ADDED 2026-07-16 (user: "in the slide about context. Add
  some math development to show why the context ladder helps the model to
  improve"). Two frames between 22 (ladder design) and 23 (ladder result); deck
  109 -> 111, sub_06 8 -> 10. SCOPE WAS CHOSEN DELIBERATELY: the user was offered
  the reach / W = 2*tau + w exchange as an option and picked "exact ladder
  argument only", so this chain stays clear of the territory deleted earlier the
  same day (entry below) and ASSUMES NOTHING about the scene. LAYOUT IS BOTH
  FRAMES' EQUATIONS IN TWO COLUMNS, NO FIGURES — the user cut the two TikZ
  visualisations (22b's nested-fields diagram, 22c's 20x20 boxcar grid) outright:
  "remove the 2 visualizations on both slides, and do a 2 column equations". Do
  not re-add them. A "The symbols, once" strip defining x, y(x), B(x), F and g_F
  was added to 22b after the user rejected the first draft for using F/F1/F2
  unintroduced ("How can you use F, F1, F2, and never defined them, do you think
  the reader should guess?") — and was then DELETED by the user too, along with
  22b's closing block ("Settled before a single step of training"). Both are gone.
  KNOWN CONSEQUENCE, left standing on the user's instruction: F, F1, F2, g_F and y
  are once again undefined on the frame; frame 22 immediately before it introduces
  the boxcar and the three tiers, but not these symbols. When told to delete them
  the correct move was to delete ONLY those parts — an attempt to compensate by
  folding the definitions inline was itself reverted ("i did not ask you to change
  the text, just to delet those parts"). TYPE DISCIPLINE, and it
  matters: F is a set of PIXELS, g_F = {g(p) : p in F} is what the field DELIVERS
  (stack vectors), and every conditioning is written on g_F, never on F. The first
  draft mixed these — it captioned F_px = {g(x)} (looks) while asserting
  B(x) subset F (pixels) — which is the same
  centres-vs-looks conflation flagged in the entry below; fixed, do not let it back
  in. The argument: (i) the label is B(x)-measurable — y(x) = Phi(g_B(x)) — exact
  by construction, since Capon and the parameter fit are deterministic maps;
  (ii) for any predictor built from a field F, E[(y-yhat)^2] = E[Var(y|g_F)] +
  E[(E[y|g_F]-yhat)^2], a floor set by F alone plus what training buys; (iii) the
  conditional variance decomposition Var(y|g_F1) = E[Var(y|g_F2)|g_F1] +
  Var(E[y|g_F2]|g_F1) >= E[Var(y|g_F2)|g_F1] gives E[Var(y|g_F1)] >=
  E[Var(y|g_F2)] for F1 subset F2 — so the floor is monotone non-increasing along
  the NESTED rungs (F_px = {x} subset F_loc subset F_patch); and (iv) the vanishing
  step, as a THREE-IMPLICATION DISPLAY rather than prose (user: "show this in the
  equations: once B(x) subset F, y is a function of gF and the floor vanishes"):
  B(x) subset F ==> g_B(x) subset g_F ==> y = Phi(g_B(x)) fixed by g_F ==>
  Var(y|g_F) = 0. Keep the middle implication: B(x) subset F ==> g_B(x) subset g_F
  IS the pixels-to-looks bridge, and it is the step that makes the type discipline
  above visible rather than assumed. y is scalar: the argument runs per label
  component. NOTE the frame states only that the floor VANISHES when B(x) subset F;
  it does NOT claim the floor is > 0 when B(x) not-subset F — that needs the unseen
  looks to not be determined by the seen ones, which is a scene property. The block
  says "stays at the variance of the looks the model never sees", which is the
  honest form; do not upgrade it to a strict inequality.
  22c "The floor of any field — the fluctuation of the looks it misses" is stated
  for an ARBITRARY field F, not for the control (user: "the equations on top are
  kind of making the argument only for the MLP but it should be more generic for a
  receptive field F" — an earlier draft split off g(x) specifically and derived the
  control's case, which then failed to match the generic table below it). The
  window splits into delivered and missed looks:
  w_a w_r Rhat(x) = sum_{p in B(x) ∩ F} g(p)g(p)^H + sum_{p in B(x) \ F} g(p)g(p)^H.
  Conditioning on g_F makes the FIRST sum a constant (every look in it is handed
  over), so only the misses fluctuate:
  Var(Rhat(x)|g_F) = (1/(w_a w_r)^2) Var( sum_{p in B(x) \ F} g(p)g(p)^H | g_F ).
  Exact — no scene assumption, no independence. WHY THIS FORM EARNS ITS KEEP: the
  three rungs are now special cases of ONE equation, read off B(x) \ F — the
  control is "all but one look", the local CNN is "those outside 13x13", the
  enc-dec is the EMPTY SET, which makes the sum empty and the floor 0. So 22b's
  vanishing step falls out of this same equation rather than being a separate
  claim, and the table's first row is literally B(x) \ F. Do not re-specialise it
  to g(x).
  (This entry's w^2 is the old square-w notation; the section is now per-axis,
  w_a w_r — see the per-axis entry above.)
  The frame CLOSES ON A COMPARISON TABLE, not prose (user: "substitute this What
  the control can and cannot do ... for a table of comparison, the E(y-yhat)^2
  label variance error, model error, and the effect of gF -> g(x) on the variance
  of y | gf"). Rows = the terms of 22b's decomposition (does the field hold B(x)?;
  label variance E[Var(y|g_F)]; model error E[(E[y|g_F]-yhat)^2]; total
  E[(y-yhat)^2]). The point the table makes visually: only the label-variance row
  (shaded argA!18) moves along the ladder — model error reads "what training buys"
  in EVERY column — so that one row is the entire ladder. Keep the shading on that
  row; it is the argument, not decoration.
  THREE COLUMNS, ONE PER RUNG — added because the user caught a real gap: "it
  explains why the MLP is clearly the worst, but it does not explain why the UNet
  like encoder/decoder is better than the local CNN". Monotonicity alone does NOT
  explain that gap: if F_loc already held B(x) its floor would be zero too and the
  theory would predict a TIE with enc-dec, leaving the measured curve-R^2
  0.47 -> 0.59 unexplained. The resolution is that the floor reaches zero ONLY on
  the top rung. RECEPTIVE FIELDS, computed from the code, not guessed:
  - F_px  = 1x1: PixelMLPNet is all kernel_size=1 convs (models/backbone/
    pixel_baselines.py).
  - F_loc = 13x13: LocalCNN is 3 ConvBlocks (LocalCNNConfig.features =
    [832,832,832]) x two 3x3 stride-1 convs each = six 3x3 convs -> RF = 1 + 6*2 =
    13; the conv head is 1x1 (models/blocks.py conv_head_kernel_size = 1) so adds
    nothing.
  - F_patch = the training patch (deck's own framing: "full patch context").
  So B(x) is NOT contained in F_loc — the window is longer than 13 in azimuth under
  EITHER candidate boxcar value (w_a = 20 repo, or 64 user; both > 13), which is
  why the claim survives the unresolved value dispute. Its floor therefore falls
  but does not vanish, and that remainder IS the enc-dec's margin. If w_a were ever
  <= 13 this frame's middle column becomes false — recheck it against the boxcar
  before trusting it.
  THE MISSES ROW IS A CLOSED FORM, not prose (user: "you can be more formal in the
  number of misses ... the cardinality of the difference between B and F"). Row =
  |B(x) \ F|: pixel MLP w_a w_r - 1; local CNN
  w_a w_r - min(w_a, r) min(w_r, r) with r = 13; enc-dec 0. JUSTIFICATION, stated
  in the footer because the min() is otherwise an assertion: B(x) and F_loc are
  BOTH rectangles centred on x, so they intersect in exactly
  min(w_a, r) x min(w_r, r). The formula also states the widening condition
  exactly — the local CNN's misses vanish iff r >= w_a AND r >= w_r — which is the
  design rule for the open follow-up below.

- OPEN FOLLOW-UP (raised by the user 2026-07-16, not yet run): "a local cnn with a
  local F > B(x) could perform better?" — yes, and it is the sharpest falsifiable
  prediction the frame makes. LocalCNN's RF = 1 + 4*(blocks) (2 conv3x3 per
  ConvBlock), so blocks >= (w_a - 1)/4 gives r >= w_a: 5 blocks (r = 21) under the
  repo's w_a = 20, 16 blocks (r = 65) under the user's 64. Only
  LocalCNNConfig.features changes. THE POINT IS THAT IT DISCRIMINATES: a zero-floor
  local CNN that still trails the enc-dec would prove context BEYOND window
  containment matters, which the measurability argument cannot explain — and the
  existing sweep already hints at exactly that, since the knee sits near 2w_a while
  containment only needs ~w_a. That gap is the reach/exchange territory of the
  deleted chain. Note stacked 3x3 grows the RF SQUARE, so r must be sized by the
  longer axis (azimuth) and overshoots in range; (3,1) kernels would grow azimuth
  only.
  TWO SIMPLIFICATIONS THE USER FORCED, both right, do not undo either:
  (1) an earlier draft routed this through a defined Rhat_{-x} = mean of the other
  w^2-1 looks ("their is no reason to define R^-x, just writhe the full equation")
  — the direct sum kills the (1-1/w^2)^2 and w^2/(w^2-1) factors that only
  cancelled against each other anyway;
  (2) a later draft added an INDEPENDENT-LOOKS IDEALISATION to reach
  Var(Rhat|g(x))/Var(Rhat) = 1 - 1/w^2 = 99.75%, i.e. "own look worth 0.25% of its
  own target" — cut ("we dont need this to argue, we dont need explicit numbers").
  Correct: the idealisation bought only a headline number while being the single
  assumption left anywhere in 22b/22c, and it was FALSE-ish besides — looks are
  spatially correlated, so g(x) does say something about its neighbours and the
  true ratio is scene-dependent. The frame now says so explicitly ("how much its
  own look buys is a property of the scene, not ours to assume"). NO NUMBERS on
  either frame: w^2 appears only symbolically. The control's best possible
  prediction is E[y|g(x)] — which is what frame 23 already reports
  as median per-pixel R^2 = -0.03. WHY THIS IS SAFE where the deleted chain was
  not: nothing here reasons about R(p) being constant on B(x), or about two
  pixels' windows bridging through a shared p; the ladder claim is a measurability
  + tower-property statement about the label as it is actually constructed. It
  says only that context CAN help and that the floor vanishes once the window
  fits — it does NOT explain the measured knee at 40-56 px, and must not be
  stretched to.

- CONTEXT-SECTION THEORY CHAIN DELETED 2026-07-16 (user: "delet slides 6-11, and
  any notes rellated to this falty demostration"). REMOVED: 24b3 risk decomposition
  + the five "why the shared component helps" step frames (covariance law /
  certificate / certified pool / knee / tail), and their three backup row-by-row
  frames (risk split; window variance+covariance; certified pool variance). Deck
  118 -> 109 pages, sub_06 14 -> 8. WHY, so nobody rebuilds it: the whole chain
  rested on (H) "R(p) = R(x) for every p in B(x), at every x" — the boxcar's
  homogeneity premise. (H) is DEGENERATE: the certificate step R(x_i) = R(x)
  needs (H) at TWO pixels at once (bridging through a shared p), which is exactly
  the chaining that, applied across adjacent windows, forces R constant over the
  whole scene. A proof from that premise proves anything. The user caught it:
  "R is a summation on the local neighbourhood of a pixel, I don't see how that
  argument can be true." SECOND, INDEPENDENT ERROR found while checking: the deck
  claimed "the reach of the certificate = the centred patch W = 2w", conflating a
  set of window CENTRES with a patch of LOOKS — certifying a centre at reach w
  needs looks out to 1.5w, so the certificate reading actually demands W = 3w = 60,
  which does NOT match the measured knee of 40-56. The honest geometry is
  W = 2*tau + w (patch W holds pool centres out to tau = (W-w)/2). WHAT WAS
  VERIFIED before deletion, kept here only as a record: an honest model targeting
  Rbar(x) = (1/w^2) sum_{p in B(x)} R(p) (what the boxcar is exactly unbiased for,
  no assumption) with delta DERIVED from a real smooth scene reproduces the
  measurement — monotone always, knee80 = 28-48 px (median 40, flat across scene
  roughness and correlation length) vs measured 40-56 — WITHOUT (H). Under it the
  exact identity Rbar(x_i) - Rbar(x) = (1/w^2)[sum_{private i} R - sum_{private x} R]
  holds (shared pixels cancel identically), so rho(tau) does double duty: it is both
  the labels' noise correlation AND their target agreement, and the private looks are
  simultaneously the fresh information and the disagreement. The knee then comes from
  the freshness/price exchange turning at reach tau ~ w/2 (robust across every scene
  parameter swept), giving W = 2*tau + w ~ 2w. NOT REBUILT — the user deleted the
  section rather than re-derive it; if it is ever revived, start from Rbar and the
  W = 2*tau + w geometry, never from (H). The measured verdict frame (24c) survives
  and was restored to observation-only: "registered 2w = 40" there now leans on the
  correlation-law / reading-the-reach frames, which STILL carry the same 2w
  centres-vs-looks geometry error and were left in place because the user scoped the
  deletion to slides 6-11 — flagged, not fixed.

- LEVER-2 PRESENCE-BALANCE FRAME REBUILT PER-SLOT 2026-07-13 (user, after the
  loss change de93800 made presence_balance per-Gaussian-slot with clamped
  fractions): frame 17b now presents the per-slot design. Left column: three
  bullets (amplitude trains every slot / the imbalance is per slot, not one
  number --- a single batch-wide fraction 16/30~0.5 on the example batch gives
  weights ~1 and cannot see the tail / inverse class frequency slot by slot),
  aligned equation pair w_{p,k} = 0.5/f_k m + 0.5/(1-f_k)(1-m) and
  f_k = clip(<m>_p, 1e-3, 1-1e-3), clip caption. Right column: occupancy-ladder
  visualization --- three slot rows of ten cells (slot 1 = 10/10 active with
  f_1 -> 0.999 clip and actives x0.50; slot 2 = 5/10, weights 1.00 both,
  "already balanced"; slot 3 = 1/10, accent) --- then the uniform-vs-balanced
  gradient-share bars REPLAYED ON SLOT 3 ONLY (10/90 -> x5.0 / x0.56 -> 50/50),
  closing line "every slot balanced against itself". The old single global
  f = <m> = 3/10 batch strip + 30/70 bars are gone. Fit: tikz scale 0.86,
  annotations at x=4.50/4.57, no overfull. CAVEAT FLAGGED: the next frame's
  12-run grid ("presence balance over-corrects") measured the GLOBAL variant;
  its conclusion awaits the per-slot presence-trials rerun.
  SECOND PASS same day (user: "less explicit with the numbers, keep it more
  generic"): all example numbers stripped to symbolic form --- ladder
  annotations now f_1 -> 1 clipped / moderate f_2 / f_3 << 1, arrow labels
  x0.5/f_3 and x0.5/(1-f_3) without evaluated values, bar labels active/empty
  without fractions, balanced bar says half/half not 50%/50%, the 16/30
  batch-fraction aside deleted from bullet 2, and the clip displays epsilon
  with epsilon = 1e-3 stated in the caption. Cell proportions still carry the
  quantities visually. Same rule as the "output is a set" frame: symbolic
  slides, constants live in captions.

- L1-VS-MSE HONESTY RECUT, FIFTEEN FRAMES 2026-07-12 (review pass found the
  core solid but four overclaims; user: "Fix it"): the section now claims
  exactly what the code and runs support. (1) Step 1 rederived as
  spike-plus-slab: the spike at 0 is exact, the occupied law p(a|a>0,x) is
  a continuous density, not a second point mass — now consistent with
  09_error_theory's presence-times-brightness hedge (Pr·E). Step 5 lands on
  c*_MSE = Pr(a>0|x)·E[a|a>0,x] (occupied mean dimmed by the posterior
  weight, split into assignment + bounds rows); step 6 solves the mixture
  CDF: the median is 0 or the slab's quantile at (1/2−Pr(a=0|x))/Pr(a>0|x)
  — "a value the label takes" softened to "zero, or a real occupied
  amplitude". Step 2 keeps the two-point position law but states the
  idealisation (modes carry the model's real error width; only the
  separation is used). (2) Balanced-mass flatness glosses added to steps 4,
  6, 8: at exactly half mass the median condition holds on an interval and
  the L1 risk is flat — near 50/50 L1's edge is only the absence of MSE's
  active pull toward the hedge. (3) Step 12 rebuilt for the actual
  optimizer (trainer: Adam; scheduler: cosine to eta_min 1e-6) and split
  into two frames: "one update under Adam" derives drift = −λ·pull/
  sqrt(pull²+η²) and kick ≈ λη/sqrt(·) from the running moments (9 rows);
  "the two floors" evaluates both slopes — MSE's drift balances its kick at
  |r| = η/2, a λ-free floor annealing cannot move; L1's drift never fades
  and circles the kink in a band ≈ λ that the schedule sends to zero. The
  unmeasured "a tuned run keeps η<1" premise is gone; the diagonal-picture
  and limit-cycle idealisations are stated in glosses. Step 13 gloss:
  "fades quadratically" corrected (the pull fades linearly with the
  residual) + Adam-rescaling note. (4) Step 9 gloss now flags the pricing
  premise: sorted fixes the target by GT rank (steps 3–8 exact), Hungarian
  re-derives it from the prediction each step (pricing an approximation);
  "0 one step and A the next" → occupied value. Step 11's numeric gloss
  marked "illustrative scales, not measurements", histograms named as the
  measurement. (5) Checks frame split in two. "Huber as a gap probe" states
  the configured delta = 0.5 (configuration/training/general/loss.py),
  adds a Huber's-share column ((Huber−MSE)/(L1−MSE): 53/64/67/−2/0/42/91/
  86 %, computed from the existing table), and turns the Att-UNet anomaly
  into a falsifiable prediction (its tail gaps must sit inside delta — the
  histograms adjudicate); the gate check reframed on the real spread: gated
  rows collapse L1−MSE to a uniform 0.049–0.058 band, ungated scatter
  0.045–0.127 (the old "spread up to 0.081/0.127" hid that ResUNet
  conv-sorted, 0.045, sits below the gated band). New closing frame "what
  the checks cannot yet say": Score is a min-max composite in rank units +
  one seed per cell (the sign test over eight cells is the evidence, not
  gap sizes), the fixed-law premise vs Hungarian, and the two runs that
  separate hedge from floor-and-fade — the loss-pair sweep's residual
  histograms (measure sigma_r, R, eps, kurtosis) and its weight axis
  (param-MSE at larger w lowers the floor to eta/2w but moves no wrong
  target). Section 13 → 15 frames; full deck 107 → 109 pages, sub_07
  30 → 32. Verified: both decks built with zero overfull from the section
  (new-row overfulls fixed: frame-1 vbox via row merge + gloss trims;
  step-5/6 hboxes via row splits and annotation trims — in an aligned
  display the widest math row and widest annotation row jointly set the
  width, so trimming the same row that holds both is not enough); pdftoppm
  render of all fifteen sub_07 pages (18–32) plus full-deck pages 69 and
  83 inspected clean.
- L1-VS-MSE ONE MANIPULATION PER LINE, THIRTEEN FRAMES 2026-07-12 (user: the
  equations are still jumping steps, like the pull of the tail — it goes
  from the definition to a full equation without any context; step by step,
  stop skipping steps, I don't care how big it gets): every chain in
  11b_l1_vs_mse now advances one manipulation per row, each row annotated
  with the manipulation's name in the 09_error_theory third-column grammar,
  and derivation-heavy steps get full-width frames (the memory-approved
  "one step across the frame" form). Steps 1-2: the point-mass laws are
  derived via an explicit total-probability row (p(a|x) = Pr(a=0|x)
  p(a|a=0) + Pr(a=A|x) p(a|a=A)) and a conditional-density row before the
  assembled law. Step 3 (own frame): plug l into R, expand the square,
  carry it inside the expectation, apply linearity, differentiate, set to
  zero. Step 4 (own frame): plug in, resolve the absolute value, split the
  integral, Leibniz on each piece with the (c-c)p(c) boundary terms written
  and cancelled, masses remain, balance, median. Step 5 (own frame):
  E[a|x] evaluated by inserting the law, pulling weights out by linearity,
  sifting each integral (int a delta(a) da = 0, int a delta(a-A) da = A),
  evaluating. Step 6 (own frame): the median's two-sided definition stated,
  the CDF tested at m=0 and m=A explicitly, both cases concluded. Step 7
  (own frame): E[mu|x] by insertion + sifting, then the mode gaps derived
  by subtraction, the masses-sum-to-one row, collection, symmetry,
  betweenness, and the min. Step 8: median cases + the match-rule pricing
  chain (restating step 7's result as its own row before comparing to 5).
  Step 9: matcher frame unchanged in content. Step 10 (new frame):
  both slopes differentiated from their definitions (chain rule row,
  sign row, flatness row). Step 11 (the flagged frame): S defined
  full-width as the explicit sum ratio, then each population's sum built
  row by row (tail: eps N terms each 2R => sum = eps N * 2R; core
  likewise), the ratio assembled, 2N cancelled, and the sigma_r -> 0 limit
  taken in three rows ((1-eps)sigma_r -> 0 => S_MSE -> eps R/eps R = 1);
  the numeric gloss now shows the arithmetic (eps R = 0.05x0.5 = 0.025,
  (1-eps)sigma_r = 0.95x0.02 = 0.019, 0.025/0.044 ~ 0.57). Step 12: the
  SGD comparison spelled out (pull wins iff lambda|dl/dc| > lambda eta,
  cancel lambda, apply each slope). Step 13: L_total differentiated by
  linearity before the two pulls are evaluated. Checks frame unchanged
  (gate reference now steps 5-6). Section: 8 -> 13 frames, steps 1-13 +
  checks; full deck 102 -> 107 pages, sub_07 25 -> 30. Verified: probe +
  both decks, zero overfull from the section (culprit rows probe-bisected:
  reassembled amplitude-law row split, differentiate-frame tail row split,
  noise-floor iff row shortened and its L1 row split, frame-1 vbox
  reclaimed via 1pt continuation skips); pdftoppm rasters of all thirteen
  pages (full 69-81) inspected, double-arrow continuation on the step-8
  frame fixed and re-rendered — chains fit, annotations aligned, Huber
  table intact, winner frames start at 82.

- L1-VS-MSE SHORTHAND VARIABLES ELIMINATED 2026-07-12 (user: i don't like
  this created extra variables like q, use the definitions): every
  abbreviation symbol the recut had introduced is replaced by its defining
  expression, carried through the chains. q is gone from all frames — the
  laws read p(a|x) = Pr(a=0|x) delta(a) + Pr(a=A|x) delta(a-A) and
  p(mu|x) = Pr(mu=mu_1|x) delta(mu-mu_1) + Pr(mu=mu_2|x) delta(mu-mu_2)
  (each split over two aligned rows with an \hspace continuation), the mean
  rows expand as E[a|x] = Pr(a=0|x)*0 + Pr(a=A|x)*A and E[mu|x] =
  Pr(mu=mu_1|x) mu_1 + Pr(mu=mu_2|x) mu_2, the median cases condition on
  Pr(a=0|x) >= 1/2 / Pr(a=A|x) > 1/2 (mirrored for mu via
  Pr(mu <= mu_1|x) = Pr(mu=mu_1|x)), and the mode distances become
  E[mu|x] - mu_1 = Pr(mu=mu_2|x)(mu_2-mu_1) etc., ending in
  min_j Pr(mu=mu_j|x)(mu_2-mu_1), split over two rows. Delta is gone —
  separations are written mu_2-mu_1 everywhere (including step 7's mode-gap
  gloss, and the near-tie jump is now |theta_k - theta_k'|); tau_a and
  tau_mu are gone — the mask indicator reads 1[a_k^gt > 1e-3] and the match
  rule |mu-hat - mu_j| <= 5 with the units named in the row. sigma_r, R,
  eps and S stay: they are the defining parameters of the residual
  populations and the share, not abbreviations of expressions. Frame and
  page counts unchanged (8 frames; full 102, sub_07 25). Verified: probe +
  both decks rebuilt, zero overfull from the section (the amplitude law row
  split after a 2.5pt overfull); pdftoppm rasters of the four touched pages
  (full 69, 71, 72, 73) inspected — split law rows align under their
  right-hand sides, chains fit their columns, columns [c]-centred.

- L1-VS-MSE NOTATION MADE RAW, SECTION EXPANDED TO EIGHT FRAMES 2026-07-12
  (user: a lot of mathematical notation was simplified, I would rather have
  it raw, despite making it much bigger): every compressed prose-annotation
  row of the argument-line recut is re-expanded into explicit derivation
  rows. The supervision-rule display now writes the full double sum
  l_param = sum_k sum_{p in {a,mu,sigma}} m_k^(p) l(theta-hat_pi(k)^(p),
  theta_k^(p)) with the mask cases on their own row and tau_a = 1e-3 named
  in the gloss; steps 1-2 define q as an explicit posterior
  (q = Pr(a_k^gt = A | x), q = Pr(mu = mu_1 | x)) before stating each
  mixture law. The single evaluation frame is split in two: step 5
  (amplitude) integrates the mean row by row (int a p da = (1-q)*0 + q*A,
  then 0 < qA < A) and derives the median from the CDF with both q-cases as
  separate rows; step 6 (position) restores the betweenness row
  mu_1 < E < mu_2, derives both mode distances E - mu_1 = (1-q)Delta and
  mu_2 - E = q Delta as separate rows, takes the min over modes, then prices
  it against the match rule |mu-hat - mu_j| <= tau_mu = 5 in a second chain
  ending at one false alarm + one miss. The matcher/gradient frame is split
  in two: step 7 writes both pairing rules raw (pi_sort = rank of mu^gt;
  pi_t = argmin_pi sum_k m_k |theta-hat_pi(k) - theta_k|, the cost mask's
  empty-slot consequence in the gloss) and states the two residual
  populations with the pinned-vs-shrinking row; step 8 differentiates both
  losses explicitly (|dl/dc| = 2|c-theta| = 2|r| and |sign(r)| = 1), defines
  the share S as the sum ratio, then evaluates S_MSE (sum form, simplified
  form with the sigma_r -> 0 limit as an xrightarrow) and S_L1 = eps with
  its full denominator, restoring the 0.025/0.044 ~ 0.57 arithmetic in the
  gloss. The endgame is renumbered step 9; steps 3-4 and the checks frame
  are unchanged. Section now 8 frames, steps 1-9 + checks; full deck
  100 -> 102 pages, sub_07 23 -> 25. Verified: probe + full + sub_07 builds,
  zero overfull from the section after splitting the mode-distance double
  row and tightening the match-rule chain (12.5pt -> 8.5 -> 3.4 -> 0, probe
  bisection); pdftoppm rasters of all eight full pages 69-76 inspected, sub
  pages 21 and 25 spot-checked (shared section files, identical geometry) —
  chains fit their columns, columns [c]-centred, Huber table intact, winner
  frames start at 77.

- L1-VS-MSE ARGUMENT LINE INVERTED 2026-07-12 (user: the derivations seem too
  generic, don't argue why that would be the case in this network — change
  the argumentation line): 11b_l1_vs_mse no longer opens with textbook risk
  minimisation; it opens with the pipeline, and the generic lemmas become
  instruments applied to it. Steps 1-2 derive the label laws from the head's
  supervision rule (w^(a)=1 on every slot, so the amplitude law is
  spike-and-slab BY CONSTRUCTION: p(a|x)=(1-q)delta(a)+q delta(a-A); mu/sigma
  masked to occupied slots, so the position law is two-valued between true
  elevations under layover with separation Delta set by the scene) — grounded
  in pipelines/backbone/training/loss.py param_mask and tied to the recall
  paradox's measured ambiguous slice (activity recall cap ~0.96 phi-only).
  Steps 3-4 keep the mean/median conditional-risk chains verbatim but
  repositioned as "one instrument from decision theory" applied to those
  point-mass laws. Steps 5-6 price both targets with the benchmark: qA is the
  recall-paradox hedge reproduced at the loss (param-L1 deletes it for gated
  and ungated heads alike); the position mean sits min(q,1-q)Delta from the
  nearer mode, and past tau_mu = 5 elevation units (inference metrics
  match_tol) it is matched to neither GT — one precision hit AND one recall
  hit, while the median always sits on a real scatterer. Step 7 derives the
  residual tails from the matcher itself (sorted = GT-rank wiring, a drifted
  slot trains on the wrong target, the K=3 instability; Hungarian = per-step
  argmin, a near-tie flips the target) and adds the convergence limit:
  S_MSE -> 1 as sigma_r -> 0 while S_L1 = epsilon at every stage — the tails
  are pinned by scene and matcher, so MSE's wasted gradient share GROWS as
  training succeeds (the assumed-scales instantiation 0.57 is kept as the
  "already at" illustration). Step 8 keeps the parking/fade derivations but
  names the real competing terms (the gated head's presence BCE, the sweep
  composites' curve terms) and states the handback: a fading param-MSE
  returns its pixels to the broadcast curve terms — the composition ambiguity
  the param loss was chosen to remove. The generic Gaussian-vs-Laplace
  ML/asymptotic-variance frame is cut; its one useful sentence lives on in
  the closing kurtosis check. The check frame gains the gate cross-check read
  off the existing table: the L1-MSE gap compresses to 0.049-0.058 on the
  set-pred rows vs spreads up to 0.081/0.127 ungated — the surviving band is
  the position-and-matcher tail no gate can remove. Still six frames; page
  counts unchanged (full 100, sub_07 23). Verified: tectonic full + sub_07
  builds, zero overfull from the section after probe-bisecting three culprit
  rows (Delta row tightened, drifted-slot row shortened, S_MSE double
  fraction split across chain rows); \mathbb 1 rendered a broken glyph (msbm
  has no blackboard digit) and was replaced by \mathbf 1; pdftoppm rasters of
  full pages 69-74 and sub pages 18+23 inspected — columns [c]-centred,
  chains fit their columns, Huber table intact, winner frames untouched at
  75+.

- L1-VS-MSE EQUATIONS REBUILT FROM DEFINITIONS 2026-07-12 (user: the
  equations are jumping steps, are not starting from the definitions, focus
  on the equations): every column of 11b_l1_vs_mse is now a \Longrightarrow
  chain that opens with an explicit definition, in the 09_error_theory
  grammar. The frame of steps 1-2 opens with the conditional risk
  R(c|x) = E[l(c,theta)|x] = int l p dtheta above the columns; step 1 states
  l=(c-theta)^2, expands the square, differentiates, sets to zero; step 2
  states l=|c-theta|, splits the integral at c (boundary terms vanish),
  differentiates to the two masses, balances them at the median. Steps 3-4
  now write the two-point label laws p(a|x) = (1-q)delta(a)+q delta(a-A) and
  p(mu|x) = q delta(mu-mu1)+(1-q)delta(mu-mu2) and integrate to mean/cdf/
  median, with the strict betweenness line mu1 < E < mu2. Step 5 defines the
  batch loss L = (1/N) sum l, takes per-element slopes, states the two
  residual populations, and computes the artefact share as a sum ratio
  (0.025/0.044 ~ 0.57 vs epsilon = 0.05). Step 6 defines the SGD update
  Delta c = -lambda dl/dc + xi with E|xi| = lambda eta before deriving the
  parking level, and derives the composite fade from L_total = w l_param +
  sum w_t l_t. Step 7 writes both densities N(r), Lap(r) explicitly, takes
  -log to recover each loss, then derives Var[med] = pi s^2/2n and b^2/n
  from Var[med] = 1/(4 f(0)^2 n) case by case to reach pi/2 and 2. Frame
  and page counts unchanged (6 frames; full 100, sub_07 23). Verified:
  rebuilt both decks, zero overfull from the section after three row-splits
  (S_MSE double fraction, the density pair, the Laplace variance row);
  pdftoppm rasters of full pages 69-74 and sub pages 18+23 inspected ---
  chains fit their columns, columns stay [c]-centred, Huber-check frame
  untouched.

- L1-VS-MSE SECTION RE-CUT TO SIX FRAMES 2026-07-12 (user: too rushed, needs
  to be more step by step; slides too crowded, too cramped; make the two
  columns always vertically centralized): the three dense frames of
  11b_l1_vs_mse re-cut into six — steps 1-2 (mean vs median targets), steps
  3-4 (two-valued amplitude / layover position), step 5 (residual kinds and
  the artefact share of the gradient), step 6 (noise-floor parking and the
  composite fade), step 7 (Gaussian vs Laplace ML, variance ratios), and the
  Huber board check as its own closing frame. One step per column, at most
  two short displays per column, glosses as soft scriptsize lines under each
  column, punchline as an accent line at frame bottom, lead sentence at the
  top with \vfill air around the column block; every columns environment in
  the section is now [c] (vertically centred), no [T] remains. Verified:
  tectonic full deck -> 100 pages (97+3), sub_07 -> 23 pages (20+3), zero
  overfull warnings from the section; pdftoppm rasters of full pages 69-75
  and sub pages 18-23 inspected — all six frames airy, columns centred,
  winner frame intact after the shift.

- L1-VS-MSE SECTION ADDED 2026-07-12 (user: turn the "why does param-L1 beat
  param-MSE" benchmark discussion into its own section, own file): new
  sections/11b_l1_vs_mse.tex, three two-column step frames in the
  error-theory grammar. Steps 1-2 derive each loss's closed-form target
  (conditional mean vs conditional median) and evaluate both on the
  two-valued labels — spike-and-slab amplitude E=qA vs med=0-or-A, layover
  position E=q·mu1+(1-q)·mu2 vs a true mode; tied to step 8's dimming.
  Steps 3-4 the gradient budget: slope 2|r| vs 1, pairing artefacts take
  ~57% of the MSE batch gradient vs 5% under L1 at stated scales (R=0.5,
  sigma_r=0.02, eps=0.05), and MSE parks at |r|~eta/2 under the optimiser
  noise floor eta while L1 pushes to zero and never fades inside the
  composite objective. Step 5 the likelihood view: MSE = Gaussian ML, L1 =
  Laplace ML, median-vs-mean asymptotic variance pi/2 vs 2, so the board
  margin itself measures the residual law; closed by the Huber
  interpolation check — all eight completed grid cells tabled, p-MSE <=
  p-Huber <= p-L1 holds in seven, the exception a 0.002 tie (Att-UNet
  conv-sorted). Remaining check flagged on the slide: matched-residual
  kurtosis from the loss-pair sweep. Wired after 11_loss_design in the
  full deck; appended to sub_07_loss_theory, its subtitle extended.
  Verified: tectonic full deck -> 97 pages (94+3), sub_07 -> 20 pages
  (17+3); two width fixes during the pass (per-row gloss columns moved to
  gloss lines under the displays on the steps-1-2 frame, likelihood display
  split to two rows on the step-5 frame) leave zero overfull warnings from
  the new file; pdftoppm rasters of full pages 69-72 and sub pages 17-20
  inspected — new frames clean, winner frame intact after the +3 shift.

- SUB-PRESENTATIONS RE-CUT ALONG CONTENT THEMES 2026-07-12 (user: don't use
  the act structure, isolate themes by content): the ten act-based sections
  re-cut into 18 theme sections at frame-level boundaries — Act II split into
  dataset_labels vs representation (normalization belongs with the channel
  representation, not the stack); Act III into stability (clamps, warmup,
  clipping), set_prediction (the small-K capacity frame opens it — its own
  text hands over to the sorting mechanism — through the 12-run imbalance
  grid) and augmentation; Act IV into context_information vs error_theory
  (steps 1-9); Act V into benchmark_board, loss_design (the 29b-f
  composition-ambiguity run is theory, not board reading) and winner; Act VI
  into physics_losses vs unrolled (the where-does-the-physics-live bridge
  frame opens the unrolled chunk). The \act/\subact divider lines moved
  inline into full_project_story.tex so no sub-deck carries an act title.
  Ten thematic sub-decks compose these: 04_stability = stability +
  augmentation, 07_loss_theory = error_theory + loss_design, 08_benchmark =
  board + winner, 09_physics = losses + unrolled; the rest map one-to-one.
  Old sub_01..07 act decks deleted. Verified: full deck rebuilt -> 94 pages,
  all 94 pdftoppm rasters bit-identical to the pre-split reference; all ten
  sub-decks compile and every content page raster is bit-identical to its
  full-deck page (sub_01 -> 3,5-7; sub_02 -> 9-15; sub_03 -> 16-19; sub_04 ->
  22-24,36-38; sub_05 -> 25-35; sub_06 -> 40-46; sub_07 -> 47-57,64-68;
  sub_08 -> 59-63,69-72; sub_09 -> 74-83; sub_10 -> 85-88; dividers, front,
  close and backup remain full-deck-only).

- DECK SPLIT INTO SHARED SECTIONS + SEVEN SUB-PRESENTATIONS 2026-07-12 (user:
  keep the full deck but create per-topic sub-presentations in separate files
  that the full one reads from): full_project_story.tex (4773 lines) factored
  into preamble.tex (old lines 2-66) + ten sections/ files cut exactly at the
  `% ==== ACT` comment lines (content byte-identical, structural comments
  kept) — the full deck is now a 23-line assembly of \input statements, and
  wrappers sub_01_problem ... sub_07_jepa give each act a standalone deck.
  Verified: tectonic on the reassembled full deck -> 94 pages, all 94
  pdftoppm rasters bit-identical to a pre-split reference build; all seven
  sub-decks compile and every content page raster is bit-identical to its
  consecutive full-deck page range (sub_01 -> pages 04-07, sub_02 -> 08-19,
  sub_03 -> 20-38, sub_04 -> 39-57, sub_05 -> 58-72, sub_06 -> 73-83,
  sub_07 -> 84-88; front 01-03 and close/backup 89-94 remain full-deck-only).

- VARIANCE-DECOMPOSITION CLOSING ADDED TO SUMMED-UP FRAME 2026-07-10 (user:
  add the variance decomposition as the step-8 closing): 25f-a3 (page 56)
  right column — the table gloss trimmed to its first sentence (columns span
  the ablation runs) and followed by "The closing identity": law of total
  variance display Var(a2) = Var(E[a2|K]) + E[Var(a2|K)] with underbraces
  "between --- does it exist?" / "within --- how bright?", then the reading:
  phase separates the buckets and kills the between term (its residual is
  the pure within spread, a-MAE 0.420), amplitude carries the pixel scale
  and is the only input that shrinks within, no lone input kills both — the
  full stack can once the gate (step 9) stops one wire from re-mixing the
  two questions (this replaces the old "each network answers the question
  its input informs" sentence, which it subsumes). Verified: tectonic,
  page 56 rendered and inspected — identity and closing text inside the
  column, page count 94.

- FREEBIE LEDGER ADDED TO THE FUNNEL FRAME 2026-07-10 (user: add the freebies
  framing to the funnel slide): 25f-a2b right column's closing soft gloss
  ("MSE itself pays for the hedge...") replaced by "The freebie ledger" —
  the 74% inactive pixels are free for whoever can identify them; four
  what-each-strategy-pays bullets: hedge rents them (0.26 a2_bar apiece, the
  funnel never lets it), collapse takes them paid with every active,
  phi-only takes them AND keeps the actives (can tell the two apart), gate
  takes them through the logit with the brightness wire untouched. The old
  gloss's optimiser-vs-loss message is folded into the hedge line. Verified:
  tectonic, page 55 re-rendered and inspected — ledger inside the column,
  page count 94.

- STEP 8 MECHANISM FRAME (COLLAPSE FUNNEL) ADDED 2026-07-10 (user: the "why
  does MSE collapse channel 2" answer landed — add it to the slides as an
  arrow sequence): new 25f-a2b "Step 8, the mechanism --- how training falls
  past the mean" (page 55, between the continued frame and summed-up), deck
  93 -> 94 pages (81 content frames, header re-measured). Left column
  (0.56): five-box TikZ arrow chain (chainbox style, flow arrows) — 74% of
  labels agree push slot 2 to 0 (coherent majority vs scattered 26%) ->
  output shoved past the floor at 0 (dominant mode first) -> below the floor
  the clamp leaks, recovery gradient x1e-3 (three stacked leaky floors
  0.1^2 x 0.1, GaussianClamp tools/data/gaussians.py:154-162 + normalisation
  compress/decompress, as documented in the webui equation library) -> no
  reward to climb back, Lambda(x)~1 cannot find the pairs (only the hedge's
  sliver 0.26->0.19 a2bar^2 through a damped gradient) -> slot parks at the
  floor, collapse (argA-tinted terminal box, under-count 0.31 ~ pair mass).
  Right column (0.42): "Two ways out of the funnel" — two good-tinted boxes:
  phi-only never falls in (can identify the pairs, full 0.26 a2bar^2 targeted
  reward holds pair pixels above the floor; under-count 0.05) and the gate
  removes the door (gradient to a2_hat scaled by sigma(e2) -> 0 on empty
  pixels, the 74% rerouted into the logit; 0.284 -> 0.019), closed by a soft
  gloss "MSE itself pays for the hedge --- the collapse is the optimiser's
  doing". Two render passes: first had node distance 0.22cm (flow arrowheads
  invisible, gap smaller than the 2mm Stealth head) and the tikzpicture
  baseline sinking the left column below the [T] top — fixed with node
  distance 0.42cm + baseline=(current bounding box.north). Verified:
  tectonic zero warnings in the region, page 55 rendered with pdftoppm and
  inspected (arrows visible, columns top-aligned), pages 54-57 titles in
  order (8-continued / 8-mechanism / 8-summed-up / 9), page count 94.

- STEP 8 SUMMED-UP FRAME ADDED 2026-07-10 (user: after a long didactic Q&A on
  the paradox, add the more didactic explanation to the step-8 slides with a
  summary table): new 25f-a3 "Step 8, summed up --- the paradox side by side"
  (page 55) between the continued frame and step 9, deck 92 -> 93 pages (80
  content frames, header re-measured). Left column, two arguments born in the
  Q&A: (1) "MSE never wants the majority call" --- the classifier intuition
  (output 0, win the 74%) vs the hedge, costed among constants with pairs at
  a2_bar: 0.26·a2_bar^2 vs 0.19·a2_bar^2, so the MSE optimum is always the
  hedge and 0.26·a2_bar >> tau — a head AT the optimum would have no recall
  paradox; (2) "Collapse needs all three" --- 74% zero mass (the force) +
  blind presence Lambda(x)~1 (nothing to earn back on the 26%) + one wire for
  two questions (nowhere to park the doubt), with the two measured controls
  in the gloss proving each is necessary (phi-only shares the imbalance and
  holds at under-count 0.05; the gate shares inputs and imbalance and holds,
  0.284 -> 0.019). Right column, side-by-side table (house layout:
  bullets-left + table-right): rows evidence Lambda(x) / presence Pr /
  brightness E / MSE pull / trained slot lands / under-count / recall /
  a-MAE; columns amplitude-informed (cluster ranges 0.26-0.31, 0.65-0.67,
  0.33-0.36 spanning 5|A|/30|A|/reduced) vs phi-only (0.05, 0.87, 0.420);
  closing gloss "each network answers well exactly the question its input
  informs --- one wire forces the trade; the gate (step 9) removes it".
  Verified: tectonic zero warnings in the region, page 55 rendered with
  pdftoppm and inspected (table inside the column), pages 53-56 titles
  confirmed in order (8 / 8-continued / 8-summed-up / 9), page count 93.

- LAMBDA REGIME QUANTIFIERS + BOTH EXTREMES 2026-07-10 (user: the regime rows
  dropped Lambda's argument, inviting the reading "Pr(K=2|x) -> 1 means every
  input has a second scatterer"; make each regime's pixel set explicit and add
  the Lambda << 1 case): 25f regime block reworked — |A| row now "Lambda(x) ~ 1
  on every pixel => Pr ~ pi"; the phi-only block gains a dichotomy header
  ("every pixel lands at an extreme:") and both cases — Lambda(x) >> 1 on true
  pairs => Pr -> 1 => undimmed, and NEW Lambda(x) << 1 on singles => Pr -> 0 =>
  a2_hat -> 0, the slot closes (decisive both ways — the mechanism behind
  phi-only's 0.85 precision quoted on 25f-a2). First pass clipped: the one-row
  << 1 case ran past the column edge with NO overfull warning (gotcha
  re-confirmed — pdftoppm render-verify is the only reliable check); reflowed
  to two rows per case, row gaps 2/3pt -> 1/2pt, pre-gloss smallskip dropped,
  handoff gloss shortened to one clause. Verified: tectonic, page 53 rendered
  and inspected — all rows inside the column, page count unchanged at 92.

- STEP 8 SHRINKAGE-TO-COLLAPSE CORRECTION 2026-07-10 (audit: the "pi·E < tau"
  bridge fails arithmetically — pi = 0.26 measured, amplitude scale O(0.1–1)
  per the ablation table's a-MAE 0.33–0.72, tau = ACTIVE_AMP_THR = 1e-3, so
  conditional-mean shrinkage lands at ~0.03–0.13, two orders ABOVE the
  threshold; a head that truly converged to the conditional mean would keep
  every second slot active and the paradox would not exist. The measured
  state is stronger than shrinkage: the trained conv head collapses slot 2
  below 1e-3 on nearly all pair pixels): both step-8 frames now present the
  conditional mean as the PULL and the measured collapse as where the trained
  slot LANDS. 25f "Step 8 — the recall paradox" (page 53): "and the trained
  head converges to it" -> "the value the loss pulls every output toward";
  coupling sentence extended ("Doubt can only dim … the head has no wire for
  present but uncertain"); pi stated as 0.26 measured; the Lambda regime
  block no longer concludes "< tau => missed / > tau => detected" — the |A|
  rows end at "a2_hat pulled to 0.26·E[a2|K=2,x]" and the phi rows at
  "-> E, undimmed", closed by a soft handoff gloss ("the fourfold dimming is
  only the pull — where the trained slot actually lands … is the next
  frame"). 25f-a2 "Step 8, continued" (page 54): the inversion line gains the
  operating-point inoculation (phase-only wins precision too, 0.85 vs 0.82,
  both from the ablation table — recall inversions only bind at matched
  precision); the faint-pixel burial paragraph REPLACED by "Past the mean,
  onto the floor" — even total doubt stops the mean at 0.26·E, orders above
  tau = 1e-3, shrinkage cannot bury a slot, yet the measured active counts
  (results/extra_tracks_exp/metrics_comparison(1).md: baseline
  active_count_pred_mean 1.01 vs GT 1.2605, count_under_frac 0.31438;
  phi-only 1.2965 / 0.052993) show the slot buried on ~every pair pixel:
  collapse past the mean onto the zero branch, not calibration. Right column
  retitled "Two brightness branches (steps 3–4)": the reduced branch is
  pixel-calibrated (brightness and presence both vary, one wire carries their
  product, the wire goes dark against the zero mass) while the phase branch
  is the single constant a2_bar with Pr·a2_bar >> tau unless Pr ~ 0; the
  check gloss now decomposes recall — activity alone caps it at ~0.75
  (reduced, (1.2605-0.314)/1.2605) vs ~0.96 (phi-only), and both sides lose
  ~9 further points to the 5-unit mu tolerance (match_tol = 5.0 in
  pipelines/backbone/inference/metrics.py; detection = active slot AND
  matched |dmu| <= 5), so the whole 21-point gap is the buried slot, not
  localisation; the un-shrunk-guess cost kept (a-MAE 0.420 vs 0.33–0.36,
  recall 0.87). Step-9 frame untouched — its "step 8's multiplicative
  coupling is cut by construction" line still holds, and its gate result is
  the causal patch for "must be the brightness factor" (same inputs,
  decoupled head, recall recovers). Verified: tectonic clean in the region
  (9 overfulls, all pre-existing, none in 25f/25f-a2), pages 53–54 rendered
  with pdftoppm and inspected — columns inside bounds, all glosses visible,
  page count unchanged at 92.

- STEP 5/6 REBALANCE 2026-07-10 (user: step 6 heatmap bigger and vertically
  centred in the right column; step 5 drop the "Which mistake is dearer?"
  text and move the left column's first 2 rows to the top of the right
  column): 25e "Step 5" (page 49) — the 2-row partition/R^2 summary block
  (a<-|A| only ... 0.61 united) moved from the top of the left column to the
  top of the right column, above "Take the derivative"; the bold question
  reduced to "Set the two errors equal:" so the equal-error display keeps its
  intro and the misplacing-vs-misscaling punchline stays grounded. The move
  made the right column overfull by 29.6pt, reclaimed without content loss:
  frame displayskips 3pt -> 1pt, right-column row gaps [2pt]/[3pt] -> [1pt],
  the two right-column smallskips dropped, and the soft derivative-peak/2E
  ceiling aside relocated to the bottom of the now-short left column as a
  footnote-style remark — overfull gone. 25e-b "Step 6" (page 50) — columns
  rebalanced 0.55/0.43 -> 0.52/0.46 with the heatmap at full linewidth
  (~13% larger), and the columns environment switched [T] -> [c] so the
  image centres vertically against the taller math column. Verified:
  tectonic clean for both frames (no new overfulls; 3425/3782 pre-existing
  elsewhere), pdftoppm render of pages 49-50 inspected — nothing clipped,
  heatmap enlarged and centred, page count unchanged at 92.
- REDUCED-STACK DETECTION FRAME ADDED 2026-07-10 (user: the deck has no section
  explaining why the reduced stack does not detect as well as phase-only): new
  25f-a2 "Step 8, continued — why the reduced stack loses detections" (page 54)
  between the paradox and the gate, deck 91 -> 92 pages (79 content frames,
  header re-measured). Left column: the information-inequality opener — the
  reduced stack contains the phase channels and an ideal detector may ignore
  inputs, so ideal recall(x_red) >= ideal recall(x_phi); measured 0.66 vs 0.87,
  hence the inversion lives in the trained estimator, and since both inputs
  carry the same phase evidence the difference must sit in step 8's brightness
  factor; then the amplitude side — E[A_n^2] = P (step 3) means the stack's
  E[a2|K=2, x_red] is calibrated per pixel, small on faint pixels, and the
  posterior multiplies it under tau (missed). Right column: the phase side —
  scale-blindness (step 4) fixes the split but never the pixel's brightness,
  so E[a2|K=2, x_phi] ~ its prior mean a2_bar and Pr·a2_bar clears tau
  (detected): phase-only detects more BECAUSE it knows less about brightness;
  one estimate serves two masters, calibration trades against detection. This
  is the first-pass 25g "worst a-MAE and best recall are the same un-shrunk
  guess" mechanism, now given its own frame with the class-mean argument made
  explicit. Check gloss: amplitude-informed columns cluster at a-MAE 0.33–0.36
  / recall 0.65–0.67 while phase-only sits at 0.420 / 0.87 (clusters used
  deliberately — the per-column ordering is not perfectly monotone: 5|A| has
  both the best a-MAE 0.328 and recall 0.67 above 30|A|'s 0.65), gate pointer
  to step 9. Verified: tectonic zero warnings in the region, page 54 rendered
  with pdftoppm and inspected, pages 53/55 titles confirmed in order (8 /
  8-continued / 9).

- STEP 8 DERIVED INSTEAD OF ASSERTED, GATE SPLIT TO STEP 9, PRIOR CORRECTED
  2026-07-10 (user: step 8 has equations that come out of nothing, results not
  justified like amp-only Lambda ~ 1 and phase-only Lambda >> 1 — improve): the
  old single frame stated the factorised estimator, Bayes posterior, and both
  Lambda regimes as bare display rows with all symbols dumped in one intro
  sentence. Now 25f "Step 8 — the recall paradox: MSE turns doubt into dimming"
  (page 53) derives the chain: left column starts from the loss (squared error
  is minimised by the conditional mean, so the trained head converges to
  a2_hat(x) = E[a2|x]; gloss declares x and that the a2 label is defined 0 on
  K=1 pixels), then total expectation over the two cases with the zero case
  dropping, giving the underbraced presence-times-brightness product; right
  column writes presence as the Bayes posterior pi Lambda/(1 - pi + pi Lambda)
  with Lambda(x) = p(x|K=2)/p(x|K=1) DEFINED as the likelihood ratio, then
  justifies the regimes from steps 3–4: amplitudes see a second scatterer only
  as a small speckle-buried dent in |rho| (both hypotheses explain the input
  almost equally well => Lambda ~ 1 => a2_hat ~ pi E[a2|.] < tau, missed) while
  phases see it bend the line arg rho(k_n) = k_n mu (=> Lambda >> 1, detected).
  PRIOR CORRECTED: the old frame claimed pi = Pr(K=2) ~ 0.04 (the first-pass
  95.4/4.1 prior bar); both cited results files measure slot_1_active_gt_frac =
  0.26048 (results/extra_tracks_exp and results/benchmark_experiment
  metrics_comparison(1).md, active_count_gt_mean 1.2605) — the slide now says
  26% and the check gloss ties conv-head under-count 0.284 ~ the whole 0.26
  second-slot mass (shrinkage buries nearly every second scatterer), keeping
  the matched-pairs information-order check. The 45.3% on the best-K-map slide
  is a different measurement (penalised-MSE model selection over the full
  param_extraction scene, not these runs' acre) — no conflict. The gate moved
  to its own 25f-b "Step 9 — the fix: the set-prediction gate" (page 54): left
  column the gate equation and its closed/open limits, right column the
  saturation argument (doubt moves the logit, not the brightness — step 8's
  multiplicative coupling cut by construction) and the head-comparison check
  numbers. Deck 90 -> 91 pages (78 content frames, header re-measured); the
  25e-c "(step 8)" gloss still points at the paradox frame. Verified: tectonic
  zero warnings in the region, pages 53–54 rendered with pdftoppm and inspected
  — columns inside bounds, all glosses visible.

- K-SCATTERER PAIR CONDENSED TO ONE FRAME 2026-07-10 (user: keep it more simple —
  argue the overlap term cancels assuming Gaussian separation big enough): the
  two-frame version below (25e-d exact expansion + 25e-e coupling bound) becomes
  a single 25e-d "Step 7 — the full profile: separated scatterers decouple"
  (page 52), deck 91 -> 90 pages (77 content frames, header re-measured). Left
  column unchanged (residuals, exact bilinear split, diagonal = K step-6
  surfaces). Right column replaced by the assumption argument: the pair overlap
  in proportional form <a_j N(z;m,sigma_j), a_k N(z;m',sigma_k)> prop
  e^{-(m-m')^2/2 Sigma_jk^2} (gloss keeps only the joint width and
  product-of-Gaussians note; the j=k/step-5 cross-check dropped for space), then
  a Longrightarrow-per-row chain: |mu_j - mu_k| >= 3 Sigma_jk ==> factor <=
  e^{-9/2} ~ 0.01 ==> <r_j, r_k> ~ 0 ==> ||p_hat - p||^2 ~ sum ||r_k||^2, closing
  with each scatterer keeping its own valley, ceiling 2E_k, and bowl, and the
  same 2-scatterer-rows check gloss. Dropped relative to the rigorous version:
  the explicit four-overlap expansion, the B_jk bound with the s = 2/3/4 numbers,
  the gradient corollary, and the count-mismatch ledger. 25f stays "Step 8"; the
  25e-c handoff now reads "step 7 shows that separated scatterers simply add".
  Two overfull rounds fixed en route: the one-line implication chain 23pt too
  wide (split to one implication per row) then the column 25pt too tall (j=k
  gloss check dropped, lead-in tightened, last two chain rows merged — that
  width fits). Verified: tectonic clean in the region, pages 51–53 rendered with
  pdftoppm and inspected — columns inside bounds, closing gloss visible, step
  numbering consistent (7 then 8).

- K-SCATTERER DEVELOPMENT MADE RIGOROUS 2026-07-10 (user: the amplitude-only math
  slides argue for 1 Gaussian then super-simplify K Gaussians at once — make the
  full development rigorous and complete): the one-display "+ cross-overlaps"
  shortcut at the bottom of 25e-c is replaced by two new frames, deck 89 -> 91
  pages (78 content frames, header re-measured). 25e-c retitled "Step 6, continued
  — the bowl and the training pull", its bottom block now a two-line handoff and
  its gloss's "(next slide)" fixed to "(step 8)". New 25e-d "Step 7 — the full
  profile: the exact K-scatterer error" (page 52): left column writes truth and
  prediction as K Hungarian-paired components, residuals r_k = (1+eps_k)
  N_{mu_k+delta_k} - N_{mu_k} (energy E_k = a_k^2/(2 sqrt(pi) sigma_k) declared in
  a gloss), then the exact bilinear expansion ||p_hat - p||^2 = sum ||r_k||^2 +
  2 sum_{j<k} <r_j, r_k> with the diagonal identified as K exact copies of the
  step-6 surface; right column derives the closed-form pair overlap O(u) =
  sqrt(E_j E_k) w_jk e^{-u^2/2 Sigma_jk^2} with Sigma_jk^2 = sigma_j^2 + sigma_k^2
  and w_jk = sqrt(2 sigma_j sigma_k)/Sigma_jk <= 1 (gloss checks j=k recovers step
  5's E e^{-delta^2/4 sigma^2}), then expands one cross term exactly into its four
  overlaps at Delta_jk + delta_j - delta_k, Delta_jk + delta_j, Delta_jk - delta_k,
  Delta_jk. New 25e-e "Step 7, continued — the coupling dies with separation"
  (page 53): left column proves |<r_j,r_k>| <= B_jk = (2+eps_j)(2+eps_k)
  sqrt(E_j E_k) e^{-s_jk^2/2} with s_jk the shift-discounted separation in joint
  widths (coefficients sum to (2+eps_j)(2+eps_k) since (x+1)(y+1) = xy+x+y+1), and
  puts in numbers 4e^{-2} ~ 0.54 / 4e^{-4.5} ~ 0.04 / 4e^{-8} ~ 0.001 for s =
  2/3/4; right column states the decoupled form sum ||r_k||^2 + R with |R| <=
  2 sum B_jk, the gradient corollary (d O/d u = -(u/Sigma^2) O keeps the
  exponential, so each scatterer descends its own bowl), the count-mismatch ledger
  (unpaired true bump costs E_k, unpaired prediction (1+eps)^2 E — step 5's 2E
  ceiling is one of each), and a check line tying O(1) coupling to the 2-scatterer
  table rows (mu-MAE 3.75–4.58 vs 0.50–1.16, a-MAE 0.53–0.72 vs 0.14–0.21, every
  modality pays it). 25f retitled Step 7 -> Step 8. First build overfulled: the
  overlap display 36pt too wide (fixed by naming O(u) in the display over three
  short rows, which also shortened the cross-term lead-in) and 25e-e's right
  column 7pt too tall plus two ~3pt hboxes (fixed by compacting the decoupled
  display to sum ||r_k||^2 + R with the bound inline, dropping its gloss, and
  removing the Longrightarrow from the gradient display). Verified: tectonic
  clean in the new region (remaining warnings are the pre-existing set), pages
  51–54 rendered with pdftoppm and inspected — all columns inside bounds, check
  glosses fully visible, titles and step numbering consistent.

- JOINT ERROR LANDSCAPE FRAME ADDED 2026-07-10 (user: also add a shifted-and-scaled
  version, calculate the derivatives, and plot a heatmap of the error over a range
  of shifts and scales): new frame 25e-b "Step 6 — shift and scale together: the
  joint error" (page 50) between the isolated-errors slide and the paradox/gate
  frame, which is retitled Step 6 -> Step 7; deck goes 87 -> 88 pages (header
  re-measured, 75 content frames). Left column (0.55): the joint chain
  ||(1+eps) N_{mu+delta} - N_mu||^2 = (1+eps)^2 E + E - 2(1+eps)<N_{mu+delta}, N_mu>
  = E[(1+eps)^2 + 1 - 2(1+eps) e^{-delta^2/4 sigma^2}], with the eps=0 / delta=0
  slices identified as the step-5 curves; then both partials in Leibniz form —
  d/d delta = (1+eps)(E delta/sigma^2) e^{-delta^2/4 sigma^2}, d/d eps =
  2E(1+eps - e^{-delta^2/4 sigma^2}) ==> d/d eps = 0 at 1+eps = e^{-delta^2/
  4 sigma^2} < 1 ==> a misplaced bump prefers to be dim (the MSE shrinkage of the
  recall paradox, now visible in the geometry). Right column (0.43): heatmap
  results/error_landscape/joint_error_heatmap.png over delta/sigma in [0,4], eps in
  [-1,1.5], value in units of E; single-hue rose ramp built from the deck accent
  (#B03052), soft-gray contours 0.25E/0.5E/1E/3E (a 5E level clipped at the border
  and was dropped), white emphasized 2E ceiling contour (hits delta=0 at
  eps=sqrt(2), flattens onto eps=0 as delta grows), dashed deck-green shrinkage
  valley eps = e^{-delta^2/4 sigma^2} - 1 anchored at a "truth" dot on (0,0);
  colorbar labelled with the true norm expression; 300 dpi, serif/cm mathtext.
  Generated by joint_error_heatmap.py NEXT TO THE DECK (class JointErrorHeatmap,
  Dune-env python, tools.monitoring.logger Logger, repo-root walk-up bootstrap
  so it runs from a staging worktree or the real checkout; rerun: Dune python
  joint_error_heatmap.py from the deck directory). VERIFY: warning set identical
  to the shipped 8 (the 0.55-column Leibniz lines fit), pages 50-51
  pdftoppm-verified, 88 pages.
  SECOND PASS same day (user: "in the heatmap add the d/depsilon = 0 curve too" —
  the d/d-eps valley was already plotted, so read as completing the critical-curve
  picture): the heatmap gains the d/d delta = 0 locus as blue (deck argA #285FA5)
  dash-dot lines on its two branches — the delta = 0 axis and the 1+eps = 0 line
  (bump erased; the surface is flat in delta there at height E) — each branch
  labelled; the green valley annotation now carries its explicit equation
  1+eps = e^{-delta^2/4 sigma^2} (the bottom blue label moved to delta ~ 0.4 after
  colliding with the valley tail at first render). Slide caption extended: names
  the blue locus and closes with "the two meet only at the truth — the surface's
  one minimum" (delta=0 meets the valley exactly at (0,0); the 1+eps=0 branch is
  not stationary in eps, so no other crossing). Warning set unchanged, page 50
  pdftoppm-verified, 88 pages.
  THIRD PASS same day (user: below the heatmap add one more view of the same map
  for small deviations, to see the behaviour when the network converges): the
  script now renders TWO figures (render_landscape / render_convergence sharing
  compute_surface + decorate) and the slide stacks them in the right column at
  0.85 linewidth. The convergence view joint_error_convergence.png zooms to
  delta/sigma in [0, 0.5], eps in [-0.25, 0.25]: near the truth the surface is
  the quadratic bowl E(eps^2 + delta^2/2 sigma^2) (annotated on-plot; Taylor of
  the exact law — elliptic contours 0.02E/0.05E/0.1E, sqrt(2) wider along
  delta/sigma than eps), the valley enters tangent to eps = 0, and there is no
  plateau in sight — gradients stay informative at convergence. Both figures
  regenerated flatter for the stacked layout (5.2x2.9 and 5.2x2.0 in, base font
  10, vertical d/d delta label moved down to eps ~ 0.3; the 0.1E contour label
  manually placed at (0.39, -0.20) and the 0.15E level dropped — both clipped at
  the top border on auto placement). Slide caption compressed to top/bottom
  reading (the curve explanations live on-plot as labels). Warning set unchanged,
  page 50 pdftoppm-verified, 88 pages. WORKFLOW NOTE: the staging worktree copies
  and the results symlink are deleted by end-of-turn cleanup — every editing turn
  must re-stage BEFORE touching files; Write silently recreates the deck dir, so
  a skipped staging does not error, it just strands outputs in a real
  worktree-local results/ dir (happened this pass; symlink restored, figures
  regenerated to the real results path).
  FOURTH PASS same day (user: add one more small-scale view with gradient arrows
  in many points): third figure joint_error_gradient.png — same zoom backdrop
  (unlabelled thin contours 0.02/0.05/0.1E, valley, truth dot) with a 10x9 quiver
  of the DESCENT field -grad = (-(1+eps) (delta/sigma) e^{-delta^2/4 sigma^2},
  -2(1+eps - e^{-delta^2/4 sigma^2})) in ink, angles/scale in data units
  (scale=16), "-nabla (descent)" label on a white backing box (first render had it
  colliding with arrows). Three stacked maps could not fit one column readably,
  so the frames were re-split: 25e-b keeps the derivation chains + the landscape
  solo (0.95 linewidth, one-line caption) and NEW frame 25e-c "Step 6, continued
  — near convergence: the bowl and the training pull" (page 51) shows the
  convergence + gradient maps side by side at 0.48 textwidth each (both
  regenerated at 4.6x2.6 in for the pairing). Right caption states the
  two-time-scale reading off the Hessian of the quadratic bowl: curvature 2E in
  eps vs E in delta/sigma, so brightness is pulled onto the valley twice as fast
  as position slides along it into the truth; the dim drift feeds the recall
  paradox (next slide). Deck goes 88 -> 89 pages (header re-measured, 76 content
  frames); paradox/gate frame stays "Step 7", now page 52. Warning set unchanged,
  pages 50-51 pdftoppm-verified.
  FIFTH PASS same day (user: update the argument to the error of multiple
  superimposed gaussians — same thing with summations and indexes): the continued
  frame (retitled "Step 6, continued — the bowl, the training pull, and the full
  profile") closes with a full-width superposition block: ||sum_k (1+eps_k)
  N_{mu_k+delta_k} - sum_k N_{mu_k}||^2 = sum_k E_k[(1+eps_k)^2 + 1 -
  2(1+eps_k) e^{-delta_k^2/4 sigma_k^2}] + cross-overlaps, gloss declaring
  N_{mu_k} = a_k N(z; mu_k, sigma_k) with energy E_k, the exact cross-overlap
  decay e^{-(mu_k-mu_j)^2/2(sigma_k^2+sigma_j^2)} (gone a few widths apart;
  couples only when scatterers crowd — the hard 2-scatterer rows), and the
  per-component conclusion: each scatterer pays its own ceiling 2E_k and owns
  its own valley 1+eps_k = e^{-delta_k^2/4 sigma_k^2}. To make room the two
  per-column figure captions merged into one spanning 2-line caption and the
  maps went to 0.9 linewidth. Warning set unchanged, page 51 pdftoppm-verified,
  89 pages.

- STEP 5 SPLIT INTO ITS OWN ERROR-COST SLIDE, PARADOX MERGED INTO THE GATE FRAME
  2026-07-10 (user: separate "put the two invariances together" into its own slide
  with detailed step-by-step derivations of each error component — the maximum of the
  shift error, and how to see which mistake is more impactful): frame count unchanged
  (4 math frames, deck still 87 pages) because the displaced recall-paradox column was
  merged into the gate frame, whose old left column was already a recap of the same
  posterior-mean product.
  25e "Step 5 — put the two invariances together" (page 49): full-width partition
  display (a <- |A| only, mu <- phi only, sigma/Delta-mu <- both ==> R^2 0.41-0.47
  alone, 0.61 united), then two derivation columns. Left, the shift error:
  ||N_{mu+delta} - N_mu||^2 = 2E - 2<N_{mu+delta}, N_mu> ==> overlap = E e^{-delta^2/
  4 sigma^2} ==> cost(delta) = 2E(1 - e^{-delta^2/4 sigma^2}) ==> small-delta
  E delta^2/2 sigma^2, large-delta -> 2E; scriptsize gloss declares <f,g> = int fg and
  ||f||^2 = <f,f>; conclusion names 2E as the maximum any prediction can pay (E for
  the missed true bump + E for the spurious one). Right, the scale error:
  ||(1+eps)N - N||^2 = eps^2 E exact, 2E only at eps = sqrt(2) (141%); then "which
  mistake is dearer?" equates the costs eps^2 = 2(1 - e^{-delta^2/4 sigma^2}):
  delta = sigma <=> eps = 0.66, delta = 3 sigma <=> eps = 1.34 — no run misscales
  that badly (a-MAE 0.420 worst), so misplacement dominates the curve metrics.
  25f "Step 6 — the recall paradox, and the fix: the set-prediction gate" (page 50):
  left column is now the full paradox chain (MSE => a2_hat = underbraced
  Pr(K=2|x) x E[a2|K=2,x], Pr = pi Lambda/(1-pi+pi Lambda), |A|: Lambda~1 => missed,
  phi-only: Lambda>>1 => detected) with x, K, pi, Lambda, tau declared in the intro;
  the old coupling column's content (doubt multiplies brightness, cannot be read
  apart) is its conclusion sentence, deduplicating the product formula that appeared
  on both old slides; right column (gate chain) unchanged. The old "misses of step 5"
  cross-reference disappeared with the merge.
  VERIFY: warning set identical to the shipped log (8 pre-existing, none in the
  25c-25f range; only their log line numbers shifted with the longer source), pages
  49-50 pdftoppm-verified, 87 pages re-measured unchanged.
  SECOND PASS same day (user: "keep on the left column the 2 errors derivation the
  shift and scale, and use the right one for discussion"): 25e columns reorganised —
  left = both derivation chains (shift with the <f,g> gloss, then scale), right =
  discussion only ("Two very different laws" saturation-vs-parabola paragraph with
  the 2E-maximum decomposition, then the "which mistake is dearer?" cost-equating
  chain and the table-grounded conclusion). The move made the frame 5.9pt too tall
  (right column was the tall box — identical overfull before/after left-side trims
  gave it away): the two discussion paragraphs merged into one and the comparison
  chain gaps went 3pt -> 2pt; warning set back to the shipped 8, page 49
  pdftoppm-verified, still 87 pages.
  THIRD PASS same day (user: delete the top "each parameter..." text, break the
  partition line below it into 2 and place it at the top of the left column): the
  frame-level intro sentence is gone and the full-width partition display now opens
  the left column as a two-line chain (a <- |A| only, mu <- phi only, sigma/Delta-mu
  <- both ==> R^2 0.41-0.47 alone, 0.61 united; \qquad separators narrowed to \quad
  for column width), \smallskip before "The shift error". Shipped warning set
  unchanged, page 49 pdftoppm-verified, 87 pages.
  FOURTH PASS same day (user: "cant we use the derivate of the error to justify
  it?"): the "Two very different laws" prose paragraph became a marginal-cost chain —
  cost'(delta) = (E delta/sigma^2) e^{-delta^2/4 sigma^2} -> 0 for delta >> sigma
  ==> the shift cost flattens, 2E is a true maximum; cost'(eps) = 2 eps E > 0 at
  every eps ==> the scale cost never flattens. Scriptsize gloss keeps the two prose
  facts: the marginal cost of misplacing peaks at delta = sqrt(2) sigma (the
  expensive widths are the first ones) and 2E = E (missed true bump) + E (spurious
  one). Deliberately NOT claimed: a small-error derivative comparison — at equal
  relative errors the scale slope 2rE is actually the steeper one; the dominance
  argument stays with the cost-equating chain at the error sizes the runs commit.
  Shipped warning set unchanged, page 49 pdftoppm-verified, 87 pages.
  FIFTH PASS same day (user: don't use the term cost, use the true expression with
  derivative symbols): every cost(.) / cost'(.) on 25e replaced by the explicit
  objects — shift chain line 3 restates ||N_{mu+delta} - N_mu||^2 = 2E(1 - e^{...}),
  scale chain closes eps^2 E = 2E only at eps = sqrt(2), and the derivative chain
  reads d/d delta ||N_{mu+delta} - N_mu||^2 = (E delta/sigma^2) e^{-delta^2/
  4 sigma^2} -> 0 and d/d eps ||(1+eps)N_mu - N_mu||^2 = 2 eps E > 0 (\tfrac
  Leibniz form, xrightarrow{delta >> sigma} fits the column). Prose de-costed too:
  "Take the derivative of each error", "the shift/scale error flattens/never
  flattens", gloss "the derivative of the shift error peaks at...", "set the two
  errors equal", conclusion "misplacing by one width matches misscaling by 66%".
  Shipped warning set unchanged, page 49 pdftoppm-verified, 87 pages.

- MATH DISCUSSION REBUILT AS ARROW-CHAIN DERIVATIONS 2026-07-10 (user: the discussion
  after the amplitude-only experiment is hard to follow because "never defined,
  undeclared, never shown before variables are called everywhere"; wants the
  output-clamp-slide grammar — a series of equations, one per line, joined by arrows,
  a short intro text and a short concluding text): frames 25c-25f (pages 47-50)
  rewritten in place, count unchanged (4 frames, deck still 87 pages). Every chain
  line either uses symbols already declared or carries its declaration.
  25c "Steps 1-2 — one chain from the label to the channels, and back": single
  full-width 6-line ==>-chain, p = sum a_k N(mu_k, sigma_k) -> E[gamma gamma*] =
  p delta -> g_n = int gamma e^{jk_n z} dz = A_n e^{j theta_n} -> nine channels
  (A_0..A_4, phi_n = theta_0 - theta_n) -> R_hat_mn = <A_m A_n e^{j(phi_n-phi_m)}>_B(x)
  ~ P rho(k_m - k_n) -> Capon -> fit -> the same a_k, mu_k, sigma_k. Each line carries
  a soft scriptsize right-column gloss declaring its new symbols (a/mu/sigma_k, gamma,
  k_n prop b_perp, A_n/theta_n, B(x), P = int p, rho = FT[p/P]) — the frame doubles as
  the symbol table for the whole discussion. <.>_B(x) replaces the (1/w^2)sum_B form
  (width; w no longer needed anywhere).
  25d "Steps 3-4 — two exact invariances": per column intro line, 4-line invariance
  chain (p -> p(z-Delta) ==> gamma(z-Delta) ==> e^{jk_n Delta} g_n ==> A_n unchanged |
  p -> c p ==> sqrt(c) gamma ==> sqrt(c) g_n ==> phi_n unchanged), then a short
  "what they still carry" chain (Siegert E[A^2] = P, cov = P^2 |rho|^2 ==> sigma,
  Delta-mu | rho(kappa) = e^{j kappa mu} e^{-kappa^2 sigma^2/2} ==> arg rho = k_n mu,
  |rho| ==> sigma; kappa declared in the lead-in sentence), one-line conclusion,
  scriptsize table check.
  25e "Step 5 — the verdict follows": left = partition chain (a <- |A| only,
  mu <- phi only, sigma/Delta-mu <- both ==> 0.41-0.47 alone, 0.61 united) plus the
  misplace-vs-misscale chain (2E vs eps^2 E ==> misplacing >> misscaling); right =
  the shrinkage chain (MSE ==> a2_hat = Pr(K=2|x) E[a2|K=2,x], Pr = pi Lambda /
  (1 - pi + pi Lambda), |A|: Lambda ~ 1 => missed, phi-only: Lambda >> 1 => detected)
  with x, K, pi, Lambda, tau all declared in the intro sentence.
  25f "Step 6 — the fix: the set-prediction gate": left = coupling chain (underbraced
  a2_hat = presence x brightness ==> doubt shrinks ==> under tau ==> the step-5
  misses); right = gate chain (a2 = sigmoid(e2) a2_hat + (1 - sigmoid(e2)) a_off,
  closed/open lines, ==> presence in e2, brightness whole); e2, a_off, sigmoid
  declared in the lead-in; same table checks as before.
  FIT NOTES: three fresh overfulls cleared — 25c covariance line (38pt: the <.>_B(x)
  form + trimmed gloss), 25e Lambda lines (7pt: inner Longrightarrow -> Rightarrow),
  25f right column (16pt: conclusion line shortened to "presence in e2, brightness
  whole", closed-line tail dropped). Warning set now identical to the shipped log
  (8 pre-existing warnings, none in the 25c-25f range). Pages 47-50 pdftoppm-verified
  after the final build; 87 pages unchanged.

- FIVE MATHEMATICS FRAMES ADDED AFTER THE MODALITY-ABLATION VERDICT 2026-07-10 (user:
  "add the slides, focus on the equations, and add visualizations to each step" — the
  slides derive the amplitude-only / phase-only verdict table mathematically): markers
  25c-25g, pages 47-51, inserted between the verdict frame and ACT V. Colour key
  carried over from the verdict bullets: amplitude rail argB, phase rail argC.
  25c "The mathematics — nine channels, one function": forward model
  g_n = int gamma e^{jk_n z} dz = A_n e^{j theta_n} (k_n = 4pi b_perp/(lambda r sin
  theta), per tools/sar/geometry_field.py), the nine channels as the polar split,
  R_hat_mn = <A_m A_n e^{j(phi_n - phi_m)}>_B — the input is a sufficient statistic
  for its own label (theta_0 cancels as a gauge). Figure: vertical loop — profile card
  -> stack node -> amplitude chip + phase chip -> R_hat box -> Capon + fit -> label
  card, dashed return elbow "the same parameters — the loop closes".
  25d "Shift blindness — position never reaches an amplitude": gamma(z-Delta) =>
  g_n -> e^{jk_n Delta} g_n => A_n unchanged (plus conjugate-flip); Siegert relation
  cov(A_m^2, A_n^2) = P^2 |rho(k_m - k_n)|^2 — scale at 1st order, shape at 4th,
  position never (phase-retrieval limit: magnitudes = autocorrelation). Figure: three
  bands — shifted profiles with Delta dimension arrow, phasor circle with k_n Delta
  arc (angle moves, radius pinned), two identical 5-square amplitude rows joined
  per-pair by "=". Footnote anchors the theorem in the table: amp-only 1-scatterer
  mu-MAE 1.16 m ~ GT prior std 1.27 m (computed this day from
  test_data/params/params_k5.../parameters.npy over active single-scatterer pixels;
  single/double shares 95.4%/4.1%), and 30 |A| moves mu-MAE 2.76 -> 2.66 m only.
  25e "Scale blindness — and phase is linear in position": p -> c p leaves every
  phi_n; the phase law sees only rho(kappa) = FT[p/P]; rho = e^{j kappa mu}
  e^{-kappa^2 sigma^2/2} so arg rho(k_n) = k_n mu (sub-bin: 0.50 m vs the 0.67 m
  grid) and the window phase spread reads |rho| => sigma. Figure: three panels —
  p and c·p mapping to the same four fringe squares, arg-rho-vs-kappa slope plot
  (two slopes, four sample dots each), two phase-cloud circles (tight = high |rho|
  small sigma, spread = low |rho| large sigma).
  25f "The partition — every row of the verdict, predicted": misplaced-vs-misscaled
  cost pair (shift error saturates at 2E, scale error stays eps^2 E — why phase-only
  edges every amplitude run on curve metrics and only the union reaches 0.61).
  Figure: source chips 5|A| / 4 phi -> parameter cards a / mu / sigma,Delta-mu, each
  card carrying its winning-vs-losing numbers; arrow grammar solid = measured,
  dashed = 4th-order noisy, dotted-x = provably invisible (shift-/scale-invariant).
  25g "The recall paradox — shrinkage, not information": phase channels are a subset
  of the full stack so the 0.87-vs-0.66 recall inversion is an estimation gap;
  a_2_hat = Pr(2 layers | in) · E[a_2 | 2 layers, in] shrinks toward the 95.4%-empty
  prior for amplitude-informed nets (under the activity threshold — missed) while
  scale-blind phase-only stays at the class mean (cleared): worst a-MAE and best
  recall are the same un-shrunk guess; once matched, information wins again
  (2-scatterer mu/sigma: full 3.75/1.46 vs 4.02/1.61); the set-prediction gate is
  the fix. Figure: 95.4/4.1 prior bar, tau line, accent arrow dragging the posterior
  mean below tau vs the good phase-only dot beyond it, a_2_hat axis underneath.
  All table numbers from results/extra_tracks_exp/metrics_comparison(1).md; the
  forward model matches notes/DLR-TomoSAR/Overview/SAR Tomography Signal Model.md and
  pipelines/processing/generation/interferogram.py (angle_only IFG channels).
  FIT NOTES: the rotated loop-closure label is anchored inside the dashed elbow (the
  first cut sat on the column edge and clipped, 4.6pt overfull); E[A^2]=P moved
  inline so the Siegert display fits the 0.46 column; the phasor annotation and
  "identical channels" became explicit-linebreak nodes (justified text-width nodes
  kept overfulling); 25f/25g bottom captions un-justified with hand line breaks
  (killed the "or-der"/"re-call" hyphenations). Pages 47-51 pdftoppm-verified, zero
  overfull warnings in the new-frame range; the pre-existing deck warnings are
  unchanged against the shipped log.
  SECOND PASS same day (user: "the visualization are too much, drop them, make it a
  formal, derivation equation focus argument"): ALL five tikzpictures on 25c-25g are
  GONE (loop diagram / shifted-profiles+phasor+channel-rows / scaled-profiles+slope+
  phase-clouds / partition arrows / prior-bar+threshold scene — reintroduce from the
  first-pass entry above if ever wanted) and the frames are rebuilt as single-column
  formal derivations, bold-led Theorem/Proof/Corollary blocks with full-width
  displays: 25c "Setup — the measurement model and a sufficiency lemma" (model +
  polar channels, distributed-scatterer statistics R_mn = P rho(k_m - k_n), Lemma:
  R_hat from the nine channels, theta_0 a cancelling gauge); 25d "Theorem 1 —
  amplitudes are shift-blind" (proof via the unimodular factor e^{jk_n Delta}, flip
  corollary, Siegert block: P at 1st order, |rho| at 4th — autocorrelation class,
  phase-retrieval ambiguity); 25e "Theorem 2 — phases are scale-blind, and linear
  in position" (proof via gamma -> sqrt(c) gamma, corollary a structural, positional
  encoding rho = e^{j kappa mu} e^{-kappa^2 sigma^2/2}, mod-2pi unwrapping remark,
  spread-reads-|rho| at 2nd order); 25f "Corollaries — the identifiability
  partition" (invariance groups G_|A| = shifts x flip, G_phi = scalings, trivial
  intersection => only the union identifies; curve-cost pair
  2E(1 - e^{-delta^2/4 sigma^2}) vs eps^2 E; numeric check block); 25g "Proposition
  — the recall inversion is estimation, not information" (Bayes-risk ordering,
  posterior-odds display Pr(K=2|x) = pi Lambda/(1 - pi + pi Lambda) with pi ~ 0.041,
  three-bullet mechanism: Lambda = O(1) shrinkage miss / Lambda >> 1 un-shrunk
  detection / conditional-on-match consistency + set-pred pointer). Each frame keeps
  a "check against the table" line tying the statement to the ablation numbers.
  Titles renamed to the formal ladder (Setup / Theorem 1 / Theorem 2 / Corollaries /
  Proposition). Pages 47-51 re-verified via pdftoppm, zero overfull in the new
  range, deck still 88 pages.
  THIRD PASS same day (user: "keep two column and step by step, less step"): the five
  single-column formal frames CONDENSED TO THREE two-column frames, the argument
  numbered as five steps, two per slide where they pair — markers 25c-25e, pages
  47-49, deck 88 -> 86 pages. 25c "Steps 1-2 — the model, and a sufficiency lemma"
  (left = step 1: forward model + polar channel split + second moments
  R_mn = P rho(k_m - k_n) with P and rho defined inline after a 14pt display
  overfull; right = step 2: R_hat from the nine channels, theta_0 gauge,
  sufficiency conclusion). 25d "Steps 3-4 — two exact invariances" (columns [T],
  headers colour-keyed to the verdict bullets: step 3 argB amplitudes shift-blind —
  unimodular factor + Siegert what-survives block; step 4 argC phases scale-blind —
  sqrt(c) proof line + arg rho = k_n mu positional encoding; each column ends in a
  scriptsize check line). 25e "Step 5 — the verdict follows" (left = trivial
  intersection of the invariance groups + saturating-vs-quadratic curve-cost pair +
  R^2 ladder; right = recall-inversion proposition with the posterior-odds display
  and the shrinkage bullets compressed to prose; matched-pairs consistency +
  set-pred pointer as the check line). The Theorem/Proof/Corollary long-form
  wording and the 25f/25g standalone frames are GONE — reintroduce from the
  SECOND PASS entry if wanted. Pages 47-49 pdftoppm-verified, zero overfull in
  the new range.
  FOURTH PASS same day (user: "you started using variables without
  declaring/explaning them"): every symbol on 25c-25e is now declared at first
  use. 25c: g_n named as the SLC value of pass n at a pixel, z as elevation, a
  scriptsize gloss line under the model display (b_perp,n perpendicular baseline
  to the primary / lambda wavelength / r range / theta incidence; A_n = |g_n|
  amplitude, theta_n phase), the channel sentence names the primary-referenced
  phase differences, a_k/mu_k/sigma_k spelled (amplitude/elevation/width), P and
  rho named (total power / coherence), and step 2 ties R_hat to "the same w x w
  window B(x) that built the GT". 25d: Delta introduced as an elevation shift,
  the Siegert covariance carries its full argument |rho(k_m - k_n)|^2 (was bare
  |rho|^2), c > 0 declared, and the positional-encoding display is qualified
  "for a single scatterer at elevation mu with width sigma". 25e: the curve-cost
  pair is prefixed "for one component N of energy E = ||N||^2, a position error
  delta and a relative scale error eps", R* is named the Bayes risk, a_2_hat /
  activity threshold tau / prior pi = Pr(K=2) / likelihood ratio Lambda(x) are
  all named in the sentence introducing the posterior-odds display. Layout
  unchanged (three two-column step frames); pages 47-49 re-verified via
  pdftoppm, zero overfull, deck still 86 pages.
  FIFTH PASS same day (user: "the text is super confusing, make a more direct,
  simple text to follow"): all prose on 25c-25e rewritten as short declarative
  sentences, one idea each, academic phrasing dropped ("samples the Fourier
  transform ... at a frequency set by the pass geometry" -> "Take one pixel.
  gamma(z) is its reflectivity along elevation z. Pass n measures one Fourier
  coefficient of it."). Steps 3 and 4 now mirror each other line for line:
  "Shift the profile up by Delta. Each g_n only rotates. No amplitude moves." /
  "Scale the profile by any c > 0. Each g_n only grows. No phase moves." — then
  a "What amplitudes/phases still carry" block each. Step 2 is a four-sentence
  argument ending "Removing channels is the only way information is lost.
  Which information each cut removes — next slide." Step 5 leads with plain
  questions ("Why phase-only still wins the curve metrics" / "Why phase-only
  beats the full stack at detection — it should be impossible: the full stack
  contains the phases"), the k_n formula and the x/pi/Lambda definitions moved
  to scriptsize gloss lines under their displays, and the shrinkage story is
  three two-word-verdict sentences (under tau. Missed. / over tau. Detected. /
  One effect, both signs). Equations, declarations, check lines and the
  three-frame two-column layout are unchanged. Columns [c] -> [T] on 25c for
  top alignment. Pages 47-49 re-verified via pdftoppm, zero overfull, 86 pages.
  SIXTH PASS same day (user: add a section about "the set-prediction gate
  decouples detection from a-hat — exactly this fix"): NEW FRAME 25f "Step 6 —
  the fix: the set-prediction gate" after step 5 (page 50, deck 86 -> 87
  pages), and step 5's closing check line now ends "— next slide" instead of
  "— exactly this fix". Two mirrored columns in the step voice: LEFT (argB)
  "The conv head couples two questions" — the shrinkage product re-displayed
  with underbrace labels "is it there?" / "how bright?", "every doubt scales
  the amplitude down"; RIGHT (argC) "The gate gives each question its own
  output" — the frame-15a blend a_2 = sigma(e_2) a_2_hat + (1 - sigma(e_2))
  a_off re-displayed (a_off ~ 0 moved into the following sentence after a 15pt
  display overfull), closed-gate/open-gate reading, sigmoid-saturation point
  ("hedging is expensive and committing is cheap"), and a check line quoting
  the Act III head-comparison sorted column (recall 0.68 -> 0.89, 1-scatterer
  0.67 -> 1.00, under-count 0.284 -> 0.019, a-MAE 0.334 -> 0.285). ALSO FIXED:
  step 5's shrinkage display carried a 15.8pt overfull that the fifth pass's
  too-narrow warning grep had missed — the posterior-odds formula
  Pr(K=2|x) = pi Lambda/(1 - pi + pi Lambda) moved from the display into the
  x/pi/Lambda gloss line. Pages 47-50 verified with the full-range grep plus
  pdftoppm, zero overfull across all four math frames.
  SEVENTH PASS same day (user: "you are not using consistent notation. In the
  profile shift you used gamma as the profile, in the scale profile you used
  p"): both invariance displays on 25d now start from the profile p and show
  the gamma realisation explicitly — step 3 "p(z-Delta): gamma(z-Delta) =>
  g_n -> e^{jk_n Delta} g_n => A_n unchanged" (lead sentence "Shift the
  profile p up by Delta — the reflectivity shifts with it"), step 4 "c p(z):
  sqrt(c) gamma(z) => g_n -> sqrt(c) g_n => phi_n unchanged" (lead "Scale the
  profile p by any c > 0 — the reflectivity scales by sqrt(c)"). The missing
  gamma-to-p link is stated on 25c step 1: "The scatterers are uncorrelated:
  E[gamma(z) gamma*(z')] = p(z) delta(z - z'). The power profile p of gamma is
  what the label stores." Display-width fit: the new p(...) prefixes cost two
  spacing shrinks on step 3 (quad -> thin space, exponent thin space dropped)
  after 14.3pt then 1.6pt overfulls. Pages 47-48 re-verified via pdftoppm,
  zero overfull across the math frames, deck still 87 pages.

- Verdict frame fourth pass 2026-07-10 (user: "stack them like a pile of paper"): the
  pooling-level stacks reoriented from in-plane offset squares to FLAT SHEETS in shallow
  perspective — each level is three parallelogram sheets (front edge s, depth edge
  +0.5s/+0.25s, vertical lift 0.06 between sheets, top sheet drawn last), footprints to
  scale W = 40 -> 20 -> 10 -> 5 descending, div-2 arrows (head down) on the centre line,
  size labels on the left rail, W=40 as the top label. Flat piles are much shorter than
  upright stacks, so the caption returned to BELOW the diagram at full column width.
  Geometry emitted programmatically (scratchpad paper_pile.py). Page 44 pdftoppm-verified.

- Verdict frame third pass 2026-07-10 (user: no old-run mentions — "the presentation is
  the final version of the project, not a history of changes" — and "stacked stacks
  vertically on the layer pooling"): ALL four-level-pyramid / step-16 history removed
  from the frame (grep-verified deck-wide: no four-level / step 16 / never admitted
  left). Pooling diagram redrawn VERTICAL: four feature-map stacks (3 offset sheets
  each, back sheets up-left so the label rails stay clean), spatial size to scale
  W = 40 -> 20 -> 10 -> 5 top to bottom, soft "div 2" arrows on the right rail, size
  labels 20/10/5 on the left rail, W = 40 on top. The dashed accent "2.5 --- not
  integer" fourth halving is GONE (it existed only to explain the old grid). Caption
  moved BESIDE the stack (minipage 0.40/0.58 [c]) to fit the column: "Each pooling
  halves the patch --- three levels, integer all the way down: the grid runs in
  multiples of 8, and the ceiling 2w = 40 sits on it." Page 44 pdftoppm-verified, all
  content inside the frame (the previous below-diagram caption clipped past the frame
  bottom WITHOUT any overfull warning — render-verify is mandatory, warnings do not
  catch column overflow in [c] columns).

- Verdict frame follow-up 2026-07-10 (user: "reduce a lot the bullet points text, add a
  visualization about the 3 pooling operations on the patch size"): right column reworked
  — bullets cut 4 -> 3 terse (setup / climb-over-by-40 + n=5 exact / knee-does-not-move),
  and the four-level-vs-three-level prose bullet REPLACED by a pooling-chain diagram:
  four ink squares to scale (W = 40 -> 20 -> 10 -> 5 at 0.04 cm/px, vertically centred on
  a common axis) joined by soft "div 2" arrows, plus a dashed accent fourth halving
  dropping to "div 2 -> 2.5 --- not integer". Scriptsize caption: integer at every level
  for multiples of 8, so 2w = 40 is on the grid; the four-level pyramid's fourth halving
  breaks 40 (step 16 never admitted it). Footnote compressed to one sentence pair.
  itemsep 5pt, no overfull, page 44 re-verified via pdftoppm.

- "The verdict --- saturation at the window scale, for every track count" — RE-PLOTTED
  2026-07-10 from the second (post-fix, step-8) sweep run, report
  results/patch_sweep_experiment/report_new_patches.md (the report's w*sqrt(N/n)
  prediction column is the refuted floor its generator still draws — only the loss
  values were extracted). Five curves n = 2/3/5/9/15 (line styles solid/dashed/dotted/
  dash-dot/dash-dot-dot, one ink colour), x-axis 16..72 step 8, y = raw held-out test
  loss mapped y=(loss-0.09)*36, dashed 2w=40 ceiling line (the shaded 40->48
  first-admissible band is retired — the 3-level U-Net puts 40 itself on the grid).
  Registered-vs-observed table now 3 rows: knee (80% of gain) registered 2w=40 every n
  / observed 40-56 px flat in n; gain captured at 40 px 73-100%; loss left past 40 px
  <= 0.006. Bullets: step-8 grid + why 40 was untestable before (4-level pyramid,
  multiples of 16); climb over by 40 px, n=5 argmin exactly 40; knee does not move
  with n. Old n=2 kz~0 bullet REMOVED (post-fix n=2 = median-baseline pair, planner
  picks the middle candidate at a single secondary — half aperture, not near-zero kz).
  Footnote flags the shared unexplained 16->24 px rise on every curve. itemsep 4pt to
  clear a 4.3pt overfull vbox.

- "Run-health tooling: no GPU-hour starts unhealthy" — EXPANDED INTO THE BENCHMARK-SETUP
  FRAME 2026-07-10 (user: delete the four design/grid frames, fold the setup facts in
  here). The gate pipeline gained a FIRST box "capacity match — every backbone
  width-scaled to ~30M parameters" (now 4 boxes at 26mm, node distance 6mm: capacity
  match -> overfit gate -> max-batch finder -> DataLoader tuner). Below it a footnotesize
  r|l axes table: 17 backbones (full name list, two lines), 8 objectives (curve MSE, L1,
  Huber, Charbonnier, cosine / parameter MSE, L1, Huber), 2 heads (conv regression /
  gated set prediction), 2 matchings (sorted-GT / Hungarian — head x matching swept on
  the strongest backbones). Bullets: capacity-matched + identical recipe + reference cell
  UNet+param-MSE (differences architectural, not capacity); overfit-gate rationale merged
  with the honest-VRAM-probe point; unattended-grid payoff. Deck 87 -> 83 pages.

- "Padding brings no new looks --- the patch runs past 2w" — DELETED 2026-07-10 (user).
  The frame (section marker 24c0, clipped-window-rim scene: split-border strip, W=48 vs
  W=96 patches with core percentages, clipped 2w window at a border pixel, three-bullet
  right column on filler/rim/first-admissible-step) is gone from the deck; the patch-size
  act now runs straight from the shared-looks overlap frame into "The verdict ---
  saturation at the window scale". Reintroduce from git-less history: this entry is the
  only record — the full source was ~70 lines, recover from a PDF backup or rebuild if
  ever wanted.

- "The answer to slot imbalance: two levers" — SPLIT INTO TWO FRAMES 2026-07-10 (user:
  one slide per lever, clearer visualizations, especially active normalization). Both new
  frames use bullets LEFT (0.40, itemsep 10pt) with the lever's equation + scriptsize
  footnote under them, figure RIGHT (0.58, tikz scale 0.95), columns [c]. FRAME 1
  "--- lever 1: active normalization": the active-slot definition strip stays on top.
  Figure = three bands. Band A: the 10-slot batch strip (pixel-1/pixel-2 braces, bumps in
  the 3 active slots, m = 1/0 mask row --- the old left panel's geometry, unchanged).
  Bridge line: "w=m keeps only the active errors --- same numerator, the reductions
  differ in what they count". Bands B/C draw the two reductions as LITERAL VISUAL
  FRACTIONS sharing the same numerator (the 3 active bump-boxes joined by "+"): band B
  "mean reduction" puts all 10 mini-squares under the bar, solid (3 accent, 7 grey,
  caption "counts every slot: div 10") arrowing to a short accent!45 bar "diluted 3.3x"
  under a "gradient per active slot" header; band C "active normalization" (accent
  header) ghosts the 7 empty squares to DOTTED OUTLINES --- not counted, caption "counts
  only the active: div sum m = 3" --- arrowing to a full-length accent bar "full
  strength". The old cryptic division-bar pair is gone; the fraction is the picture.
  FRAME 2 "--- lever 2: presence balance": one-line reminder strip (mask m as on the
  previous frame; f = <m> = the batch's active fraction). Figure: band A = the same
  batch strip but with the amplitude TARGET row (a in active slots, 0 in empty --- the
  amplitude term trains every slot) + "active fraction f = <m> = 3/10"; band B = "share
  of the amplitude-term gradient": uniform-w bar split 30/70 (class labels above, side
  caption "the empty majority owns the objective"); crossing multiplier arrows
  x 0.5/f = 1.67 (accent) and x 0.5/(1-f) = 0.71 (soft); band C = balanced-w bar split
  50/50, side caption "each class carries exactly half"; footer "silence stops winning
  by head count --- the two classes pull with equal total weight". All numbers derive
  from the shared toy batch (3 active in 10, f = 0.3), tying the two frames together.
  Bullets rewritten per lever. Comment markers 17 -> 17a/17b. Header page count
  re-measured for this entry (86 was stale; 87 before the split, 88 after). Both pages
  render clean, no new overfulls.
  SECOND PASS same day (user: the w = m story contradicts the code --- a Gaussian
  matched to an empty GT slot still gets an amplitude loss; verified in
  pipelines/backbone/training/loss.py:89-97: param_mask is 1 for the a channel on EVERY
  slot and m-masked only for mu/sigma; presence/focal defaults are 1): frame 1 rebuilt at
  ELEMENT level. The "w = m" equation tag and pure-on/off footnote are GONE; a per-term
  WEIGHT GRID (rows w_a / w_mu=m / w_sigma=m x the 10 slot columns, aligned under the
  batch strip, a 1/0 printed in every cell: accent = active slot, grey = empty-slot
  amplitude --- supervised toward a=0, what switches false positives off, dotted =
  mu/sigma with no target) replaces the old slot-level m mask row; annotation "16 of 30
  terms carry weight". The two visual fractions now share the symbolic numerator
  sum w e and their denominators are compact 3x10 mini-grids: mean counts all 30 cells
  (all solid, div 30), active-norm ghosts the 14 zero-weight cells (div sum w = 16).
  The dilution figure is CORRECTED 3.3x (slot-level fiction) -> 1.9x (30/16,
  element-level truth); result-bar lengths follow 16/30. Bullets rewritten
  (not-every-term-has-a-target incl. the false-positive story / mean-counts-zero-weight-
  terms / divide-by-weight-mass, noting it stays correct when lever 2 reweights w).
  Bars header "gradient per active slot" -> "gradient per supervised term" (staggered
  below the band header after two collision fixes). tikz scale 0.95 -> 0.88 (taller
  figure). Frame 2 verified CONSISTENT with the code as framed: its 30/70 and 50/50 are
  amplitude-channel shares, which is exactly where the class balance lives; the presence
  factor also rescales active-slot mu/sigma uniformly --- a magnitude side effect, not a
  class rebalance, deliberately left off the slide.
  THIRD PASS 2026-07-10 (user: remove some right-column visualization to add per-component
  equations showing a uses all data / only mu+sigma use the filter; swap the visualization
  to 2 slots x 3 params where the placeholder GT slot leaves the predicted mu/sigma
  label-less): frame 1's right column rebuilt. The 10-slot batch strip, the per-term
  weight grid AND the two visual-fraction reduction scenes are GONE; in their place (top)
  a K=2 GT-vs-prediction card scene in the set-output slide's card grammar --- GT row =
  slot 1 active card (accent, bump, (a1,mu1,sigma1), tag m=1) + slot 2 placeholder card
  (soft, flat line, (0,--,--), tag m=0); prediction row = two ink cards with symbolic
  hat-triplets; per-param connectors between rows: slot 1 all three accent double-arrows
  ("all three supervised" left of them), slot 2 = accent a-arrow ("target a=0: switches
  it off") + two grey DASHED mu/sigma lines with x marks and the note "mu2-hat,
  sigma2-hat: no label to predict --- w=0" --- and (bottom) the three component equations
  as an align* block with soft right-side annotations: l_a = sum|a-hat - a-GT| / (N_pix K)
  "every slot, no mask --- empties supervise a-hat -> 0"; l_mu and l_sigma =
  sum m|.| / sum m (m in accent) "active slots only --- m filters the sum AND the count".
  Matches tools/loss/param_loss.py + loss.py param_mask exactly (per_param terms of
  ParamLoss.l1 under use_active_normalization; presence/focal scales abstracted as
  before). Left column untouched except bullet 2's figure numbers: 16-of-30 / 1.9x ->
  4-of-6 / 1.5x (the new single-pixel scene). The mean-vs-active fraction contrast now
  lives only in the left bullets + the total equation; reintroduce from the SECOND PASS
  entry if ever wanted back. Label collisions fixed same pass (param letters beside the
  connectors, side labels anchored east). Page 33 renders clean, no new overfull.
  FOURTH PASS same day (user: delete the 3 bullet points, move the "Active slot = ..."
  part to the left column): the frame-top definition strip is GONE and the left column
  is now just two blocks --- the active-slot definition (lead line + displayed
  m_{p,k} = 1[a^GT > tau] + the scriptsize pixel/tau subline) and, 18pt below, the lever
  (one lead line "one reduction, dividing by the weight mass, not the cell count" +
  displayed l = sum w e / sum w + the w-per-term caption). All three bullets deleted ---
  the not-every-term-has-a-target story lives in the right column's card scene, the
  dilution argument (4-of-6, 1.5x) left the slide entirely (BUILD.md history only).
  Columns [c] rebalance; page 33 clean. SPACING PASS same day (user): slot-2 group
  shifted +0.7 units right (card gap 1.2 -> 1.9, the "target a=0" label now breathes
  between the groups) and the "one pixel, K=2 slots" header raised 5.55 -> 5.95, clear
  of the slot tags.
- "Set-prediction lineage: DETR" — gate glyphs REDESIGNED 2026-07-10 (user: "make the
  gate a box instead of a dot, colored the area between the end of the gate to the gt
  points and called it hungarian assignment"): the three gate dots at the slot outputs
  are now small boxes (4mm, good!12 fill for open, black!8 for closed, local
  gateon/gateoff styles), a rounded argA!12 band fills the space from the gate boxes'
  east edge to the GT points (dots sit on the band's right edge), the dashed matching
  lines now start at the gate box edges, and the "Hungarian assignment" label moved
  under the band in argA so it names the region. Page 29 rendered clean. SECOND PASS
  (user: "make each gate drop a dot, the final output of the network and this dots and
  the gt dots should be inside the colored area"): each gate box now emits an output
  dot via a short soft connector (good fill for open slots, black!35 for the closed
  one, coordinates o1-o3 at d_k.east + 0.45); the GT dots moved from the band edge to
  x = 4.45 and the empty-set symbol to (4.45, 0), so outputs, GT points and the
  no-match target all sit INSIDE the band (band = d1.east + 0.12 to x = 4.62); the
  dashed matching lines now run dot-to-dot within the band and the GT 1 / GT 2 text
  labels sit outside at x = 4.95. Rendered clean again. THIRD PASS (user: "separate
  the points horizontally a bit more"): output-dot / GT-dot gap widened 0.68 -> 1.30
  units — output dots pulled toward the gates (d_k.east + 0.32), GT dots and the
  empty-set symbol pushed right to x = 4.62, band right edge to 4.8, GT labels to
  x = 5.12; the slot pills shifted left (x = 2.2 -> 2.05, gate open/closed labels
  followed) so the diagram keeps its column footprint. Rendered clean.

- "Reading the ceiling" — RESTRUCTURED TO PURE FIGURES 2026-07-10 (user, supersedes the
  same-day four-step-derivation rebuild below): "delete the right column, move the
  equations that are under the visualization to the right column, and in the middle of
  the slide, below both columns, repeat the correlation triangle plot, but from -w to w
  and add a line that measures between the -w to +w and equal to W". New layout: LEFT
  (0.56) = the 2-D grid scene + its one-line caption (unchanged); RIGHT (0.42, centred)
  = the two-row sum decomposition tikz (w^2 R_hat(x) / w^2 R_hat(x') = shared S(Delta)
  block + private sums) that previously sat under the grid; BELOW BOTH COLUMNS,
  centred = a symmetric version of the correlation-law triangle, rho_label(tau) over
  tau in [-w, +w] (accent triangle, soft axes, ticks -w / w) with an ink |-| dimension
  line under the base labelled "W = 2w", caption "the label correlates over +-w around
  its pixel --- the centred patch that spans the whole triangle is W = 2w". The
  four-step estimator derivation (noisy estimate / can-only-pool / overlap certificate
  / L(W) = min(W,2w)^2) is GONE from the deck — reintroduce from the entry below if
  ever wanted. The frame is now visualization-only: no prose bullets, the triangle's
  -w..+w span carries the radius-to-diameter argument graphically (mirrors the
  correlation-law frame's one-sided triangle). Page 42 rendered clean, no overfull.
  SECOND PASS same request thread (user: "keep only the visualization on the left
  column and move the triangle plot to the right column under the equation"): the
  below-columns centred triangle block was folded INTO the right column — right column
  is now decomposition tikz, \\[14pt], triangle tikz at scale 0.85 (fits the 0.42
  column), 2-line wrapped caption; left column = grid scene + caption only. Rendered
  clean again. THIRD PASS (user: "add a small text to introduce the equations and the
  triangle"): two scriptsize intro lines added in the right column — "two labels at
  offset Delta split into shared and private looks:" above the decomposition, "the
  shared fraction at every offset tau is the correlation law:" above the triangle
  (bridges the Delta-notation equations to the tau-axis plot). Gap 14pt -> 12pt to
  absorb the lines; rendered clean. FOURTH PASS (user: "make the visualization canvas
  vertically bigger"): the left grid canvas grew 4.9x2.9 -> 4.9x4.3 units (~3.4cm ->
  ~5.1cm at scale 1.18); the scene content is wrapped in scope shift={(0,0.8)} so it
  sits vertically centred in the taller field (margins ~1.1 below / ~1.15 above,
  0.8 is a grid-step multiple so cell phase is preserved). Rendered clean. FIFTH PASS
  (user: "swap the left and right column position"): columns exchanged — equations +
  triangle column (0.42) now LEFT, grid-scene column (0.56) now RIGHT; content and
  widths untouched. Rendered clean. SIXTH PASS (user: "move both columns to the
  right"): an empty spacer column added at the far left and the grid column narrowed
  to keep the total at 0.98. First cut used spacer 0.10 / grid 0.46 — user: "that was
  a bit too much, do it half as much" — FINAL: spacer 0.05 / grid 0.51 (equations
  block shifted right ~0.7cm, grid ~0.4cm, comfortable clearance at the page edge).
  Grid caption wraps to two lines at the narrower width. Rendered clean, no overfull.
- "The correlation law" — HARD ARGUMENT RESTORED, FRAME NOW CONCLUDES 2w 2026-07-10
  (user, third pass on this frame, SUPERSEDES the same-day LANGUAGE FIX below): after
  the "Reading the ceiling" right column was rebuilt on the radius-w cutoff, the user
  established the two frames make the exact same argument — "w away there is 0
  correlation, w for each side goes to 2w patch size centralized in the pixel" — so
  the softened wording was un-hedged and the frame now carries the geometry to its
  conclusion. (1) caption: "disjoint windows — zero correlation"; (2) bullet: "past w
  the windows are disjoint and the labels share nothing — the correlation is exactly
  zero"; (3) closing REPLACED: "From the law to the patch: the label's correlation
  reaches exactly w to each side of x and nothing beyond — the patch that captures
  all of it spans 2w, centred on the pixel." The frame is no longer the w-floor
  statement (the earlier "minimum patch is w" / "do not correct this frame to 2w"
  framing is superseded); it now derives the 2w patch geometrically, and "Reading
  the ceiling" quantifies the same cutoff as a look budget (min(W,2w)^2). Page 41
  rendered clean, no overfull. NOTE: the user's instructing message was cut off
  ("...bring back that hard argument from the beggining in slide 1, and") — a further
  instruction may follow.
- "Reading the ceiling" — RIGHT COLUMN REBUILT AS A DERIVATION 2026-07-10 (user: "the
  mathematical development of the second slide, in the right column is unclear, make a
  more solid case with equations"). The old three bullets asserted disconnected facts
  (label = f_theta(R_hat) — wrong symbol, the fit makes the label, not the network;
  shared block shrinks; a Var law with an UNDEFINED L_eff whose saturation at 2w was
  stated, never derived). New form: four bold-led steps, each one display equation,
  each following from the previous — (1) the label is a noisy estimate,
  label(x) = fit(R_hat(x)) inline, display R_hat = (1/w^2) sum y y^H with E R_hat = R(x)
  appended (w^2 = 400 looks); (2) the network can only pool — noise seen once cannot be
  reproduced; averaging L looks of the same statistics leaves Var[R_hat_L] =
  (1/L) Var[y y^H]; (3) a look counts while windows overlap — its label shares the
  block S(Delta) (left graphic) with the label at x, display L_cap(Delta) =
  (w-|D_az|)+(w-|D_rg|)+, positive iff |Delta| < w, so counting looks fill a 2w x 2w
  square; (4) the ceiling — a centred patch of side W reaches |Delta| <= W/2, so it
  holds L(W) = min(W, 2w)^2 counting looks: 400 -> 1600 by W = 2w = 40 px, flat past
  it. The min(W,2w)^2 closed form replaces the undefined L_eff and gives the hard knee
  at exactly 2w; at W = w it returns the label's own 400 looks (floor consistency with
  the correlation-law frame). FIT: the four-step column overflowed at \small (two
  iterations clipped bullet 4 and pushed the left column) — final fit = \footnotesize
  (matches the ladder-result bullet column), itemsep 5pt, frame displayskips 6pt -> 3pt,
  fit relation + unbiasedness folded inline/into displays. Page 42 rendered clean, no
  overfull. Left column untouched.
- "The correlation law" — LANGUAGE FIX 2026-07-10 (user: the slide claimed labels
  farther than w apart are "statistically independent" / "share nothing" / "the
  information that built the label reaches exactly w, no farther", contradicting the
  very next frame ("Reading the ceiling"), whose whole point is that estimates stay
  related through shared looks out to the 2w ceiling — the user's thesis: past w a
  pixel's statistics no longer enter its neighbour's label directly, but covariances
  still carry shared components, so the labels are NOT unrelated). Fix is MINIMAL by
  explicit user instruction ("you are bringing the next slide argument in the first
  one, there is no need of that" — a first version that explained the shared-look
  chain on this frame was reverted): the frame now states only facts and drops the
  wrong statistical conclusion. (1) diagram caption "disjoint windows — independent"
  -> "disjoint windows — no shared inputs"; (2) correlation-law bullet ends "past w
  the windows are disjoint and no input pixel enters both labels" (the "share
  nothing — statistically independent" clause deleted); (3) closing "The minimum
  patch is w" now says the network "must see AT LEAST the whole wxw window that made
  it" — the "information reaches exactly w, no farther" claim deleted, w reads as a
  floor. The law itself (rho_label = (1 - tau/w)+, triangle plot, tau = w zero
  crossing) is untouched — it is the direct-overlap correlation; only the
  independence reading was wrong. No layout change; ceiling argument stays entirely
  on the next frame.
- "The imbalance failure (high K)" — REDESIGNED 2026-07-10 (user: improve the
  visualization, after the K=2/K=3 failure-slide passes). The old layout (left
  components-per-pixel histogram, right ch-1..5 chips with status dots and an
  "unmatched" dot trail, bullets below) showed cause and outcome disconnected. New
  layout matches the failure-slide family: bullets LEFT (0.42, itemsep 10pt — the old
  intro line folded into bullet 1, plus the slot-k >= k-layers matching rule, the
  loss-optimum bullet, and the K=2-sidesteps / K=5-confronts exit), figure RIGHT (0.56,
  scale 0.95, bbox (-0.05,0)-(7.9,6.95)) — THREE BANDS sharing one five-column slot
  axis (columns x = 1.0/2.2/3.4/4.6/5.8) so cause flows into consequence per column:
  band A "layers per pixel --- one or two almost everywhere" (accent bars 70/20/6/3/1%
  with % labels, count labels 1-5); an accent mapping line "slot k is matched only when
  the pixel holds >= k layers" bridges to band B, each slot's training diet: full-height
  stacked bars, accent bottom = share of steps with a real target (survival 100/30/10/
  4/1%, % labels above), grey remainder = matched to emptyset, with a two-swatch legend
  at right ("real target" / "emptyset --- nothing"); band C "what each slot converges
  to": five verdict cards in the K=3 slot-card language — slot 1 accent card with a
  strong bump ("live"), slot 2 faint accent card with a small bump ("fading"), slots
  3-5 dashed grey cards with a flat thick silence line ("off"). Two soft footer lines:
  "zeroing the deep slots is a loss optimum, not a training accident" and "when a deep
  pixel finally arrives, its extra layers have no living slot --- the count
  undershoots". Numbers are illustrative (70/20/6/3/1 and its survival), not measured.
  SECOND PASS 2026-07-10 (user: not good enough --- stop repeating patterns from the
  other failure slides, this is a different thing): the bands/bars/cards design was
  replaced by a TRAINING-DYNAMICS figure, the shape the failure actually has in a run.
  Two panels sharing the training-steps axis: TOP "slot activity over training --- the
  deep slots die, one by one" (y = share of pixels where the slot fires, ticks 0/1):
  slot 1 ink settles high ~0.95, slot 2 ink!40 settles ~0.3, slots 3/4/5 an accent
  bundle (accent!85/!60/!40) that decays gaussian-fast to zero at staggered times, each
  death marked with a dot on the zero line, a dotted guide line through BOTH panels,
  and a white-filled "slot k off" label; accent annotation in the empty mid-panel space
  "a deep slot meets a real target in ~1% of steps --- the silence gradient outweighs
  it 99:1". BOTTOM: the training-loss curve over the same steps, falling smoothly
  through all three death lines, annotated "keeps falling through every death ---
  collapse IS the optimum". Footer: "the tail is abandoned --- the predicted component
  count undershoots the truth". Bullet 3 updated to the dynamic reading (slots die
  early, loss keeps falling, never return). Curves are schematic (tikz exp/sin), not
  run data --- swap in real slot-activity TensorBoard traces if available before the
  talk. tikz scale 1.05, bbox (-0.15,-0.3)-(6.95,5.35).
  THIRD PASS 2026-07-10 (user: delete the 2nd bullet, vertically centre the rest):
  the slot-k >= k-layers matching-rule bullet was DELETED (its content survives in
  the figure's accent annotation "a deep slot meets a real target in ~1% of steps");
  three bullets remain (nothing-to-match, loss-optimum, K=2-sidesteps exit).
  Vertical centring needed no code change --- columns [c] centres the shorter
  bullet block against the figure automatically; verified on the rendered page.

- "The sorting instability, concretely (K=3)" — REDESIGNED 2026-07-09 (user: improve the
  K=3 failure-mode visualization, after the K=2 redesign). The old two-panel pixel-A /
  pixel-B sketch with ch-1/2/3 chips and a dashed "same scatterer, different output
  channel" flow was rebuilt in the K=2 frame's established language: bullets LEFT (0.42,
  itemsep 10pt: sorted_gt semantics / canopy ch 2 -> ch 3 / discontinuous ch-2 target /
  score-the-set exit), figure RIGHT (0.56, scale 0.9, bbox (-0.15,-6.0)-(6.0,1.55), same
  row geometry: axes y 0 / -2.5 / -5.0, headers +1.35, under-axis notes). Row 1 "pixel A:
  ground + canopy --- sorted by mu, ch 3 stays empty" (ground ink at mu 1.0, canopy accent
  at 4.3; under-axis channel tags ch 1 / ch 2 (accent bold) / "(ch 3 empty)" soft). Row 2
  "pixel B, one pixel over: a mid-height volume appears" (same ground and canopy, good
  volume at 2.6 labelled "volume --- new"; tags shift to ch 1 / ch 2 (good) / ch 3 (accent
  bold); accent under-axis note "the canopy did not move --- its channel did: ch 2 ->
  ch 3"). The accent guide at the canopy mu spans rows 1-2 (the scatterer is pinned while
  its tag changes); a short good guide marks the volume in row 2 only. Row 3 is NEW ---
  "channel 2's target across the scene: discontinuous": a step function over image
  position x (accent segment at canopy height labelled "target: canopy", dropping via a
  dashed arrow to a good segment labelled "target: volume", annotation "+1 layer below ---
  the target jumps", pixel A / pixel B ticks under the axis). Bottom caption: "the channel
  index counts layers below --- bookkeeping, not identity; no convention is stable". The
  chips are gone; the discontinuity the bullets used to describe is now drawn.
  SECOND PASS 2026-07-09 (user: argument too similar to K=2 and not strong — the real
  mechanism is SLOT SPECIALIZATION: with most pixels ground+canopy the head specializes
  slot 1 ground / slot 2 canopy / slot 3 silence; a third Gaussian pushes the canopy into
  the always-silent slot and drops the volume onto the canopy specialist): the frame was
  rebuilt around that story, same three-row geometry. Row 1 "the training diet: almost
  every pixel is ground + canopy" (typical two-layer pixel; under-axis specialization tags
  "slot 1: ground" / "slot 2: canopy" + note "slot 3 almost never sees a Gaussian --- it
  learns silence"). Row 2 "a three-layer pixel: sorting reshuffles every target" (volume
  appears at 2.6; tags become slot 1 ground / slot 2 VOLUME / slot 3 CANOPY; note "the
  volume lands on the canopy specialist; the canopy lands on the silent slot"). Row 3
  "the prediction: each slot answers from habit" — drawn prediction = ground + slot 2's
  habitual canopy bump (accent very thick at the canopy mu) vs the volume target as a good
  dotted ghost, slot-2 error arrow between the guides (label above the arrow); under-axis
  note "slot 3: trained on silence, stays silent --- the canopy loses its predictor".
  Guides: canopy mu spans all three rows, volume mu spans rows 2-3. Caption: "one rare
  pixel, wrong in every slot --- the specialists were trained for a different label".
  Bullets rewritten to the specialization argument. The row-3 step-function
  (target-discontinuity) plot of the first pass was dropped with the old argument.
  THIRD PASS 2026-07-09 (user: still the exact same visualization as K=2 — make one
  custom to this argument): the aligned-profile-rows template was replaced by a
  SLOT-GRID — columns = the head's slots 1/2/3 (bold headers), two card rows. Top row
  "what each slot learns": a stacked-ghost pixel card "almost every pixel" (ground +
  canopy, deck's repeated-card motif) arrows into three habit cards — "always: ground" /
  "always: canopy" / a DASHED empty card "never fed / always: nothing". Bottom row "the
  rare pixel": a three-layer pixel card arrows into three outcome cards showing the
  sorted target (dotted) against the slot's habitual answer (solid): slot 1 good-tinted
  "match" (ground on ground); slot 2 accent-tinted "volume missed" (good dotted volume
  target vs the accent canopy habit, side by side in the card); slot 3 accent-tinted
  "canopy missed" (accent dotted canopy target over a thick flat silence line). Between
  the rows, a dashed accent elbow arrow from the row-1 canopy card into the row-2 slot-3
  card: "the canopy is pushed / to the dead slot". Two soft footer lines: the
  dotted/solid legend and the punchline "one rare pixel, wrong in every slot --- the
  specialists were trained for a different label". tikz scale 0.95, bbox
  (-0.05,0.05)-(7.95,5.15); z-position encoding inside every card is consistent
  (ground left, volume centre, canopy right). Same day (user: increase the vertical
  separation): row 1 lifted +1.0 and row 2 +0.35 so the inter-row gap grew ~1.1 -> 1.75
  (shove arrow elbow at y 3.5 with its two label lines fully above it, footer lines
  respaced, bbox now (-0.05,0.3)-(7.95,6.15)). Same day (user: rename + slot identity
  colours): pixel caption "almost every pixel" -> "the standard case"; two-line
  specialization labels under the slot headers ("specialized in / ground search" ink,
  "specialized in / canopy search" accent, "non-specialized / head" soft); each slot
  column now carries ONE identity styling in BOTH rows — slot 1 ink!40 border /
  black!4 fill, slot 2 accent!60 border / accent!6 fill, slot 3 dashed soft / black!2
  (row 2's outcome-coloured good/accent card tints replaced; verdict words below the
  cards keep the good/accent outcome colours). Bbox top 6.15 -> 6.55. 2026-07-10 (user):
  slot-3 row-1 caption "always: nothing" -> "always: placeholder".

- "The small-K failure, concretely (K=2)" — REDESIGNED 2026-07-09 (user: better
  visualization of the problem). The old two-panel sketch (typical pixel vs
  volume-outshines-canopy, chips linked by a dashed "same channel" flow) showed the two
  label scenarios but never drew the actual failure — the network's wrong prediction lived
  only in the bullets. New layout: bullets LEFT (0.35, prior / label truncation / misfire /
  pointer-to-sorting), figure RIGHT (0.63) — a THREE-ROW story on one shared z-axis
  (rows vertically aligned, dotted colour guides at the volume and canopy mu positions
  spanning all rows): row 1 "the scene" = three layers (ground ink, volume good, canopy
  accent; amplitude-rank note at right), row 2 "the label: K=2 keeps the two strongest" =
  ground + volume solid, canopy dashed with "3rd strongest --- dropped, no slot" under the
  axis, row 3 "the network: the ground + canopy prior misfires" = prediction drawn at
  canopy height (accent, very thick) vs the label target as a good dotted ghost, a
  double-headed "slot-2 error" arrow spanning the mu gap between the guides. Bottom
  caption: "the label has no room for a third layer --- the miss is built into K=2".
  The failure is now visible as a horizontal offset between aligned rows rather than
  described in text. SECOND PASS 2026-07-09 (user: drop the ch 1 / ch 2 boxes): the slot
  chips beside rows 2 and 3, their "neq" dashed connector and the "same slot, different
  layer" label were all removed — the aligned rows and the slot-2 error arrow carry the
  story alone; figure bounding box narrowed 8.95 -> 8.6. THIRD PASS 2026-07-09 (user:
  more vertical separation, amplitude rank below the first plot): row spacing widened
  2.05 -> 2.5 (axes at y 0 / -2.5 / -5.0, headers/annotations/caption shifted along,
  bounding box deepened to -5.9), and the amplitude-rank note became a single centred
  line under row 1's axis (white-filled, matching the row-2 dropped-canopy note) instead
  of the two-line block at the right. FOURTH PASS 2026-07-09 (user: columns too close to
  the left border, balance them): the bounding box still reserved the width of the
  removed chips / side note (right edge 8.6 vs actual content ending ~5.9), pinning the
  figure left inside its column; trimmed to 6.0 so the column's \centering seats the
  figure mid-column and the two blocks sit symmetrically on the slide. FIFTH PASS
  2026-07-09 (user: wider bullet reach; slot-2 error under the axis lines): columns
  rebalanced 0.35/0.63 -> 0.42/0.56 so the bullets wrap in three lines instead of four,
  and the "slot-2 error" label moved off the arrow (which is now unbroken between the
  guides) to below row 3's axis at the arrow's midpoint x, joining the under-axis note
  language of rows 1-2; bottom caption dropped to -5.75, bounding box deepened to -6.0.
  Same day: bullets given \itemsep 10pt so the left block breathes and matches the
  figure's vertical extent.

- "Per-channel normalization, one source of truth" — REDESIGNED 2026-07-09 (user: show the
  old normalization strategy against the new; reorganize the table and the plots). Retitled
  "Per-channel normalization --- old -> new, channel by channel". Layout is now a full-width
  five-column migration table on top (channel | old | -> | new | rationale), rows in the
  ablation-ladder order (pass/mag, ifg/phase, then \addlinespace, out/amp, out/sigma,
  out/mu): old column all soft grey, new column bold and strategy-coloured (accent robust
  IQR over log1p, good fixed div-pi, soft z-score kept), arrow column "->" for moved rows
  and "=" for the kept out/mu row. Old scheme per the user + the normalization_exp base:
  zscore-log1p pass mag, min-max (p0.1-p99.9, code preset min_max_p999) ifg phase, z-score
  all outputs. Below, the two schematics became a MINIMAL PAIR on the same heavy-tailed
  curve: left soft-keyed "old --- moments fit the tail" (mean dashed dragged to x=1.05,
  wide sigma double-arrow 0.15-1.95, accent outlier dots labelled "+12 to +39 sigma") vs
  right accent-keyed "new --- quantiles fit the bulk" (the original robust-IQR sketch:
  median dashed at 0.62, shaded IQR band, same dots labelled "ignored" in soft); z-score /
  robust-IQR formula captions under each with the consequence (tail drags mu, inflates
  sigma 1.4-2.75x, bulk compressed / median+IQR ignore the tail, bulk keeps its spread).
  The old well-behaved-Gaussian z-score sketch was DROPPED (out/mu's kept-z-score rationale
  lives in its table row). Table footnotesize, arraystretch 1.32, tabcolsep 4pt.
  SECOND PASS 2026-07-09 (user: delete the bullet points, make the plots bigger and closer
  to the page centre): the two footnote bullets (one-source-of-truth / active-pixels stats
  and the non-stationarity roadmap line, both inherited from the old frame) were DELETED —
  the non-stationarity point still lives on the roadmap frame ("Interleaved splits");
  the panels grew tikz scale 1.05 -> 1.4 in columns narrowed 0.48 -> 0.38 so the pair sits
  as a tight centred group (table gap 10pt above, header/caption gaps 3pt); the left
  caption dropped the "1.4--2.75x" figure (kept on the preceding heavy-tails frame) to
  stay two lines at the narrower width.
  THIRD PASS 2026-07-09 (user: remove the "old ---" / "new ---" headers, give more focus
  to the equations): the two text headers were deleted and each panel is now headlined by
  its equation in \large display math -- soft grey z-score dfrac (x - mu)/sigma left,
  accent robust-IQR dfrac (log(1+x) - med)/(Q_3 - Q_1) right; the formulas were removed
  from the scriptsize captions, which keep only the consequence sentences (tail drags mu,
  inflates sigma, bulk compressed / median+IQR ignore the tail, bulk keeps its spread).
  The left equation carries a \vphantom of the right one so both headline boxes have
  identical height and the two plot axes stay vertically aligned. Panel identity (old vs
  new) is now carried by colour alone, matching the table's grey-old / coloured-new key.
  FOURTH PASS 2026-07-09 (user: two columns --- table in one, the other with two rows of
  equation -> plot): layout flipped from stacked (table top, panel pair bottom) to
  columns [c]: table LEFT (0.62, scriptsize, arraystretch 1.65, tabcolsep 4pt; strategy
  names shortened to fit the column --- "z-score (log1p)" / "robust IQR (log1p)", the
  full "over log(1+x)" spelling now lives only in the right-column equation), plots
  RIGHT (0.36) as two stacked equation-over-plot rows (grey z-score row top, accent
  robust-IQR row bottom, tikz scale 1.3, 14pt between rows). The scriptsize consequence
  captions and the \vphantom were dropped (rows are stacked, no cross-alignment needed);
  the plots' internal annotations (mean dragged / sigma / +12..+39 sigma; median / IQR /
  ignored) carry the interpretation alone.
  FIFTH PASS 2026-07-09 (user: transpose the table to better fit, fonts too big in a lot
  of places): the table is now TRANSPOSED --- channels as columns (pass/mag, ifg/phase,
  out/amp, out/sigma, out/mu), rows old / new / rationale (\addlinespace before
  rationale) --- and dropped scriptsize -> \tiny (arraystretch 1.4, tabcolsep 2.5pt,
  last column p{1.6cm} with an explicit \newline in ">50% ties: / IQR collapses" to
  avoid the col-lapses hyphenation; min--max shortened to (p_99.9)). The two equation
  headlines dropped \large -> \normalsize. Column split now 0.64 table / 0.34 plots.
  SIXTH PASS 2026-07-09 (user: slide very bad --- not organized, space wasted, weird
  font sizes, off-theme): the fifth-pass transposed \tiny table is REVERTED to the
  row-wise fourth-pass form --- one row per channel (pass/mag, ifg/phase,
  \addlinespace, out a / out sigma / out mu), columns channel | old | arrow | new |
  why, "=" arrow on the kept out/mu row --- at the deck's standard table setting
  (\scriptsize, arraystretch 1.75, tabcolsep 3pt), old column soft grey, new column
  bold strategy-coloured; a one-line soft key under the table ("grey = old scheme,
  coloured = new strategy, keyed to the panels"). Column split 0.58 table / 0.40
  plots. The equation headlines went back to \large (third-pass decision), tikz
  scale up 1.3 -> 1.42, row gap 14 -> 13pt; the top panel's sigma label nudged
  x 1.35 -> 1.62 off the dashed mean line. No overfull boxes on the frame.
  SEVENTH PASS 2026-07-09 (user: equations under the table in the left column, with
  some text; mid-task: equation font insanely big, follow the deck's pattern): the
  two equations MOVED from the right column into the left, under the table, as two
  minipage rows (equation 0.36 centred | consequence text 0.60): grey z-score +
  "old --- moments fit the tail. Rare bright pixels drag mu and inflate sigma; the
  bulk compresses into a narrow band." / accent robust-IQR + "new --- quantiles fit
  the bulk. The median and IQR ignore the tail; the bulk keeps its spread." (the
  consequence sentences from the third pass return as the requested text). Equations
  dropped \large -> \small (deck body size; a first \large version wrapped the
  robust-IQR fraction mid-formula at 0.34 width --- \mbox now forbids the break).
  This retires the sixth pass's colour-key line (the old/new words are in the
  captions). Right column is now the two plots alone (scale 1.42, 18pt gap). The
  \large-equation-headline idea is DEAD --- deck-size math only. Same day: the
  table-to-equations gap widened 12pt -> 19pt (user: a bit more distance).
  EIGHTH PASS 2026-07-09 (user: boxes around the plots with a top-right corner
  verdict): each plot gained an enclosing rounded rectangle drawn inside its
  tikzpicture ((-0.15,-0.28) rectangle (3.72,1.88), rounded corners 3pt, thin;
  soft!70 border on the old panel, accent!70 on the new) with a \tiny corner
  label anchored north east at (3.62,1.80): "sigma inflated, component label
  over-compressed" (soft) / "representative sigma, component label correctly
  scaled" (accent). First wording used "output"; user corrected mid-task to
  "component label". No new overfull.
  NINTH PASS 2026-07-10 (user: trim the plot annotations, fix the sigma overlap,
  table tweaks): both outlier-dot labels replaced with plain "outliers"
  ("+12 to +39 sigma" on the old panel, "ignored" on the new); the corner
  verdicts dropped "component label" --- now "sigma inflated, over-compressed" /
  "representative sigma, correctly scaled"; the old panel's sigma label moved
  x 1.62 -> 0.60 (it sat on the decaying curve --- now left of the dashed mean,
  under the rising arc, clear of it); the out/mu why-cell reads "bounded to
  sampling range" (replacing ">50% ties: IQR collapses"); a soft abbreviation
  key line added above the table: "z-score --- standard score . IQR ---
  interquartile range".

- "Label generation at scale: multi-GPU parameter extraction" — REDESIGNED 2026-07-09 (user:
  two-path visualization — both paths start at the full tomogram, one through a CPU pixel
  loop, one through the multi-GPU JAX platform, both ending in the fitted param labels;
  second pass: "too simple — make a representation for the CPU and GPU, show 4 GPUs, use
  straight lines with elbows"): the old linear ghost->box->accent chain (CPU library ->
  JAX Adam -> trained network) was replaced by a fork/join schematic in pure elbow routing
  (|- / -| segments, no diagonals). A gate source "full Capon tomogram (one spectrum per
  pixel)" forks from its north/south edges; the grey top path runs through a drawn CPU chip
  icon (pinned package + inner die, caption "one pixel per fit") into the ghost node "CPU
  curve-fit library / per-pixel loop, sequential / hours per scene"; the accent bottom path
  fans through a split bus into a vertical stack of four GPU cards (GPU 0-3, caption "all
  pixels per step") that merge back into a bus feeding the box node "multi-GPU JAX platform /
  batched Adam, all pixels at once / ~4 min per scene". Both engine nodes elbow into the
  good-tinted sink "fitted parameter labels / (a_k, mu_k, sigma_k), K Gaussians per pixel"
  via lab.north/lab.south, with the reconciliation line "same classical fit, either route"
  carried inside the sink node. The old third node "trained network / one forward pass" was
  dropped from the figure (bullet 3 still points the network story to Act VI); bullets and
  the PENDING wall-time box unchanged. THIRD PASS 2026-07-09 (user): frame made TWO-COLUMN
  --- bullets LEFT (0.40), diagram RIGHT (0.58) rotated to VERTICAL flow: tomogram at the
  top forking from its west/east edges, CPU-chip lane at x=-2 and GPU-stack lane at x=+2
  (split/merge buses now vertical: entry down the left bus, exit from the right bus
  elbowing into the JAX node's north), engine nodes side by side, labels sink centred at
  the bottom entered via lab.west/lab.east elbows; icon captions moved outboard (left of
  the chip, right of the stack); PENDING box stays full-width below the columns.
  FOURTH PASS 2026-07-09 (user: GPU lines enter from the top and leave from the bottom;
  add core counts --- CPU max 120 cores, GPUs "NVIDIA A100 48GB", core count looked up):
  the GPU stack became a HORIZONTAL row of four cards with a split bus above (stub into
  each card top) and a merge bus below (stub out of each card bottom) feeding straight
  down into the JAX node's north; engine nodes are now anchor=north aligned at y=3.08.
  Hardware lines added inside the engine nodes: "up to 120 cores" (CPU) and "4x NVIDIA
  A100 / 6912 CUDA cores each" (GPU; forced two-line break after an em-dash dangle).
  NOTE: the A100 ships only in 40/80 GB --- no 48 GB variant exists (48 GB cards are
  A40/RTX A6000, 10752 cores), so the memory size was left OFF the slide pending the
  user's confirmation of the actual cluster spec. FIFTH PASS 2026-07-09 (user): lanes
  MIRRORED --- GPU branch now on the LEFT (accent), CPU branch on the RIGHT (ghost),
  captions swapped sides with them; bullet 3 ("Faster fitting does not change what the
  labels are --- classical estimates (Act VI)") DELETED; the PENDING measured-CPU-wall-time
  box DELETED (frame no longer carries a pending marker). SIXTH PASS 2026-07-09 (user):
  bullets separated with itemsep 10pt (deck convention); GPU node retitled "multi-GPU JAX
  framework" (was "platform"). SEVENTH PASS 2026-07-09 (user): GPUs confirmed as A100
  40 GB --- memory added to the GPU node spec line; branches COLOUR-CODED as named axes:
  left = "fast axis" (accent crimson, unchanged), right = "slow axis" (argA blue --- CPU
  chip icon, node border/fill, caption and all connectors recoloured from ghost grey to
  argA); a soft dotted vertical separator at x=0 runs between the axes (y 4.8 -> 1.2)
  with the italic axis labels flanking it at y=4.62. EIGHTH PASS 2026-07-09 (user: colour
  the whole background of each branch, bigger dotted line, reference = "What this project
  does"): full-height background fills on the background layer in that slide's construction
  --- accent!5 rounded rectangle behind the fast axis (x -3.85..-0.06), argA!6 behind the
  slow axis (x 0.06..3.85), both y 0..5.85; the dotted separator became full-height and
  very thick (y -0.15..6.0), passing behind the straddling tomogram and labels nodes.
  NINTH PASS 2026-07-09 (user): the italic fast/slow axis labels moved ABOVE the tinted
  areas (anchor=south at y=5.95, centred over each half at x=+-1.95), mirroring the
  reference slide's above-the-fill area captions.

- "The output is a set, not a vector" — REDESIGNED 2026-07-09 (user: better visuals for how
  the model outputs, how two different outputs mean the same thing, and why that is bad for
  training). The frame is now a two-row story on a shared per-card z-axis: an "extracted GT
  label" row (gate node) and a "network output" row (box node) hold the SAME three
  Gaussians (accent/argA/good; symbolic triplets (a_i, mu_i, sigma_i) printed inside each
  card, coloured to match their Gaussian so identity travels with the symbol) in different
  slot orders (GT 1-2-3, prediction 2-3-1); each Gaussian sits at its mu position within
  the card so the permutation is visible at a glance. Both rows arrow into identical
  reconstructed-profile thumbnails joined by "=" ("the same profile"). Between the rows,
  dashed double-arrow connectors carry per-slot errors ell_1/ell_2/ell_3 with the sum
  "ell_1+ell_2+ell_3 >> 0" flagged "on a perfect answer"; each prediction card also shows
  its slot target as a faint dotted ghost with a small accent arrow — the gradient dragging
  every slot toward a different scatterer. Closing lines state the punishment of a correct
  set and the permutation-invariance conclusion (score the set, not the slot order). Pure
  schematic, no data dependency. Replaces the older two-curve-rows + single-profile sketch.
  SECOND PASS 2026-07-09 (user): no set-pred head and no sorting convention on this frame —
  both are introduced later; the boxes read "extracted GT label" / "network output" and the
  intro says "the network fills K slots". This frame shows the weakness of the plain model
  only. THIRD PASS 2026-07-09 (user: "remove the numbers, make it more generic"): the
  concrete triplet values and per-slot errors 1.6/1.4/2.6 = 5.6 were replaced by the
  symbolic coloured triplets and ell_k / >> 0 forms described above; closing line now says
  "charges a large loss". FOURTH PASS 2026-07-09 (user): frame MOVED from before the K=2
  failure slide to BETWEEN the K=2 (small-K/capacity) and K=3 (sorting instability) failure
  slides — Act III failure block now reads K=2 capacity -> output-is-a-set (permutation
  invariance) -> K=3 sorting -> Hungarian matching; comment marker renamed 13-iv -> 13a-ii,
  and the K=2 slide's closing pointer "the sorting mechanism itself (next)" became
  "(two frames ahead)". FIFTH PASS 2026-07-09 (user: the loss in play at that point is MSE,
  not L1): the example loss is now explicitly MSE — intro defines theta_k = (a_k,mu_k,
  sigma_k), the left legend reads "slot-wise squared error ell_k = ||theta-hat_k -
  theta_k||^2", and the sum block header is "fixed slot-wise MSE" (param-MSE is the
  training loss at this stage of the deck; L1/Hungarian notation arrives later). SIXTH PASS
  2026-07-09 (user): the closing text is ONLY the permutation-invariance line ("This is the
  permutation-invariance problem --- the loss must score the set, not the slot order");
  the "charges a large loss / gradient drags every slot" sentence was deleted — the figure
  annotations carry that story. SEVENTH PASS 2026-07-09 (user): row gap widened — the whole
  GT row (cards y 2.45-3.75, headers 3.92, gate 3.1, thumb 2.35-3.85) shifted up 0.5 and
  the between-row elements (ell labels, MSE block, "=", legend) recentred at +0.25; the
  prediction row is unchanged at y 0.

- "The answer to slot imbalance: two levers" — REDESIGNED 2026-07-09 (user: the active
  normalization slide never defines what an active pixel is, and the visualization is too
  vague). A definition strip now spans the top of the frame, above both columns: "Active
  slot = the GT placed a real scatterer in it: m_{p,k} = 1[a^GT_{p,k} > tau]", with a soft
  subline giving tau = 1e-3 linear amplitude (ParamMatcher.ACTIVE_AMP_THR in
  tools/loss/param_loss.py), "empty slots carry a=0", and the pixel-level corollary (a
  pixel is active when any of its K slots is). The left panel's old strip of 10
  undifferentiated boxes (2 filled, nothing saying what a box was) became a concrete batch:
  the 10 boxes are grouped by braces into pixel 1 (one layer) and pixel 2 (two layers),
  K=5 slots each; active slots contain a small Gaussian bump, empty slots a flat baseline;
  a mask row prints m = 1 0 0 0 0 / 1 1 0 0 0 under the slots. The two division bars are
  now quantitative: "mean: / N_pix K = 10" (short bar, "gradient diluted 3.3x by empty
  slots") vs "active-norm: / sum m = 3" (full bar). Footnote ties the equation to the
  definition (w = m, pure on/off); empty boxes switched dashed -> solid soft!50 border
  (dashes read as noise at that size); right panel unchanged except "(mask m above)" and
  tightened display-skips so the taller frame still fits (definition subline scriptsize,
  vspace -3/-9pt around the left equation, footnote one line scriptsize).

- "The two levers, full 12-run grid --- active normalization helps, presence balance
  over-corrects" — UPDATED 2026-07-10 (user: reuse the ablation-ladder color scheme to mark
  each argument's evidence in the table). The flat argC!14 rowcolor wash over the six
  set-pred rows was replaced by per-cell tints, one color per bullet: PURPLE argC!14 on the
  six set-pred RUN LABELS only (first pass painted the block's rec/2sc/c-ex/und data cells,
  user: "polluting the table, paint the labels instead") — the detection sweep they mark is
  every-vs-every (verified: conv best rec 0.686 < set-pred worst 0.878, same for
  2sc/c-ex/und; conv's R2 lead is not every-vs-every so bold+bullet carry it). Label cells
  use \tintfirst (preamble macro mirroring \tintlast: multicolumn + columncolor
  [0pt][0pt]) so paint does not bleed past the @{} LEFT edge and — user follow-up — the
  right overhang is zero too, keeping a white channel between the purple label strip and
  any tinted data cell (the sort+A row has blue R2 right next to it). BLUE argA!14 on the +A cells best-or-tied within
  their head block — they concentrate in the two sort+A rows (conv sort+A: R2/rec/c-ex/und/
  mu/a; set-pred sort+A: R2/rec/2sc) plus the bullet-cited mu 1.96 on set-pred sort+A
  (best within the sorted trio, not the block — hung+AB 1.91 holds the block bold). GRAY
  soft!22 on bullet 3's cited cells: set-pred +AB c-ex 0.885 (its win) and und 0.037/0.035
  (the cost), conv hung+AB rec 0.525 and und 0.477 (the collapse). Last-column blue
  cell (conv sort+A a 0.332) uses \tintlast, a generalized two-arg preamble macro beside
  \redlast (multicolumn + columncolor [\tabcolsep][0pt]) so paint does not bleed past the
  @{} right edge. Caption legend extended with the three color meanings ("purple labels =
  the winning head"). Column headers unabbreviated per user (rec/2sc/c-ex/und -> recall /
  shortstack "2-scatterer recall" / shortstack "count exact" / undershoot) then made
  precise in a follow-up (undershoot -> shortstack "count undershoot", mu -> "mu MAE",
  a -> "a MAE"); the caption's c-ex/und definitions dropped. VERDICT bullet deleted per
  user (three color-keyed bullets remain). \tintfirst gained \hspace{5pt} after the label
  so the widest label (hung +AB) is not flush with the paint edge. The wider headers
  overflowed the 0.68 column: bullets column 0.31->0.29, table column 0.68->0.70,
  tabcolsep 4.0->3.1pt — no overfull left on this frame. Table values untouched.

- "SAR channels are heavy-tailed" — DELETED 2026-07-10 (user). Its story is already carried
  by "Two input channels, two regimes" (real histograms + heavy-tailed/bounded split) and
  the per-channel normalization migration table; its PENDING histogram figure request
  (z-score vs robust scaling) dies with it — the minimal-pair moments-vs-quantiles plots on
  the redesigned per-channel normalization slide cover that comparison.

# Pending content

Frames carrying a dashed PENDING box need results or figures before the talk
(listed by frame title, since numbering shifts as frames are added):

- "Normalization ablation --- every rung improves, the output channels most" — NEW 2026-07-09:
  added as the last Act II frame, right after "Per-channel normalization --- old -> new,
  channel by channel" (the former "one source of truth" frame, since redesigned — see
  Design log), from results/normalization_exp/ (overview.md + metrics_comparison(1).md,
  generated 12:15). A 5-rung CUMULATIVE ladder over one shared run
  (unet-conv-sorted_gt-K_2-noaug-none-param_mse, per-slot stats): base (z-score outputs,
  zscore-log1p pass, minmax-p99 ifg) -> +pass mag robust-IQR-log1p -> +ifg phase fixed
  div-pi -> +out amp robust-IQR-log1p -> +out sigma robust-IQR-log1p (out mu stays z-score,
  matching the strategy table). STORY (framed per user 2026-07-09 as "each step improved,
  outputs most"): EVERY rung improves reconstruction -- the input rungs give small gains
  (curve R2 0.44->0.47->0.51, pixel R2 0.061->0.073->0.087, SSIM/mu-MAE/sigma-MAE all tick
  up) while detection waits; the two OUTPUT-channel rungs then carry the large gains AND
  unlock detection/localisation (matched recall 0.363->0.679, F1 0.474->0.740, count-exact
  0.252->0.615, peak err 10.4->4.2, pixel R2 ->0.417), lowest val loss overall 0.188->0.138
  (-27%). out-amp is best on curve fidelity (R2 0.62, PSNR 49.0, curve MAE 0.049); out-sigma
  is best on everything else -- it trades a hair of curve fidelity for the detection jump.
  DESIGN (iterated per user): TRANSPOSED grouped table (metrics as rows in 4 families --
  reconstruction / structure / detection / localisation, 14 rows incl. SSIM range and matched a MAE (linear amp); the 5 rungs as columns),
  bullets LEFT (0.35, three soft/argA/accent-keyed points telling the each-rung-improves /
  outputs-most story; columns [c]-centered vertically) + table RIGHT (0.63). Went through delta-annotated full-width form
  first, then per user reverted to VALUE-ONLY cells + restored bullets: the monotone
  left->right value trend carries "each step improved". The two output-rung columns are
  PAINTED (columncolor accent!14, continuous band via [4pt][0pt] overhang) with accent
  headers to mark them as the drivers; best-per-row bold (out-amp bold on curve fidelity,
  out-sigma bold on everything else). scriptsize, arraystretch 1.14, tabcolsep 4pt. Empirical
  evidence behind the strategy table's out/amp & out/sigma robust-IQR-over-log1p rows; the
  final rung IS the ladder/context/aug unet baseline (val loss 0.138). Frame text staged via
  scratchpad new_frame4.tex. UPDATED 2026-07-10 (user, four passes): (1) the three
  input-rung columns (base / +pass mag / +ifg phi) gained a cellcolor argA!14 tint on the
  rows the INPUT rungs already improve — the four reconstruction rows, the three SSIMs, and
  matched mu/sigma MAE (monotone base->ifg-phi gains; matches bullet 2's claim list);
  detection/count/peak/a-MAE rows stay untinted (flat or worse across the input rungs).
  (2) The full accent!14 columncolor band on the two output columns was replaced by
  PER-CELL accent!14 paint on only the cells that improve on the rung to their left — the
  header row is now unpainted and the out-sigma cells for curve R2 / curve MAE / PSNR
  (where out-sigma retreats from out-amp) stay white; every other output cell is painted.
  (3) Blue and red must not touch: a !{hspace 4pt} spacer column between ifg-phi and
  out-amp leaves a 4pt white channel between the default cellcolor overhangs; the LAST
  column's painted cells use \redlast (PREAMBLE macro: multicolumn + columncolor accent!14
  [\tabcolsep][0pt]) so paint does not bleed past the @{} right edge. GOTCHAS hit:
  \cellcolor does NOT take overhang args (they print literally in the PDF), and a
  parameterized \newcommand inside a frame breaks beamer ("Illegal parameter number in
  definition of \iterate") — hence the preamble definition. (4) Spacing: bullet itemsep
  3pt->9pt, arraystretch 1.14->1.22, tabcolsep 4pt->6pt (still fits, no new overfulls).
  Caption legend now: blue = metrics the input rungs already improve; red = cells the
  output rungs improve further.

- "Why clamp the output --- the exponential overflows" — UPDATED 2026-07-09 (user: show the
  normalization and denormalization routes mathematically): the left column's single boxed
  read-back equation was replaced by the two routes as mirrored arrow chains in a 3-column
  array (physical / log space / normalized): normalize p ->log(1+.)-> u ->(.-m)/s-> y (top,
  green legend: dataset build, once, bounded GT in) and denormalize p <-e^(.)-1<- z <-s.+m<-
  y-hat (bottom, accent legend: every prediction, unbounded network output in). Math matches
  pipelines/backbone/dataset/normalizer.py + tools/data/transforms.py exactly (log1p then
  z-score forward; un-scale then clamped expm1 inverse). Footnote keeps p in {a,sigma}, notes
  mu skips the log, and states the composition p = e^(s y-hat + m) - 1 so the right column's
  z/h notation still lands. The blow-up implication chain was compressed from four lines to
  two to fit; right column (clamp fix + gated-exp plot) untouched.
  SECOND ITERATION 2026-07-09 (user): the two routes are now PAINTED blocks — the array became
  a 5-column tabular with rowcolor bands, normalize chain + its caption on good!10 (green),
  denormalize chain + caption on accent!10 (red), 4pt white gap between blocks, soft column
  headers unpainted above. The implication chain went back to the original four-line aligned
  form (one implication per line, leading arrow), [3pt] spacing; fits after the user's
  same-day trims of the intro and read-route sentences.
  THIRD ITERATION 2026-07-09 (user: rowcolor stripes of two different heights per block read
  weird): the tabular was replaced by a tikzpicture — each route is ONE rounded rectangle
  (fill=good!10 / accent!10, on background layer) wrapping chain + caption as a single uniform
  area; fit includes the other caption's west/east projections so both boxes have identical
  width; chain nodes at fixed x (0 / 2.7 / 5.4) keep p, u/z, y/y-hat column-aligned across
  blocks, arrow labels above the lines, soft headers outside the fills. Captions shortened
  ("input is ground truth, bounded" / "input is y-hat, any real"). Denorm row lifted
  -1.75 -> -1.42, chain vspace 8->4pt and aligned spacing 3->2pt to unclip the bottom
  footnote line. FOURTH ITERATION 2026-07-09 (user): right column trimmed — the "Fix ---
  clamp the exponent, before the exp:" intro line and the accent closing note "A clamp
  placed after the exp is useless..." were deleted; the column now opens directly with the
  boxed min(z,h) equation and ends at the gated-exp plot caption. The left column's intro
  line "Two routes connect the spaces --- exact mirrors:" was deleted too — both columns
  now open with their visual (route boxes left, boxed equation right). With the intro lines
  gone the frame dropped [t] and the columns became [c] — both columns vertically centered.
- "Augmentation is the regularizer" — schematic train/val sketch is in place (kept as the
  concept slide). RESOLVED 2026-07-08: a measured verdict frame "Augmentation, measured ---
  flips regularize the single scene" added right after it (p30), same results-frame design
  as the ladder/amplitude tables (bullets left + painted table right). Data from
  results/augmentation_experiment/ (hv-flips vs noaug, same unet-conv-sorted_gt-K_2-param_mse
  run; noaug IS the ladder/context unet baseline). Story: flips lower val loss 0.138->0.135
  and lift every curve metric (R2 0.595->0.605, PSNR 48.73->48.85, SSIM elev 0.958->0.959),
  and help the hardest 2-scatterer pixels (recall 0.696->0.705, mu MAE 3.79->3.72, sigma MAE
  1.49->1.47); the trade is a more conservative detector (precision 0.812->0.824 up, overall
  recall 0.679->0.652 and count-exact 0.615->0.591 down). Painted argA blue = reconstruction,
  soft grey = detection trade-off, argC purple = 2-scatterer. Small but consistent, single
  seed. The schematic loss-curve on the concept slide stays (still illustrative, not the run).
  FOLLOW-UP (user: add training info / best epoch to show early stopping): added a "training
  (single scene)" group atop the table with best epoch (val) 24 (noaug) -> 49 (flips), +25 --
  the overfit turn-off delayed ~2x, the quantitative version of the concept slide's sketch;
  best val loss moved into it; bullet 1 rewritten to cite epoch 24 (early-stop) vs 49. Also
  added a curve RMSE row (0.268->0.265). Best-epoch = the checkpoint "best" epoch from each
  run's overview.md (noaug ckpt 24, flips ckpt 49). Table arraystretch trimmed 1.12 -> 1.0 to
  fit the extra rows; caption now "training rows from the validation split". SECOND PASS
  2026-07-10 (user): the unkeyed closing bullet "Small but consistent, and free..." deleted —
  the three remaining bullets are the three color-keyed arguments — and itemsep 5pt->14pt.
- "Set prediction, controlled — the head is the lever, not the matching" — NEW 2026-07-08:
  added after the DETR set-prediction lineage slide (p25) from results/hungarian_experiment/,
  a 2x2 head (conv/set_pred) x matching (sorted_gt/hungarian) factorial (ResUNet, K=2, flips,
  param-L1). Head is first-order (set-pred lifts recall 0.889->0.904, 2sc recall
  0.742->0.770, count-exact 0.910->0.923); matching is second-order (set-pred+sorted-GT beats
  set-pred+Hungarian on detection). Conv keeps a curve-R2/mu edge under param-L1. This is the
  controlled cell the third-axis grid frame asked for. Bullets left / 2x2 table right. UPDATED 2026-07-09 (user): the 2x2 table gained a structure (SSIM) block (elevation/range/azimuth -- set-pred+sorted bold on all three: 0.958/0.934/0.946) and the component-error block was completed with a sigma-MAE row (0.84/0.89/0.80/0.78, set-pred+Hungarian best) beside the existing mu and amplitude MAE; caption units note extended to mu,sigma elevation / a linear amplitude; bullet 2 now credits set-pred with the SSIMs. Table font scriptsize + arraystretch 1.06 to seat the extra rows on one frame. Source: results/hungarian_exp/ metrics_comparison(1).md (ssim_gt_* and matched_sig_mae). UPDATED 2026-07-10 (user): detection block extended with four rows from the same source — 1-scatterer recall (matched_recall_gt1: 0.667/0.602/0.996/0.997, set-pred+Hungarian bold) above the existing 2-scatterer row, precision (matched_precision: 0.812/0.774/0.824/0.822, set-pred+sorted bold), and count under/over (count_under_frac 0.284/0.331/0.019/0.022, count_over_frac 0.101/0.121/0.121/0.131) as sub-rows of count exact. Sub-rows and precision unhighlighted (argC!18 stays on the headline recall/2sc/count-exact rows). Still fits the frame at scriptsize; verified rendered page 32. SECOND PASS 2026-07-10 (user): bullets itemsep 9pt->4pt and the table stretched horizontally — column split 0.49/0.49 -> 0.44/0.54, tabcolsep 4->7pt (font stays scriptsize). THIRD PASS 2026-07-10 (user: improve the coloring-arguments visualization): the argA!18/argC!18 full-row washes were replaced with the per-cell argument scheme from the ablation-ladder/two-levers slides — PURPLE argC!14 on the set-pred columns of the detection rows it sweeps (recall/1sc/2sc/precision/count-exact/under; "over" left white, conv-sort wins it); BLUE argA!14 on each head's remaining edge (conv-sort R2+PSNR cells; set-pred SSIM block and mu/sigma/a component block — all best-or-tied vs both conv columns); GRAY soft!22 down the conv-Hung column on the detection and component rows where it breaks (SSIM rows left white, differences trivial there). A !{hspace 4pt} spacer between the conv and set-pred column pairs keeps gray from touching purple/blue; set-pred-Hung column (last, @{} edge) uses \tintlast. Caption legend gained the three color meanings. Values and bolds untouched. FOURTH PASS 2026-07-10 (user): the uncolored "Identity comes from the head..." closing bullet deleted — the three remaining bullets are exactly the three color-keyed arguments — and itemsep 4pt->14pt.
- "Ladder result: context grounds the prediction" — RESOLVED 2026-07-08: filled from
  the context_exp trio (pixel_mlp / local_cnn / unet, conv head, K=2, param-MSE, no
  aug; comparison report generated 2026-07-07 23:26, post re-regularization controls).
  Layout per user: bullets in the left column, one 20-row grouped metric table
  (reconstruction / structure / detection / localisation blocks, tiny + booktabs,
  arraystretch 1.32) covering the right column; earlier bar charts removed, then the
  recipe caption above the table removed too — the only annotation is a one-line soft
  legend under the table (held-out 5.0M px; bold = best; delta color convention).
  Each of the local CNN and enc-dec columns carries a color-coded step delta vs the
  previous tier (green improvement, accent regression, soft tie). Honest nuance in bullets and table:
  the local CNN edges overall detection (recall 0.69 vs 0.68, count exact 0.68 vs
  0.62, F1 0.75 vs 0.74) but the full window wins two-scatterer pixels (recall 0.70
  vs 0.63) and all reconstruction/localisation metrics; the pixel MLP's median
  per-pixel R2 is negative (-0.03) — the conditional-mean regression argument made
  visible. 2026-07-08 (complete batch, results/context_experiment/, report 14:28): all
  20 existing rows re-verified identical to the deck; added a "matched a MAE" row to the
  localisation block — 0.531 / 0.326 (local, best) / 0.334 (enc-dec, +0.008), amplitude
  error saturates at the local-CNN tier like the other metrics the local CNN edges.
  FOLLOW-UP (user: "component errors, I am just seeing some of them"): only mu MAE had its
  per-scatterer breakdown; completed the component block so sigma and a each also carry
  1-scatterer / 2-scatterer split rows — sigma 0.43/0.26/0.24 (1sc) + 2.01/1.59/1.49 (2sc);
  a 0.276/0.135/0.133 (1sc) + 0.797/0.557/0.551 (2sc); enc-dec best on every split. Table
  grew 21->25 body rows, so arraystretch trimmed 1.32 -> 1.14 to keep it on one frame (no
  overfull). Source report in results/ (gitignored). TRIM PASS 2026-07-10 (user): bullets 2
  ("Not a departure from theory...") and 4 ("justifies the benchmark's scope") deleted —
  two color-keyed bullets remain (argA climb-with-context, argC detection nuance). Table cut
  24 -> 14 rows, keeping the SSIMs and the radical movers (curve R2, per-pixel R2 median,
  peak error median, recall / 2sc / count exact, mu MAE + 2sc, sigma MAE + 2sc, a MAE);
  dropped val loss, curve MAE, PSNR, cosine, peak mean, F1, all 1-scatterer sub-rows and the
  sigma/a... 2sc kept. Font tiny -> scriptsize. GOTCHA: with r@{hskip 3pt}l value/delta
  column pairs, a rowcolor'd delta cell OVERPAINTS the last digit of the value cell (cell
  backgrounds paint in cell order with tabcolsep overhang > the 3pt gap) — the hskip must
  be >= tabcolsep (now 6pt >= 5pt). FIFTH PASS 2026-07-10 (user): the two remaining bullets
  tightened (bullet 1 lost the E[label|g(x)] inline math and "exactly what/cannot compute
  it" verbiage; bullet 2 lost the parenthetical local-CNN detail), itemsep 5->18pt, and the
  columns switched [T] -> [c] so the bullet block centers vertically beside the table.
  SIXTH PASS 2026-07-10 (user): bullet column narrowed 0.46 -> 0.38, table column 0.52 ->
  0.60 with \centering -> \raggedright so the table sits flush left, closer to the bullets.
- "How much context is enough? (patch-size sweep)" — RESOLVED 2026-07-08: the n=2..7
  sweep run landed and the section was simplified to three frames (sweep intro /
  information ceiling / verdict). SWEEP-INTRO FRAME DELETED 2026-07-10 (user) — the
  section now opens directly with "The correlation law" after the ladder result; the
  intro's registration framing survives on the verdict frame ("exactly as registered" +
  registered-vs-observed table, curves labelled n=2..7). NOTE: the constant-pixel-budget
  detail (why curves saturate rather than peak) now lives NOWHERE on the slides — it was
  only on the deleted intro; keep it as a talking point or restore a caption line if the
  saturating shape is challenged. The two registered sample-budget floors
  (track-count w sqrt(N/n) and kz-aperture w sqrt(S_N/S_n)) were refuted by the data:
  the saturation knee sits at ~48 px for every track count (48 px captures 80-94% of
  each curve's total gain) instead of climbing 41 -> 76 px as the floors demanded, and
  the near-zero-baseline n=2 pair still trains to within 23% of the best loss. The
  adopted reading: the label-window ceiling 2w = 40 governs; 48 is the first
  admissible size past it (border padding), the slow tail follows the shrinking padded
  fraction. Verdict frame plots the six normalized curves (real numbers hardcoded from
  the sweep run). Full derivations and the refuted floors preserved in vault note
  patch-size-optimum-derivation-2026-07-07; comparison tooling
  scripts/tmp_sweep_vs_kz_theory.py (temporary) + scripts/measure_kz_aperture.py.
  Caveats to keep in mind if challenged: single seed, pre-selection-fix subsets,
  argmin censored at the 96 px grid edge — the knee statistic is the reading.
  LATER SAME DAY: the floors were removed from the slides entirely (ceiling-only
  story; sweep-intro registration bullet reworded, verdict plot bars/label and table
  row dropped). The ceiling frame gained a labelled 1-D overlap diagram and the
  denoiser framing (overlap = certificate of shared statistics); the verdict frame
  gained a padding-effect diagram (W = 48/64/96 patches, full-context core
  (W-2w)^2/W^2 = 3/14/34%). The three-readings argument (born from the user's
  objections: why 2w not w; doesn't influence chain forever; one-shot labels break
  the naive argument) is a full CONTENT frame "Three readings of the ceiling ---
  replicate, estimate, or chain" between the information-ceiling and verdict frames
  (a first backup-frame version was promoted and removed; a 1-D-strip figure version
  was rejected — "keep the full square"): three 2-D grid scenes in the deck's
  established window-diagram language, scenes 1-2 sharing identical geometry so only
  the patch boundary and shading differ (1: W=w patch hugging B(x), overlapping
  neighbour window present but unshaded and marked "ignored"; 2: W=2w patch, the
  B(x') n B(x) intersection shaded and labelled "shared"; 3: fading chain of
  pairwise-overlapping windows, then cdots and a disjoint B(x'') with the label
  B(x) n B(x'') = emptyset). Right column states per reading WHAT is shared and WHY
  it matters, with equations (label(x) = fit(R_hat(x)) from B(x) alone — nothing
  else entered it; shared pixels feed both labels so B(x') samples the same R(x),
  extra looks that average away label speckle noise, true for |x'-x| < w;
  Cov(label(x), label(x'')) = 0 — the homogeneity bridge is assumption, not data);
  knees 32/48/none, one 16 px sweep step past each ceiling. The verdict frame's
  table now adjudicates the three readings (32 refuted / 48 matches every n / none
  refuted). Talking points if
  pressed: correlation is not transitive (disjoint boxcar averages of independent
  speckle are exactly independent), and one overlap step is anchored by shared
  samples while a second step is pure prior.
  THIRD PASS 2026-07-08 (user): the three-readings frame was cut to the winning
  reading only and retitled "Reading the ceiling --- the network estimates
  statistics". Layout per user: visualization-first, text minimal. Left column
  (0.56) = one enlarged 2-D grid scene (scale 1.40; B(x) with x, overlapping
  B(x') with the shaded shared set annotated "shared looks", a near-limit
  neighbour x'' whose shared set is a sliver annotated "overlap -> 0 at
  |Delta| = w", W = 2w patch outline) + one-line caption. Right column (0.40) =
  new pooled-shared-looks accrual curve on top (rise 2v - v^2, hard knee dot at
  2w = 40 px, flat tail, 48 px tick; labels "every step adds looks" / "nothing
  new enters") with three terse bullets below (label(x) = fit(R_hat(x)), one
  speckle draw, w^2 = 400 looks; shared-look count
  (w-|Delta_az|)_+(w-|Delta_rg|)_+ positive iff |x'-x| < w; pooling shrinks
  variance as 1/L_eff which stops growing at 2w) and the registered-prediction
  line (knee 48 px, same at every n). The verdict frame's three-reading
  adjudication table became a registered-vs-observed check (knee 48 px every n
  vs observed 48 px every n; gain at the knee 80-94%). The refuted readings now
  live only in the vault derivation note.
- "Padding brings no new looks --- the patch runs past 2w" — MECHANISM CORRECTED
  2026-07-10 (user: "the affirmation that the padding thickness is always w is
  really weird"). The old frame claimed a physical w-thick symmetric-padding
  mirror band at the split border ("identical 2w mirror cap", hatched band drawn
  w thick, 71%/31% redundant badges from the schematic geometry); verification
  against the pipeline refuted it. Facts: the only symmetric padding in the patch
  path is Patcher.extract (pipelines/backbone/dataset/spatial.py), whose amount
  is the grid-tiling remainder (W + (n-1)*stride) - H — bounded by stride-1 per
  axis, split across opposite borders, carried only by the outermost grid
  patches. For the 12000x3500 sweep split at stride_ratio 0.5 that is 0 px on
  the azimuth axis for every sweep size and 2-10 px per side on range — never a
  constant w=20 band, often zero. Interior patch borders get no mirror at all:
  model convs use default zeros padding (models/blocks.py). The backing script
  analyze_patch_padding_redundancy.py (branch worktree-patch-padding-analysis)
  HARDCODED padding_thickness = w and (W+2w)^2 padded footprints — a model, not
  a measurement; its numbers left the slide. The corrected frame keeps the title
  and conclusion but swaps the mechanism to clipped windows: the network sees one
  WxW patch, a pixel's 2w window clips at the patch edge, the filler (conv zeros,
  or the few mirrored remainder px) adds no new looks, the clipped rim is fixed
  at w per side, and only the full-window core (W-2w)^2 grows — at W = 2w only
  the centre pixel keeps a full window, so the first size with a real core is 48,
  the first admissible step past 40 (this now carries the +8 knee offset on the
  slide itself). New diagram: withheld split above a dashed border, W=48 and
  W=96 patches to scale (0.03 u/px) with accent w-rims and green cores (core
  badges 3% / 34% = (W-2w)^2/W^2), one pixel x near the border with its dashed
  2w window overhanging into a hatched "clipped — zeros or mirrored duplicates"
  region, w rim measure on the left. Caption notes interior edges are softened
  by the half-patch stride overlap while the split border is not. The sound
  version of the old claim (fixed band, patch-size independent) survives as the
  w rim; the vault note patch-size-optimum-derivation-2026-07-07 already stated
  it correctly ("at W = 2w only the centre pixel holds its full window").
- "Amplitude-only reconstruction" — RESOLVED 2026-07-08: results landed from the
  2026-07-07 15:49 comparison report over runs
  tracks_exp/unet-conv-sorted_gt-K_2-hv-A-param_mse_1_{5amp,all_amps,baseline}.
  The phase-input baseline wins 8/10 headline metrics (grouped score 0.846 vs
  0.495 vs 0.000): curve R^2 0.61 / 0.46 / 0.41, curve MAE 0.048 / 0.059 / 0.066,
  PSNR 48.9 / 47.5 / 47.1 dB, matched F1 0.73 / 0.71 / 0.71 for baseline /
  30-amp / 5-amp. Amplitude-only keeps baseline-level matched recall (0.65-0.67)
  but single-scatterer mu-MAE nearly doubles (1.07 vs 0.60 elevation units); its
  only wins are peak-location mean (30-amp) and count-exact fraction (5-amp).
  PENDING box replaced by the held-out metric table, bullets carry the numbers,
  axis dots repositioned R^2-proportional. LATER SAME DAY: split into two frames
  at user request — "Amplitude-only reconstruction --- no interferograms"
  (the test: three input cards + protocol card, hypothesis axis with hollow
  H1/H2 markers vs the reference dot, right column registers H1 phase-is-
  everything vs H2 priors-carry-part) and "The verdict --- amplitudes
  reconstruct, but the phase gap stands" (8-column table: curve R^2 / MAE /
  PSNR / SSIM elev + matched recall / precision / F1 / mu-MAE_k=1, best per
  column bold, discussion bullets + act exit line). SECOND ITERATION (user):
  protocol card removed from the test sketch (protocol folded into the test
  bullet), cards rearranged — the two amplitude-only cards on the top row, the
  reduced-stack reference card centred on the second row; verdict frame is now
  bullets LEFT / transposed table RIGHT (metrics as rows in reconstruction /
  scatterers blocks, runs as columns, best per row bold). THIRD ITERATION (user):
  verdict table enlarged (column 0.45 -> 0.47, tabcolsep 4.5pt) and a third block
  "component error (matched pairs)" added — mu MAE 2.76/2.66/2.16 with per-k splits
  1.16/1.07/0.60 (1 sc) and 4.58/4.28/3.75 (2 sc), sigma MAE 0.938/0.945/0.850 with
  splits 0.280/0.284/0.245 and 1.69/1.62/1.46 (5amp/30amp/baseline, baseline bold
  everywhere); the old mu-MAE_k=1 row moved into this block. FOURTH ITERATION
  2026-07-08: the complete results batch (results/tracks_experiment/, the amplitude-only
  5amp/all_amps/baseline runs) now carries matched_amp_mae, so the "not exported by the
  report" placeholder was replaced with the real amplitude block — a MAE 0.328/0.345/0.356
  (5|A| best/bold), per-k 0.142/0.164/0.151 (1 sc, 5|A| best) and 0.540/0.530/0.564
  (2 sc, 30|A| best); the amplitude-only inputs actually edge the phase baseline on
  amplitude error (they see |A| directly). Caption units line extended to "a in linear
  amplitude"; table arraystretch trimmed 1.05 -> 0.97 + act-exit vspace 4pt -> 1pt to keep
  the taller table on one frame. Source reports live in results/ (gitignored).
  FIFTH ITERATION 2026-07-10 (user: delete the not-phase-free-tomography point,
  vertically centre the rest): the soft caveat bullet (one scene, scene-specific
  prior, cross-scene transfer unproven, full stack remains the default) was
  DELETED from the verdict frame; three argument bullets remain (H2 verdict /
  amplitude drives the profile / phase places the scatterers). Centring needed
  no code change --- columns [c] centres the shorter bullet block against the
  table; verified on the rendered page.
- "The grid" / "Reading the two axes" / "The third axis" / "Does the ambiguity
  argument survive set prediction?" / "Winner vs baseline" —
  findings and the delta table from the comparison report are in as text. UPDATED
  2026-07-08 from the COMPLETED benchmark (results/benchmark_experiment/, overview.md +
  metrics_comparison, 145 cells, all four head x matching corners now filled). The
  two missing corners (set_pred+sorted_gt, conv+hungarian) were run for resunet +
  attention_unet, so head and matching are now DEconfounded. Verdict: the gain is the
  HEAD, not Hungarian. Three new benchmark-results frames placed BEFORE the
  mechanism/gradient block (right after "The third axis", before "Why parameter losses
  win") so the empirical answer precedes the why. Bullets left / painted table right,
  argA/argB/argC rows keyed to bullet markers:
  (1) "The completed board: the top TEN configurations" — top-10 ranked on the FIVE CORE
  GROUPS with Gain-vs-Capon DROPPED (user 2026-07-08: re-rank ignoring Capon, too many
  runs lack it). Full model + head + matching + loss + score columns (user: always show
  model+head+loss). Co-leaders tied at 0.992: resunet set_pred sorted param_l1 AND
  deeplabv3plus set_pred hungarian param_l1 (deeplab was masked in the grouped-score
  headline for want of a Capon reference). Rows 3-4 = same model Hungarian vs sorted,
  identical 0.991 (Charbonnier). Ranks 1-8 all set_pred; only conv runs in top-10 are
  ranks 9-10, both param_l1 sorted. Two backbones own the top-10: resunet 6, deeplabv3plus
  4; attention_unet just misses (0.964). Colors: argB gold = co-leaders (rows 1-2), argA
  blue = tied pair (3-4), argC purple = conv rows (9-10). Slides 2-3 and the grid frame
  already use the same Capon-free composite, so all consistent.
  ITERATED 2026-07-08 (later): the ranking table grew top-10 -> top-15 -> TOP TWENTY
  ("add more ranks to fill the column"), gained an SSIM_az column, font \tiny->\scriptsize,
  table col 0.56->0.62 / bullets 0.37 \footnotesize. Six backbones in top-20 (resunet,
  deeplab, nafnet, linknet, swin, dense); attention_unet still just misses (rank 22). Note
  the conv-only-via-param claim now holds only for ranks <=16 (swin/nafnet conv+Charbonnier
  enter at 18/20), so the bullet was reworded to "conv appears only from rank 9 down,
  almost always param-L1". A SECOND full-frame dense-table slide was ADDED right after the
  ranking (user: "huge table, no text, as many rows/cols as fit, small font"): "Every
  headline metric, top 32 configurations" -- \tiny, 32 rows x 19 columns (# model hd mt
  loss | R2 PSNR cos | SSel SSrg SSaz | F1 rec rec2 prec | muMAE sigMAE pkE | score),
  per-column best bolded, conv rows shaded argC!12, one-line legend only, no prose. Metrics
  from metrics_comparison(1).md (Curve-Level, SSIM, Per-Pixel R2/Cosine, Peak Location,
  Matched Gaussian) + score5 from overview; this report has NO matched_amp_mae column so
  amplitude error omitted. arraystretch 0.95 + tabcolsep 2.4pt fits 32x19 w/o overflow.
  Generator: scratchpad mega2.py / splice_dense.py. Deck now 68 content frames / 84 pages.
  HEADERS SPELLED OUT 2026-07-10 (user: "dont use abreviations like SSel, SSrg, use full
  names"): hd/mt/cos/SSel/SSrg/SSaz/rec/rec2/prec/muMAE/sigMAE/pkE -> head / matching /
  cosine / SSIM elev / SSIM range / SSIM azim / recall / 2-scatterer recall / precision /
  mu MAE / sigma MAE / peak error, compound names as two-line \shortstack headers; legend
  line updated (head:/matching: decode) and its metric-name enumeration dropped as
  redundant with the readable headers (units + matched/per-pixel qualifiers kept). Fits at
  the same tabcolsep 2.4pt / arraystretch 0.95, no overfull (16:9 slide had ~25% width
  slack). Data codes SP/S/H and loss short forms (p-L1, Charb, p-Hub) kept with the legend.
- "Before the fit: prominence-ranked initialisation" — SYMBOLS DEFINED 2026-07-10 (user:
  "add to the text what is the deltaE and span/8K variables"): the sigma_0 =
  max(2 Delta-xi, span/8K) bullet now defines Delta-xi = elevation grid step and
  span = xi_N - xi_1 = full grid extent (100 m here, the z in [-20, 80] m Capon window),
  and reads the 8K denominator as "an eighth of a component's even share of the axis,
  floored at two grid bins". Matches the code: initialiser.py sigma_base =
  max(2*dh, h_span/(8*K)) (pipelines/processing/param_extraction/sigma/initialiser.py:69).
  Frame fits, no overfull (page 10).
  (2) "Where the gain comes from: the head, not the matching" — UPDATED 2026-07-08 (user
  "show the pairs that were compared"): the aggregate curve/param table was replaced with
  the eight actual completing cells (ResUNet + Attention-UNet x {mse,huber,Charb,param-L1}),
  each row = c+S (conv+sorted) / SP+S (set-pred+sorted) / SP+H (set-pred+Hungarian) with
  Delta-head (SP+S - c+S, bolded) and Delta-match (SP+H - SP+S, grayed). Curve rows tinted
  argA blue, param-L1 rows argB gold (keyed to bullets). ResUNet curve Delta-match = exactly
  0.000 (bit-identical); the lone -0.136 is att mse-curve noise. Original summary numbers:
  head effect +0.23 on
  curve losses vs +0.01 on param, matching ~0 both; proof = resunet set_pred curve runs
  are bit-identical under sorted vs Hungarian (RMSE 0.20329 both, permutation-invariant).
  EXPANDED 2026-07-08 (user "add 2 more param examples -> 3 per space, add a third model"):
  table grew to 3 curve (mse/huber/Charb) + 3 param (p-MSE/p-Hub/p-L1) x 2 decomposable
  models, columns loss | c.S | c.H | SP.S | SP.H | Dtot | Dhead | Dmatch. KEY DATA
  CONSTRAINT: the completing corner was run only for resunet+attention; set_pred+sorted
  exists ONLY for {mse,huber,Charb,param_l1}, conv+hungarian ONLY for {param_mse,huber,l1}.
  So curve losses are decomposed via the SP.S corner (Dhead=SP.S-c.S, Dmatch=SP.H-SP.S) and
  param losses via the c.H corner (Dhead=SP.H-c.H, Dmatch=c.H-c.S) -- different completed
  corners, blanks (--) mark the un-run one. A THIRD model (DeepLabV3+) was added but has
  NEITHER completing corner (only conv+sorted and set_pred+hungarian), so it shows Dtot
  (=SP.H-c.S) ONLY, Dhead/Dmatch blank, labeled "total only -- 2x2 not completed". Deeplab
  Dtot follows the same shape (curve +0.11/+0.13/+0.32, param +0.08/+0.01/+0.01). No other
  backbone can be decomposed without new runs (would need deeplab set_pred+sorted or
  conv+hungarian cells). Dtot bolded, Dmatch grayed; Dtot vs Dhead+Dmatch off by <=0.001
  (3-decimal rounding). Generator: scratchpad corners.py / splice_gain.py.
  ABBREVIATIONS FIXED + BULLETS REWRITTEN 2026-07-10 (user): headers c.S/c.H/SP.S/SP.H/
  Dtot/Dhead/Dmatch -> two-line \shortstack conv sorted / conv Hungarian / set-pred
  sorted / set-pred Hungarian / Delta total / Delta head / Delta matching; loss rows
  mse/huber/Charb/p-MSE/p-Hub/p-L1 -> MSE-curve / Huber-curve / Charbonnier / param-MSE /
  param-Huber / param-L1 (top-20-table naming); legend's c/SP + S/H decode dropped
  (headers self-describing), now states the split formula in full words + row-color key.
  Bullets: the old two-bullet ellipsis pair ("...but <= +0.06...") became three
  self-contained color-keyed bullets --- argA curve losses: the head carries the gain
  (+0.08..+0.32, curve loss gives slots no identity, the gated head supplies it); argB
  parameter losses: the head barely matters (<= +0.06, per-slot targets already assign
  identity); soft-grey NEW third square keying the grayed Delta-matching column: matching
  is a wash, exactly 0.000 on ResUNet curve rows, bit-identical runs, a curve loss never
  sees the matcher. DeepLab footnote unchanged. Page 51, fits, no new overfull. NOTE the
  sibling frame "Hungarian at K=2" still carries the same short forms (mse/Charb/p-L1,
  att-unet, Hung.) --- not asked, left as is.
  (3) "Hungarian at K=2: costs nothing --- and that is the point" — UPDATED 2026-07-08
  (user "do the same"): aggregate 3-row table replaced with all 14 sorted->Hungarian
  pairs, split into a set-pred block (8: resunet+att x mse/huber/Charb/param-L1) and a
  conv block (6: resunet+att x param-MSE/Huber/L1, the only conv+hungarian cells run).
  Columns model|loss|sorted|Hung.|Delta. Colors keyed to bullets: argA blue = set-pred
  curve pairs (all Delta=0.000, permutation-invariant), argB gold = set-pred param-L1,
  argC purple = whole conv block. Delta 0 to slightly negative in every pair; lone -0.136
  (att mse) flagged as mse-curve instability. Summary numbers were: match effect 0.000
  (curve) / -0.004 (param-L1) / -0.006 (conv param); framed as free insurance for K>2
  where the slot-capacity ceiling forces more scatterers and sorted ordering goes
  unstable (callback to the K=3 slide).
  Reconciled the earlier frames to match: "third axis" caveat now says the completing
  cells were run and confirm head-not-matching; gradient-view (29f) pi relabelled
  assign() with "sorted or Hungarian, identical at K=2", bottom line credits the gated
  head not the matcher; grid frame winner updated to param-L1 #1 (Charbonnier/L1 within
  0.01). NOTE the Act III controlled 2x2 frame "Set prediction, controlled" (line ~1103,
  hungarian_exp, single scene, UNet param-MSE) makes the same head>matching point at
  probe scale; the new Act V frames are the full-grid confirmation + top-5 + K>2 argument
  (complementary, not duplicate). Still needed: the heatmap and the two marginal charts,
  plus the outstanding cells (hrnet / fpn / u2net columns, deeplabv3plus set-pred
  inference, cosine on four backbones). Scoring caveat: rank on the five core groups, not
  the sparse Gain-vs-Capon column. (The composition-ambiguity explainer, the two-column
  gradient-derivation slide and the valley-geometry slide need no numbers.)
  DELETED 2026-07-10 (user): "Benchmark design: architecture x objective, jointly",
  "The grid", "Reading the two axes" and "The third axis: sorted regression vs Hungarian
  set prediction" are all GONE from the deck — Act V now goes tooling -> completed board
  (top twenty) -> dense top-32 board -> head-vs-matching decomposition. The heatmap and
  marginal-chart pendings died with "The grid"/"Reading the two axes" (OBSOLETE unless
  the frames come back); their unique prose findings (axis marginals: ResUNet leads /
  param > curve on average / interaction via robust curve losses; UNETR 0.55 failure;
  cosine recall-by-overprediction caveat; third-axis 22-of-23 pairs + practical rule)
  survive only in this entry now. The setup facts moved onto the improved tooling frame
  (see its entry below).
- "Winner under the microscope" (structural fidelity + profile reconstruction) —
  pending boxes now cite the FIVE-SEED board winner's inference run 20260720_193330
  (unet_skip · set_pred · L1-curve, seed 0; five-seed SSIM 0.968/0.950/0.961). The
  run dirs live on the server, so the figures cannot be produced locally — pull the
  inference figures from that run when regenerating. (Historical: an earlier pass
  targeted the single-seed grid winner's run 20260707_165557.)
- "Choosing K: fit every order, penalise complexity" — LAYOUT PASS 2026-07-09 (user):
  columns swapped (bullets + score plot now LEFT 0.56, the three fitted-curve sketches
  RIGHT 0.42), bullets sit ON TOP of the score plot, fitted-curve sketches enlarged
  (xscale 1.02->1.45, yscale 0.72->1.02; score plot 1.15/0.95 -> 1.45/1.15 to fill the
  freed space). The "best K" label moved away from the curves (upper-middle, above the
  penalised curve) and is connected to the circled K=2 point by a thin good-coloured
  leader line — it previously overlapped the descending score curves. SECOND PASS same
  day (user): the penalised score is now a displayed equation on its own line inside the
  second bullet, with underbraces naming MSE_K "fitting component" and lambda_K K
  "regularization component"; the prose in the lambda_K=0 bullet renamed to match
  (fitting component decides / regularization component becomes load-bearing).
- "Hungarian matching" — EDIT PASS 2026-07-13 (user): in the left (a,mu,sigma)-space
  sketch the w=0 annotation is REMOVED (the caption already says the grey match is
  zero-weighted) and the third predicted point — the one linked to the empty slot —
  moved from a-height 1.1 down to 0.35, hovering just above the a=0 plane; the
  "predicted" label moved with it (anchor west, beside the point). In the right column
  the two-matrix cost/matching group shifted ~1.25em toward the slide centre (trailing
  \hspace*{2.5em} after the tikzpicture under \centering).
- "The answer to slot imbalance — lever 1: active normalization" — EDIT PASS 2026-07-13
  (user): the right-column loss equations now show the lever as a transformation:
  ell_mu and ell_sigma each display the pre-lever form (denominator N_pix K) -> arrow ->
  the active-normalized form with the denominator sum m highlighted in accent; ell_a
  stays single-form (unchanged by the lever). Operand subscripts p,k dropped inside the
  numerators to fit two fractions per row (index kept on the sums); the old inline
  row comments folded into a soft scriptsize note under the align (ell_a untouched /
  numerator stays, denominator swaps cell count for active count). Verified: tectonic
  full + sub_05 with zero overfull from the section (full 113pp, sub_05 12pp),
  pdftoppm render of both frames inspected.
  SECOND PASS same day (user): the soft note under the align ("ell_a is untouched ...
  cell count") is DELETED, and the header line above the two-slot sketch ("one pixel,
  K=2 slots --- each slot outputs (a,mu,sigma)") is DELETED; the equations and the
  sketch now stand uncommented. Verified: tectonic full + sub_05, render inspected.
- "The imbalance failure (high K)" — REDESIGN 2026-07-13 (user asked for a better
  visualization of the failure plus equations). LEFT: bullet 2 ("silence is almost
  always the right answer") replaced by the mechanism math — match rate rho_k (deep
  slots rho ~ 0.01), the mean-reduced two-point mixture
  E[ell_k(a-hat)] = rho_k e(a-hat,a+) + (1-rho_k) e(a-hat,0), and its minimizer as a
  cases display: rho_k E[a+] under MSE (the hedge, dimmed to ~0) vs exactly 0 under L1
  once rho_k < 1/2 — echoes the step-5 Pr*E hedge and step-4 median machinery of the
  L1-vs-MSE section without forward-referencing it; soft note carries "zeroing the
  slot is the loss optimum, not a training accident". RIGHT: figure grew a third panel
  ON TOP — "the imbalance": per-slot horizontal supervision-mix bars (accent = share of
  cells where the matcher hands the slot a real target, gray = silence target a=0),
  slot 1 near-full, slot 2 partial, slots 3-5 slivers tagged "~1% --- outweighed 99:1";
  in-bar direct labels replace a legend. The old two panels (deaths + falling loss)
  kept below, compressed (activity y-span 2.9 -> 1.9, scale 1.05 -> 0.92); the old
  99:1 annotation moved into panel A, panel B annotation now reads "starved of real
  matches, each deep slot slides to its optimum: silence --- for good" (white-filled,
  clear of the slot 1/slot 2 curve labels). Panel titles: "the imbalance ---" /
  "the consequence ---". Bar shares beyond the known ~1% deep-slot rate are sketches,
  same rigor level as the pre-existing activity curves. Verified: tectonic full (113pp)
  + sub_05, zero overfull from the frame, render inspected (two passes: fixed
  overlapping displays from a -14pt vspace, moved the panel-B annotation off the
  slot labels).
  SECOND PASS same day (user: "the visualization is a bit much"): the supervision-mix
  bar panel and the training-loss panel are DELETED (with them the "keeps falling
  through every death" and "tail is abandoned" notes and the ~1%/99:1 bar tag); the
  right column keeps ONLY the slot-activity-over-training panel (re-expanded, value
  span 1.9 -> 2.4, steps arrow restored) and gains the gradient-pull equations below
  it: the mixture gradient split with underbraces "pull toward the target"
  (rho_k d e(a-hat,a+)) vs "push toward silence" ((1-rho_k) d e(a-hat,0)), then the L1
  sign evaluation rho_k(-1)+(1-rho_k)(+1) = 1-2 rho_k ~ 0.98; soft note: constant slope
  points back to silence on the whole run-up, MSE balances at the dimmed optimum.
  Left column unchanged. Verified: tectonic full (113pp) + sub_05, zero overfull from
  the frame, render inspected.
  THIRD PASS same day (user: only param-MSE exists at this point of the story): all
  L1 forms purged from the two frames. "The imbalance failure" left column: mixture
  now E[ell_k] = rho_k E[(a-hat - a+)^2] + (1-rho_k) a-hat^2, single minimizer
  a-hat* = rho_k E[a+] ~ 0.01 E[a+] (cases display with the L1 branch REMOVED), note
  says "shrinking the slot to effective silence is the loss optimum" (shrinkage, not
  hard zero — MSE-accurate). Right column gradient block: pull = 2 rho_k (a-hat -
  E[a+]), push = 2 (1-rho_k) a-hat, collapsing to 2 (a-hat - rho_k E[a+]); note reads
  the 1%/99% weights and the 1%-amplitude parking spot. "Lever 1" right-column
  equations: |.| -> (.)^2 in all three losses (ell_a, ell_mu, ell_sigma), old->new
  arrow forms kept, denominator highlight kept. NOTE: the "Hungarian matching" frame
  still shows the matching cost with ||.||_1 — matching-cost norm vs training loss is
  a separate choice, left untouched pending user confirmation. Verified: tectonic
  full (113pp) + sub_05, zero overfull from both frames, renders inspected.
  FOURTH PASS same day (user): equations moved OUT of the left column into the right
  column, equations on top / plot on the bottom. Right column = one three-row align
  (mixture E[ell_k]; gradient split with pull/push underbraces; net force
  2(a-hat - rho_k E[a+]) => a-hat* = rho_k E[a+] ~ 0.01 E[a+]) above the slot-activity
  plot; the separate below-plot gradient block and both soft notes are GONE (their
  content lives in the align and the bullets). Left column is prose-only, four
  bullets: nothing-to-match / matcher hands a+ with rho_k~0.01, mixture on the right /
  push outweighs pull 99:1, shrinking to effective silence is the loss optimum /
  K=2 sidesteps, K=5 confronts. Verified: tectonic full (113pp) + sub_05, zero
  overfull from the frame (tightened the arrow spacing on the align last row),
  render inspected.
  FIFTH PASS same day (user): origin row added at the top of the align --- the
  per-cell matched loss ell_k(a-hat) = (a-hat - a_k^match)^2 with the two-outcome
  match target as a cases (a+ with prob. rho_k, 0 with prob. 1-rho_k); the E[ell_k]
  mixture row now reads as its expectation over the match outcome. Verified: tectonic
  full (113pp) + sub_05, zero overfull from the frame, render inspected.
- "The answer to slot imbalance --- lever 2: presence balance" — EDIT PASS 2026-07-13
  (user: reduce text, add equations showing how the lever counters the gradient pull).
  The frame-top f_k definition strip is DELETED and the three heavy bullets cut to two
  short ones (per-slot imbalance / inverse class frequency); the old "amplitude term
  trains every slot" bullet content is gone. The w_{p,k} + f_k align kept; its note
  shortened to "m as on lever 1; f_k measured per batch (~rho_k), clipped at 1e-3".
  NEW counter-effect chain below (continues the imbalance-frame notation): E[w ell_k]
  = f_k (0.5/f_k) E[(a-hat - a+)^2] + (1-f_k)(0.5/(1-f_k)) a-hat^2 -> the frequencies
  cancel to 1/2 + 1/2 -> gradient = (a-hat - E[a+]) + a-hat with underbraces "pull,
  now 1/2" / "push, now 1/2" -> optimum shift row a-hat*: rho_k E[a+] -> 1/2 E[a+]
  (new 1/2 in accent, lever-1 arrow motif). Columns rebalanced 0.40/0.58 ->
  0.44/0.54, occupancy-strip tikz scale 0.86 -> 0.82. Verified: tectonic full (113pp)
  + sub_05, zero overfull from the frame, render inspected.
  SECOND PASS same day (user): chain split across the columns --- the E[w ell_k]
  cancellation rows (product form -> 1/2 + 1/2) stay on the left under the lead-in;
  the gradient row (pull/push underbraces at 1/2 each) and the optimum-shift row
  (rho_k E[a+] -> 1/2 E[a+]) moved to the RIGHT column directly below the slot-3
  gradient bars they formalize; the tikz closing line "every slot balanced against
  itself --- silence no longer buys the tail" is DELETED (the a-hat* row states it).
  Verified: tectonic full (113pp) + sub_05, zero overfull, render inspected.
  THIRD PASS same day (user): the lead-in sentence "The counter-effect --- the weights
  cancel the class frequencies and the tug-of-war returns to 1:1:" is DELETED (the
  E[w ell_k] align now follows the f_k note directly), and the two diagonal
  uniform-w -> balanced-w arrows in the slot-3 gradient diagram are now straight
  vertical lines (accent at x=0.22 under the active sliver, soft at x=3.00 in the
  empty region; x0.5/f_3 and x0.5/(1-f_3) labels beside them at mid-gap). Verified:
  tectonic full (113pp) + sub_05, zero overfull, render inspected.
  FOURTH PASS same day (user): the occupancy strips are explicitly pixel-indexed ---
  10 columns at pitch 0.44 replaced by 8 columns at pitch 0.56 (box width unchanged,
  wider gaps) with px_1..px_8 headers above the first row, so each column reads as
  one pixel; slot-2 pattern now 4/8 active, slot-3 1/8; row width and right-side f
  annotations unchanged; strip title nudged up to clear the header row. Verified:
  tectonic full (113pp) + sub_05, zero overfull, render inspected.
- "Hungarian matching" — MSE-CONSISTENCY FIX 2026-07-13 (user: only param-MSE exists
  at this stage; closes the flag raised in the imbalance-frame third-pass entry): the
  matching-cost norm switched from ||.||_1 to ||.||^2 in both the displayed objective
  L = min_pi sum w_k ||theta-hat - theta||^2 and the cost-matrix caption
  C_ij = w_j ||theta-hat_i - theta_j||^2 --- now matches the slot-wise squared error
  already shown on the small-K frame (line ~88). Illustrative cost numbers unchanged.
  Verified: tectonic full (113pp) + sub_05, zero overfull, render inspected.
  FIFTH PASS same day (user: px_i headers too big/ugly): the px_1..px_8 column headers
  shrunk (node scale 0.7 on tiny) and lightened (soft -> soft!75), nudged down 0.02 ---
  subtle index marks instead of full-size labels. Verified: tectonic full (113pp) +
  sub_05, zero overfull, render inspected.
- VOCABULARY PASS 2026-07-20 (user: humanizer standing rule now in force; deck swept
  for AI-coded vocabulary, punctuation/symbols explicitly out of scope): all 19
  section files checked (AI word list, promotional adjectives, copula avoidance,
  -ing tack-ons, intensity adverbs, significance inflation, filler, weasel
  attributions). Three word-level fixes, no numbers touched: "Normalization
  ablation" outputs bullet "unlocks detection & localisation" -> "makes detection
  & localisation work"; by-count frame lead-in "k=1 pixels are the unlock" ->
  "k=1 pixels carry the gain" (mirrors the benchmark section's "the head carries
  the gain"); Take-home item 2 "measured, mitigated, and honestly reported" ->
  "measured and mitigated, the residual reported". Kept as legitimate register:
  "robust" (robust-IQR scheme name), "critical curves" (math), "faithfully
  reconstructed" (standard DSP collocation), "stability keystone" (single
  authorial metaphor). Verified: tectonic full (122pp) + sub_03, only the
  pre-existing overfull at 04_representation:149.
