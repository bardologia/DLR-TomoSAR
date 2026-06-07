# AI Agent Specification — Equation Walkthrough Animation Generator

## 1. Objective

Add or modify the animated equation walkthroughs in the DLR-TomoSAR web console (`webui/`) — the **Animated walkthrough** mode of the Model tab. The agent takes a pipeline (its documented equations and the variables they consume and produce) and produces a guided, 3Blue1Brown / Manim-style derivation: a left-to-right conveyor of equations that the viewer steps or plays through, with colour-coded terms, a contextual sketch and tip beside each step, and dotted right-angle connectors linking variables and equations.

The agent operates under four guarantees:

1. **Faithful**: every equation, term, role, and variable shown traces to a real operation in the pipeline source (`pipelines/`) or its reference entry in `webui/equation_library.py`; illustrative tile values are clearly illustrative and never presented as measured data.
2. **Geometrically clean**: connector lines use right angles only, are routed so they never cross or overlap equation glyphs, and the equation that is in focus is never obscured. Text never leaves its box or the canvas.
3. **Standard-conformant**: the layout, colour roles, transition timing, highlight behaviour, arrow semantics, tip styling, and iteration rules follow Sections 4–9 exactly; these are codified preferences, not stylistic choices available to the agent.
4. **Regression-safe**: the static Reference grid and every other Model-tab feature render unchanged; the verification protocol of Section 11 passes in full before delivery.

The owned files are `webui/flow_library.py`, `webui/static/js/flow_view.js`, the cinematic CSS block of `webui/static/css/styles.css` (delimited by the `Model tab: guided cinematic walkthrough` banner), and the asset-version query strings plus the MathJax config in `webui/static/index.html`. The agent does not modify `equation_library.py` (the Reference grid), the pipeline code, or any unrelated CSS.

---

## 2. Inputs Required

| Input | Type | Description |
|-------|------|-------------|
| `pipeline_key` | Text | Stable key for the flow (`processing`, `param`, `dataset`, `training`, `inference`, `tuning`). Matches the pipeline registry naming. |
| `equations` | Source | The documented equations — `equation_library.py` group entries and the implementing modules under `pipelines/`. Ground truth for every `tex` string and constant. |
| `variables` | Derived | The data arrays/tensors/profiles/parameter vectors moving between steps, each with a role and a representation kind. |
| `step_order` | Derived | The execution order of the steps; the conveyor advances in this order. |

A new pipeline beyond the two seeded flows (`processing`, `param`) is added by appending a `_<pipeline>()` builder to `FlowLibrary.collect()`; the frontend is data-driven and needs no per-pipeline code.

---

## 3. System Architecture (what is edited where)

| File | Role | Typical change |
|---|---|---|
| `webui/flow_library.py` | `FlowLibrary` served at `/api/flows`. One builder method per pipeline returning `{key, name, blurb, nodes, steps}` | A new `_<pipeline>()` builder, or edits to an existing flow's nodes/steps |
| `webui/static/js/flow_view.js` | `FlowView`: conveyor, transitions, highlight, carry/progression/tip wires, sketch library, iteration readout, controls | The `GUIDE` map (sketch + tip per step) and `_sketchMarkup` cases for any new step ids; behaviour only when a rule in Sections 4–9 changes |
| `webui/static/css/styles.css` | The `.cine*` cinematic block | Style only; geometry of arrows/highlights is computed in JS, never fixed in CSS |
| `webui/static/index.html` | MathJax config, script/style includes, cache-bust versions | Bump `?v=` on every edited static asset; the MathJax `html` extension must stay enabled |

The backend is pure data. All animation logic is in `FlowView`. The `GUIDE` object keys by `step.id`; a step with no `GUIDE` entry simply renders without a sketch or tip.

---

## 4. Data Model (the flow schema)

Each flow is `{key, name, blurb, nodes, steps}`.

**Node** (a variable):

```
{"id", "tex", "role", "kind", "shape", "sample", "desc"}
```

- `role` is one of `measured`, `intermediate`, `calculated`, `final` (Section 5).
- `kind` is `scalar`, `vector`, `set`, `matrix`, or `tensor` — drives the numeric-tile representation.
- `sample` is a small illustrative value (one number, a short list, or a small grid); strictly illustrative.
- `shape` and `desc` are short human strings.

**Step** (one transformation):

```
{"id", "title", "phase"?, "note", "inputs":[node_id], "outputs":[node_id],
 "lines":[ [term, ...], ... ], "iterative"? }
```

- `lines` is a list of display lines; each line is an ordered list of **terms**. A multi-line step is a stacked equation.
- A **term** is `{"id"?, "tex", "role"?}`. A term with a `role` is a variable token; it is tagged so it can be coloured and matched across equations. A term without a `role` is structural glue (operators, fractions, sums) and is rendered plain.
- A term's `id` is the matching key. The **same `id` in two steps is the same variable** and enables the carry box + arrow. Never reuse an `id` for a different quantity (a past defect: `s_0` tagged with id `si`).
- `iterative` (optional) marks a loop step: `{"var", "steps", "label", "trace":[...]}` — `trace` is the short illustrative convergence sequence the counter walks.

Validation (enforced before delivery): every `inputs`/`outputs` entry resolves to a defined node; every role term's `id` is either a defined node id or an intentionally local token.

---

## 5. Roles and Colour

Exactly four data roles, each a fixed colour used for the term, its tile representation, and any arrow originating from it:

| Role | Meaning | Colour |
|---|---|---|
| `measured` | External input / given data | `#6ea8ff` (blue) |
| `intermediate` | Transient quantity produced and consumed mid-pipeline | `#f5b971` (amber) |
| `calculated` | A real computed output of a step | `#4fd6c4` (teal) |
| `final` | Pipeline output / supervised target | `#c4a3ff` (violet) |

These are defined as CSS variables on the `.cine` root and mirrored in `FlowView.COLOR`. Colour is applied to MathJax sub-terms by setting `element.style.color` on the `\cssId` element (the SVG inherits via `currentColor`) — **never** `\color{#hex}` in TeX (the `#` is a macro-parameter character and fails). Config constants (thresholds, learning rates) stay inside the equation glue, not as separate coloured tokens.

---

## 6. Stage Layout

- **Cinematic dark stage.** The walkthrough is a dark stage that **breaks out to the full viewport width** (`width: 100vw; margin-left: calc(50% - 50vw)`), is tall (`78vh`), and has a **fullscreen** toggle (Fullscreen API; `:fullscreen` scales equations and tips up). The equation is the hero, not a card.
- **Three equations, current centred.** A `previous · current · next` window: the previous step sits left (`s0`), the **current is centred** (`s1`), the next step previews on the right (`s2`). Advancing slides everything one slot left; the next becomes the centred current. First step has no left neighbour; the **final step finishes centred** with no right neighbour.
- **All three fully visible.** No opacity dimming of the flanking equations — every visible equation is at full opacity.
- **Current is marked** by the highlight box of Section 8 (not by dimming the others).
- **Breadcrumb trail** of every step's output symbol runs across the top, dotted-linked, clickable, with the active step highlighted.
- **Controls**: restart, previous, play/pause, next, a progress track, and fullscreen.
- **Equation font** is enlarged (`~1.26em`, larger in fullscreen). Stacked lines within one step are **tight** (`~5px` gap) so they read as one grouped step.

---

## 7. Connectors (arrows and links)

All connectors are drawn in one SVG overlay and recomputed after any slide settles and on resize/fullscreen change. Three distinct kinds, visually separable:

- **Progression arrow** (equation → next equation): a **long, headless**, brighter dashed line with a continuous flowing (marching-dash) animation, drawn in the gap between consecutive equations at their shared vertical centre. It conveys direction by motion, not an arrowhead. One per adjacent visible pair.
- **Variable carry** (a variable shared by two equations): the variable is **boxed** in the earlier equation, with a **dotted, right-angle (90°-only) arrow** to the same variable's position in the later equation, coloured by the variable's role. The arrow is **routed up into a channel above the equations**, across, then down onto the target — it must never cross equation glyphs. Drawn for **one pair only**: the `previous → current` transition (the step that produced the current equation); on the final step, the `previous → final` pair. Never on the `current → next` pair.
- **Tip link** (equation → its tip): a short dotted line in a muted neutral from the equation to its tip box (up to a top tip, down to a bottom tip).

The carry box (role-coloured) and the highlight box (Section 8, blue) are different things and must stay visually distinct.

---

## 8. Focus Highlight

- The centred (current) equation carries a **highlight box**: a soft blue rounded rectangle with a subtle glow, framing the equation.
- For a **single-line** current equation the box is static around that line.
- For a **stacked** current equation the box **scans line by line, top to bottom, on a loop**, dwelling on each line (`~1.2s`), with a smooth transition between lines.
- The highlight starts only once the equation has settled in the centre (after the slide), and is cleared when the viewer moves to another step.

---

## 9. Iteration Readout

- A step marked `iterative` (e.g. the Adam width fit) shows a small readout under its equation.
- It stays **idle** (a static `sigma fit · t = 1 … N` label) while the step is the off-centre next/previous equation.
- The live counter (`iterating · t = … · value → …`, walking the `trace`) **starts only when the equation is centred (the focus)** and stops when the viewer leaves the step. The iteration is a focus-only animation; it never runs on a flanking equation.

---

## 10. Tips and Sketches

- Each step's tip is a **boxed card** (border, dark translucent fill, soft shadow) holding a small **draw-on sketch** and a concise text line.
- Tips are placed **above or below** the equation, alternating by step parity so the two visible tips never collide and both the top and bottom free space is used. Each tip is joined to its equation by a tip link (Section 7).
- **Tip text** is one informative, precise sentence describing what the step actually does — phrased for a reader who knows the domain, polished, no filler. (Example, the Adam step: the widths are fit by gradient descent while the means and amplitudes stay fixed at the profile's peak maxima.)
- **Sketches** are hand-built inline SVG micro-diagrams in a `240×150` viewBox, one per `GUIDE` entry, drawn on by animating `stroke-dashoffset` of the stroke paths (`.skl-draw`, `.skl-axis`) and fading in dots/fills (`.skl-pop`, `.skl-dash`). They use the role-colour utility classes (`c-meas`, `c-mid`, `c-cal`, `c-fin`, `c-faint`, `f-*`). The catalogue includes profile, smoothed profile, prominence peaks, top-K bars, Gaussian-mixture fit, convergence curve, penalty-vs-K arg-min, sort bars, R-squared heatmap, rotating phasor, unit phasor, Capon spectrum, and subsection concat. New steps reuse an existing type or add a clean, label-light one.

---

## 11. Technical Requirements and Gotchas

- **MathJax `html` extension** must be loaded (`loader: { load: ["[tex]/html"] }`, `tex.packages "[+]": ["html"]`) so `\cssId{...}{...}` term tagging works; without it the macro renders literally.
- **Colour via `currentColor`**, never `\color{#hex}` (Section 5).
- The walkthrough root element must carry the **`cine` class** so the role-colour CSS variables resolve for all descendants (a past defect made sketch strokes invisible because the variables were undefined).
- **Draw connectors after the slide settles** (~720 ms after the entering group is ready) so `getBoundingClientRect` reads final positions; redraw on `fullscreenchange` and `resize`.
- **Full-width breakout** via `width: 100vw; margin-left: calc(50% - 50vw)` on the `.cine` container.
- **Cache busting**: bump the `?v=` query on every edited static asset in `index.html`. Backend (`flow_library.py`) changes require a **server restart** (`bash webui/run.sh 8765`); static JS/CSS changes need only a refresh.
- **Visual iteration is mandatory** (Section 12). Drive a headless Chromium through the steps and read back the rendered frames; tune spacing, sizing, and routing from what is actually rendered, not from the source. The reusable screenshot harness (global `playwright-core` plus the cached chromium binary) is recorded in user memory; reuse it rather than reinventing.

---

## 12. Verification Protocol (mandatory, in order)

1. `node --check` on `flow_view.js`; `py_compile` on `flow_library.py`.
2. **Endpoint shape**: `/api/flows` returns every flow; for each, every step `inputs`/`outputs` resolves to a defined node id; no role term reuses an id for a different quantity.
3. **Console clean**: a full play of every flow produces **no `pageerror` and no console error** (the `/favicon.ico` 404 is the only allowed network miss).
4. **Lifecycle clean**: after a full play, step-back, and trail-jump — the conveyor holds the correct visible count (3 mid-pipeline, 2 at the ends), exactly one highlight exists on the current equation, and no orphaned ghost/box/arrow nodes remain.
5. **Layout checks** (read back from rendered frames): all visible equations at full opacity; current centred (`s1`); final step centred with no right neighbour; stacked-line highlight advances through the lines; carry box+arrow present only on the correct pair and routed clear of glyphs; progression arrows present and headless.
6. **Iteration check**: the iterative readout is idle while off-centre and only runs when centred.
7. **Regression**: the Reference grid still renders its full set of equation cards; no other Model-tab feature changed.

A failure at any step is fixed before delivery. Throwaway capture scripts are removed after use; the final report states what was checked (flows played, steps captured, regression status).

---

## 13. Output Deliverable

- The owned files updated, conforming to Sections 4–10, with asset versions bumped.
- Capture scripts removed; results summarised in the final report.
- When a new pipeline flow or sketch type is added, the relevant vault notes are updated so the walkthrough and the written documentation agree.

---

## 14. Coding Conventions (vault rules)

- No comments, no docstrings; all logic in classes with methods (`FlowLibrary`, `FlowView`); a scheduler/dispatch method sequences the work.
- Vertically align runs of `=`/`:` into columns; separate logical groups with blank lines; one statement per line.
- No emojis or decorative Unicode anywhere — in code, tips, sketches, logs, or notes. Plain professional wording.
- Reuse before reimplementing; never copy a helper between modules.
- No backward-compatibility shims or silent fallbacks: when the flow schema changes, update producer and consumer to the new form only and regenerate; break loudly rather than tolerate stale formats.

---

## 15. Constraints and Forbidden Patterns

- No diagonal connector segments; no line crossing or overlapping an equation glyph; no connector obscuring the focus equation.
- No `\color{#hex}` in TeX; no reliance on a colour without the `cine` class present.
- No reused term `id` for a different variable; no fabricated equation, constant, or role — everything traces to `pipelines/` or `equation_library.py`.
- No dimming of flanking equations to mark the current one; the highlight box is the only focus marker.
- No carry arrow on the `current → next` pair; carry is the `previous → current` (or `previous → final`) pair only.
- No iteration animation on an off-centre equation.
- No arrowhead on the equation-to-equation progression line.
- The Reference grid and unrelated Model-tab features must not change as a side effect.

---

## 16. Cooperation with Other Agents

**Sibling:**
- [[model_diagram_agent]] — the other `webui/` visualisation surface (architecture diagrams). Both share the orthogonal-routing, no-overlap, faithful-to-source discipline and the same verification mindset; they own disjoint files.

**Receives from (upstream):**
- The pipeline implementation under `pipelines/` and its `equation_library.py` reference entries — the equation-level ground truth. This agent runs after the equations are documented, never before.

**Handoff contract:** the walkthrough is a *view* over the documented equations. Whenever a pipeline's mathematics changes, the equations are updated at the source first, then this agent reconciles the flow; disagreement between the walkthrough and the implementation is always resolved toward the implementation.
