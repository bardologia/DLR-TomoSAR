# AI Agent Specification — Model Architecture Diagram Generator

## 1. Objective

Add or modify model architecture diagrams in the DLR-TomoSAR web console (`webui/`). The agent takes a model architecture (its implementation under `models/` and its vault note under `notes/DLR-TomoSAR/Models/`) and produces its interactive gallery entry: the backend family record, the network diagram blueprint, and the zoomable per-component block definitions — all conforming to the diagram standard codified in this specification.

The agent operates under four guarantees:

1. **Faithful**: every box, glyph, connector, and skip path on a diagram corresponds to a real operation or tensor route in the model implementation; nothing is invented or omitted at the granularity the diagram shows.
2. **Geometrically clean**: orthogonal routing only, no line ever crosses or overlaps another line collinearly, no line passes through a box, no box overlaps a box, no text leaves its box or the canvas.
3. **Standard-conformant**: orientation, flow connectors, glyph usage, skip styling, spacing, and zoom scale follow the rules in Sections 4–7 exactly; deviations are not stylistic choices available to the agent.
4. **Regression-safe**: existing models render byte-identically after the change; the verification protocol of Section 8 passes in full before delivery.

The three owned files are `webui/model_library.py`, `webui/static/js/diagram.js`, and `webui/static/js/models.js`. The agent does not modify model implementations, configuration dataclasses, or CSS except where Section 7 explicitly permits.

---

## 2. Inputs Required

| Input | Type | Description |
|-------|------|-------------|
| `model_key` | Text | Registry key of the architecture (must exist in `models/__init__.py` `CONFIG_REGISTRY`). |
| `model_source` | Path | The implementation file under `models/` — the ground truth for op sequences inside blocks. |
| `model_note` | Path | The vault note under `notes/DLR-TomoSAR/Models/` — the ground truth for design intent, family placement, and the "when to use" guidance. |
| `family` | Text | Gallery family: `CNN encoder-decoder`, `Transformer`, `Multi-head CNN`, or `Context and resolution`. A new family requires explicit user approval. |
| `params` | Text | Measured default-config parameter count (e.g., `~19M`), taken from a real instantiation, never estimated. |

---

## 3. System Architecture (what is edited where)

| File | Role | Typical change |
|---|---|---|
| `webui/model_library.py` | Backend data served at `/api/models`: `CONFIG_CLASSES` registry plus `_families()` records with `key`, `name`, `skip`, `head`, `params`, `recommended`, `when` | One registry line, one family entry |
| `webui/static/js/diagram.js` | `ModelDiagram`: `KEYSPEC` dispatch, network blueprints (`_uGraph` or custom), block definitions, the two block renderers, the orientation/flow engine | Blueprint + blocks for the new model |
| `webui/static/js/models.js` | Gallery list, detail panel, popup management, legend | A legend note only if a new `skipKind` is introduced |

Dispatch is **key-based**: new models are added to `KEYSPEC` with an explicit `{skipKind, type}`; string-sniffing of `skip`/`head` text is the legacy path for the original models and must not be extended.

The backend resolves `activation`/`normalization` live from the real config dataclasses — the model's config class must exist before the gallery entry is added.

---

## 4. Network Diagram Rules

### 4.1 Layout selection

- Architectures with a symmetric encoder–bottleneck–decoder shape reuse `_uGraph` (labels-only blueprint).
- Architectures that are structurally not a U (pyramids, parallel branches, context modules) get a dedicated custom graph rendered by `_netCustom`, using the same visual language: `_node`, `_gridStack`, `_path`, `_label`, the shared colour constants, ordinal animation delays, and a content-sized `viewBox`.

### 4.2 Routing

- **Orthogonal only.** Routes `H` and `V` insert a mid-bend when endpoints are not aligned; `HV`/`VH` are the two elbow forms; `P` takes explicit `via` waypoints plus `fromSide`/`fromPt` and `toSide`/`toPt`. A diagonal segment is a defect.
- Multi-feed merges use **dedicated channels**: vertical channel x-positions distinct per feed, or shared x with disjoint y-spans. Outer feeds take outer channels so paths never cross.
- Merge boxes receiving several feeds are drawn tall, with **distinct ports on the receiving edge, in feed order** (top-to-bottom matching the sources). Port y-values must not coincide with any neighbouring row's y, or arrival segments become collinear with departure segments.
- No segment may pass through any box; no two boxes may overlap; arrival/departure points on a box edge must be unique per edge.

### 4.3 Tensor grids

- The input grid sits **at the top of the first column**, entered vertically; the output grid sits **above the output head**, with flow upward — params always exit at the top right.
- A grid stack visually extends about 12 px beyond its nominal 50 px box and carries labels up to 53 px above its centre: grids are never placed horizontally adjacent to a block, and grid `cy` is at least 70 so label baselines stay ≥ 10 px from the canvas top.

### 4.4 Edge labels

- Placed on the **longest segment** of the path: beside vertical segments (`x + 8`, anchor start), above horizontal ones.
- A label needs roughly 100 px of run; shorter edges carry no label. Redundant labels on repeated parallel edges are written once.

### 4.5 Canvas

- Height is sized to content with at least 14 px bottom margin; width fits the rightmost element. Custom layouts target the 800–880 px width band of the existing diagrams.

### 4.6 Per-node flow attributes

- The orientation engine annotates every network node with `data-flow="<in sides:counts>;<out sides:counts>;<upward>"`. This is what makes **edge instances** of a shared block (first/last stage of a chain) open popups showing only their own connectors. Custom blueprints must route all edges through the standard edge list so this annotation stays automatic.

---

## 5. Block Popup Rules (the zoom standard)

### 5.1 Orientation

A block renders as a **horizontal row** when its network attachments (all edges, skips included, aggregated over every instance of the block) satisfy `left + right >= top + bottom`; otherwise it is a **vertical stack**. Ties stay horizontal. Fan heads (multi-head outputs) are exempt and keep the fan layout. Nested sub-blocks inherit the **parent popup's** orientation and flow direction.

### 5.2 Flow connectors

- `flowIn`/`flowOut` are ordered **side-sets**: every side with at least one attachment appears; the primary side (majority of main-flow edges, skips excluded, axis-preferred on ties) comes first. `upward` is derived — primary input `bottom` means upward — never hand-set.
- Entry and exit connector lines are drawn for **every** side in the set, in its true direction.
- **Multiplicity**: the per-side connector count equals the maximum count over the block's node instances (cap 4), drawn as parallel lines — 14 px apart on left/right, 18 px on top/bottom, tightened to 8 px when converging into a circle glyph so all lines touch it.
- Non-primary inputs attach at the **merge op** (the concat glyph or first concat-labelled op) where the tensor actually joins, not at the chain start.
- A vertical block's chain-start carries the primary vertical entry; the chain-end carries the vertical exit; horizontal sides attach at the merge op (entries) or last op (exits). Horizontal rows: left entries at the row start, right exits at the row end, vertical sides drop into the first op / leave the last op.

### 5.3 Operation representation

- **One operation per box.** Composite titles (`Conv 1x1 -> Add`, `GAP + 1x1`) are forbidden: split into op boxes and glyph nodes, or give the box a descriptive single-concept title with the op chain in the sub line (`Image pool` / `GAP, 1x1, up`).
- Glyphs: addition is a `+` circle, elementwise product an `x` circle, **concatenation a `||` circle** with its annotation beside it — to the right of the circle in vertical stacks, 12 px **below the row** in horizontal ones (never inside neighbouring boxes' vertical span).
- Titles must fit their boxes: 126 px boxes (horizontal) and 208 px boxes (vertical); detail belongs in the smaller sub line.

### 5.4 Skip connections and projections

- **Identity skips** (pure residuals, token re-use) are plain labelled green lines.
- **Parameterized skip paths** (1×1 projections, upsample-then-add) carry a small op box (70 × 22) sitting on the path — the visual encodes whether the skip carries parameters. The box is declared with `op: "<layer>"` on the shortcut.
- The skip channel runs on the side **opposite the vertical input** (input from below → channel above), so it never laces an incoming connector; with no vertical input it defaults below.
- A shortcut originating at a sum/cat node taps the **main flow line 8 px past the circle** in flow direction, never a floating point at the arrival row.
- Vertical channel offset: 48 px past the op column, so the 70 px op box clears the column by ≥ 8 px.

### 5.5 Geometry and scale

- Vertical popups use a constant 392 px canvas for uniform zoom; the op column is **centred (92/92 margins)** when the block has no shortcuts, and sits left-of-channel otherwise.
- Horizontal popups are content-sized: the pop width is `natural width x 1.1 + 40`, capped at 1100 px; the CSS display scale is 1.1x, matching the vertical density. Fixed-width pops around short rows are a defect.
- Margins follow the actual connector directions (a bottom arrow needs 26 px below; a top channel needs 40 px above), computed, not assumed.

---

## 6. Backend Entry Rules

- `CONFIG_CLASSES` keeps aligned colons; family records keep the exact dict layout and field order of the existing entries.
- `params` is the measured default-config count. `recommended` is `True` for at most one model (the benchmark winner).
- The `when` text is 1–2 terse sentences in the established register, condensed from the Model Zoo note's selection guidance — first fragment states the use case.
- `skip` and `head` strings follow the data contract exactly as the frontend expects them; changing them on an existing model requires checking `spec()` consequences.

---

## 7. Style Configuration

- Colour constants come from `ModelDiagram.C` (`accent #35e6d0`, `accent2 #7cff9b`, shared fills/strokes) — never hardcode new colours.
- Animation: nodes stagger by ordinal delay; skip paths settle after the flow (existing timing constants).
- CSS changes are limited to the popup scale/width rules of Section 5.5; diagram geometry is never fixed in CSS.
- No emojis or decorative Unicode anywhere; the only non-ASCII glyphs are the operation symbols inside circles.
- Code style: compact static methods, aligned object literals, section divider comments only (`/* ---------- x ---------- */`), no other comments.

---

## 8. Verification Protocol (mandatory, in order)

1. `node --check` on every edited JS file; `py_compile` on the backend file.
2. **Build sweep**: `ModelDiagram.build()` for all gallery models with real `/api/models` data — balanced `<svg>`/`<g>` tags; every `data-block` and nested `data-subblock` reference resolves in the block map.
3. **Network geometry harness** (intercept `_path`/`_node`/`_gridStack`/`_label`): orthogonal segments only; no segment through or overlapping any box (grids tested at full visual extent including labels); no box–box overlap; no collinear line overlaps; no interior crossings between different paths; no label or grid text clipped at the viewBox.
4. **Popup sweep**: every block of every model rendered in **both orientations** (children inherit either); every line and rect inside its viewBox.
5. **Multiplicity assertions**: known multi-feed blocks show the expected connector counts; edge instances (first/last of a chain) show only their own sides via `data-flow`.
6. **Regression**: all pre-existing models produce byte-identical network SVG against a baseline captured before the edit.
7. Runtime note: backend Python changes require a server restart; static JS/CSS changes need only a browser refresh.

A failure at any step is fixed before delivery — the agent does not deliver a diagram that fails its own harness.

---

## 9. Output Deliverable

- The three owned files updated (or fewer, when the change is narrower), conforming to Sections 4–7.
- The throwaway verification scripts removed after use; results summarized in the final report (counts of renders checked, regression status).
- When the change introduces a new family, skip kind, or layout pattern, the vault notes ([[Model Zoo]], [[Control Console]]) are updated accordingly.

---

## 10. Constraints and Forbidden Patterns

- No diagonal lines, no collinear path overlaps, no path-through-box, no box-box overlap, no clipped text — ever.
- No composite operation titles; no operation rendered as a bare text box when a glyph exists for it (`+`, `x`, `||`).
- No identity-styled line for a parameterized skip, and no op box on an identity skip.
- No hand-set `upward`/`horizontal` flags on network blocks — both derive from the flow engine; only nested sub-block inheritance and the fan exemption bypass it.
- No string-sniffed dispatch for new models; `KEYSPEC` only.
- No fixed-size popups independent of content; no zoom-scale divergence between orientations.
- No fabricated parameter counts, op sequences, or architectural claims — everything traces to `models/*.py` and the model notes.
- Existing models must not change, visually or structurally, as a side effect.

---

## 11. Cooperation with Other Agents

**Receives from (upstream):**
- The model implementation and config (created by the developer or a model-authoring session) — the op-level ground truth.
- [[notes_auditor]] — audited model notes mean the `when` guidance and design rationale carried into the gallery are trustworthy.

**Hands off to (downstream):**
- [[slides_generator]] — presentation decks may reference the gallery's family structure and capacity figures; both must agree with [[Model Zoo]].

**Handoff contract:** the gallery is a *view* over `models/` and the vault notes. Whenever an architecture is added or restructured, this agent runs after the registry and notes exist, never before; disagreement between diagram and implementation is always resolved toward the implementation.
