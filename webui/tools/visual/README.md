# Animation visual harness

Renders the pipeline process animations (`webui/static/js/process_anim.js`) headlessly with MathJax loaded and dumps per-timestep PNG frames of the canvas. Used to audit equation placement, text overlap, sizing, and scene pacing without driving the full web UI.

## Setup

```
npm install
```

Requires a Playwright Chromium under `~/.cache/ms-playwright` (any `chromium*` build). Override the executable with `PW_CHROMIUM=/path/to/chrome`.

## Usage

```
node dump_frames.js <process_anim.js> <scene-key> <duration-s> <step-s> <out-dir>
```

Example — full sweep of the inference scene at 2 s steps:

```
node dump_frames.js ../../static/js/process_anim.js inference 124 2 /tmp/frames
```

Scene keys and loop durations: `param` 88 · `processing` 122 · `dataset` 118 · `training` 210 · `inference` 124 · `tuning` 96.

The dumper performs two warm passes first so MathJax equation sprites are rasterized before screenshots, and prints sprite cache stats (`ready/total/failed`). Use a 2 s step for full audits and 0.5 s around transitions.

`mathjax-tex-svg.js` is a vendored copy of MathJax 3.2.2 `tex-svg` so the harness works offline; the live UI loads the same bundle from the jsdelivr CDN in `index.html`.
