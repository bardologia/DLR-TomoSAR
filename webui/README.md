# DLR-TomoSAR Control Console

A dark-tech, radar-themed web console for the DLR-TomoSAR project. It renders the
project's signal-model equations, lets you browse and configure the nine entry-point
scripts, and launches them with live-streamed output.

## Run

```bash
webui/run.sh            # starts on http://127.0.0.1:8765
webui/run.sh 9000       # custom port
```

`run.sh` auto-selects the `Dune` conda interpreter (the env that carries the project
dependencies) and falls back to `conda:base` then system `python3`.

The backend is **standard-library only** (`http.server` + server-sent events), so it
needs no extra installation. The frontend pulls KaTeX, GSAP, and highlight.js from a
CDN, so a network connection is needed for equation typesetting and syntax highlighting
(both degrade gracefully if offline).

## What it does

| Surface | Backed by | Behaviour |
|---|---|---|
| Hero | `radar.js` | Animated radar sweep and a live Gaussian-mixture elevation spectrum. |
| Model | `/api/equations` | KaTeX-rendered signal model, mixture target, losses, and optimiser. |
| Pipelines | `/api/pipelines` | The six pipelines as a staged flow; each maps to a launchable entry point. |
| Architectures | `/api/models` | Ten backbones across three families, with selection guidance. |
| Scripts | `/api/scripts` | The nine `main/*.py` entry points. Open one to inspect source, edit its constant block, pick an interpreter, and launch. |
| Configuration | `/api/configs` | Every configuration dataclass, field, type, and default, parsed live from `configuration/*.py`. |
| Console | `/api/jobs/<id>/stream` | Real-time stdout of launched jobs over SSE, with stop control. |

## How launching works

The `main/*.py` scripts are configured through a block of module-level constants at the
top of each file, then run as `python main/<script>.py`. The console mirrors that model:

- **Edit configuration** rewrites only the constant assignments you changed, preserving
  formatting. The original file is copied to `webui/.backups/<script>.<timestamp>.bak`
  before any write.
- **Launch run** executes the script from the repository root with the selected
  interpreter and streams its output. Exit codes are reported honestly; nothing is
  faked.

Scripts expect the F-SAR dataset paths and GPUs referenced inside their constant blocks.
On a machine without those, a launched job will stream its real traceback to the console.

## Architecture

Backend, one class per file (no comments, per the vault coding rules):

```
serve.py            ServeEntry        CLI entry, builds and runs the server
web_ui_server.py    WebUIServer       orchestrator; threaded HTTP server
request_router.py   RequestRouter     path dispatch, JSON, static, SSE
script_catalog.py   ScriptCatalog     lists scripts, parses constant blocks (AST)
script_editor.py    ScriptEditor      applies constant edits with backup
config_registry.py  ConfigRegistry    parses configuration dataclasses (AST)
equation_library.py EquationLibrary   curated KaTeX equation set
model_library.py    ModelLibrary      architecture families and selection guidance
pipeline_library.py PipelineLibrary   the six pipelines and their stages
process_manager.py  ProcessManager    launches scripts, fan-out SSE streams
project_paths.py    ProjectPaths      paths and interpreter discovery
web_logger.py       WebLogger         console logging
```

Frontend, `static/`:

```
index.html
css/styles.css
js/{radar,equations,pipelines,models,scripts,configs,console,app}.js
```
