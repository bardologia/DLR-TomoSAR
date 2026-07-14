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
| Scripts | `/api/scripts` | The nine `main/*.py` entry points as cards; each opens its launch page. |
| Launch | `/api/scripts/<key>/config` | Full-screen launch control per script: config sections grouped by dataclass, typed controls, override manifest, command preview, interpreter, launch. |
| Configuration | `/api/configs` | Every configuration dataclass, field, type, and default, parsed live from `configuration/*.py`. |
| Console | `/api/jobs/<id>/stream` | Real-time stdout of launched jobs over SSE, with stop control. |
| Leaderboard | `/api/leaderboard` | Every saved inference `metrics.json` under the runs directory as one sortable, filterable table; run-name axes parsed into filter dropdowns; selecting two rows opens a metric-by-metric diff plus the resolved-config differences. |

## How launching works

Each `main/*.py` script builds its configuration as `ConfigCli(<EntryConfig>()).apply()`,
with all defaults living in `configuration/` dataclasses. The console mirrors that
contract:

- **Configuration** resolves the script's entry dataclass in a subprocess (using the
  preferred project interpreter) and lists every leaf field with its default, exactly
  as `--help-config` would. Results are cached and invalidated when `configuration/`,
  the script, or `tools/config_cli.py` change.
- **Edited fields** never touch any file. They are appended to the launch command as
  `--<dotted.path> <value>` overrides, which `ConfigCli` coerces to the field's type.
  The command preview always shows exactly what will run.
- **Launch run** executes the script from the repository root with the selected
  interpreter and streams its output. Exit codes are reported honestly; nothing is
  faked.

Scripts expect the F-SAR dataset paths and GPUs referenced in the configuration
defaults. On a machine without those, a launched job will stream its real traceback to
the console.

## Architecture

Backend, one class per file (no comments, per the vault coding rules):

```
serve.py                   ServeEntry            CLI entry, builds and runs the server
web_ui_server.py           WebUIServer           orchestrator; threaded HTTP server
request_router.py          RequestRouter         path dispatch, JSON, static, SSE
script_catalog.py          ScriptCatalog         lists scripts with their entry config class
script_config_resolver.py  ScriptConfigResolver  detects ConfigCli entry configs (AST), resolves defaults in a subprocess, caches by mtime
config_registry.py         ConfigRegistry        parses configuration dataclasses (AST)
equation_library.py        EquationLibrary       curated KaTeX equation set
model_library.py           ModelLibrary          architecture families and selection guidance
pipeline_library.py        PipelineLibrary       the six pipelines and their stages
process_manager.py         ProcessManager        launches scripts with overrides, fan-out SSE streams
project_paths.py           ProjectPaths          paths and interpreter discovery
web_logger.py              WebLogger             console logging
```

Frontend, `static/`:

```
index.html
css/styles.css
js/{radar,equations,pipelines,models,scripts,launch,configs,console,app}.js
```
