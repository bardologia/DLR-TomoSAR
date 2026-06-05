# Product

## Register

product

## Users

A single AI researcher (the project author) running SAR tomography experiments on a Linux workstation, usually in a dark room during long training sessions. The console is the cockpit for the DLR-TomoSAR codebase: inspect the signal model, pick an architecture, tune configuration, launch entry-point scripts, and watch live output. DLR colleagues and supervisors see it during demos and screen-shares; the home page is the first impression of the research.

## Product Purpose

A local control console for the DLR-TomoSAR deep-learning pipeline. It parses equations, pipelines, architectures, configuration dataclasses, and entry-point scripts live from the codebase and lets the researcher launch and monitor runs. Success means any value, script, or result is reachable in seconds without opening an editor.

## Brand Personality

Instrument-grade, calm, precise. A scientific radar console: dark surfaces, one teal accent, monospace data. Dense where data demands it, never decorative.

The home page is the one exception in register: a cinematic observatory. Deep blacks, slow sweeping motion, the data as spectacle, layered as cinematic hero, then live telemetry, then navigation depth. Every visual on it is data-backed (the actual Gaussian-mixture signal model), never decorative noise; a colleague watching for five seconds should see something move that is real.

## Anti-references

- Generic SaaS dashboards: hero metrics, icon-card grids, gradient buttons.
- Consumer-app playfulness: bouncy motion, emoji, rounded-everything.
- Walls of undifferentiated rows; data without hierarchy or scannability.

## Design Principles

- The tool disappears into the task: every screen serves one workflow (inspect, configure, launch, monitor).
- Parsed from source, never duplicated: the UI renders what the codebase says, live.
- Findability over decoration: search, grouping, and keyboard paths beat visual flourish.
- Accent means state or selection, never ornament.
- Honest output: exit codes, tracebacks, and defaults are shown as they are.

## Accessibility & Inclusion

Keyboard-reachable controls, visible focus states, `prefers-reduced-motion` alternatives for all animation, contrast at WCAG AA on dark surfaces.
