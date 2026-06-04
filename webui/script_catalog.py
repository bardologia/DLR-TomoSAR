from __future__ import annotations

import ast
from pathlib import Path

from project_paths import ProjectPaths


class ScriptCatalog:

    META = {
        "pre_process": {
            "title"   : "Pre-process",
            "category": "Data",
            "purpose" : "Ingest raw F-SAR products, beamform the tomogram, and form interferograms.",
        },
        "extract_params": {
            "title"   : "Extract Parameters",
            "category": "Data",
            "purpose" : "Fit per-pixel Gaussian mixtures to build the supervised parameter targets.",
        },
        "single_train": {
            "title"   : "Single Train",
            "category": "Training",
            "purpose" : "Train one model configuration end to end with EMA, warmup, and scheduling.",
        },
        "batch_train": {
            "title"   : "Batch Train",
            "category": "Training",
            "purpose" : "Train several model configurations in sequence for comparison.",
        },
        "overfit_test": {
            "title"   : "Overfit Test",
            "category": "Training",
            "purpose" : "Overfit a single batch to verify model capacity and wiring.",
        },
        "single_infer": {
            "title"   : "Single Inference",
            "category": "Inference",
            "purpose" : "Run sliding-window prediction, stitch cubes, and generate the report.",
        },
        "batch_inference": {
            "title"   : "Batch Inference",
            "category": "Inference",
            "purpose" : "Evaluate inference across multiple trained runs.",
        },
        "benchmark": {
            "title"   : "Benchmark",
            "category": "Analysis",
            "purpose" : "Benchmark inference speed and capacity-matched architecture trade-offs.",
        },
        "tune": {
            "title"   : "Tune",
            "category": "Tuning",
            "purpose" : "Run the Optuna two-phase hyperparameter search.",
        },
    }

    ORDER = [
        "pre_process",
        "extract_params",
        "single_train",
        "batch_train",
        "overfit_test",
        "single_infer",
        "batch_inference",
        "benchmark",
        "tune",
    ]

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def list_scripts(self) -> list[dict]:
        entries = []
        for key in self.ORDER:
            path = self.paths.main_dir / f"{key}.py"
            if not path.exists():
                continue
            meta      = self.META.get(key, {"title": key, "category": "Other", "purpose": ""})
            constants = self._parse_constants(path)
            entries.append({
                "key"         : key,
                "file"        : f"main/{key}.py",
                "title"       : meta["title"],
                "category"    : meta["category"],
                "purpose"     : meta["purpose"],
                "n_constants" : len(constants),
            })
        return entries

    def get_script(self, key: str) -> dict | None:
        path = self.paths.main_dir / f"{key}.py"
        if not path.exists():
            return None

        meta      = self.META.get(key, {"title": key, "category": "Other", "purpose": ""})
        source    = path.read_text(encoding="utf-8")
        constants = self._parse_constants(path)

        return {
            "key"       : key,
            "file"      : f"main/{key}.py",
            "title"     : meta["title"],
            "category"  : meta["category"],
            "purpose"   : meta["purpose"],
            "source"    : source,
            "language"  : "python",
            "constants" : constants,
            "command"   : f"python main/{key}.py",
        }

    def _parse_constants(self, path: Path) -> list[dict]:
        try:
            source = path.read_text(encoding="utf-8")
            tree   = ast.parse(source)
        except (OSError, SyntaxError):
            return []

        boundary = self._boundary_line(tree)
        consts   = []

        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if boundary is not None and node.lineno >= boundary:
                continue

            name = self._assign_name(node)
            if name is None or name.isupper() or name.startswith("_"):
                continue
            if node.value is None:
                continue

            value = self._safe_unparse(node.value)
            consts.append({
                "name"  : name,
                "value" : value,
                "type"  : self._infer_kind(node.value),
                "line"  : node.lineno,
            })
        return consts

    def _boundary_line(self, tree: ast.Module) -> int | None:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node.lineno
        return None

    def _assign_name(self, node: ast.Assign | ast.AnnAssign) -> str | None:
        if isinstance(node, ast.AnnAssign):
            return node.target.id if isinstance(node.target, ast.Name) else None
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None

    def _infer_kind(self, value: ast.expr) -> str:
        if isinstance(value, ast.Constant):
            return type(value.value).__name__
        if isinstance(value, (ast.List, ast.Tuple)):
            return "list"
        if isinstance(value, ast.Call):
            func = value.func
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                return func.attr
        if isinstance(value, ast.UnaryOp) and isinstance(value.operand, ast.Constant):
            return type(value.operand.value).__name__
        return "expr"

    def _safe_unparse(self, node: ast.expr) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "?"
