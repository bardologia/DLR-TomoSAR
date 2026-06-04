from __future__ import annotations

import ast
from pathlib import Path

from project_paths import ProjectPaths


class ConfigRegistry:

    MODULE_TITLES = {
        "processing_config"       : "Processing",
        "param_extraction_config" : "Parameter Extraction",
        "dataset_config"          : "Dataset",
        "norm_config"             : "Normalization",
        "training_config"         : "Training",
        "models_config"           : "Models",
        "inference_config"        : "Inference",
        "tuning_config"           : "Tuning",
    }

    MODULE_ORDER = [
        "processing_config",
        "param_extraction_config",
        "dataset_config",
        "norm_config",
        "training_config",
        "models_config",
        "inference_config",
        "tuning_config",
    ]

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def collect(self) -> list[dict]:
        groups = []
        for module in self.MODULE_ORDER:
            path = self.paths.config_dir / f"{module}.py"
            if not path.exists():
                continue
            classes = self._parse_module(path)
            if not classes:
                continue
            groups.append({
                "module" : module,
                "title"  : self.MODULE_TITLES.get(module, module),
                "classes": classes,
            })
        return groups

    def _parse_module(self, path: Path) -> list[dict]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return []

        classes = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not self._is_dataclass(node):
                continue
            fields = self._parse_fields(node)
            if not fields:
                continue
            classes.append({"name": node.name, "fields": fields})
        return classes

    def _is_dataclass(self, node: ast.ClassDef) -> bool:
        for deco in node.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            if isinstance(target, ast.Name) and target.id == "dataclass":
                return True
            if isinstance(target, ast.Attribute) and target.attr == "dataclass":
                return True
        return False

    def _parse_fields(self, node: ast.ClassDef) -> list[dict]:
        fields = []
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if not isinstance(item.target, ast.Name):
                continue

            name       = item.target.id
            annotation = self._safe_unparse(item.annotation)
            default    = self._format_default(item.value)

            fields.append({
                "name"    : name,
                "type"    : annotation,
                "default" : default,
            })
        return fields

    def _format_default(self, value: ast.expr | None) -> str:
        if value is None:
            return "required"
        if isinstance(value, ast.Call):
            func = value.func
            is_field = (isinstance(func, ast.Name) and func.id == "field") or (isinstance(func, ast.Attribute) and func.attr == "field")
            if is_field:
                for kw in value.keywords:
                    if kw.arg == "default_factory":
                        inner = self._safe_unparse(kw.value)
                        return f"{inner}()"
                    if kw.arg == "default":
                        return self._safe_unparse(kw.value)
                return "factory"
        return self._safe_unparse(value)

    def _safe_unparse(self, node: ast.expr | None) -> str:
        if node is None:
            return ""
        try:
            return ast.unparse(node)
        except Exception:
            return "?"
