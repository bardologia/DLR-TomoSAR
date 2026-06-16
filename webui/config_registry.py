from __future__ import annotations

import ast
from pathlib import Path

from project_paths import ProjectPaths


class ConfigRegistry:

    SECTION_TITLES = {
        "sar"              : "SAR Processing",
        "param"            : "Parameter Extraction",
        "normalization"    : "Normalization",
        "dataset"          : "Dataset",
        "architectures"    : "Model Architectures",
        "training"         : "Training",
        "inference"        : "Inference",
        "benchmark"        : "Benchmark",
        "cross_validation" : "Cross-validation",
        "tuning"           : "Tuning",
    }

    SECTION_ORDER = ["sar", "param", "normalization", "dataset", "architectures", "training", "inference", "benchmark", "cross_validation", "tuning"]

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

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
        fields   = []
        group    = 0
        prev_end = None
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            if not isinstance(item.target, ast.Name):
                continue

            if prev_end is not None and item.lineno - prev_end > 1:
                group += 1
            prev_end = item.end_lineno

            name       = item.target.id
            annotation = self._safe_unparse(item.annotation)
            default    = self._format_default(item.value)

            fields.append({
                "name"    : name,
                "type"    : annotation,
                "default" : default,
                "group"   : group,
            })
        return fields

    def _format_default(self, value: ast.expr | None) -> str:
        if value is None:
            return "required"
        if isinstance(value, ast.Call):
            func     = value.func
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

    def _sections(self) -> list[str]:
        root  = self.paths.config_dir
        found = [p.name for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(("_", "."))]
        known = [s for s in self.SECTION_ORDER if s in found]
        extra = [s for s in found if s not in self.SECTION_ORDER]
        return known + extra

    def _section_files(self, section: str) -> list[Path]:
        section_dir = self.paths.config_dir / section
        return [p for p in sorted(section_dir.rglob("*.py")) if p.name != "__init__.py"]

    def _rel_module(self, path: Path) -> str:
        return path.relative_to(self.paths.config_dir).with_suffix("").as_posix()

    def collect(self) -> list[dict]:
        groups = []
        for section in self._sections():
            classes = []
            for path in self._section_files(section):
                for cls in self._parse_module(path):
                    cls["module"] = self._rel_module(path)
                    classes.append(cls)
            if not classes:
                continue
            groups.append({
                "module" : section,
                "title"  : self.SECTION_TITLES.get(section, section.replace("_", " ").title()),
                "classes": classes,
            })
        return groups
