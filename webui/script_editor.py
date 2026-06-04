from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path

from project_paths import ProjectPaths


class ScriptEditor:

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def apply(self, key: str, values: dict) -> dict:
        path = self.paths.main_dir / f"{key}.py"
        if not path.exists():
            return {"ok": False, "error": "script not found"}

        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return {"ok": False, "error": f"parse error: {exc}"}

        boundary = self._boundary_line(tree)
        targets  = self._locate_targets(tree, boundary)

        invalid = self._validate(values)
        if invalid:
            return {"ok": False, "error": f"invalid expression for: {', '.join(invalid)}"}

        lines   = source.splitlines(keepends=True)
        changed = []

        for name, new_value in values.items():
            node = targets.get(name)
            if node is None:
                continue
            updated = self._rewrite_line(lines, node, new_value)
            if updated:
                changed.append(name)

        if not changed:
            return {"ok": False, "error": "no matching constants were updated"}

        backup = self._write_backup(key, source)
        path.write_text("".join(lines), encoding="utf-8")

        return {"ok": True, "changed": changed, "backup": str(backup.name)}

    def _validate(self, values: dict) -> list[str]:
        bad = []
        for name, expr in values.items():
            try:
                ast.parse(str(expr), mode="eval")
            except SyntaxError:
                bad.append(name)
        return bad

    def _boundary_line(self, tree: ast.Module) -> int | None:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node.lineno
        return None

    def _locate_targets(self, tree: ast.Module, boundary: int | None) -> dict:
        found = {}
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if boundary is not None and node.lineno >= boundary:
                continue
            name = self._assign_name(node)
            if name is not None and node.value is not None:
                found[name] = node.value
        return found

    def _assign_name(self, node: ast.Assign | ast.AnnAssign) -> str | None:
        if isinstance(node, ast.AnnAssign):
            return node.target.id if isinstance(node.target, ast.Name) else None
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None

    def _rewrite_line(self, lines: list[str], node: ast.expr, new_value: str) -> bool:
        if node.lineno != node.end_lineno:
            return False

        index   = node.lineno - 1
        line    = lines[index]
        newline = "\n" if line.endswith("\n") else ""
        body    = line[:-1] if newline else line

        start   = node.col_offset
        end     = node.end_col_offset
        rebuilt = body[:start] + str(new_value) + body[end:]

        lines[index] = rebuilt + newline
        return True

    def _write_backup(self, key: str, source: str) -> Path:
        self.paths.ensure_backups()
        stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = self.paths.backups_dir / f"{key}.{stamp}.bak"
        backup.write_text(source, encoding="utf-8")
        return backup
