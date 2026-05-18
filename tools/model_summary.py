from pathlib import Path

import jax
import numpy as np


class ModelSummary:
    def __init__(self, logger, params, docs_dir: str):
        """
        params : JAX pytree of model parameters (e.g. init_vars["params"]).
        """
        self.params       = params
        self.rows         = []
        self.total_params = 0
        self.logger       = logger
        self.docs_dir     = Path(docs_dir)

    def _walk(self, tree, prefix=""):
        """Recursively walk a pytree, yielding (path_str, leaf) pairs."""
        if isinstance(tree, dict):
            for k, v in tree.items():
                yield from self._walk(v, f"{prefix}/{k}" if prefix else k)
        else:
            yield prefix, tree

    def to_markdown(self, title="Model Summary") -> str:
        if not self.rows:
            return f"# {title}\n\nNo parameters found."

        rows_fmt = [(name, shape, f"{params:,}") for name, shape, params in self.rows]

        col1 = max(len("Parameter"), *(len(r[0]) for r in rows_fmt))
        col2 = max(len("Shape"),     *(len(r[1]) for r in rows_fmt))
        col3 = max(len("Count"),     *(len(r[2]) for r in rows_fmt))

        def line(a, b, c):
            return f"| {a:<{col1}} | {b:<{col2}} | {c:>{col3}} |"

        table = [line("Parameter", "Shape", "Count"),
                 f"| {'-'*col1} | {'-'*col2} | {'-'*col3} |"]
        for name, shape, count in rows_fmt:
            table.append(line(name, shape, count))

        md = [f"# {title}\n"]
        md.extend(table)
        md.append(f"\n**Total Parameters:** `{self.total_params:,}`")
        return "\n".join(md)

    def run(self):
        self.logger.section("[Model Summary]")
        self.logger.subsection("Generating model parameter summary")
        self.total_params = 0
        self.rows = []

        for path, leaf in self._walk(self.params):
            leaf_np   = np.asarray(leaf)
            n_params  = int(leaf_np.size)
            self.total_params += n_params
            self.rows.append((path, str(leaf_np.shape), n_params))

        self.logger.subsection(f"Total parameters : {self.total_params:,}")

    def save_markdown(self, filename="model_summary.md", title: str = "Model Summary"):
        md = self.to_markdown(title=title)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        path = self.docs_dir / filename
        path.write_text(md, encoding="utf-8")
        self.logger.subsection(f"Model summary saved to {path} \n")
