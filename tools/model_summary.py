from pathlib import Path
import torch
import torch.nn as nn


class ModelSummary:
    def __init__(self, logger, model: nn.Module):
        self.model        = model
        self.rows         = []
        self.total_params = 0
        self.logger       = logger
     
    def count_params(self, module: nn.Module):
        return sum(p.numel() for p in module.parameters())

    def to_markdown(self, title="Model Summary") -> str:
        if not self.rows:
            return f"# {title}\n\nNo layers found."
        
        rows_fmt = [(name, typ, f"{params:,}") for name, typ, params in self.rows]
        
        col1 = max(len("Layer"), *(len(name) for name, _, _ in rows_fmt))
        col2 = max(len("Type"), *(len(typ) for _, typ, _ in rows_fmt))
        col3 = max(len("Parameters"), *(len(p) for _, _, p in rows_fmt))

        def line(a, b, c):
            return f"| {a:<{col1}} | {b:<{col2}} | {c:>{col3}} |"

        table = []
        table.append(line("Layer", "Type", "Parameters"))
        table.append(f"| {'-'*col1} | {'-'*col2} | {'-'*col3} |")
        for name, typ, params in rows_fmt:
            table.append(line(name, typ, params))

        total = f"{self.total_params:,}"

        md = []
        md.append(f"# {title}\n")
        md.extend(table)
        md.append(f"\n**Total Parameters:** `{total}`")
        return "\n".join(md)

    def run(self):
        self.logger.section("[Model Summary]")
        self.logger.info("Generating model architecture summary")
        self.total_params = 0

        for name, module in self.model.named_modules():
            if name == "":
                continue

            n_params = self.count_params(module)
            self.total_params += n_params
            
            self.rows.append((name, module.__class__.__name__, n_params))
    
    def save_markdown(self, path: str, title: str = "Model Summary"):
        md = self.to_markdown(title=title)
        Path(path).write_text(md, encoding="utf-8")
        self.logger.info(f"Model summary saved to {path}")