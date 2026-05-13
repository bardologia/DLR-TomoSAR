from pathlib import Path


class ShapeLogger:
    def __init__(self, model, logger, include_types, docs_dir: str):
        self.model         = model
        self.include_types = include_types
        self.logger        = logger
        self.docs_dir      = Path(docs_dir)
        self.records       = []
        self.hooks         = []
    
    def _hook(self, name):
        def fn(module, inputs, output):
            x = inputs[0]
            in_shape  = tuple(x.shape) if hasattr(x, "shape") else str(type(x))
            if isinstance(output, tuple):
                if hasattr(output[0], "shape"):
                    out_shape = tuple(output[0].shape)
                else:
                    out_shape = f"tuple[{len(output)}]"
            else:
                out_shape = tuple(output.shape) if hasattr(output, "shape") else str(type(output))
            
            self.records.append((name, module.__class__.__name__, in_shape, out_shape))
        
        return fn

    def attach(self):
        self.logger.subsection("Hooks attached to layers for shape logging. \n")
        
        for name, module in self.model.named_modules():
            if name == "":
                continue
           
            if isinstance(module, self.include_types):
                self.hooks.append(module.register_forward_hook(self._hook(name)))
        
        return self

    def detach(self):
        if self.hooks == []:
            return

        self.logger.subsection("Hooks detached from layers. \n")

        for h in self.hooks:
            h.remove()
        
        self.hooks.clear()

    def clear(self):
        self.records.clear()
    
    def to_markdown(self, title: str = "Shape Log", sort_by_layer: bool = False) -> str:
        rows = list(self.records)
        if sort_by_layer:
            rows.sort(key=lambda r: r[0])

        def s(x):
            return str(x)

        def layer_cell(name: str) -> str:
            return f"`{name}`"

        col_names = ["Layer", "Type", "Input shape", "Output shape"]
        col_data = [
            [layer_cell(r[0]) for r in rows],
            [str(r[1]) for r in rows],
            [s(r[2]) for r in rows],
            [s(r[3]) for r in rows],
        ]

        widths = []
        for header, data in zip(col_names, col_data):
            widths.append(max([len(header)] + [len(v) for v in data]) if rows else len(header))

        def fmt_row(cells):
            return "| " + " | ".join(f"{c:<{w}}" for c, w in zip(cells, widths)) + " |"

        def fmt_sep():
            return "| " + " | ".join((":" + "-" * (w - 1)) if w > 1 else "-" for w in widths) + " |"

        lines = []
        lines.append(f"# {title}\n")
        lines.append(fmt_row(col_names))
        lines.append(fmt_sep())

        for (name, typ, ins, outs), layer_txt in zip(rows, col_data[0]):
            lines.append(fmt_row([layer_txt, str(typ), s(ins), s(outs)]))

        lines.append(f"\n**Records:** {len(rows)}")
        return "\n".join(lines)

    def save_markdown(self, filename="tensor_shape.md", title: str = "Shape Log", sort_by_layer: bool = False):
        md = self.to_markdown(title=title, sort_by_layer=sort_by_layer)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        path = self.docs_dir / filename
        path.write_text(md, encoding="utf-8")
        self.logger.subsection(f"Shape log saved to {path} \n")
