from typing import Any, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class LiveMonitor:
    def __init__(self, console: Console, title: str = "Training Monitor") -> None:
        self.console = console
        self.title = title
        self._metrics: dict[str, Any] = {}
        self._live: Optional[Live] = None

    def __enter__(self):
        self._live = Live(self._render(), console=self.console, refresh_per_second=4, transient=False)
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None
        return False

    def update(self, **kwargs: Any) -> None:
        self._metrics.update(kwargs)
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Panel:
        tbl = Table(show_header=True, header_style="bold cyan", box=None, expand=False)
        tbl.add_column("Metric", style="key", no_wrap=True)
        tbl.add_column("Value", style="value", justify="right")

        for k, v in sorted(self._metrics.items()):
            if isinstance(v, float):
                tbl.add_row(k, f"{v:.6f}" if abs(v) < 1000 else f"{v:.2f}")
            else:
                tbl.add_row(k, str(v))

        return Panel(tbl, title=f"[bold cyan]{self.title}[/bold cyan]", border_style="cyan")
