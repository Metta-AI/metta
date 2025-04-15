from rich.table import Table

from .dashboard import ROUND_OPEN, DashboardComponent, b2, c1, c2

class UserStats(DashboardComponent):
    def __init__(self, stats: dict, max_stats=5):
        super().__init__()
        self.max_stats = max_stats
        self.stats = stats

    def render(self):
        table = Table(box=ROUND_OPEN, expand=True, pad_edge=False)
        table.add_column(f"{c1}Env Stats", justify="left", width=20)
        table.add_column(f"{c1}Value", justify="right", width=10)
        i = 0
        for metric, value in self.stats.items():
            if i >= self.max_stats:
                break
            try:  # Discard non-numeric values
                int(value)
            except Exception as _:
                continue

            table.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")
            i += 1
        return table
