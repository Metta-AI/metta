import numpy as np
from rich.table import Table

import metta.rl.pufferlib.utilization as utilization

from .dashboard import DashboardComponent, b2, c1, c3

class Utilization(DashboardComponent):
    def __init__(self):
        super().__init__()
        self.utilization = utilization.Utilization(delay=10)

    def update(self):
        pass

    def render(self):
        table = Table(box=None, expand=True, pad_edge=False)
        cpu_percent = np.mean(self.utilization.cpu_util)
        dram_percent = np.mean(self.utilization.cpu_mem)
        gpu_percent = np.mean(self.utilization.gpu_util)
        vram_percent = np.mean(self.utilization.gpu_mem)
        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)
        table.add_row(
            f":blowfish: {c1}PufferLib {b2}1.0.0",
            f"{c1}CPU: {c3}{cpu_percent:.1f}%",
            f"{c1}GPU: {c3}{gpu_percent:.1f}%",
            f"{c1}DRAM: {c3}{dram_percent:.1f}%",
            f"{c1}VRAM: {c3}{vram_percent:.1f}%",
        )
        return table

    def stop(self):
        self.utilization.stop()
