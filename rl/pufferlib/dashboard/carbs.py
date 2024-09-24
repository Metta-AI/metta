
import numpy as np
from rich.table import Table

from rl.pufferlib.utilization import Utilization

from .component import DashboardComponent
from .component import c1, b1, c2, b2, c3

class CarbsComponent(DashboardComponent):
    def __init__(self, dashboard):
        super().__init__(dashboard)
        self.num_observations = 0
        self.num_suggestions = 0
        self.last_metric = 0
        self.last_run_time = 0
        self.last_run_success = False
        self.num_failures = 0

    def update(self):
        pass

    def render(self):
        table = Table(box=None, expand=True, pad_edge=False)
        table.add_row(
            f' {c1}Carbs: {b2}' +
            f'o: {self.carbs["num_observations"]} ' +
            f's: {self.carbs["num_suggestions"]} ' +
            f'm: {self.carbs["last_metric"]} ' +
            f't: {self.carbs["last_run_time"]} ' +
            f't: {self.carbs["last_run_success"]} ' +
            f'f: {self.carbs["num_failures"]}')
        return table
