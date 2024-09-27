
import numpy as np
from rich.table import Table
from rl.carbs.carbs_controller import CarbsController
from .dashboard import DashboardComponent
from .dashboard import c1, b1, c2, b2, c3, ROUND_OPEN
from .training import Training
from .dashboard import abbreviate

class Carbs(DashboardComponent):
    def __init__(self, carbs_controller: CarbsController):
        super().__init__()
        self.carbs = carbs_controller

    def render(self):
        c = Table(box=ROUND_OPEN, pad_edge=False)
        c.add_column(f"{c1}Carbs", justify='left', vertical='top')
        c.add_column(f"{c1}{self.carbs._stage}", justify='right', vertical='top')
        c.add_row(f'{c2}Suggestions', abbreviate(self.carbs._num_suggestions))
        c.add_row(f'{c2}Observations', abbreviate(self.carbs._num_observations))
        c.add_row(f'{c2}Failures', abbreviate(self.carbs._num_failures))

        table = Table(box=ROUND_OPEN, expand=True, pad_edge=False)
        components = []
        if self.carbs._trainer:
            components.append(Training(self.carbs._trainer).render())
        components.append(c)

        table.add_row(*components)

        return table
