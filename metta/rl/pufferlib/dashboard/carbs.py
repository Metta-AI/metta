from rich.table import Table

from metta.rl.carbs.carbs_controller import CarbsController

from .dashboard import ROUND_OPEN, DashboardComponent, abbreviate, c1, c2
from .training import Training

class Carbs(DashboardComponent):
    def __init__(self, carbs_controller: CarbsController):
        super().__init__()
        self.carbs = carbs_controller

    def render(self):
        c = Table(box=ROUND_OPEN, pad_edge=False)
        c.add_column(f"{c1}Carbs", justify="left", vertical="top")
        c.add_column(f"{c1}{self.carbs._stage}", justify="right", vertical="top")
        c.add_row(f"{c2}Suggestions", abbreviate(self.carbs._num_suggestions))
        c.add_row(f"{c2}Observations", abbreviate(self.carbs._num_observations))
        c.add_row(f"{c2}Failures", abbreviate(self.carbs._num_failures))
        if self.carbs._last_rollout_result:
            r = self.carbs._last_rollout_result
            c.add_row(f"{c1}Last Observation")
            c.add_row(f"{c2}Score", abbreviate(r["score"]))
            c.add_row(f"{c2}Rollout Time", abbreviate(r["rollout_time"]))
            c.add_row(f"{c2}Train Time", abbreviate(r["train_time"]))
            c.add_row(f"{c2}Eval Time", abbreviate(r["eval_time"]))

        table = Table(box=ROUND_OPEN, expand=True, pad_edge=False)
        components = []
        if self.carbs._trainer:
            components.append(Training(self.carbs._trainer).render())
        components.append(c)

        table.add_row(*components)

        return table
