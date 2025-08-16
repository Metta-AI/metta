from experiments.evals.suites.navigation import make_navigation_eval_suite
from metta.sim.simulation_config import SimulationConfig


def get_eval_suite(name: str) -> list[SimulationConfig]:
    if name == "navigation":
        return make_navigation_eval_suite()
    else:
        raise ValueError(f"Unknown suite: {name}")
