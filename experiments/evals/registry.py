from metta.sim.simulation_config import SimulationConfig

from experiments.evals.ada import make_ada_eval_suite
from experiments.evals.navigation import make_navigation_eval_suite


def get_eval_suite(name: str) -> list[SimulationConfig]:
    if name == "navigation":
        return make_navigation_eval_suite()
    if name == "ada":
        return make_ada_eval_suite()
    else:
        raise ValueError(f"Unknown suite: {name}")
