import metta.sim.simulation_config

import experiments.evals.navigation


def get_eval_suite(name: str) -> list[metta.sim.simulation_config.SimulationConfig]:
    if name == "navigation":
        return experiments.evals.navigation.make_navigation_eval_suite()
    else:
        raise ValueError(f"Unknown suite: {name}")
