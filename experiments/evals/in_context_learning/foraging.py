from metta.sim.simulation_config import SimulationConfig

from experiments.recipes.in_context_learning.foraging import make_eval_suite


def make_foraging_eval_suite() -> list[SimulationConfig]:
    """Aggregate foraging evaluation suite.

    Includes:
      - A default foraging map
      - Directional foraging maps (small/medium)
      - Biased foraging maps across sizes
    """
    suite: list[SimulationConfig] = []
    suite.extend(make_eval_suite(suite_type="biased"))
    suite.extend(make_eval_suite(suite_type="directional"))
    return suite
