from experiments.evals.in_context_learning.assemblers.assembly_lines import (
    make_assembly_line_eval_suite,
)
from experiments.evals.in_context_learning.assemblers.foraging import (
    make_foraging_eval_suite,
)

from metta.sim.simulation_config import SimulationConfig


def make_assembler_eval_suite() -> list[SimulationConfig]:
    return make_assembly_line_eval_suite() + make_foraging_eval_suite()
