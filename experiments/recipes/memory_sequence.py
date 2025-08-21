import metta.mettagrid.config.envs as eb
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.sim import SimTool

from experiments.evals import memory_sequence as mem_evals


def make_env(num_agents: int = 1) -> EnvConfig:
    return eb.make_memory_sequence(num_agents=num_agents)


def make_memory_eval_suite() -> list[SimulationConfig]:
    return mem_evals.make_memory_sequence_eval_suite()


def evaluate() -> SimTool:
    return SimTool(simulations=make_memory_eval_suite())
