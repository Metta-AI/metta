from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.builder.envs import make_arena


def mettagrid() -> MettaGridConfig:
    # Lightweight config generation; no environment execution
    return make_arena(num_agents=2)


def simulations() -> list[SimulationConfig]:
    env = mettagrid()
    return [SimulationConfig(suite="demo", name="basic", env=env)]


def evaluate() -> EvaluateTool:
    """Explicit evaluate tool for testing."""
    return EvaluateTool(simulations=simulations())


def train_shaped() -> TrainTool:
    """Explicit train tool for testing."""
    ...
