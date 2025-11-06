import metta.sim.simulation_config
import metta.tools.eval
import metta.tools.train
import mettagrid
import mettagrid.builder.envs


def mettagrid() -> mettagrid.MettaGridConfig:
    # Lightweight config generation; no environment execution
    return mettagrid.builder.envs.make_arena(num_agents=2)


def simulations() -> list[metta.sim.simulation_config.SimulationConfig]:
    env = mettagrid()
    return [metta.sim.simulation_config.SimulationConfig(suite="demo", name="basic", env=env)]


def evaluate() -> metta.tools.eval.EvaluateTool:
    """Explicit evaluate tool for testing."""
    return metta.tools.eval.EvaluateTool(simulations=simulations())


def train_shaped() -> metta.tools.train.TrainTool:
    """Explicit train tool for testing."""
    ...
