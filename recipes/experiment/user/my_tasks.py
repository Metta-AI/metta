# experiments/user/my_tasks.py
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_arena
from recipes.experiment.arena import BASELINE as ARENA_BASELINE


def my_train(baseline: TrainTool | None = None) -> TrainTool:
    if baseline is None:
        baseline = ARENA_BASELINE.model_copy(deep=True)
    else:
        baseline = baseline.model_copy(deep=True)

    baseline.evaluator.simulations = [SimulationConfig(suite="arena", name="arena/basic", env=make_arena(num_agents=6))]

    return baseline
