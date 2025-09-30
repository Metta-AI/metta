# experiments/user/my_tasks.py
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_arena


def my_train() -> TrainTool:
    return TrainTool(
        training_env=TrainingEnvironmentConfig(),
        evaluator=EvaluatorConfig(
            simulations=[
                SimulationConfig(
                    suite="arena", name="arena/basic", env=make_arena(num_agents=6)
                )
            ]
        ),
    )
