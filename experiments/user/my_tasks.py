# experiments/user/my_tasks.py
from softmax.training.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from softmax.training.sim.simulation_config import SimulationConfig
from softmax.training.tools.train import TrainTool
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
