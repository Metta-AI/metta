# experiments/user/my_tasks.py
import metta.rl.training
import metta.sim.simulation_config
import metta.tools.train
import mettagrid.builder.envs


def my_train() -> metta.tools.train.TrainTool:
    return metta.tools.train.TrainTool(
        training_env=metta.rl.training.TrainingEnvironmentConfig(),
        evaluator=metta.rl.training.EvaluatorConfig(
            simulations=[
                metta.sim.simulation_config.SimulationConfig(
                    suite="arena",
                    name="arena/basic",
                    env=mettagrid.builder.envs.make_arena(num_agents=6),
                )
            ]
        ),
    )
