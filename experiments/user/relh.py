import metta.mettagrid.config.envs as eb
from experiments import arena, navigation
from metta.mettagrid.map_builder.ascii import AsciiMapBuilder
from metta.tools.play import PlayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

########################################################
# Arena
########################################################

# Obstacles
obstacles_env = eb.make_navigation(num_agents=1)
obstacles_env.game.map_builder = AsciiMapBuilder.Config.from_uri(
    "configs/env/mettagrid/maps/navigation/obstacles0.map",
)
obstacles_env.game.max_steps = 100


def train(run: str = "relh.mb.1") -> TrainTool:
    """
    Training configuration that accepts run name as parameter.
    
    Usage:
        ./tools/run.py experiments.user.relh.train --args run=relh.mb.1
        ./tools/run.py experiments.user.relh.train --args run=relh.test.2
    """
    env = navigation.make_env()
    env.game.max_steps = 100
    cfg = navigation.train(
        run=run,  # Now uses the parameter instead of hardcoded value
        curriculum=navigation.make_curriculum(env),
    )
    # Set workers to match CPU count for optimal performance
    cfg.trainer.rollout_workers = 16
    return cfg


def play() -> PlayTool:
    env = arena.make_evals()[0].env
    env.game.max_steps = 100
    # env.game.agent.initial_inventory["battery_red"] = 10
    cfg = arena.play(env)
    return cfg


def evaluate(policy_uri: str = "file://./train_dir/relh.mb.1/checkpoints") -> SimTool:
    """
    Evaluation configuration that accepts policy URI as parameter.
    
    Usage:
        ./tools/run.py experiments.user.relh.evaluate --args policy_uri=file://./train_dir/relh.mb.1/checkpoints
        ./tools/run.py experiments.user.relh.evaluate --args policy_uri=wandb://run/relh.mb.1
    """
    cfg = arena.evaluate(policy_uri=policy_uri)
    return cfg
