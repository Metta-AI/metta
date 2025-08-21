from typing import List, Optional, Sequence

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

# TODO(dehydration): make sure this trains as well as main on arena
# it's possible the maps are now different


def make_env(num_agents: int = 24) -> EnvConfig:
    arena_env = eb.make_arena(num_agents=num_agents)
    return arena_env


def make_curriculum(
    arena_env: Optional[EnvConfig] = None, use_learning_progress: bool = True
) -> CurriculumConfig:
    arena_env = arena_env or make_env()

    # make a set of training tasks for the arena
    arena_tasks = cc.bucketed(arena_env)

    # arena_tasks.add_bucket("game.map_builder.root.params.agents", [1, 2, 3, 4, 6])
    # arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30, 40])
    # arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30, 40])
    # arena_tasks.add_bucket("game.map_builder.instance_border_width", [0, 6])

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # sometimes add initial_items to the buildings
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    # Use the updated to_curriculum method that defaults to learning progress
    return arena_tasks.to_curriculum(
        num_tasks=16, use_learning_progress=use_learning_progress
    )


def make_evals(env: Optional[EnvConfig] = None) -> List[SimulationConfig]:
    basic_env = env or make_env()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(name="arena/basic", env=basic_env),
        SimulationConfig(name="arena/combat", env=basic_env),
    ]


def train(
    run: str,
    curriculum: Optional[CurriculumConfig] = None,
    use_learning_progress: bool = True,
) -> TrainTool:
    trainer_cfg = TrainerConfig(
        curriculum=curriculum
        or make_curriculum(use_learning_progress=use_learning_progress),
        evaluation=EvaluationConfig(
            simulations=[
                SimulationConfig(
                    name="arena/basic", env=eb.make_arena(num_agents=24, combat=False)
                ),
                SimulationConfig(
                    name="arena/combat", env=eb.make_arena(num_agents=24, combat=True)
                ),
            ],
        ),
    )

    return TrainTool(trainer=trainer_cfg)


def train_shaped(
    run: str,
    rewards: bool = True,
    converters: bool = True,
) -> TrainTool:
    env_cfg = make_env()
    env_cfg.game.agent.rewards.inventory.heart = 1
    env_cfg.game.agent.rewards.inventory.heart_max = 100

    if rewards:
        env_cfg.game.agent.rewards.inventory.ore_red = 0.1
        env_cfg.game.agent.rewards.inventory.ore_red_max = 10
        env_cfg.game.agent.rewards.inventory.battery_red = 0.1
        env_cfg.game.agent.rewards.inventory.battery_red_max = 10
        env_cfg.game.agent.rewards.inventory.laser = 0.1
        env_cfg.game.agent.rewards.inventory.laser_max = 10
        env_cfg.game.agent.rewards.inventory.armor = 0.1
        env_cfg.game.agent.rewards.inventory.armor_max = 10
        env_cfg.game.agent.rewards.inventory.blueprint = 0.1
        env_cfg.game.agent.rewards.inventory.blueprint_max = 1

    if converters:
        # Set altar input resources for battery_red conversion
        altar_obj = env_cfg.game.objects["altar"]
        if hasattr(altar_obj, "input_resources"):
            altar_obj.input_resources["battery_red"] = 1

    curriculum = cc.env_curriculum(env_cfg)

    trainer_cfg = TrainerConfig(
        curriculum=curriculum,
        evaluation=EvaluationConfig(
            simulations=make_evals(),
        ),
    )

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    eval_env = env or make_env()
    return PlayTool(sim=SimulationConfig(env=eval_env, name="arena"))


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    eval_env = env or make_env()
    return ReplayTool(sim=SimulationConfig(env=eval_env, name="arena"))


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    simulations = simulations or make_evals()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
