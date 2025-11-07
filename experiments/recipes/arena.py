# ruff: noqa: E501
import typing

import metta.cogworks.curriculum as cc
import mettagrid as mettagrid_pkg
import mettagrid.builder.envs as eb
import metta.cogworks.curriculum.curriculum
import metta.cogworks.curriculum.learning_progress_algorithm
import metta.rl.training
import metta.sim.simulation_config
import metta.tools.eval
import metta.tools.play
import metta.tools.replay
import metta.tools.train
import mettagrid.builder.envs as eb

# TODO(dehydration): make sure this trains as well as main on arena
# it's possible the maps are now different


def mettagrid(num_agents: int = 24) -> mettagrid_pkg.MettaGridConfig:
    arena_env = eb.make_arena(num_agents=num_agents)
    return arena_env


def make_curriculum(
    arena_env: typing.Optional[mettagrid_pkg.MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumAlgorithmConfig
    ] = None,
) -> metta.cogworks.curriculum.curriculum.CurriculumConfig:
    arena_env = arena_env or mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    # arena_tasks.add_bucket("game.map_builder.instance.params.agents", [1, 2, 3, 4, 6])
    # arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30, 40])
    # arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30, 40])
    # arena_tasks.add_bucket("game.map_builder.instance_border_width", [0, 6])

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])
    arena_tasks.add_bucket("game.agent.initial_inventory.ore_red", [0, 1, 3])
    arena_tasks.add_bucket("game.agent.initial_inventory.battery_red", [0, 3])

    if algorithm_config is None:
        algorithm_config = metta.cogworks.curriculum.learning_progress_algorithm.LearningProgressConfig(
            use_bidirectional=True,  # Default: bidirectional learning progress
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(
    env: typing.Optional[mettagrid_pkg.MettaGridConfig] = None,
) -> list[metta.sim.simulation_config.SimulationConfig]:
    basic_env = env or mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        metta.sim.simulation_config.SimulationConfig(
            suite="arena", name="basic", env=basic_env
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="arena", name="combat", env=combat_env
        ),
    ]


def train(
    curriculum: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumConfig
    ] = None,
    enable_detailed_slice_logging: bool = False,
) -> metta.tools.train.TrainTool:
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    return metta.tools.train.TrainTool(
        training_env=metta.rl.training.TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=metta.rl.training.EvaluatorConfig(simulations=simulations()),
    )


def train_shaped(rewards: bool = True) -> metta.tools.train.TrainTool:
    env_cfg = mettagrid()
    env_cfg.game.agent.rewards.inventory["heart"] = 1
    env_cfg.game.agent.rewards.inventory_max["heart"] = 100

    if rewards:
        env_cfg.game.agent.rewards.inventory.update(
            {
                "ore_red": 0.1,
                "battery_red": 0.8,
                "laser": 0.5,
                "armor": 0.5,
                "blueprint": 0.5,
            }
        )
        env_cfg.game.agent.rewards.inventory_max.update(
            {
                "ore_red": 1,
                "battery_red": 1,
                "laser": 1,
                "armor": 1,
                "blueprint": 1,
            }
        )

    return metta.tools.train.TrainTool(
        training_env=metta.rl.training.TrainingEnvironmentConfig(
            curriculum=cc.env_curriculum(env_cfg)
        ),
        evaluator=metta.rl.training.EvaluatorConfig(simulations=simulations()),
    )


def evaluate(
    policy_uris: str | typing.Sequence[str] | None = None,
) -> metta.tools.eval.EvaluateTool:
    return metta.tools.eval.EvaluateTool(
        simulations=simulations(),
        policy_uris=policy_uris,
    )


def replay(policy_uri: typing.Optional[str] = None) -> metta.tools.replay.ReplayTool:
    return metta.tools.replay.ReplayTool(sim=simulations()[0], policy_uri=policy_uri)


def play(policy_uri: typing.Optional[str] = None) -> metta.tools.play.PlayTool:
    return metta.tools.play.PlayTool(sim=simulations()[0], policy_uri=policy_uri)
