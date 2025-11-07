# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot
# You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.alex.train`
# The VSCode "Run and Debug" section supports options to run these functions.
import typing

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
import experiments.recipes
import metta.agent.policies.vit_sliding_trans
import metta.agent.policy
import metta.cogworks.curriculum.curriculum
import metta.cogworks.curriculum.learning_progress_algorithm
import metta.rl.loss.losses
import metta.rl.loss.ppo
import metta.rl.trainer_config
import metta.rl.training
import metta.sim.simulation_config
import metta.tools.eval
import metta.tools.play
import metta.tools.replay
import metta.tools.train
import mettagrid


def make_mettagrid(num_agents: int = 24) -> mettagrid.MettaGridConfig:
    arena_env = eb.make_arena(num_agents=num_agents)

    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    return arena_env


def make_curriculum(
    arena_env: typing.Optional[mettagrid.MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: typing.Optional[metta.cogworks.curriculum.curriculum.CurriculumAlgorithmConfig] = None,
) -> metta.cogworks.curriculum.curriculum.CurriculumConfig:
    arena_env = arena_env or make_mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    if algorithm_config is None:
        algorithm_config = metta.cogworks.curriculum.learning_progress_algorithm.LearningProgressConfig(
            use_bidirectional=True,  # Enable bidirectional learning progress by default
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def make_evals(env: typing.Optional[mettagrid.MettaGridConfig] = None) -> typing.List[metta.sim.simulation_config.SimulationConfig]:
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        metta.sim.simulation_config.SimulationConfig(suite="arena", name="basic", env=basic_env),
        metta.sim.simulation_config.SimulationConfig(suite="arena", name="combat", env=combat_env),
    ]


def train(
    curriculum: typing.Optional[metta.cogworks.curriculum.curriculum.CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: typing.Optional[metta.agent.policy.PolicyArchitecture] = None,
) -> metta.tools.train.TrainTool:
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    eval_simulations = make_evals()
    trainer_cfg = metta.rl.trainer_config.TrainerConfig(
        losses=metta.rl.loss.losses.LossesConfig(ppo=metta.rl.loss.ppo.PPOConfig()),
    )
    # policy_config = FastDynamicsConfig()
    # policy_config = FastLSTMResetConfig()
    # policy_config = FastConfig()
    # policy_config = ViTSmallConfig()
    policy_config = metta.agent.policies.vit_sliding_trans.ViTSlidingTransConfig()
    training_env = metta.rl.training.TrainingEnvironmentConfig(curriculum=curriculum)
    evaluator = metta.rl.training.EvaluatorConfig(simulations=eval_simulations)

    return metta.tools.train.TrainTool(
        trainer=trainer_cfg,
        training_env=training_env,
        evaluator=evaluator,
        policy_architecture=policy_config,
    )


def play() -> metta.tools.play.PlayTool:
    env = experiments.recipes.arena.make_evals()[0].env
    env.game.max_steps = 100
    cfg = experiments.recipes.arena.play(env)
    return cfg


def replay() -> metta.tools.replay.ReplayTool:
    env = experiments.recipes.arena.make_mettagrid()
    env.game.max_steps = 100
    cfg = experiments.recipes.arena.replay(env)
    # cfg.policy_uri = "wandb://run/daveey.combat.lpsm.8x4"
    return cfg


def evaluate(run: str = "local.alex.1") -> metta.tools.eval.EvaluateTool:
    cfg = experiments.recipes.arena.evaluate(policy_uris=[f"wandb://run/{run}"])

    # If your run doesn't exist, try this:
    # cfg = arena.evaluate(policy_uri="wandb://run/daveey.combat.lpsm.8x4")
    return cfg
