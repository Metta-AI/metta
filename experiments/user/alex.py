# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot
# You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.alex.train`
# The VSCode "Run and Debug" section supports options to run these functions.
from typing import List, Optional

import softmax.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from experiments.recipes import arena
from metta.agent.policies.vit_sliding_trans import ViTSlidingTransConfig
from metta.agent.policy import PolicyArchitecture
from softmax.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from softmax.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from softmax.training.rl.loss.loss_config import LossConfig
from softmax.training.rl.loss.ppo import PPOConfig
from softmax.training.rl.trainer_config import TrainerConfig
from softmax.training.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from softmax.training.sim.simulation_config import SimulationConfig
from softmax.training.tools.play import PlayTool
from softmax.training.tools.replay import ReplayTool
from softmax.training.tools.sim import SimTool
from softmax.training.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.config import ConverterConfig


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
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

    # Easy converter: 1 battery_red to 1 heart (instead of 3 to 1)
    altar = arena_env.game.objects.get("altar")
    if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
        altar.input_resources["battery_red"] = 1

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or make_mettagrid()

    arena_tasks = cc.bucketed(arena_env)

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

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Enable bidirectional learning progress by default
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="arena", name="basic", env=basic_env),
        SimulationConfig(suite="arena", name="combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    eval_simulations = make_evals()
    trainer_cfg = TrainerConfig(
        losses=LossConfig(loss_configs={"ppo": PPOConfig()}),
    )
    # policy_config = FastDynamicsConfig()
    # policy_config = FastLSTMResetConfig()
    # policy_config = FastConfig()
    # policy_config = ViTSmallConfig()
    policy_config = ViTSlidingTransConfig()
    training_env = TrainingEnvironmentConfig(curriculum=curriculum)
    evaluator = EvaluatorConfig(simulations=eval_simulations)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=training_env,
        evaluator=evaluator,
        policy_architecture=policy_config,
    )


def play() -> PlayTool:
    env = arena.make_evals()[0].env
    env.game.max_steps = 100
    cfg = arena.play(env)
    return cfg


def replay() -> ReplayTool:
    env = arena.make_mettagrid()
    env.game.max_steps = 100
    cfg = arena.replay(env)
    # cfg.policy_uri = "wandb://run/daveey.combat.lpsm.8x4"
    return cfg


def evaluate(run: str = "local.alex.1") -> SimTool:
    cfg = arena.evaluate(policy_uri=f"wandb://run/{run}")

    # If your run doesn't exist, try this:
    # cfg = arena.evaluate(policy_uri="wandb://run/daveey.combat.lpsm.8x4")
    return cfg
