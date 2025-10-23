"""A Cogs vs Clips version of the arena recipe.

This is meant as a basic testbed for CvC buildings / mechanics, not as a full-fledged recipe.
"""

from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.train import TrainTool
from mettagrid.builder import building
from mettagrid.config import AssemblerConfig, MettaGridConfig


def mettagrid(num_agents: int = 24) -> MettaGridConfig:
    arena_env = eb.make_arena(num_agents=num_agents)

    arena_env.game.objects.update(
        {
            "altar": building.assembler_altar,
            "mine_red": building.assembler_mine_red,
            "generator_red": building.assembler_generator_red,
            "lasery": building.assembler_lasery,
            "armory": building.assembler_armory,
        }
    )

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or mettagrid()

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
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Default: bidirectional learning progress
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_slice_axes=5,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    basic_env = env or mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="cvc_arena", name="basic", env=basic_env),
        SimulationConfig(suite="cvc_arena", name="combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    resolved_curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=[
            SimulationConfig(
                suite="cvc_arena", name="basic", env=mettagrid(num_agents=24)
            ),
            SimulationConfig(
                suite="cvc_arena", name="combat", env=mettagrid(num_agents=24)
            ),
        ],
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def train_shaped(rewards: bool = True, assemblers: bool = True) -> TrainTool:
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

    if assemblers:
        # Update altar recipe to require battery_red input
        altar_config = env_cfg.game.objects["altar"]
        assert isinstance(altar_config, AssemblerConfig)
        altar_config.recipes[0][1].input_resources["battery_red"] = 1

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    curriculum = cc.env_curriculum(env_cfg)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=simulations(env_cfg)),
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    return EvaluateTool(
        simulations=simulations(),
        policy_uris=policy_uris,
    )
