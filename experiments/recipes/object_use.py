import os
from datetime import datetime
from typing import Optional

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import Span
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.builder import building, empty_converters
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    GameConfig,
    MettaGridConfig,
)
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen

from experiments.evals.object_use import make_object_use_eval_suite


def _get_user_identifier() -> str:
    """Get user identifier from USER environment variable."""
    return os.getenv("USER", "unknown")


def _default_run_name() -> str:
    """Generate a robust run name following the pattern: object_use.{user}.{date}.{unique_id}

    Format: object_use.{username}.MMDD-HHMMSS.{git_hash_short} or object_use.{username}.MMDD-HHMMSS
    Example: object_use.alice.0820-143052.a1b2c3d or object_use.alice.0820-143052"""
    user = _get_user_identifier()
    now = datetime.now()
    timestamp = now.strftime("%m%d-%H%M%S")

    # Try to get git hash (7 chars like CI) for better tracking
    try:
        from gitta import get_current_commit

        git_hash = get_current_commit()[:7]
        return f"object_use.{user}.{timestamp}.{git_hash}"
    except Exception:
        # Fallback: use timestamp
        return f"object_use.{user}.{timestamp}"


def mettagrid(num_agents: int = 1, num_instances: int = 4) -> MettaGridConfig:
    """Create a base object use environment for training."""

    # Configure objects with their resource chains
    mine = building.mine_red.model_copy()
    mine.cooldown = 50

    generator = building.generator_red.model_copy()
    generator.cooldown = 25

    altar = building.altar.model_copy()
    altar.cooldown = 10

    lasery = building.lasery.model_copy()
    lasery.cooldown = 10

    armory = building.armory.model_copy()
    armory.cooldown = 10

    # Additional prototype objects configured from empty converters
    temple = empty_converters.temple.model_copy()
    temple.output_resources = {"heart": 1}
    temple.cooldown = 30

    lab = empty_converters.lab.model_copy()
    lab.input_resources = {"ore_red": 2, "battery_red": 1}
    lab.output_resources = {"blueprint": 1}
    lab.cooldown = 20

    factory = empty_converters.factory.model_copy()
    factory.input_resources = {"blueprint": 1, "ore_red": 1}
    factory.output_resources = {"armor": 2}
    factory.cooldown = 25

    objects = {
        "wall": building.wall,
        "block": building.block,
        "altar": altar,
        "mine_red": mine,
        "generator_red": generator,
        "lasery": lasery,
        "armory": armory,
        # Include additional objects so buckets can toggle their counts from 0..N
        "temple": temple,
        "lab": lab,
        "factory": factory,
    }

    env = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents * num_instances,
            max_steps=1000,
            objects=objects,
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(),
                get_items=ActionConfig(),
                put_items=ActionConfig(),
                swap=ActionConfig(),
            ),
            agent=AgentConfig(
                default_resource_limit=50,
                resource_limits={
                    "heart": 255,
                },
                rewards=AgentRewards(
                    inventory={
                        "heart": 1,
                    },
                ),
            ),
            map_builder=MapGen.Config(
                instances=num_instances,
                border_width=6,
                instance_border_width=3,
                instance=RandomMapBuilder.Config(
                    agents=num_agents,
                    width=25,
                    height=25,
                    objects={
                        "altar": 2,
                        "mine_red": 3,
                        "generator_red": 2,
                        "armory": 1,
                        "lasery": 1,
                        "wall": 5,
                        # Start disabled; curriculum buckets can raise to 0..2
                        "temple": 0,
                        "lab": 0,
                        "factory": 0,
                    },
                ),
            ),
        ),
    )
    return env


"""Default mettagrid() defined above (parameterized version)."""


def make_curriculum(
    object_use_env: Optional[MettaGridConfig] = None,
) -> CurriculumConfig:
    """Create curriculum for object use training."""
    object_use_env = object_use_env or mettagrid()

    # Create training tasks with varying difficulties
    tasks = cc.bucketed(object_use_env)

    # Vary map sizes
    tasks.add_bucket("game.map_builder.instance.width", [Span(15, 50)])
    tasks.add_bucket("game.map_builder.instance.height", [Span(15, 50)])

    # Vary object counts
    tasks.add_bucket("game.map_builder.instance.objects.altar", [Span(1, 3)])
    tasks.add_bucket("game.map_builder.instance.objects.mine_red", [Span(1, 5)])
    tasks.add_bucket("game.map_builder.instance.objects.generator_red", [Span(1, 3)])
    tasks.add_bucket("game.map_builder.instance.objects.armory", [Span(0, 2)])
    tasks.add_bucket("game.map_builder.instance.objects.lasery", [Span(0, 2)])
    tasks.add_bucket("game.map_builder.instance.objects.wall", [Span(0, 10)])
    tasks.add_bucket("game.map_builder.instance.objects.block", [Span(0, 5)])

    # Vary object cooldowns to change difficulty
    tasks.add_bucket("game.objects.altar.cooldown", [Span(10, 60)])
    tasks.add_bucket("game.objects.generator_red.cooldown", [Span(10, 60)])
    tasks.add_bucket("game.objects.mine_red.cooldown", [Span(10, 60)])

    # Vary initial resource counts
    tasks.add_bucket("game.objects.altar.initial_resource_count", [0, 1])
    tasks.add_bucket("game.objects.generator_red.initial_resource_count", [0, 1])

    # Toggle additional objects directly on the base env; single unified generator
    tasks.add_bucket("game.map_builder.instance.objects.temple", [Span(0, 2)])
    tasks.add_bucket("game.map_builder.instance.objects.lab", [Span(0, 2)])
    tasks.add_bucket("game.map_builder.instance.objects.factory", [Span(0, 2)])

    return CurriculumConfig(task_generator=tasks)


def train(
    run: Optional[str] = None, curriculum: Optional[CurriculumConfig] = None
) -> TrainTool:
    """Create a training tool for object use."""
    # Generate structured run name if not provided
    if run is None:
        run = _default_run_name()

    resolved_curriculum = curriculum or make_curriculum()

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=make_object_use_eval_suite(),
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
        run=run,
    )


def simulations() -> list[SimulationConfig]:
    return list(make_object_use_eval_suite())


def evaluate(policy_uris: Optional[list[str]] = None) -> EvaluateTool:
    """Evaluate policies on object use tasks."""
    return EvaluateTool(simulations=simulations(), policy_uris=policy_uris or [])


def play(policy_uri: Optional[str] = None) -> PlayTool:
    """Interactive play with a policy."""
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    """Generate replay from a policy."""
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)
