"""Level 5 - Expert: Maximum difficulty with curriculum learning.

This recipe represents the full complexity of the arena:
- Only heart reward (no intermediate rewards)
- Full combat enabled
- Maximum agents
- Curriculum learning with task variations
- Agents must discover entire resource chain independently
- Most challenging benchmark for architecture evaluation
"""

from typing import List, Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
    """Create expert difficulty arena with no reward shaping."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Only reward for heart - no intermediate rewards
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
    }

    # Combat enabled
    arena_env.game.actions.attack.consumed_resources["laser"] = 1

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    """Create curriculum with task variations.

    This enables the agent to learn progressively through different
    task configurations, discovering strategies through curriculum learning.
    """
    arena_env = arena_env or make_mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    # Vary combat difficulty
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # Vary initial resources in buildings (sometimes give hints)
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations."""
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="benchmark_arch", name="level_5_basic", env=basic_env),
        SimulationConfig(suite="benchmark_arch", name="level_5_combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """Train on Level 5 - Expert difficulty."""
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=make_evals()),
        policy_architecture=policy_architecture,
    )


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    """Interactive play tool."""
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="level_5_expert")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(
            suite="benchmark_arch", env=eval_env, name="level_5_expert"
        )
    )


def evaluate(
    policy_uri: str | None = None,
    simulations: Optional[Sequence[SimulationConfig]] = None,
) -> SimTool:
    """Evaluate a policy on Level 5 - Expert."""
    simulations = simulations or make_evals()
    policy_uris = [policy_uri] if policy_uri is not None else None

    return SimTool(
        simulations=simulations,
        policy_uris=policy_uris,
    )
