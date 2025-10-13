"""Sparse rewards × Hard task complexity.

2-Axis Grid Position:
- Reward Shaping: Sparse (very sparse intermediate rewards 0.01-0.05)
- Task Complexity: Hard (25×25 map, 24 agents, full combat)

This configuration tests:
- Very sparse intermediate rewards
- Only heart reward is substantial
- Large map with many agents for increased competition
- Combat enabled
- Agents must discover effective strategies with minimal guidance

Use case: Test exploration and credit assignment at high complexity
"""

from typing import List, Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig

from experiments.recipes.benchmark_architectures.benchmark import ARCHITECTURES


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
    """Create a hard complexity arena with sparse rewards."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Large map for complex multi-agent interactions
    arena_env.game.map_builder.width = 25
    arena_env.game.map_builder.height = 25

    # Very sparse intermediate rewards - mostly just heart
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.01,  # Minimal reward
        "battery_red": 0.05,  # Minimal reward
        "laser": 0.05,
        "armor": 0.05,
        "blueprint": 0.01,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # Combat enabled
    arena_env.game.actions.attack.consumed_resources["laser"] = 1

    return arena_env


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations with both basic and combat modes."""
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(
            suite="benchmark_arch", name="sparse_hard_basic", env=basic_env
        ),
        SimulationConfig(
            suite="benchmark_arch", name="sparse_hard_combat", env=combat_env
        ),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    arch_type: str = "fast",
) -> TrainTool:
    """Train on sparse rewards × hard complexity."""
    if curriculum is None:
        env = make_mettagrid()
        curriculum = cc.env_curriculum(env)

    architecture_config = ARCHITECTURES[arch_type]

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=make_evals()),
        policy_architecture=architecture_config,
    )


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    """Interactive play tool."""
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="sparse_hard")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="sparse_hard")
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    """Evaluate a policy on sparse rewards × hard complexity."""
    return EvaluateTool(
        simulations=make_evals(),
        policy_uris=policy_uris,
    )
