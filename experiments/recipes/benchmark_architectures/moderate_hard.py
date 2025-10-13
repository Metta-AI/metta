"""Moderate rewards × Hard task complexity.

2-Axis Grid Position:
- Reward Shaping: Moderate (medium intermediate rewards 0.2-0.7)
- Task Complexity: Hard (25×25 map, 24 agents, combat enabled)

This configuration tests architecture performance with:
- Moderate reward guidance
- Maximum task complexity
- Realistic challenge combining both axes
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

from experiments.recipes.benchmark_architectures.adaptive import ARCHITECTURES


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
    """Create hard complexity arena with moderate reward shaping."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Large map size
    arena_env.game.map_builder.width = 25
    arena_env.game.map_builder.height = 25

    # Moderate rewards for intermediate items
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.2,  # Moderate reward for mining
        "battery_red": 0.7,  # Moderate reward for conversion
        "laser": 0.4,
        "armor": 0.4,
        "blueprint": 0.3,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 2,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # Standard converter ratios (3:1) - no modification needed
    # No initial resources in buildings - default behavior

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
            suite="benchmark_arch", name="moderate_hard_basic", env=basic_env
        ),
        SimulationConfig(
            suite="benchmark_arch", name="moderate_hard_combat", env=combat_env
        ),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    arch_type: str = "fast",
) -> TrainTool:
    """Train on moderate rewards × hard complexity."""
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
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="moderate_hard")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="moderate_hard")
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    """Evaluate a policy on moderate rewards × hard complexity."""
    return EvaluateTool(
        simulations=make_evals(),
        policy_uris=policy_uris,
    )
