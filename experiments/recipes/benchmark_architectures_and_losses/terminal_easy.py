"""Terminal-only rewards × Easy task complexity.

2-Axis Grid Position:
- Reward Shaping: Terminal-only (only heart reward, no intermediate rewards)
- Task Complexity: Easy (15×15 map, 12 agents, no combat)

This configuration tests architecture performance with:
- No intermediate reward guidance
- Minimal task complexity
- Pure exploration test on simple tasks
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


def make_mettagrid(num_agents: int = 12) -> MettaGridConfig:
    """Create easy complexity arena with terminal-only rewards."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=False)

    # Small map for easier learning
    arena_env.game.map_builder.width = 15
    arena_env.game.map_builder.height = 15

    # Terminal-only rewards - only heart matters
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0,
        "battery_red": 0,
        "laser": 0,
        "armor": 0,
        "blueprint": 0,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 0,
        "battery_red": 0,
        "laser": 0,
        "armor": 0,
        "blueprint": 0,
    }

    # Standard converter ratios (3:1) - no modification needed
    # No initial resources in buildings - default behavior

    # Combat disabled
    arena_env.game.actions.attack.consumed_resources["laser"] = 100

    return arena_env


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations."""
    basic_env = env or make_mettagrid()
    return [
        SimulationConfig(suite="benchmark_arch", name="terminal_easy", env=basic_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    arch_type: str = "fast",
) -> TrainTool:
    """Train on terminal-only rewards × easy complexity."""
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
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="terminal_easy")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="terminal_easy")
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    """Evaluate a policy on terminal-only rewards × easy complexity."""
    return EvaluateTool(
        simulations=make_evals(),
        policy_uris=policy_uris,
    )
