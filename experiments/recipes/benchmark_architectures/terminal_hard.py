"""Terminal-only rewards × Hard task complexity.

2-Axis Grid Position:
- Reward Shaping: Terminal-only (only heart reward, no intermediate rewards)
- Task Complexity: Hard (25×25 map, 24 agents, full combat)

This configuration represents maximum difficulty:
- Only heart reward (no intermediate rewards)
- Full combat enabled
- Maximum agents and large map
- Agents must discover entire resource chain independently
- Most challenging benchmark for architecture evaluation

Use case: Test architecture's ability to learn complex behaviors from sparse feedback
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
    """Create hard complexity arena with terminal-only rewards."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Large map for complex multi-agent interactions
    arena_env.game.map_builder.width = 25
    arena_env.game.map_builder.height = 25

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


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations with both basic and combat modes."""
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(
            suite="benchmark_arch", name="terminal_hard_basic", env=basic_env
        ),
        SimulationConfig(
            suite="benchmark_arch", name="terminal_hard_combat", env=combat_env
        ),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    arch_type: str = "fast",
) -> TrainTool:
    """Train on terminal-only rewards × hard complexity."""
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
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="terminal_hard")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="terminal_hard")
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    """Evaluate a policy on terminal-only rewards × hard complexity."""
    return EvaluateTool(
        simulations=make_evals(),
        policy_uris=policy_uris,
    )
