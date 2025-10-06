"""Level 2 - Easy: Reduced reward shaping, still beginner-friendly.

This recipe reduces some guidance while maintaining accessibility:
- Moderate intermediate rewards (lower than Level 1)
- Standard converter ratios (3:1)
- Combat still disabled
- No initial resources in buildings
- Standard map size
"""

from typing import List, Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


def make_mettagrid(num_agents: int = 16) -> MettaGridConfig:
    """Create an easy arena environment with moderate reward shaping."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=False)

    # Standard map size
    arena_env.game.map_builder.width = 20
    arena_env.game.map_builder.height = 20

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

    # Combat disabled
    arena_env.game.actions.attack.consumed_resources["laser"] = 100

    return arena_env


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations."""
    basic_env = env or make_mettagrid()
    return [
        SimulationConfig(suite="benchmark_arch", name="level_2_easy", env=basic_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """Train on Level 2 - Easy difficulty."""
    if curriculum is None:
        env = make_mettagrid()
        curriculum = cc.env_curriculum(env)

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=make_evals()),
        policy_architecture=policy_architecture,
    )


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    """Interactive play tool."""
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="level_2_easy")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="level_2_easy")
    )


def evaluate(
    policy_uri: str | None = None,
    simulations: Optional[Sequence[SimulationConfig]] = None,
) -> SimTool:
    """Evaluate a policy on Level 2 - Easy."""
    simulations = simulations or make_evals()
    policy_uris = [policy_uri] if policy_uri is not None else None

    return SimTool(
        simulations=simulations,
        policy_uris=policy_uris,
    )
