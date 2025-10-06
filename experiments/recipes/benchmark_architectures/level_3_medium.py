"""Level 3 - Medium: Combat enabled with reduced reward shaping.

This recipe introduces combat and reduces reward shaping:
- Low intermediate rewards for resources
- Standard converter ratios
- Combat enabled (normal laser cost)
- Agents must learn both resource gathering and combat
- Standard arena map size
- Dual evaluation: basic (no combat) and combat modes
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


def make_mettagrid(num_agents: int = 20) -> MettaGridConfig:
    """Create a medium difficulty arena with combat enabled."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Standard map size
    arena_env.game.map_builder.width = 25
    arena_env.game.map_builder.height = 25

    # Low intermediate rewards
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.3,
        "laser": 0.2,
        "armor": 0.2,
        "blueprint": 0.1,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # Combat enabled with normal cost
    arena_env.game.actions.attack.consumed_resources["laser"] = 1

    return arena_env


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations with both basic and combat modes."""
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="benchmark_arch", name="level_3_basic", env=basic_env),
        SimulationConfig(suite="benchmark_arch", name="level_3_combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """Train on Level 3 - Medium difficulty."""
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
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="level_3_medium")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(
            suite="benchmark_arch", env=eval_env, name="level_3_medium"
        )
    )


def evaluate(
    policy_uri: str | None = None,
    simulations: Optional[Sequence[SimulationConfig]] = None,
) -> SimTool:
    """Evaluate a policy on Level 3 - Medium."""
    simulations = simulations or make_evals()
    policy_uris = [policy_uri] if policy_uri is not None else None

    return SimTool(
        simulations=simulations,
        policy_uris=policy_uris,
    )
