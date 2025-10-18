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

from experiments.recipes.benchmark_architectures.benchmark import ARCHITECTURES


def make_mettagrid(num_agents: int = 20) -> MettaGridConfig:
    """Create arena with moderate reward shaping and hard task complexity.

    Task Complexity (Hard):
    - 3:1 converter ratio (complex resource chain - default)
    - No initial resources (standard start)

    Reward Shaping (Moderate):
    - Medium intermediate rewards (0.2-0.5)
    """
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Standard map size across all recipes
    arena_env.game.map_builder.width = 20
    arena_env.game.map_builder.height = 20

    # Moderate reward shaping
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.2,
        "battery_red": 0.5,
        "laser": 0.3,
        "armor": 0.3,
        "blueprint": 0.2,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 2,
        "battery_red": 2,
        "laser": 2,
        "armor": 2,
        "blueprint": 2,
    }

    # Hard task complexity: 3:1 converter (3 battery_red → 1 heart - default)
    # No need to modify - this is the default converter ratio

    # Hard task complexity: No initial resources in buildings
    # (buildings start empty - default behavior)

    # Combat enabled (standard across all)
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
