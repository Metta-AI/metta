"""Dense rewards × Easy task complexity.

2-Axis Grid Position:
- Reward Shaping: Dense (high intermediate rewards 0.5-0.9)
- Task Complexity: Easy (15×15 map, 12 agents, no combat)

This configuration provides the most guidance through:
- High intermediate rewards for all resources
- Easy converter ratios (1:1 instead of 3:1)
- Combat disabled (high laser cost)
- Initial resources in buildings
- Smaller map size for faster learning

Use case: Test basic learning capabilities and fast prototyping
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
from mettagrid.config import ConverterConfig

from experiments.recipes.benchmark_architectures.benchmark import ARCHITECTURES


def make_mettagrid(num_agents: int = 12) -> MettaGridConfig:
    """Create a basic arena environment with dense reward shaping."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=False)

    # Small map for faster learning
    arena_env.game.map_builder.width = 15
    arena_env.game.map_builder.height = 15

    # High rewards for all intermediate items
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.5,  # High reward for mining
        "battery_red": 0.9,  # High reward for conversion
        "laser": 0.7,
        "armor": 0.7,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 2,
        "battery_red": 2,
        "laser": 2,
        "armor": 2,
        "blueprint": 2,
    }

    # Easy converter: 1 battery_red to 1 heart
    altar = arena_env.game.objects.get("altar")
    if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
        altar.input_resources["battery_red"] = 1
        altar.initial_resource_count = 2  # Start with resources

    # Add initial resources to all buildings
    for obj_name in ["mine_red", "generator_red", "lasery", "armory"]:
        obj = arena_env.game.objects.get(obj_name)
        if obj and hasattr(obj, "initial_resource_count"):
            obj.initial_resource_count = 2

    # Combat disabled
    arena_env.game.actions.attack.consumed_resources["laser"] = 100

    return arena_env


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    """Create evaluation configurations."""
    basic_env = env or make_mettagrid()
    return [
        SimulationConfig(suite="benchmark_arch", name="dense_easy", env=basic_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    arch_type: str = "fast",
) -> TrainTool:
    """Train on dense rewards × easy complexity."""
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
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="dense_easy")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="dense_easy")
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    """Evaluate a policy on dense rewards × easy complexity."""
    return EvaluateTool(
        simulations=make_evals(),
        policy_uris=policy_uris,
    )
