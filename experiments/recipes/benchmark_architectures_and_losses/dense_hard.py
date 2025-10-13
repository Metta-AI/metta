"""Dense rewards × Hard task complexity.

2-Axis Grid Position:
- Reward Shaping: Dense (high intermediate rewards 0.5-0.9)
- Task Complexity: Hard (25×25 map, 24 agents, combat enabled)

This configuration tests architecture performance with:
- Maximum reward guidance
- Maximum task complexity
- Tests capacity and scaling with strong reward signal
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


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
    """Create hard complexity arena with dense reward shaping."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Large map size
    arena_env.game.map_builder.width = 25
    arena_env.game.map_builder.height = 25

    # Dense rewards for all intermediate items
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
        altar.initial_resource_count = 2

    # Add initial resources to all buildings
    for obj_name in ["mine_red", "generator_red", "lasery", "armory"]:
        obj = arena_env.game.objects.get(obj_name)
        if obj and hasattr(obj, "initial_resource_count"):
            obj.initial_resource_count = 2

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
            suite="benchmark_arch", name="dense_hard_basic", env=basic_env
        ),
        SimulationConfig(
            suite="benchmark_arch", name="dense_hard_combat", env=combat_env
        ),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    arch_type: str = "fast",
) -> TrainTool:
    """Train on dense rewards × hard complexity."""
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
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="dense_hard")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="dense_hard")
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    """Evaluate a policy on dense rewards × hard complexity."""
    return EvaluateTool(
        simulations=make_evals(),
        policy_uris=policy_uris,
    )
