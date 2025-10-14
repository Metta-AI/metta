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
from mettagrid.config import ConverterConfig

from experiments.recipes.benchmark_architectures.benchmark import ARCHITECTURES


def make_mettagrid(num_agents: int = 20) -> MettaGridConfig:
    """Create arena with terminal-only rewards and easy task complexity.

    Task Complexity (Easy):
    - 1:1 converter ratio (simple resource chain)
    - Initial resources in buildings (easier start)

    Reward Shaping (Terminal):
    - Only terminal reward (heart=1.0, all intermediates=0.0)
    """
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Standard map size across all recipes
    arena_env.game.map_builder.width = 20
    arena_env.game.map_builder.height = 20

    # Terminal reward shaping: Only heart reward
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
    }

    # Easy task complexity: 1:1 converter (1 battery_red → 1 heart)
    altar = arena_env.game.objects.get("altar")
    if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
        altar.input_resources["battery_red"] = 1
        altar.initial_resource_count = 2

    # Easy task complexity: Initial resources in all buildings
    for obj_name in ["mine_red", "generator_red", "lasery", "armory"]:
        obj = arena_env.game.objects.get(obj_name)
        if obj and hasattr(obj, "initial_resource_count"):
            obj.initial_resource_count = 2

    # Combat enabled (standard across all)
    arena_env.game.actions.attack.consumed_resources["laser"] = 1

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
