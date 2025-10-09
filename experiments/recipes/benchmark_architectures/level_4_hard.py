"""Level 4 - Hard: Sparse rewards with combat and more agents.

This recipe significantly reduces reward shaping:
- Very sparse intermediate rewards
- Only heart reward is substantial
- Combat enabled
- More agents for increased competition
- Larger map with more complexity
- Agents must discover effective strategies with minimal guidance
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
from experiments.recipes.benchmark_architectures.level_1_basic import ARCHITECTURES


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
    """Create a hard difficulty arena with sparse rewards."""
    arena_env = eb.make_arena(num_agents=num_agents, combat=True)

    # Standard map size
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
    """Create evaluation configurations."""
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="benchmark_arch", name="level_4_basic", env=basic_env),
        SimulationConfig(suite="benchmark_arch", name="level_4_combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    arch_type: str = "fast",  # (vit | vit_sliding | vit_reset | transformer | fast | fast_lstm_reset | fast_dynamics | memory_free | agalite | gtrxl | trxl | trxl_nvidia | puffer)
) -> TrainTool:
    """Train on Level 4: Hard difficulty."""
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
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="level_4_hard")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Replay tool for recorded games."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="benchmark_arch", env=eval_env, name="level_4_hard")
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    """Evaluate a policy on Level 4 - Hard."""
    return EvaluateTool(
        simulations=make_evals(),
        policy_uris=policy_uris,
    )
