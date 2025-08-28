"""
Arena experiment with trader NPCs that offer instant trades at higher prices.

This experiment tests whether agents prefer:
- Generator: 3 ore -> 1 battery with cooldown (cheaper but slower)
- Trader NPC: 4 ore -> 1 battery instantly (more expensive but no wait)
"""

from typing import List, Optional, Sequence

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.mettagrid.mettagrid_config import AgentConfig, EnvConfig, GroupConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

from experiments.recipes.arena_basic_easy_shaped import (
    make_evals as make_standard_evals,
)


def make_env(num_agents: int = 24, num_traders: int = 4) -> EnvConfig:
    """Create environment with trader NPCs.

    Args:
        num_agents: Total number of agents (including traders)
        num_traders: Number of trader NPCs
    """
    arena_env = eb.make_arena(num_agents=num_agents)

    # Configure rewards (same as arena_basic_easy_shaped)
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # Easy converter: 1 battery_red to 1 heart (instead of 3 to 1)
    arena_env.game.objects["altar"].input_resources = {"battery_red": 1}

    # Enable transfer action for trading
    arena_env.game.actions.transfer.enabled = True
    arena_env.game.actions.transfer.input_resources = {"ore_red": 4}  # Cost: 4 ore
    arena_env.game.actions.transfer.output_resources = {
        "battery_red": 1
    }  # Receive: 1 battery
    arena_env.game.actions.transfer.trader_only = True
    arena_env.game.actions.transfer.trader_group_id = 99  # Special group for traders

    # Configure trader NPCs as a special group
    if "trader" not in arena_env.game.groups:
        # Create proper GroupConfig for traders
        trader_agent_config = AgentConfig()
        # Traders shouldn't earn rewards from inventory
        trader_agent_config.rewards.inventory = {}
        trader_agent_config.rewards.inventory_max = {}

        arena_env.game.groups["trader"] = GroupConfig(
            id=99,
            sprite=15,  # Different visual for traders
            group_reward_pct=0.0,  # Traders don't share rewards
            props=trader_agent_config,
        )

    # Place both regular agents and trader agents on the map
    try:
        root_cfg = arena_env.game.map_builder.root
        if hasattr(root_cfg, "params") and hasattr(root_cfg.params, "agents"):
            # Distribute agents by group for map generation
            non_traders = max(0, num_agents - num_traders)
            root_cfg.params.agents = {"agent": non_traders, "trader": num_traders}
    except Exception:
        # If map builder isn't MapGen.Random, skip adjusting agent placement
        pass

    return arena_env


def make_curriculum(arena_env: Optional[EnvConfig] = None) -> CurriculumConfig:
    """Create curriculum with trading scenarios."""
    arena_env = arena_env or make_env()

    # Make a set of training tasks for the arena
    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2])

    # Enable or disable attacks
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # Vary trader exchange rates to test sensitivity
    arena_tasks.add_bucket("game.actions.transfer.input_resources.ore_red", [3, 4, 5])

    # Sometimes add initial items to buildings
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    # Vary generator cooldown to test time pressure effects
    arena_tasks.add_bucket("game.objects.generator_red.cooldown", [10, 20, 30])

    return CurriculumConfig(task_generator=arena_tasks)


def make_evals(env: Optional[EnvConfig] = None) -> List[SimulationConfig]:
    """Create evaluation scenarios with and without combat."""
    basic_env = env or make_env()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100  # No combat

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1  # Combat enabled

    # Scenario with expensive traders (5 ore for 1 battery)
    expensive_trader_env = basic_env.model_copy()
    expensive_trader_env.game.actions.transfer.input_resources = {"ore_red": 5}

    return [
        SimulationConfig(name="trader/basic", env=basic_env),
        SimulationConfig(name="trader/combat", env=combat_env),
        SimulationConfig(name="trader/expensive", env=expensive_trader_env),
    ]


def train(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    """Configure training with trader NPCs."""
    # Combine standard arena evals with trader-specific evals
    eval_sims = make_standard_evals() + make_evals()

    trainer_cfg = TrainerConfig(
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(
            simulations=eval_sims,
            skip_git_check=True,
        ),
    )

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    """Play with trader NPCs."""
    eval_env = env or make_env()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="trader_arena",
            npc_policy_uri="local://metta.sim.trader_npc_policy:create_trader_npc_policy",
        )
    )


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    """Replay with trader NPCs."""
    eval_env = env or make_env()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="trader_arena",
            npc_policy_uri="local://metta.sim.trader_npc_policy:create_trader_npc_policy",
        )
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    """Evaluate a policy with trader NPCs."""
    # Combine standard arena evals with trader-specific evals, unless explicitly provided
    simulations = simulations or (make_standard_evals() + make_evals())

    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
