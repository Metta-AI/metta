"""
Arena experiment with a wandering trader NPC - Product Chain Discovery.

Setup:
- 6 agents: 5 learning agents + 1 wandering trader NPC
- Learning agents start with ZERO inventory (must figure out product chain)
- Trader NPC: Wanders randomly with 100 batteries, CANNOT collect resources (inventory limits)
- Product chain: Collect Ore -> Convert to Battery -> Convert to Hearts

Two paths to get batteries:
1. GENERATOR: 3 ore -> 1 battery (cheaper but 10-tick cooldown)
2. WANDERING TRADER: 4 ore -> 1 battery (more expensive but instant when you find them)

Key Mechanics:
- Trader has 0 capacity for ore (can't collect from mines)
- Trader only carries batteries for trading
- Learning agents must find the wandering trader to trade

Research Questions:
- Which option do agents prefer in peaceful vs combat scenarios?
- Do agents learn to find and follow the wandering trader?
- How does the instant vs cooldown trade-off affect agent strategies?
- In combat, do agents prefer the instant trader to quickly get resources?
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


def make_env(num_agents: int = 6) -> EnvConfig:
    """Create environment with a single wandering trader NPC."""
    arena_env = eb.make_arena(num_agents=num_agents)

    # Transfer action - enables ore-for-battery trading with the wandering trader
    arena_env.game.actions.transfer.enabled = True
    arena_env.game.actions.transfer.input_resources = {
        "ore_red": 4
    }  # Learning agents pay 4 ore to trade
    arena_env.game.actions.transfer.output_resources = {
        "battery_red": 1
    }  # Get 1 battery from trader
    arena_env.game.actions.transfer.trader_only = (
        True  # Restrict trades to the trader NPC
    )
    arena_env.game.actions.transfer.trader_group_id = (
        99  # Trader group id (see GroupConfig below)
    )

    # Configure rewards to incentivize finding and trading with the wandering NPC
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.05,  # Low ore value encourages conversion
        "battery_red": 1.0,  # High battery value rewards successful trades
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,  # Standard max ore reward
        "battery_red": 2,  # Increased max battery reward
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    # Easy converter: 1 battery to 1 heart
    arena_env.game.objects["altar"].input_resources = {"battery_red": 1}

    # Generator: 3 ore -> 1 battery (cheaper than trader but has cooldown)
    arena_env.game.objects["generator_red"].input_resources = {"ore_red": 3}
    arena_env.game.objects[
        "generator_red"
    ].cooldown = 10  # Add cooldown to make trader more attractive

    # PRODUCT CHAIN: Ore (collect) -> Battery (generator or trader) -> Hearts (altar)

    # Learning agents start with NOTHING - must figure out the product chain
    arena_env.game.agent.initial_inventory = {
        "battery_red": 0,  # No batteries
        "ore_red": 0,  # Must collect ore first
        "laser": 1,  # For combat scenarios
        "armor": 0,
    }

    # Configure the wandering trader NPC (different starting inventory and capabilities)
    trader_config = AgentConfig()
    trader_config.initial_inventory = {
        "battery_red": 100,  # Trader has batteries to sell
        "ore_red": 0,  # Trader doesn't need ore
        "laser": 0,  # Trader is peaceful
        "armor": 1,
    }
    trader_config.rewards = arena_env.game.agent.rewards.model_copy()
    trader_config.rewards.inventory = {}  # Trader gets no rewards

    # Trader needs ore capacity to receive ore during trades
    # Accept that trader will collect some ore while wandering (unavoidable with random policy)
    trader_config.default_resource_limit = 50  # Standard capacity
    trader_config.resource_limits = {
        "battery_red": 100,  # Has batteries to trade
        "ore_red": 20,  # Limited ore capacity (needed for trades to work)
        "heart": 10,  # Small heart capacity
        "laser": 0,  # Can't pick up weapons
        "armor": 1,  # Keep minimal armor
    }

    # Create trader group
    arena_env.game.groups["trader"] = GroupConfig(
        id=99,
        sprite=15,  # Different visual appearance
        group_reward_pct=0.0,  # No rewards for trader
        props=trader_config,
    )

    # CRITICAL: Ensure sufficient resources for the experiment
    arena_env.game.map_builder.root.params.objects["mine_red"] = (
        15  # Plenty of mines for ore collection
    )
    arena_env.game.map_builder.root.params.objects["generator_red"] = (
        8  # More generators to reduce competition
    )
    arena_env.game.map_builder.root.params.objects["altar"] = (
        6  # Sufficient altars for heart conversion
    )
    arena_env.game.map_builder.root.params.objects["battery_red"] = (
        0  # No free batteries - must use generator or trader!
    )

    # Agent assignment: 5 learning agents, 1 trader
    # NOTE: The last agent (index 5) will be the NPC with random movement
    arena_env.game.map_builder.root.params.agents = {"agent": 5, "trader": 1}

    return arena_env


def make_curriculum(env: Optional[EnvConfig] = None) -> CurriculumConfig:
    """Create curriculum for wandering trader experiment."""
    env = env or make_env()

    arena_tasks = cc.bucketed(env)

    # Test different reward structures for trading behavior
    arena_tasks.add_bucket("game.agent.rewards.inventory.battery_red", [0.5, 1.0, 2.0])
    arena_tasks.add_bucket("game.agent.rewards.inventory.ore_red", [0.01, 0.05, 0.1])

    # Vary trading cost to study price sensitivity
    arena_tasks.add_bucket(
        "game.actions.transfer.input_resources.ore_red", [3, 4, 5, 6]
    )

    # Vary generator cooldown to test trade-offs
    arena_tasks.add_bucket("game.objects.generator_red.cooldown", [5, 10, 20])

    # Combat on/off to see effect on trader interactions
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    return CurriculumConfig(task_generator=arena_tasks)


def make_evals(env: Optional[EnvConfig] = None) -> List[SimulationConfig]:
    """Create evaluation scenarios to test trader vs generator preferences."""
    basic_env = env or make_env()

    # Peaceful scenario - no combat pressure
    peaceful_env = basic_env.model_copy()
    peaceful_env.game.actions.attack.consumed_resources["laser"] = (
        100  # Combat disabled
    )

    # Combat scenario - agents need resources quickly
    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1  # Combat enabled

    # Common config: trader group agent is the wandering NPC
    npc_config = {
        "npc_policy_uri": "mock://random",  # Trader wanders randomly
        "policy_agents_pct": 5 / 6,  # 5 learning agents, 1 wandering trader NPC
        "npc_group_id": 99,  # Assign group 99 (trader) to NPC policy
    }

    return [
        SimulationConfig(
            name="trader_vs_generator/peaceful", env=peaceful_env, **npc_config
        ),
        SimulationConfig(
            name="trader_vs_generator/combat", env=combat_env, **npc_config
        ),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    total_timesteps: int = 200_000,
) -> TrainTool:
    """Configure training with trader NPCs."""
    trainer_cfg = TrainerConfig(
        total_timesteps=total_timesteps,
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(
            simulations=make_evals(),
            skip_git_check=True,
        ),
    )

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    """Play as one of the learning agents with a wandering trader NPC."""
    eval_env = env or make_env()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="wandering_trader",
            npc_policy_uri="mock://random",  # The wandering trader moves randomly
            policy_agents_pct=5
            / 6,  # You control 1 of 5 learning agents; 1 is NPC trader
            npc_group_id=99,  # Assign group 99 (trader) to NPC policy
        )
    )


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    """Replay episodes to observe agent-trader interactions."""
    eval_env = env or make_env()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="wandering_trader",
            npc_policy_uri="mock://random",  # The wandering trader moves randomly
            policy_agents_pct=5 / 6,  # 5 learning agents, 1 NPC trader
            npc_group_id=99,  # Assign group 99 (trader) to NPC policy
        )
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    """Evaluate a policy with trader NPCs."""
    simulations = simulations or make_evals()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
