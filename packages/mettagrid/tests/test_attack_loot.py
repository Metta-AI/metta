"""Tests for attack action with ordered loot configuration."""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    GameConfig,
    MettaGridConfig,
    NoopActionConfig,
    ObsConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.actions import attack
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


@pytest.fixture
def base_config():
    """Base configuration for attack loot tests."""
    return GameConfig(
        max_steps=50,
        num_agents=2,
        obs=ObsConfig(width=3, height=3, num_tokens=100),
        resource_names=["A", "B", "C"],
        actions=ActionsConfig(
            noop=NoopActionConfig(),
            attack=AttackActionConfig(enabled=True, loot=["A", "B"]),
        ),
        agent=AgentConfig(
            rewards=AgentRewards(),
            # Initial limits (will be overridden)
            resource_limits={},
            initial_inventory={"A": 0, "B": 0, "C": 0},
        ),
    )


@pytest.fixture
def make_sim(base_config: GameConfig):
    """Factory fixture that creates a configured Simulation environment."""

    def _create_sim(game_map, config_overrides=None):
        # Deep copy base config to avoid modifying fixture
        game_config = base_config.model_copy(deep=True)

        if config_overrides:
            # We need to carefully update the pydantic model
            # Converting to dict and back is safer for deep updates of nested models if we re-create
            config_dict = game_config.model_dump()

            # Update top-level keys from overrides
            for key, value in config_overrides.items():
                if key not in ["agent", "actions"]:  # Handle special nested ones separately
                    config_dict[key] = value

            if "agent" in config_overrides:
                agent_overrides = config_overrides["agent"]
                if "initial_inventory" in agent_overrides:
                    config_dict["agent"]["initial_inventory"].update(agent_overrides["initial_inventory"])
                if "resource_limits" in agent_overrides:
                    config_dict["agent"]["resource_limits"].update(agent_overrides["resource_limits"])

            if "actions" in config_overrides and "attack" in config_overrides["actions"]:
                # Update attack action config
                # Since actions is a sub-model, we need to handle it
                if config_dict["actions"]["attack"]:
                    config_dict["actions"]["attack"].update(config_overrides["actions"]["attack"])
                else:
                    config_dict["actions"]["attack"] = config_overrides["actions"]["attack"]

            game_config = GameConfig(**config_dict)

        cfg = MettaGridConfig(game=game_config)

        map_list = game_map.tolist() if hasattr(game_map, "tolist") else game_map
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_list)

        sim = Simulation(cfg, seed=42)
        return sim

    return _create_sim


def test_attack_loot_order(make_sim):
    """Test that resources are looted in the specified order (A then B)."""
    # Agent 0 at (0,0), Agent 1 at (0,1)
    game_map = [
        ["agent.team_0", "agent.team_1"],
    ]

    # Define explicit agents with team IDs to ensure they are registered
    # Target needs higher limit to hold both A and B initially

    config_overrides = {
        "agents": [
            {
                "team_id": 0,
                "resource_limits": {"shared_limit": {"limit": 5, "resources": ["A", "B"]}},
                "initial_inventory": {"A": 0, "B": 0},
            },
            {
                "team_id": 1,
                "resource_limits": {"shared_limit": {"limit": 100, "resources": ["A", "B"]}},
                "initial_inventory": {"A": 10, "B": 10},
            },
        ],
        "actions": {"attack": {"loot": ["A", "B"]}},
    }

    sim = make_sim(game_map, config_overrides)

    # Attacker (0) starts empty
    sim.agent(0).set_inventory({"A": 0, "B": 0})
    # Target (1) has 10 A, 10 B
    sim.agent(1).set_inventory({"A": 10, "B": 10})

    # Verify setup
    inv1_setup = sim.agent(1).inventory
    assert inv1_setup.get("A", 0) == 10, f"Target setup failed A: {inv1_setup}"
    assert inv1_setup.get("B", 0) == 10, f"Target setup failed B: {inv1_setup}"

    # Perform attack
    result = attack(sim, target_arg=0, agent_idx=0)

    assert result["success"], f"Attack failed: {result.get('error')}"

    # Check inventories
    # Attacker should have taken A first.
    # Shared limit is 5.
    # A takes 5 space. B takes 0 space available.
    inv0 = sim.agent(0).inventory
    inv1 = sim.agent(1).inventory

    print(f"Attacker inventory: {inv0}")
    print(f"Target inventory: {inv1}")

    assert inv0.get("A", 0) == 5, "Attacker should have 5 A (filled limit with A)"
    assert inv0.get("B", 0) == 0, "Attacker should have 0 B (no space left)"

    # Target loses 5 A (stolen), keeps 10 B
    assert inv1.get("A", 0) == 5, f"Target should have 5 A (10-5). Got {inv1.get('A')}"
    assert inv1.get("B", 0) == 10, "Target should have 10 B (none stolen)"


def test_attack_loot_reverse_order(make_sim):
    """Test that resources are looted in the specified order (B then A)."""
    game_map = [
        ["agent.team_0", "agent.team_1"],
    ]

    config_overrides = {
        "agents": [
            {
                "team_id": 0,
                "resource_limits": {"shared_limit": {"limit": 5, "resources": ["A", "B"]}},
                "initial_inventory": {"A": 0, "B": 0},
            },
            {
                "team_id": 1,
                "resource_limits": {"shared_limit": {"limit": 100, "resources": ["A", "B"]}},
                "initial_inventory": {"A": 10, "B": 10},
            },
        ],
        "actions": {"attack": {"loot": ["B", "A"]}},
    }

    sim = make_sim(game_map, config_overrides)

    sim.agent(0).set_inventory({"A": 0, "B": 0})
    sim.agent(1).set_inventory({"A": 10, "B": 10})

    result = attack(sim, target_arg=0, agent_idx=0)
    assert result["success"], f"Attack failed: {result.get('error')}"

    inv0 = sim.agent(0).inventory
    inv1 = sim.agent(1).inventory

    print(f"Attacker inventory: {inv0}")
    print(f"Target inventory: {inv1}")

    assert inv0.get("B", 0) == 5, "Attacker should have 5 B"
    assert inv0.get("A", 0) == 0, "Attacker should have 0 A"

    assert inv1.get("B", 0) == 5, "Target should have 5 B (10-5)"
    assert inv1.get("A", 0) == 10, "Target should have 10 A"
