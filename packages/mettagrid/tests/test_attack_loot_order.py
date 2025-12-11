import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.actions import attack
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


@pytest.fixture
def loot_order_config():
    """Configuration for testing loot order."""
    return GameConfig(
        max_steps=10,
        num_agents=2,
        obs=ObsConfig(width=3, height=3, num_tokens=100),
        resource_names=["A", "B"],
        actions=ActionsConfig(
            noop=NoopActionConfig(),
            move=MoveActionConfig(),
            # Attack config will be overridden in tests
            attack=AttackActionConfig(enabled=True),
        ),
        objects={
            "wall": WallConfig(),
        },
        agent=AgentConfig(rewards=AgentRewards()),
    )


def test_loot_order_with_shared_limit(loot_order_config):
    """Test that resources are looted in the specified order when there is a shared limit."""

    # Map layout: Agent 0 (attacker) next to Agent 1 (victim)
    # [Attacker, Victim]
    game_map = [["agent.red", "agent.blue"]]

    # Configure agents
    # Attacker has shared limit of 1 for A and B
    attacker_config = AgentConfig(
        team_id=0,
        inventory=InventoryConfig(
            initial={"A": 0, "B": 0},
            limits={"shared_limit": ResourceLimitsConfig(limit=1, resources=["A", "B"])},
        ),
    )

    # Victim has plenty of A and B
    victim_config = AgentConfig(
        team_id=1,
        inventory=InventoryConfig(
            initial={"A": 10, "B": 10},
            limits={"default": ResourceLimitsConfig(limit=100, resources=["A", "B"])},
        ),
    )

    # --- Case 1: loot=["A", "B"] ---
    config1 = loot_order_config.model_copy(deep=True)
    config1.agents = [attacker_config, victim_config]
    config1.actions.attack = AttackActionConfig(enabled=True, loot=["A", "B"])

    mg_config1 = MettaGridConfig(game=config1)
    mg_config1.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
    sim1 = Simulation(mg_config1, seed=42)

    # Attacker (agent 0) attacks Victim (agent 1, which is neighbor 5 in scan order for (0,0) looking East?)
    # Wait, without orientation, scan order is 0-8.
    # Neighbor at (0, 1) relative to (0, 0).
    # 3x3 grid around (0,0):
    # (-1,-1) (-1,0) (-1,1)
    # ( 0,-1) ( 0,0) ( 0,1)
    # ( 1,-1) ( 1,0) ( 1,1)
    # Indices:
    # 0: (-1,-1), 1: (-1,0), 2: (-1,1)
    # 3: ( 0,-1), 4: ( 0,0), 5: ( 0,1)
    # 6: ( 1,-1), 7: ( 1,0), 8: ( 1,1)
    # So neighbor at (0, 1) is index 5.
    # But the Attack logic iterates through neighbors and returns a list of targets.
    # "Attack takes an argument 0-8, which is the index of the target agent to attack."
    # "Target agents are those found in a 3x3 grid in front of the agent, indexed in scan order."
    # Wait, the C++ implementation:
    # "Simple 3x3 neighborhood scan... if (r==actor.r && c==actor.c) continue... targets.push_back(agent)"
    # So it collects all neighbors into a list `targets`.
    # If `arg` is 0, it attacks `targets[0]`.
    # In this map, there is only 1 neighbor (the victim). So `targets` has size 1.
    # So attacking with arg=0 should hit the victim.

    result = attack(sim1, target_arg=0, agent_idx=0)
    assert result["success"], "Attack should succeed"

    attacker_inv1 = sim1.agent(0).inventory
    print(f"Case 1 (loot=['A', 'B']): Attacker inventory: {attacker_inv1}")

    # With shared limit 1, should have taken A first, filled capacity, so B is skipped.
    assert attacker_inv1.get("A", 0) == 1
    assert attacker_inv1.get("B", 0) == 0

    # --- Case 2: loot=["B", "A"] ---
    config2 = loot_order_config.model_copy(deep=True)
    config2.agents = [attacker_config, victim_config]
    config2.actions.attack = AttackActionConfig(enabled=True, loot=["B", "A"])

    mg_config2 = MettaGridConfig(game=config2)
    mg_config2.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
    sim2 = Simulation(mg_config2, seed=42)

    result = attack(sim2, target_arg=0, agent_idx=0)
    assert result["success"], "Attack should succeed"

    attacker_inv2 = sim2.agent(0).inventory
    print(f"Case 2 (loot=['B', 'A']): Attacker inventory: {attacker_inv2}")

    # With shared limit 1, should have taken B first, filled capacity, so A is skipped.
    assert attacker_inv2.get("B", 0) == 1
    assert attacker_inv2.get("A", 0) == 0


if __name__ == "__main__":
    # Manually run the test function if executed as script
    pytest.main([__file__])
