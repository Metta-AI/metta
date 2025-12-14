import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
)
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.simulator import Action, Simulation


@pytest.fixture
def sim_with_resources() -> Simulation:
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=32),
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            resource_names=["red", "blue", "green"],
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=7),
        )
    )
    return Simulation(cfg)


@pytest.fixture
def sim_with_large_inventory_limit() -> Simulation:
    """Simulation with high resource limits to test inventory values > 255."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=64),
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            resource_names=["carbon", "gold"],
            agent=AgentConfig(inventory=InventoryConfig(default_limit=65535)),
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=7),
        )
    )
    return Simulation(cfg)


def _find_agent_id(grid_objects: dict[int, dict]) -> int:
    for obj in grid_objects.values():
        if "agent_id" in obj:
            return int(obj["agent_id"])
    raise AssertionError("No agent found in grid_objects")


def test_set_inventory_sets_and_clears(sim_with_resources: Simulation):
    sim = sim_with_resources

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # 1) Set initial inventory: red=3, blue=2
    inv1 = {"red": 3, "blue": 2}
    agent.set_inventory(inv1)

    # Step to update observations
    agent.set_action(Action(name="noop"))
    sim.step()

    # Verify using the inventory property
    inv_from_property = agent.inventory
    assert inv_from_property.get("red", 0) == 3, f"Expected red=3, got {inv_from_property.get('red', 0)}"
    assert inv_from_property.get("blue", 0) == 2, f"Expected blue=2, got {inv_from_property.get('blue', 0)}"
    assert inv_from_property.get("green", 0) == 0, f"Expected green=0, got {inv_from_property.get('green', 0)}"

    # Also verify through grid_objects for backward compatibility
    # Note: grid_objects returns inventory with integer indices, not resource names
    objs_after_first = sim.grid_objects()
    agent_inventory_1 = None
    for obj in objs_after_first.values():
        if obj.get("agent_id") == agent_id:
            agent_inventory_1 = obj.get("inventory", {})
            break
    assert agent_inventory_1 is not None
    # grid_objects uses resource indices as keys
    red_idx = sim.resource_names.index("red")
    blue_idx = sim.resource_names.index("blue")
    green_idx = sim.resource_names.index("green")
    assert int(agent_inventory_1.get(red_idx, 0)) == 3
    assert int(agent_inventory_1.get(blue_idx, 0)) == 2
    assert int(agent_inventory_1.get(green_idx, 0)) == 0

    # 2) Update inventory with only red=1; blue should be cleared (removed)
    inv2 = {"red": 1}
    agent.set_inventory(inv2)

    # Step to update observations
    agent.set_action(Action(name="noop"))
    sim.step()

    # Verify using the inventory property
    inv_from_property = agent.inventory
    assert inv_from_property.get("red", 0) == 1, f"Expected red=1, got {inv_from_property.get('red', 0)}"
    assert inv_from_property.get("blue", 0) == 0, f"Expected blue=0, got {inv_from_property.get('blue', 0)}"
    assert inv_from_property.get("green", 0) == 0, f"Expected green=0, got {inv_from_property.get('green', 0)}"

    # Also verify through grid_objects
    objs_after_second = sim.grid_objects()
    agent_inventory_2 = None
    for obj in objs_after_second.values():
        if obj.get("agent_id") == agent_id:
            agent_inventory_2 = obj.get("inventory", {})
            break
    assert agent_inventory_2 is not None
    # grid_objects uses resource indices as keys
    assert int(agent_inventory_2.get(red_idx, 0)) == 1
    # blue should be absent or zero
    assert int(agent_inventory_2.get(blue_idx, 0)) == 0
    # green remains zero
    assert int(agent_inventory_2.get(green_idx, 0)) == 0


def test_inventory_values_under_100(sim_with_large_inventory_limit: Simulation):
    """Test inventory values under 100 (only ones token, no ton token)."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set inventory with values under 100
    agent.set_inventory({"carbon": 42, "gold": 99})

    agent.set_action(Action(name="noop"))
    sim.step()

    inv = agent.inventory
    assert inv.get("carbon", 0) == 42, f"Expected carbon=42, got {inv.get('carbon', 0)}"
    assert inv.get("gold", 0) == 99, f"Expected gold=99, got {inv.get('gold', 0)}"


def test_inventory_values_100_to_9999(sim_with_large_inventory_limit: Simulation):
    """Test inventory values between 100-9999 (base token + e2 token)."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set inventory with values between 100-9999
    agent.set_inventory({"carbon": 100, "gold": 9999})

    agent.set_action(Action(name="noop"))
    sim.step()

    inv = agent.inventory
    assert inv.get("carbon", 0) == 100, f"Expected carbon=100, got {inv.get('carbon', 0)}"
    assert inv.get("gold", 0) == 9999, f"Expected gold=9999, got {inv.get('gold', 0)}"


def test_inventory_values_over_255(sim_with_large_inventory_limit: Simulation):
    """Test inventory values over 255 using exponential encoding (base + e2*100)."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set inventory with values over 255
    # 1234 = 34 (base) + 12 (e2) * 100
    # 5678 = 78 (base) + 56 (e2) * 100
    agent.set_inventory({"carbon": 1234, "gold": 5678})

    agent.set_action(Action(name="noop"))
    sim.step()

    inv = agent.inventory
    assert inv.get("carbon", 0) == 1234, f"Expected carbon=1234, got {inv.get('carbon', 0)}"
    assert inv.get("gold", 0) == 5678, f"Expected gold=5678, got {inv.get('gold', 0)}"


def test_inventory_max_value(sim_with_large_inventory_limit: Simulation):
    """Test large inventory value using all three encoding tokens (base, e2, e4)."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Test with 54321 which exercises all three tokens:
    # 54321 = 21 (base) + 43 (e2) * 100 + 5 (e4) * 10000
    # Note: InventoryQuantity is uint16_t, max 65535
    agent.set_inventory({"carbon": 54321})

    agent.set_action(Action(name="noop"))
    sim.step()

    inv = agent.inventory
    assert inv.get("carbon", 0) == 54321, f"Expected carbon=54321, got {inv.get('carbon', 0)}"


def test_inventory_observation_tokens_for_large_values(sim_with_large_inventory_limit: Simulation):
    """Test that inventory observations emit correct tokens for values > 100."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set carbon to 1234: base=34, e2=12
    agent.set_inventory({"carbon": 1234})

    agent.set_action(Action(name="noop"))
    sim.step()

    # Check raw observation tokens
    obs_tokens = {}
    for token in agent.self_observation():
        if token.feature.name.startswith("inv:"):
            obs_tokens[token.feature.name] = token.value

    # Should have inv:carbon = 34 (1234 % 100) and inv:carbon:e2 = 12 ((1234 / 100) % 100)
    assert "inv:carbon" in obs_tokens, f"Missing inv:carbon token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon"] == 34, f"Expected inv:carbon=34, got {obs_tokens['inv:carbon']}"
    assert "inv:carbon:e2" in obs_tokens, f"Missing inv:carbon:e2 token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon:e2"] == 12, f"Expected inv:carbon:e2=12, got {obs_tokens['inv:carbon:e2']}"


def test_inventory_observation_tokens_for_e4_values(sim_with_large_inventory_limit: Simulation):
    """Test that inventory observations emit correct tokens for values >= 10000."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set carbon to 54321: base=21, e2=43, e4=5
    # Note: InventoryQuantity is uint16_t, max 65535
    agent.set_inventory({"carbon": 54321})

    agent.set_action(Action(name="noop"))
    sim.step()

    # Check raw observation tokens
    obs_tokens = {}
    for token in agent.self_observation():
        if token.feature.name.startswith("inv:"):
            obs_tokens[token.feature.name] = token.value

    # Should have all three tokens
    assert "inv:carbon" in obs_tokens, f"Missing inv:carbon token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon"] == 21, f"Expected inv:carbon=21, got {obs_tokens['inv:carbon']}"
    assert "inv:carbon:e2" in obs_tokens, f"Missing inv:carbon:e2 token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon:e2"] == 43, f"Expected inv:carbon:e2=43, got {obs_tokens['inv:carbon:e2']}"
    assert "inv:carbon:e4" in obs_tokens, f"Missing inv:carbon:e4 token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon:e4"] == 5, f"Expected inv:carbon:e4=5, got {obs_tokens['inv:carbon:e4']}"


def test_inventory_no_e2_e4_token_for_small_values(sim_with_large_inventory_limit: Simulation):
    """Test that values < 100 don't emit the :e2 or :e4 tokens."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set carbon to 42 (< 100, should not have :e2 or :e4 tokens)
    agent.set_inventory({"carbon": 42})

    agent.set_action(Action(name="noop"))
    sim.step()

    # Check raw observation tokens
    obs_tokens = {}
    for token in agent.self_observation():
        if token.feature.name.startswith("inv:"):
            obs_tokens[token.feature.name] = token.value

    # Should have inv:carbon = 42, but NOT inv:carbon:e2 or inv:carbon:e4
    assert "inv:carbon" in obs_tokens, "Missing inv:carbon token"
    assert obs_tokens["inv:carbon"] == 42, f"Expected inv:carbon=42, got {obs_tokens['inv:carbon']}"
    assert "inv:carbon:e2" not in obs_tokens, "Unexpected inv:carbon:e2 token for value < 100"
    assert "inv:carbon:e4" not in obs_tokens, "Unexpected inv:carbon:e4 token for value < 100"
