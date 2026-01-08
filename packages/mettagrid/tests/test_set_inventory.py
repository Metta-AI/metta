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
from mettagrid.map_builder.random_map import RandomMapBuilder
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
    """Simulation with high resource limits to test inventory values > 256."""
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


def test_inventory_values_under_256(sim_with_large_inventory_limit: Simulation):
    """Test inventory values under 256 (only base token, no power tokens)."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set inventory with values under 255
    agent.set_inventory({"carbon": 42, "gold": 99})

    agent.set_action(Action(name="noop"))
    sim.step()

    inv = agent.inventory
    assert inv.get("carbon", 0) == 42, f"Expected carbon=42, got {inv.get('carbon', 0)}"
    assert inv.get("gold", 0) == 99, f"Expected gold=99, got {inv.get('gold', 0)}"


def test_inventory_values_over_256(sim_with_large_inventory_limit: Simulation):
    """Test inventory values over 256 using multi-token encoding (base + p1*256)."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set inventory with values over 256
    # 1234 = 210 (base) + 4 (p1) * 256
    # 5678 = 46 (base) + 22 (p1) * 256 (5678 = 46 + 22*256 = 46 + 5632)
    agent.set_inventory({"carbon": 1234, "gold": 5678})

    agent.set_action(Action(name="noop"))
    sim.step()

    inv = agent.inventory
    assert inv.get("carbon", 0) == 1234, f"Expected carbon=1234, got {inv.get('carbon', 0)}"
    assert inv.get("gold", 0) == 5678, f"Expected gold=5678, got {inv.get('gold', 0)}"


def test_inventory_max_value(sim_with_large_inventory_limit: Simulation):
    """Test large inventory value using two encoding tokens (base, p1)."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Test with 65535 (max uint16_t) which exercises two tokens with base 256:
    # 65535 = 255 (base) + 255 (p1) * 256
    agent.set_inventory({"carbon": 65535})

    agent.set_action(Action(name="noop"))
    sim.step()

    inv = agent.inventory
    assert inv.get("carbon", 0) == 65535, f"Expected carbon=65535, got {inv.get('carbon', 0)}"


def test_inventory_observation_tokens_for_large_values(sim_with_large_inventory_limit: Simulation):
    """Test that inventory observations emit correct tokens for values > token_value_base."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set carbon to 1234: with base 256, 1234 = 210 + 4*256
    # base = 1234 % 256 = 210, p1 = 1234 // 256 = 4
    agent.set_inventory({"carbon": 1234})

    agent.set_action(Action(name="noop"))
    sim.step()

    # Check raw observation tokens
    obs_tokens = {}
    for token in agent.self_observation():
        if token.feature.name.startswith("inv:"):
            obs_tokens[token.feature.name] = token.value

    # Should have inv:carbon = 210 (1234 % 256) and inv:carbon:p1 = 4 (1234 // 256)
    assert "inv:carbon" in obs_tokens, f"Missing inv:carbon token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon"] == 210, f"Expected inv:carbon=210, got {obs_tokens['inv:carbon']}"
    assert "inv:carbon:p1" in obs_tokens, f"Missing inv:carbon:p1 token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon:p1"] == 4, f"Expected inv:carbon:p1=4, got {obs_tokens['inv:carbon:p1']}"


def test_inventory_observation_tokens_for_max_values(sim_with_large_inventory_limit: Simulation):
    """Test that inventory observations emit correct tokens for max uint16 values."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set carbon to 65535 (max uint16_t): with base 256
    # 65535 = 255 + 255*256
    # base = 65535 % 256 = 255, p1 = 65535 // 256 = 255
    agent.set_inventory({"carbon": 65535})

    agent.set_action(Action(name="noop"))
    sim.step()

    # Check raw observation tokens
    obs_tokens = {}
    for token in agent.self_observation():
        if token.feature.name.startswith("inv:"):
            obs_tokens[token.feature.name] = token.value

    # Should have two tokens for max value with base 256
    assert "inv:carbon" in obs_tokens, f"Missing inv:carbon token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon"] == 255, f"Expected inv:carbon=255, got {obs_tokens['inv:carbon']}"
    assert "inv:carbon:p1" in obs_tokens, f"Missing inv:carbon:p1 token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon:p1"] == 255, f"Expected inv:carbon:p1=255, got {obs_tokens['inv:carbon:p1']}"
    # p2 should not be emitted with base 256 for max uint16
    assert "inv:carbon:p2" not in obs_tokens, "Unexpected inv:carbon:p2 token for base 256"


def test_inventory_no_power_token_for_small_values(sim_with_large_inventory_limit: Simulation):
    """Test that values < token_value_base don't emit power tokens."""
    sim = sim_with_large_inventory_limit

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set carbon to 42 (< 256, should not have power tokens)
    agent.set_inventory({"carbon": 42})

    agent.set_action(Action(name="noop"))
    sim.step()

    # Check raw observation tokens
    obs_tokens = {}
    for token in agent.self_observation():
        if token.feature.name.startswith("inv:"):
            obs_tokens[token.feature.name] = token.value

    # Should have inv:carbon = 42, but NOT power tokens
    assert "inv:carbon" in obs_tokens, "Missing inv:carbon token"
    assert obs_tokens["inv:carbon"] == 42, f"Expected inv:carbon=42, got {obs_tokens['inv:carbon']}"
    assert "inv:carbon:p1" not in obs_tokens, "Unexpected inv:carbon:p1 token for value < 256"
    assert "inv:carbon:p2" not in obs_tokens, "Unexpected inv:carbon:p2 token for value < 256"


@pytest.fixture
def sim_with_token_value_base_100() -> Simulation:
    """Simulation with token_value_base=100 for base-100 encoding tests."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=64, token_value_base=100),
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            resource_names=["carbon", "gold"],
            agent=AgentConfig(inventory=InventoryConfig(default_limit=65535)),
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=7),
        )
    )
    return Simulation(cfg)


def test_token_value_base_100_encoding(sim_with_token_value_base_100: Simulation):
    """Test that token_value_base=100 produces correct base-100 encoding."""
    sim = sim_with_token_value_base_100

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set carbon to 1234: with base 100, 1234 = 34 + 12*100
    agent.set_inventory({"carbon": 1234})

    agent.set_action(Action(name="noop"))
    sim.step()

    # Check raw observation tokens
    obs_tokens = {}
    for token in agent.self_observation():
        if token.feature.name.startswith("inv:"):
            obs_tokens[token.feature.name] = token.value

    # Should have inv:carbon = 34 (1234 % 100) and inv:carbon:p1 = 12 (1234 // 100)
    assert "inv:carbon" in obs_tokens, f"Missing inv:carbon token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon"] == 34, f"Expected inv:carbon=34, got {obs_tokens['inv:carbon']}"
    assert "inv:carbon:p1" in obs_tokens, f"Missing inv:carbon:p1 token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon:p1"] == 12, f"Expected inv:carbon:p1=12, got {obs_tokens['inv:carbon:p1']}"

    # Verify inventory property reconstructs correctly
    inv = agent.inventory
    assert inv.get("carbon", 0) == 1234, f"Expected carbon=1234, got {inv.get('carbon', 0)}"


def test_token_value_base_100_large_value(sim_with_token_value_base_100: Simulation):
    """Test that token_value_base=100 handles large values correctly."""
    sim = sim_with_token_value_base_100

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set carbon to 54321: with base 100
    # 54321 = 21 + 43*100 + 5*10000
    agent.set_inventory({"carbon": 54321})

    agent.set_action(Action(name="noop"))
    sim.step()

    # Check raw observation tokens
    obs_tokens = {}
    for token in agent.self_observation():
        if token.feature.name.startswith("inv:"):
            obs_tokens[token.feature.name] = token.value

    assert "inv:carbon" in obs_tokens, f"Missing inv:carbon token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon"] == 21, f"Expected inv:carbon=21, got {obs_tokens['inv:carbon']}"
    assert "inv:carbon:p1" in obs_tokens, f"Missing inv:carbon:p1 token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon:p1"] == 43, f"Expected inv:carbon:p1=43, got {obs_tokens['inv:carbon:p1']}"
    assert "inv:carbon:p2" in obs_tokens, f"Missing inv:carbon:p2 token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon:p2"] == 5, f"Expected inv:carbon:p2=5, got {obs_tokens['inv:carbon:p2']}"

    # Verify inventory property reconstructs correctly
    inv = agent.inventory
    assert inv.get("carbon", 0) == 54321, f"Expected carbon=54321, got {inv.get('carbon', 0)}"


def test_token_value_base_256_encoding():
    """Test that token_value_base=256 (default) produces correct base-256 encoding."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=64),
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            resource_names=["carbon"],
            agent=AgentConfig(inventory=InventoryConfig(default_limit=65535)),
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=7),
        )
    )
    sim = Simulation(cfg)

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Set carbon to 1234: with base 256, 1234 = 210 + 4*256
    agent.set_inventory({"carbon": 1234})

    agent.set_action(Action(name="noop"))
    sim.step()

    # Check raw observation tokens
    obs_tokens = {}
    for token in agent.self_observation():
        if token.feature.name.startswith("inv:"):
            obs_tokens[token.feature.name] = token.value

    assert "inv:carbon" in obs_tokens, f"Missing inv:carbon token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon"] == 210, f"Expected inv:carbon=210, got {obs_tokens['inv:carbon']}"
    assert "inv:carbon:p1" in obs_tokens, f"Missing inv:carbon:p1 token, got: {obs_tokens}"
    assert obs_tokens["inv:carbon:p1"] == 4, f"Expected inv:carbon:p1=4, got {obs_tokens['inv:carbon:p1']}"

    # Verify inventory property reconstructs correctly
    inv = agent.inventory
    assert inv.get("carbon", 0) == 1234, f"Expected carbon=1234, got {inv.get('carbon', 0)}"
