"""Test global observation configuration functionality."""

import numpy as np

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    GameConfig,
    GlobalObsConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    WallConfig,
)
from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.simulator import Action, Simulation


class ObjectNameMapBuilder(MapBuilder):
    """Map builder that uses pre-built object name maps."""

    class Config(MapBuilderConfig["ObjectNameMapBuilder"]):
        map_data: list[list[str]]

    def __init__(self, config: Config):
        self.config = config

    def build(self) -> GameMap:
        return GameMap(grid=np.array(self.config.map_data))


def create_test_sim(global_obs_config: dict[str, bool]) -> Simulation:
    """Create test simulation with specified global_obs configuration."""
    game_config = GameConfig(
        num_agents=2,
        obs_width=11,
        obs_height=11,
        num_observation_tokens=100,
        max_steps=100,
        resource_names=["item1", "item2"],
        global_obs=GlobalObsConfig(**global_obs_config),
        agent=AgentConfig(
            default_resource_limit=10, freeze_duration=0, rewards=AgentRewards(), action_failure_penalty=0
        ),
        actions=ActionsConfig(noop=NoopActionConfig(enabled=True), move=MoveActionConfig(enabled=True)),
        objects={"wall": WallConfig(swappable=False)},
    )

    game_map = [
        ["wall", "wall", "wall", "wall"],
        ["wall", "agent.agent", "agent.agent", "wall"],
        ["wall", "wall", "wall", "wall"],
    ]

    cfg = MettaGridConfig(game=game_config)
    cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

    sim = Simulation(cfg, seed=42)

    # Step once to populate observations
    for i in range(sim.num_agents):
        sim.agent(i).set_action(Action(name="noop"))
    sim.step()

    return sim


def test_all_global_tokens_enabled():
    """Test that all global tokens are present when enabled."""
    global_obs = {"episode_completion_pct": True, "last_action": True, "last_reward": True}

    sim = create_test_sim(global_obs)

    # Check both agents have all global observation tokens
    for i in range(sim.num_agents):
        agent = sim.agent(i)
        global_obs_data = agent.global_observations

        assert "episode_completion_pct" in global_obs_data, "Should have episode_completion_pct"
        assert "last_action" in global_obs_data, "Should have last_action"
        assert "last_reward" in global_obs_data, "Should have last_reward"


def test_episode_completion_disabled():
    """Test that episode completion token is not present when disabled."""
    global_obs = {"episode_completion_pct": False, "last_action": True, "last_reward": True}

    sim = create_test_sim(global_obs)

    # Check that agents have last_action and last_reward but NOT episode_completion_pct
    for i in range(sim.num_agents):
        agent = sim.agent(i)
        global_obs_data = agent.global_observations

        assert "episode_completion_pct" not in global_obs_data, "Should NOT have episode_completion_pct when disabled"
        assert "last_action" in global_obs_data, "Should have last_action"
        assert "last_reward" in global_obs_data, "Should have last_reward"


def test_last_action_disabled():
    """Test that last action tokens are not present when disabled."""
    global_obs = {"episode_completion_pct": True, "last_action": False, "last_reward": True}

    sim = create_test_sim(global_obs)

    # Check that agents have episode_completion_pct and last_reward but NOT last_action
    for i in range(sim.num_agents):
        agent = sim.agent(i)
        global_obs_data = agent.global_observations

        assert "episode_completion_pct" in global_obs_data, "Should have episode_completion_pct"
        assert "last_action" not in global_obs_data, "Should NOT have last_action when disabled"
        assert "last_reward" in global_obs_data, "Should have last_reward"


def test_all_global_tokens_disabled():
    """Test that no global tokens are present when all disabled."""
    global_obs = {"episode_completion_pct": False, "last_action": False, "last_reward": False}

    sim = create_test_sim(global_obs)

    # Check that agents have NO global observation tokens
    for i in range(sim.num_agents):
        agent = sim.agent(i)
        global_obs_data = agent.global_observations

        assert "episode_completion_pct" not in global_obs_data, "Should NOT have episode_completion_pct"
        assert "last_action" not in global_obs_data, "Should NOT have last_action"
        assert "last_reward" not in global_obs_data, "Should NOT have last_reward"

        # Global obs dict should be empty or only have other non-global tokens
        assert len(global_obs_data) == 0, f"Should have no global tokens, got {list(global_obs_data.keys())}"


def test_global_obs_default_values():
    """Test that global_obs uses default values when not specified."""
    # Test with no global_obs specified - should use defaults (all True)
    game_config = GameConfig(
        num_agents=1,
        obs_width=11,
        obs_height=11,
        num_observation_tokens=100,
        max_steps=100,
        resource_names=["item1"],
        # No global_obs specified - should use defaults
        agent=AgentConfig(
            default_resource_limit=10, freeze_duration=0, rewards=AgentRewards(), action_failure_penalty=0
        ),
        actions=ActionsConfig(noop=NoopActionConfig(enabled=True)),
        objects={"wall": WallConfig(swappable=False)},
    )

    game_map = [["agent.agent"]]

    cfg = MettaGridConfig(game=game_config)
    cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

    sim = Simulation(cfg, seed=42)

    # Step once to populate observations
    sim.agent(0).set_action(Action(name="noop"))
    sim.step()

    # Should have all global tokens by default
    agent = sim.agent(0)
    global_obs_data = agent.global_observations

    assert "episode_completion_pct" in global_obs_data, "Should have episode_completion_pct by default"
    assert "last_action" in global_obs_data, "Should have last_action by default"
    assert "last_reward" in global_obs_data, "Should have last_reward by default"
