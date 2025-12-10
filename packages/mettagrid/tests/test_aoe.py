"""Tests for Area of Effect (AOE) on grid objects.

Any grid object can have an AOE effect config that emits resource effects in a radius.
Agents standing within the radius receive the resource_deltas each tick.
"""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AOEEffectConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


def create_aoe_test_sim(
    aoe_range: int = 1,
    resource_deltas: dict[str, int] | None = None,
    initial_inventory: dict[str, int] | None = None,
) -> Simulation:
    """Create a simulation for testing AOE effects."""
    # Map with agent and wall with AOE effect
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "aoe_emitter", "empty", "wall"],
        ["wall", "agent.agent", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = GameConfig(
        max_steps=50,
        num_agents=1,
        obs=ObsConfig(width=5, height=5, num_tokens=100),
        resource_names=["energy", "health"],
        actions=ActionsConfig(
            noop=NoopActionConfig(),
            move=MoveActionConfig(enabled=True),
        ),
        agent=AgentConfig(
            rewards=AgentRewards(),
            initial_inventory=initial_inventory or {"energy": 50, "health": 100},
        ),
        objects={
            "wall": WallConfig(),
            "aoe_emitter": WallConfig(
                name="aoe_emitter",
                aoe=AOEEffectConfig(
                    range=aoe_range,
                    resource_deltas=resource_deltas or {"health": 1},
                ),
            ),
        },
    )

    cfg = MettaGridConfig(game=game_config)
    cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

    return Simulation(cfg, seed=42)


def get_agent(sim: Simulation) -> dict:
    """Get the agent object from the simulation."""
    objects = sim.grid_objects()
    agents = [obj for obj in objects.values() if "agent_id" in obj]
    return agents[0] if agents else {}


class TestAOEEffects:
    """Test AOE effects on agents."""

    def test_aoe_applies_resource_delta(self):
        """Test that AOE applies resource delta to agents in range."""
        sim = create_aoe_test_sim(
            aoe_range=2,
            resource_deltas={"health": 2},
            initial_inventory={"energy": 50, "health": 100},
        )

        health_idx = sim.resource_names.index("health")
        agent = get_agent(sim)
        initial_health = agent["inventory"][health_idx]

        # Step 5 times
        for _ in range(5):
            sim.step()

        agent = get_agent(sim)
        final_health = agent["inventory"][health_idx]

        # Agent should gain 2 health per step
        assert final_health == initial_health + 10, (
            f"Health should increase by 10 (2 per step * 5 steps). Was {initial_health}, now {final_health}"
        )

    def test_aoe_drains_resource(self):
        """Test that AOE can drain resources (negative delta)."""
        sim = create_aoe_test_sim(
            aoe_range=2,
            resource_deltas={"energy": -3},
            initial_inventory={"energy": 50, "health": 100},
        )

        energy_idx = sim.resource_names.index("energy")
        agent = get_agent(sim)
        initial_energy = agent["inventory"][energy_idx]

        # Step 5 times
        for _ in range(5):
            sim.step()

        agent = get_agent(sim)
        final_energy = agent["inventory"][energy_idx]

        # Agent should lose 3 energy per step
        assert final_energy == initial_energy - 15, (
            f"Energy should decrease by 15 (3 per step * 5 steps). Was {initial_energy}, now {final_energy}"
        )

    def test_aoe_multiple_resources(self):
        """Test that AOE can affect multiple resources."""
        sim = create_aoe_test_sim(
            aoe_range=2,
            resource_deltas={"health": 1, "energy": -1},
            initial_inventory={"energy": 50, "health": 100},
        )

        energy_idx = sim.resource_names.index("energy")
        health_idx = sim.resource_names.index("health")
        agent = get_agent(sim)
        initial_energy = agent["inventory"][energy_idx]
        initial_health = agent["inventory"][health_idx]

        # Step 5 times
        for _ in range(5):
            sim.step()

        agent = get_agent(sim)
        final_energy = agent["inventory"][energy_idx]
        final_health = agent["inventory"][health_idx]

        # Agent should gain health and lose energy
        assert final_health == initial_health + 5, (
            f"Health should increase by 5. Was {initial_health}, now {final_health}"
        )
        assert final_energy == initial_energy - 5, (
            f"Energy should decrease by 5. Was {initial_energy}, now {final_energy}"
        )

    def test_aoe_out_of_range_no_effect(self):
        """Test that agents outside AOE range don't receive effects."""
        # Create map with agent far from AOE
        game_map = [
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
            ["wall", "aoe_emitter", "empty", "empty", "empty", "agent.agent", "wall"],
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=100),
            resource_names=["energy", "health"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(),
                initial_inventory={"energy": 50, "health": 100},
            ),
            objects={
                "wall": WallConfig(),
                "aoe_emitter": WallConfig(
                    name="aoe_emitter",
                    aoe=AOEEffectConfig(
                        range=1,  # Small range - agent is at distance 4
                        resource_deltas={"health": 10},
                    ),
                ),
            },
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(cfg, seed=42)

        health_idx = sim.resource_names.index("health")
        agent = get_agent(sim)
        initial_health = agent["inventory"][health_idx]

        # Step 5 times
        for _ in range(5):
            sim.step()

        agent = get_agent(sim)
        final_health = agent["inventory"][health_idx]

        # Agent should not gain any health (out of range)
        assert final_health == initial_health, (
            f"Health should not change (agent out of range). Was {initial_health}, now {final_health}"
        )


class TestAOEStats:
    """Test AOE statistics tracking."""

    def test_aoe_stats_gained(self):
        """Test that AOE stats are tracked for resource gains."""
        sim = create_aoe_test_sim(
            aoe_range=2,
            resource_deltas={"health": 2},
            initial_inventory={"energy": 50, "health": 100},
        )

        # Step 5 times
        for _ in range(5):
            sim.step()

        stats = sim.episode_stats
        game_stats = stats["game"]
        agent_stats = stats["agent"][0]

        # Game stats should track gained health
        assert game_stats.get("aoe.health.gained", 0) == 10  # 2 per step * 5 steps
        assert game_stats.get("aoe.health.delta", 0) == 10
        assert game_stats.get("aoe.health.lost", 0) == 0

        # Agent stats should also track gained health
        assert agent_stats.get("aoe.health.gained", 0) == 10
        assert agent_stats.get("aoe.health.delta", 0) == 10

    def test_aoe_stats_lost(self):
        """Test that AOE stats are tracked for resource losses."""
        sim = create_aoe_test_sim(
            aoe_range=2,
            resource_deltas={"energy": -3},
            initial_inventory={"energy": 50, "health": 100},
        )

        # Step 5 times
        for _ in range(5):
            sim.step()

        stats = sim.episode_stats
        game_stats = stats["game"]
        agent_stats = stats["agent"][0]

        # Game stats should track lost energy
        assert game_stats.get("aoe.energy.lost", 0) == 15  # 3 per step * 5 steps
        assert game_stats.get("aoe.energy.delta", 0) == -15
        assert game_stats.get("aoe.energy.gained", 0) == 0

        # Agent stats should also track lost energy
        assert agent_stats.get("aoe.energy.lost", 0) == 15
        assert agent_stats.get("aoe.energy.delta", 0) == -15


class TestAOEEffectConfig:
    """Test AOE effect configuration."""

    def test_aoe_effect_config_defaults(self):
        """Test that AOEEffectConfig has correct defaults."""
        aoe = AOEEffectConfig()
        assert aoe.range == 1
        assert aoe.resource_deltas == {}

    def test_aoe_effect_config_custom_values(self):
        """Test that AOEEffectConfig accepts custom values."""
        aoe = AOEEffectConfig(
            range=3,
            resource_deltas={"health": 5, "energy": -2},
        )
        assert aoe.range == 3
        assert aoe.resource_deltas == {"health": 5, "energy": -2}
