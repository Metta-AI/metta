"""Tests for align and scramble actions.

Align sets a target's commons to the actor's commons.
Scramble clears a target's commons (sets to none).
Both are triggered by vibes on move.
"""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AlignActionConfig,
    ChangeVibeActionConfig,
    CommonsChestConfig,
    CommonsConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder

# Skip all tests in this module if C++ doesn't support align action
try:
    from mettagrid.mettagrid_c import AlignActionConfig as _  # noqa: F401

    HAS_ALIGN = True
except ImportError:
    HAS_ALIGN = False

pytestmark = pytest.mark.skipif(not HAS_ALIGN, reason="Align action not available in C++ bindings")

HEART_VIBE_NAME = "heart_a"
SWORDS_VIBE_NAME = "swords"


class TestAlignAction:
    """Test align action that sets target's commons to actor's commons."""

    def _create_agent_chest_sim(self, align: bool = True) -> Simulation:
        """Create a simulation with an agent and a CommonsChest for testing alignment."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
                align=AlignActionConfig(
                    vibe=HEART_VIBE_NAME,
                )
                if align
                else None,
            ),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(),
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": CommonsChestConfig(
                    name="chest",
                    # No commons initially - the chest is unaligned
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(
                    name="cogs",
                    inventory=InventoryConfig(initial={"energy": 0, "heart": 0}),
                ),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        return Simulation(cfg, seed=42)

    def _get_objects_by_type(self, sim: Simulation) -> dict:
        """Helper to get objects organized by type."""
        objects = sim.grid_objects()
        result = {"agents": [], "chests": []}
        for obj in objects.values():
            if "agent_id" in obj:
                result["agents"].append(obj)
            elif obj.get("type_name") == "chest":
                result["chests"].append(obj)
        return result

    def test_align_chest_to_agent_commons_with_heart_vibe(self):
        """Test that moving onto a CommonsChest with heart vibe aligns it to agent's commons."""
        sim = self._create_agent_chest_sim(align=True)

        # Initial state: agent has commons, chest doesn't
        objs = self._get_objects_by_type(sim)
        assert objs["agents"][0].get("commons_id") == 0, "Agent should start with commons_id=0"
        assert objs["chests"][0].get("commons_id") is None, "Chest should start unaligned"

        # Agent changes vibe to heart_a
        sim.agent(0).set_action("change_vibe_heart_a")
        sim.step()

        # Verify vibe is set
        objs = self._get_objects_by_type(sim)
        assert objs["agents"][0]["vibe"] == 10, (
            f"Agent should have heart_a vibe (id=10), got {objs['agents'][0]['vibe']}"
        )

        # Agent moves east (to empty space)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Agent moves east again onto the chest - should trigger alignment
        sim.agent(0).set_action("move_east")
        sim.step()

        # Check that chest is now aligned to agent's commons
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") == 0, (
            f"Chest should be aligned to agent's commons (id=0), got {objs['chests'][0].get('commons_id')}"
        )

    def test_no_alignment_without_heart_vibe(self):
        """Test that moving onto a CommonsChest without heart vibe does NOT align it."""
        sim = self._create_agent_chest_sim(align=True)

        # Initial state
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") is None, "Chest should start unaligned"

        # Agent moves east (to empty space) - keeping default vibe
        sim.agent(0).set_action("move_east")
        sim.step()

        # Agent moves east again onto the chest - should NOT trigger alignment (wrong vibe)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Chest should still be unaligned
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") is None, (
            f"Chest should remain unaligned without heart vibe, got commons_id={objs['chests'][0].get('commons_id')}"
        )

    def test_no_alignment_when_align_disabled(self):
        """Test that alignment doesn't happen when align is not configured."""
        sim = self._create_agent_chest_sim(align=False)

        # Initial state
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") is None, "Chest should start unaligned"

        # Agent changes vibe to heart_a
        sim.agent(0).set_action("change_vibe_heart_a")
        sim.step()

        # Agent moves east (to empty space)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Agent moves east again onto the chest - should NOT align (no align config)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Chest should still be unaligned
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") is None, (
            "Chest should remain unaligned when align not configured, got "
            + f"commons_id={objs['chests'][0].get('commons_id')}"
        )


class TestScrambleAction:
    """Test scramble action that clears commons alignment."""

    def _create_scramble_sim(self, scramble: bool = True) -> Simulation:
        """Create a simulation with an agent and an aligned CommonsChest for testing scramble."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
                scramble=AlignActionConfig(
                    vibe=SWORDS_VIBE_NAME,
                    set_to_none=True,
                )
                if scramble
                else None,
            ),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(),
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": CommonsChestConfig(
                    name="chest",
                    # Chest starts aligned to cogs commons
                    commons="cogs",
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(
                    name="cogs",
                    inventory=InventoryConfig(initial={"energy": 0, "heart": 0}),
                ),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        return Simulation(cfg, seed=42)

    def _get_objects_by_type(self, sim: Simulation) -> dict:
        """Helper to get objects organized by type."""
        objects = sim.grid_objects()
        result = {"agents": [], "chests": []}
        for obj in objects.values():
            if "agent_id" in obj:
                result["agents"].append(obj)
            elif obj.get("type_name") == "chest":
                result["chests"].append(obj)
        return result

    def test_scramble_clears_chest_commons(self):
        """Test that moving onto an aligned CommonsChest with swords vibe clears its commons."""
        sim = self._create_scramble_sim(scramble=True)

        # Initial state: both agent and chest have commons
        objs = self._get_objects_by_type(sim)
        assert objs["agents"][0].get("commons_id") == 0, "Agent should start with commons_id=0"
        assert objs["chests"][0].get("commons_id") == 0, "Chest should start aligned to commons_id=0"

        # Agent changes vibe to swords
        sim.agent(0).set_action("change_vibe_swords")
        sim.step()

        # Agent moves east (to empty space)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Agent moves east again onto the chest - should trigger scramble
        sim.agent(0).set_action("move_east")
        sim.step()

        # Check that chest's commons is now cleared (None)
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") is None, (
            f"Chest should have no commons after scramble, got commons_id={objs['chests'][0].get('commons_id')}"
        )

    def test_no_scramble_without_swords_vibe(self):
        """Test that moving onto a CommonsChest without swords vibe does NOT scramble it."""
        sim = self._create_scramble_sim(scramble=True)

        # Initial state
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") == 0, "Chest should start aligned"

        # Agent moves east (to empty space) - keeping default vibe
        sim.agent(0).set_action("move_east")
        sim.step()

        # Agent moves east again onto the chest - should NOT trigger scramble (wrong vibe)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Chest should still be aligned
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") == 0, (
            f"Chest should remain aligned without swords vibe, got commons_id={objs['chests'][0].get('commons_id')}"
        )

    def test_no_scramble_when_scramble_disabled(self):
        """Test that scramble doesn't happen when scramble is not configured."""
        sim = self._create_scramble_sim(scramble=False)

        # Initial state
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") == 0, "Chest should start aligned"

        # Agent changes vibe to swords
        sim.agent(0).set_action("change_vibe_swords")
        sim.step()

        # Agent moves east (to empty space)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Agent moves east again onto the chest - should NOT scramble (no scramble config)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Chest should still be aligned
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") == 0, (
            "Chest should remain aligned when scramble not configured, got "
            + f"commons_id={objs['chests'][0].get('commons_id')}"
        )

    def test_no_scramble_on_unaligned_chest(self):
        """Test that scramble does nothing on a chest that has no commons."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
                scramble=AlignActionConfig(
                    vibe=SWORDS_VIBE_NAME,
                    set_to_none=True,
                ),
            ),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(),
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": CommonsChestConfig(
                    name="chest",
                    # Chest starts unaligned (no commons)
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(
                    name="cogs",
                    inventory=InventoryConfig(initial={"energy": 0, "heart": 0}),
                ),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        sim = Simulation(cfg, seed=42)

        # Initial state: chest has no commons
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") is None, "Chest should start unaligned"

        # Agent changes vibe to swords
        sim.agent(0).set_action("change_vibe_swords")
        sim.step()

        # Agent moves east (to empty space)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Agent moves east again onto the chest - scramble should do nothing (already no commons)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Chest should still have no commons
        objs = self._get_objects_by_type(sim)
        assert objs["chests"][0].get("commons_id") is None, (
            f"Chest should remain unaligned, got commons_id={objs['chests'][0].get('commons_id')}"
        )


class TestCommonsStats:
    """Test commons stats tracking for aligned objects."""

    def _create_sim_with_aligned_chest(self) -> Simulation:
        """Create a simulation with an agent and a CommonsChest aligned to the same commons."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
                align=AlignActionConfig(vibe=HEART_VIBE_NAME),
            ),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(),
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": CommonsChestConfig(
                    name="chest",
                    commons="cogs",  # Start aligned to cogs
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(
                    name="cogs",
                    inventory=InventoryConfig(initial={"energy": 50, "heart": 100}),
                ),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        return Simulation(cfg, seed=42)

    def test_commons_stats_aligned_count(self):
        """Test that commons stats track aligned object counts."""
        sim = self._create_sim_with_aligned_chest()

        # Run a few steps
        for _ in range(5):
            sim.agent(0).set_action("noop")
            sim.step()

        # Get episode stats
        stats = sim.episode_stats

        # Should have commons stats
        assert "commons" in stats, "Episode stats should include 'commons'"
        commons_stats = stats["commons"]

        # Should have stats for 'cogs' commons
        assert "cogs" in commons_stats, f"Should have stats for 'cogs', got {list(commons_stats.keys())}"
        cogs_stats = commons_stats["cogs"]

        # Should track aligned chest count (agent + chest = 2 alignable objects, but type-specific)
        # The chest is type "chest", the agent is type "agent"
        assert "aligned.chest" in cogs_stats, f"Should track aligned.chest, got {list(cogs_stats.keys())}"
        assert cogs_stats["aligned.chest"] == 1, f"Should have 1 aligned chest, got {cogs_stats['aligned.chest']}"

        assert "aligned.agent" in cogs_stats, f"Should track aligned.agent, got {list(cogs_stats.keys())}"
        assert cogs_stats["aligned.agent"] == 1, f"Should have 1 aligned agent, got {cogs_stats['aligned.agent']}"

    def test_commons_stats_held_duration(self):
        """Test that commons stats track held duration (ticks)."""
        sim = self._create_sim_with_aligned_chest()

        # Run 10 steps
        num_steps = 10
        for _ in range(num_steps):
            sim.agent(0).set_action("noop")
            sim.step()

        stats = sim.episode_stats
        commons_stats = stats["commons"]["cogs"]

        # Held duration should accumulate per tick (count * ticks)
        # Each step, we have 1 chest aligned, so held should be roughly num_steps
        assert "aligned.chest.held" in commons_stats, (
            f"Should track aligned.chest.held, got {list(commons_stats.keys())}Stats: {commons_stats}"
        )
        assert commons_stats["aligned.chest.held"] == num_steps, (
            f"Should have {num_steps} held ticks for chest, got {commons_stats['aligned.chest.held']}"
        )

    def test_commons_stats_inventory(self):
        """Test that commons stats include inventory."""
        sim = self._create_sim_with_aligned_chest()

        # Run a step to ensure stats are computed
        sim.agent(0).set_action("noop")
        sim.step()

        stats = sim.episode_stats
        commons_stats = stats["commons"]["cogs"]

        # Should track inventory
        assert "inventory.heart" in commons_stats, f"Should track inventory.heart, got {list(commons_stats.keys())}"
        # Initial inventory was 100 hearts
        assert commons_stats["inventory.heart"] == 100, (
            f"Should have 100 hearts in inventory, got {commons_stats['inventory.heart']}"
        )


class TestCommonsStatsRewards:
    """Test commons_stats rewards that reward agents based on commons stats."""

    def _create_sim_with_aligned_chest_and_rewards(self) -> Simulation:
        """Create a simulation with an agent rewarded for aligned chest hold time."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
                align=AlignActionConfig(vibe=HEART_VIBE_NAME),
            ),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(
                    commons_stats={
                        "aligned.chest.held": 0.1,  # Reward 0.1 per tick per aligned chest
                    },
                ),
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": CommonsChestConfig(
                    name="chest",
                    commons="cogs",  # Start aligned to cogs
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(
                    name="cogs",
                    inventory=InventoryConfig(initial={"energy": 50, "heart": 100}),
                ),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        return Simulation(cfg, seed=42)

    def test_commons_stats_reward_for_held_duration(self):
        """Test that agents receive rewards based on commons stats like aligned.chest.held."""
        sim = self._create_sim_with_aligned_chest_and_rewards()

        # Run 10 steps
        num_steps = 10
        for _ in range(num_steps):
            sim.agent(0).set_action("noop")
            sim.step()

        # Get episode reward
        episode_rewards = sim.episode_rewards
        agent_reward = episode_rewards[0]

        # Expected reward: 0.1 * 10 ticks = 1.0 (one chest aligned for 10 ticks)
        # The reward is computed as stat_value * reward_per_unit
        # aligned.chest.held accumulates: 1 chest * 10 ticks = 10
        # So reward should be 10 * 0.1 = 1.0
        expected_reward = num_steps * 0.1
        assert abs(agent_reward - expected_reward) < 0.01, f"Expected reward ~{expected_reward}, got {agent_reward}"

    def test_commons_stats_reward_for_aligned_count(self):
        """Test that agents can be rewarded based on aligned object count."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
            ),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(
                    commons_stats={
                        "aligned.chest": 5.0,  # Reward 5.0 per aligned chest
                    },
                ),
                inventory=InventoryConfig(initial={"energy": 100}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": CommonsChestConfig(
                    name="chest",
                    commons="cogs",
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(name="cogs"),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(cfg, seed=42)

        # Run a step
        sim.agent(0).set_action("noop")
        sim.step()

        # Should get reward for 1 aligned chest
        agent_reward = sim.episode_rewards[0]
        assert abs(agent_reward - 5.0) < 0.01, f"Expected reward ~5.0, got {agent_reward}"

    def test_no_reward_when_agent_has_no_commons(self):
        """Test that agents without commons don't get commons_stats rewards."""
        game_map = [
            ["wall", "wall", "wall"],
            ["wall", "agent.agent", "wall"],
            ["wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy"],
            actions=ActionsConfig(noop=NoopActionConfig()),
            agent=AgentConfig(
                # No commons assigned
                rewards=AgentRewards(
                    commons_stats={
                        "aligned.chest.held": 100.0,  # Would be big reward if it worked
                    },
                ),
                inventory=InventoryConfig(initial={"energy": 100}),
            ),
            objects={
                "wall": WallConfig(),
            },
            commons=[
                CommonsConfig(name="cogs"),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(cfg, seed=42)

        # Run some steps
        for _ in range(5):
            sim.agent(0).set_action("noop")
            sim.step()

        # Should get no reward since agent has no commons
        agent_reward = sim.episode_rewards[0]
        assert agent_reward == 0.0, f"Expected no reward, got {agent_reward}"


class TestCommonsStatsAlignment:
    """Test commons stats updates when alignment changes."""

    def test_stats_update_on_align(self):
        """Test that commons stats update when a chest is aligned."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
                align=AlignActionConfig(vibe=HEART_VIBE_NAME),
            ),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(),
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": CommonsChestConfig(
                    name="chest",
                    # No commons initially - chest is unaligned
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(name="cogs"),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(cfg, seed=42)

        # Initially, chest is unaligned - check stats
        sim.agent(0).set_action("noop")
        sim.step()

        stats = sim.episode_stats
        cogs_stats = stats["commons"]["cogs"]
        # Should only have agent aligned (not chest)
        assert cogs_stats.get("aligned.chest", 0) == 0, "Chest should not be aligned initially"
        assert cogs_stats.get("aligned.agent", 0) == 1, "Agent should be aligned"

        # Change vibe to heart for alignment
        sim.agent(0).set_action("change_vibe_heart_a")
        sim.step()

        # Move east (empty space)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Move east onto chest - should trigger align
        sim.agent(0).set_action("move_east")
        sim.step()

        # Now chest should be aligned
        stats = sim.episode_stats
        cogs_stats = stats["commons"]["cogs"]
        assert cogs_stats.get("aligned.chest", 0) == 1, f"Chest should now be aligned, got {cogs_stats}"

    def test_stats_update_on_scramble(self):
        """Test that commons stats update when a chest is scrambled (unaligned)."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
                scramble=AlignActionConfig(vibe=SWORDS_VIBE_NAME, set_to_none=True),
            ),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(),
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": CommonsChestConfig(
                    name="chest",
                    commons="cogs",  # Start aligned to cogs
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(name="cogs"),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(cfg, seed=42)

        # Initially, chest is aligned
        sim.agent(0).set_action("noop")
        sim.step()

        stats = sim.episode_stats
        cogs_stats = stats["commons"]["cogs"]
        assert cogs_stats.get("aligned.chest", 0) == 1, "Chest should be aligned initially"

        # Change vibe to swords for scramble
        sim.agent(0).set_action("change_vibe_swords")
        sim.step()

        # Move east (empty space)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Move east onto chest - should trigger scramble
        sim.agent(0).set_action("move_east")
        sim.step()

        # Now chest should be unaligned
        stats = sim.episode_stats
        cogs_stats = stats["commons"]["cogs"]
        assert cogs_stats.get("aligned.chest", 0) == 0, f"Chest should now be unaligned, got {cogs_stats}"

    def test_held_accumulates_only_while_aligned(self):
        """Test that held duration only accumulates while object is aligned."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
                scramble=AlignActionConfig(vibe=SWORDS_VIBE_NAME, set_to_none=True),
            ),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(),
                inventory=InventoryConfig(initial={"energy": 100, "heart": 10}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": CommonsChestConfig(
                    name="chest",
                    commons="cogs",  # Start aligned
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(name="cogs"),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(cfg, seed=42)

        # Run 5 steps while aligned
        for _ in range(5):
            sim.agent(0).set_action("noop")
            sim.step()

        stats = sim.episode_stats
        held_after_5 = stats["commons"]["cogs"].get("aligned.chest.held", 0)
        assert held_after_5 == 5, f"Should have 5 held ticks, got {held_after_5}"

        # Scramble the chest
        sim.agent(0).set_action("change_vibe_swords")
        sim.step()
        sim.agent(0).set_action("move_east")
        sim.step()
        sim.agent(0).set_action("move_east")  # Onto chest
        sim.step()

        # Run 5 more steps while unaligned
        for _ in range(5):
            sim.agent(0).set_action("noop")
            sim.step()

        stats = sim.episode_stats
        # held should not have increased much (maybe +1 or +2 from the steps before scramble completed)
        # but definitely not +5 more
        held_final = stats["commons"]["cogs"].get("aligned.chest.held", 0)
        assert held_final < held_after_5 + 5, (
            f"Held should not accumulate while unaligned. Before: {held_after_5}, After: {held_final}"
        )


class TestMultipleCommonsStats:
    """Test stats tracking with multiple commons."""

    def test_stats_tracked_per_commons(self):
        """Test that each commons tracks its own stats independently."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "cogs_chest", ".", "clips_chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy"],
            actions=ActionsConfig(noop=NoopActionConfig()),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(),
                inventory=InventoryConfig(initial={"energy": 100}),
            ),
            objects={
                "wall": WallConfig(),
                "cogs_chest": CommonsChestConfig(
                    name="cogs_chest",
                    commons="cogs",
                    vibe_transfers={"default": {"energy": 100}},
                ),
                "clips_chest": CommonsChestConfig(
                    name="clips_chest",
                    commons="clips",
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(name="cogs"),
                CommonsConfig(name="clips"),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(cfg, seed=42)

        # Run some steps
        for _ in range(5):
            sim.agent(0).set_action("noop")
            sim.step()

        stats = sim.episode_stats

        # Check cogs stats
        cogs_stats = stats["commons"]["cogs"]
        assert cogs_stats.get("aligned.cogs_chest", 0) == 1, "Cogs should have 1 aligned cogs_chest"
        assert cogs_stats.get("aligned.clips_chest", 0) == 0, "Cogs should not have clips_chest"
        assert cogs_stats.get("aligned.agent", 0) == 1, "Cogs should have 1 aligned agent"

        # Check clips stats
        clips_stats = stats["commons"]["clips"]
        assert clips_stats.get("aligned.clips_chest", 0) == 1, "Clips should have 1 aligned clips_chest"
        assert clips_stats.get("aligned.cogs_chest", 0) == 0, "Clips should not have cogs_chest"
        assert clips_stats.get("aligned.agent", 0) == 0, "Clips should not have aligned agent"

    def test_held_tracked_per_commons(self):
        """Test that held duration is tracked per commons."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "cogs_chest", ".", "clips_chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ]

        game_config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy"],
            actions=ActionsConfig(noop=NoopActionConfig()),
            agent=AgentConfig(
                commons="cogs",
                rewards=AgentRewards(),
                inventory=InventoryConfig(initial={"energy": 100}),
            ),
            objects={
                "wall": WallConfig(),
                "cogs_chest": CommonsChestConfig(
                    name="cogs_chest",
                    commons="cogs",
                    vibe_transfers={"default": {"energy": 100}},
                ),
                "clips_chest": CommonsChestConfig(
                    name="clips_chest",
                    commons="clips",
                    vibe_transfers={"default": {"energy": 100}},
                ),
            },
            commons=[
                CommonsConfig(name="cogs"),
                CommonsConfig(name="clips"),
            ],
        )

        cfg = MettaGridConfig(game=game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(cfg, seed=42)

        num_steps = 10
        for _ in range(num_steps):
            sim.agent(0).set_action("noop")
            sim.step()

        stats = sim.episode_stats

        # Both should track their own held durations
        cogs_stats = stats["commons"]["cogs"]
        clips_stats = stats["commons"]["clips"]

        assert cogs_stats.get("aligned.cogs_chest.held", 0) == num_steps, (
            f"Cogs should have {num_steps} held ticks for cogs_chest"
        )
        assert clips_stats.get("aligned.clips_chest.held", 0) == num_steps, (
            f"Clips should have {num_steps} held ticks for clips_chest"
        )
