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
            f"Chest should remain unaligned when align not configured, got commons_id={objs['chests'][0].get('commons_id')}"
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
            f"Chest should remain aligned when scramble not configured, got commons_id={objs['chests'][0].get('commons_id')}"
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

