"""Tests for the new activation handler system.

This tests the data-driven target activation handler mechanism where:
- Objects can have activation_handlers configured via mettagrid_config
- Each handler has filters (VibeFilter, ResourceFilter) that must all pass
- Each handler has mutations (ResourceDelta, ResourceTransfer, Alignment, Freeze, Attack)
- Handlers are checked in registration order when an agent moves onto the target
"""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    ActivationHandler,
    AgentConfig,
    AgentRewards,
    AlignmentMutation,
    ChangeVibeActionConfig,
    ChestConfig,
    CommonsChestConfig,
    CommonsConfig,
    FreezeMutation,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    ResourceDeltaMutation,
    ResourceFilter,
    ResourceTransferMutation,
    VibeFilter,
    WallConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


def get_agent_position(sim: Simulation, agent_id: int) -> tuple | None:
    """Get an agent's position as (row, col)."""
    grid_objects = sim.grid_objects()
    for obj in grid_objects.values():
        if obj.get("agent_id") == agent_id:
            return (obj["r"], obj["c"])
    return None


def get_agent_frozen_status(sim: Simulation, agent_id: int) -> bool:
    """Check if an agent is frozen."""
    grid_objects = sim.grid_objects()
    for obj in grid_objects.values():
        if obj.get("agent_id") == agent_id:
            return obj.get("is_frozen", False)
    return False


def get_agent_inventory(sim: Simulation, agent_id: int) -> dict:
    """Get an agent's inventory as a dict mapping resource name to amount."""
    grid_objects = sim.grid_objects()
    for obj in grid_objects.values():
        if obj.get("agent_id") == agent_id:
            inv = obj.get("inventory", {})
            # inv is a dict mapping resource_id to amount
            if isinstance(inv, dict):
                return {sim.resource_names[res_id]: amount for res_id, amount in inv.items() if amount > 0}
            # Handle list format (legacy)
            return {sim.resource_names[i]: amount for i, amount in enumerate(inv) if amount > 0}
    return {}


def get_chest_inventory(sim: Simulation, chest_name: str = "chest") -> dict:
    """Get a chest's inventory as a dict mapping resource name to amount."""
    grid_objects = sim.grid_objects()
    for obj in grid_objects.values():
        if obj.get("type_name") == chest_name:
            inv = obj.get("inventory", {})
            # inv is a dict mapping resource_id to amount
            if isinstance(inv, dict):
                return {sim.resource_names[res_id]: amount for res_id, amount in inv.items() if amount > 0}
            # Handle list format (legacy)
            return {sim.resource_names[i]: amount for i, amount in enumerate(inv) if amount > 0}
    return {}


class TestVibeFilterActivation:
    """Test activation handlers with vibe-based filtering."""

    def test_vibe_filter_triggers_resource_delta(self):
        """Test that a vibe filter correctly gates a resource delta mutation."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        # Create a chest that gives energy when agent has charger vibe
        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 0, "heart": 5}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": ChestConfig(
                    name="chest",
                    activation_handlers=[
                        ActivationHandler(
                            name="charger_grant",
                            filters=[
                                VibeFilter(target="actor", vibe="charger"),
                            ],
                            mutations=[
                                ResourceDeltaMutation(target="actor", deltas={"energy": 50}),
                            ],
                        ),
                    ],
                ),
            },
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(mg_config, seed=42)

        # Verify initial state
        inv_before = get_agent_inventory(sim, 0)
        assert inv_before.get("energy", 0) == 0, "Agent should start with 0 energy"

        # First move to empty space
        sim.agent(0).set_action("move_east")
        sim.step()

        # Try to activate chest without charger vibe - should NOT trigger
        sim.agent(0).set_action("move_east")
        sim.step()

        inv_after = get_agent_inventory(sim, 0)
        assert inv_after.get("energy", 0) == 0, "Agent should still have 0 energy without charger vibe"

        # Now set charger vibe
        sim.agent(0).set_action("change_vibe_charger")
        sim.step()

        # Try to activate chest again - should trigger now
        sim.agent(0).set_action("move_east")
        sim.step()

        inv_after_vibe = get_agent_inventory(sim, 0)
        assert inv_after_vibe.get("energy", 0) == 50, "Agent should have 50 energy after charger vibe activation"


class TestResourceFilterActivation:
    """Test activation handlers with resource-based filtering."""

    def test_resource_filter_gates_activation(self):
        """Test that a resource filter correctly prevents activation when resources are insufficient."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        # Create a chest that requires 10 heart to use
        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 0, "heart": 5}),  # Only 5 heart, needs 10
            ),
            objects={
                "wall": WallConfig(),
                "chest": ChestConfig(
                    name="chest",
                    activation_handlers=[
                        ActivationHandler(
                            name="heart_consumer",
                            filters=[
                                ResourceFilter(target="actor", resources={"heart": 10}),
                            ],
                            mutations=[
                                ResourceDeltaMutation(target="actor", deltas={"energy": 100}),
                            ],
                        ),
                    ],
                ),
            },
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(mg_config, seed=42)

        # Move to empty space first
        sim.agent(0).set_action("move_east")
        sim.step()

        # Try to activate chest - should NOT trigger (insufficient heart)
        sim.agent(0).set_action("move_east")
        sim.step()

        inv_after = get_agent_inventory(sim, 0)
        assert inv_after.get("energy", 0) == 0, "Agent should not get energy (insufficient heart)"


class TestResourceTransferMutation:
    """Test ResourceTransfer mutations."""

    def test_resource_transfer_from_actor_to_target(self):
        """Test transferring resources from actor to target."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 100, "heart": 5}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": ChestConfig(
                    name="chest",
                    activation_handlers=[
                        ActivationHandler(
                            name="deposit",
                            filters=[
                                VibeFilter(target="actor", vibe="charger"),
                            ],
                            mutations=[
                                ResourceTransferMutation(
                                    from_target="actor",
                                    to_target="target",
                                    resources={"energy": 50},
                                ),
                            ],
                        ),
                    ],
                ),
            },
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(mg_config, seed=42)

        # Set charger vibe
        sim.agent(0).set_action("change_vibe_charger")
        sim.step()

        # Move to empty space
        sim.agent(0).set_action("move_east")
        sim.step()

        # Activate chest
        sim.agent(0).set_action("move_east")
        sim.step()

        agent_inv = get_agent_inventory(sim, 0)
        chest_inv = get_chest_inventory(sim)

        assert agent_inv.get("energy", 0) == 50, "Agent should have 50 energy after transfer"
        assert chest_inv.get("energy", 0) == 50, "Chest should have 50 energy after transfer"


class TestAlignmentMutation:
    """Test Alignment mutations using activation handlers."""

    def test_alignment_to_actor_commons(self):
        """Test aligning target to actor's commons."""
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", ".", "chest", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=50,
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=100),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(enabled=True),
                change_vibe=ChangeVibeActionConfig(enabled=True),
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
                    activation_handlers=[
                        ActivationHandler(
                            name="align",
                            filters=[
                                VibeFilter(target="actor", vibe="heart_a"),
                            ],
                            mutations=[
                                AlignmentMutation(target="target", align_to="actor_commons"),
                            ],
                        ),
                    ],
                ),
            },
            commons=[
                CommonsConfig(
                    name="cogs",
                    inventory=InventoryConfig(initial={"energy": 0, "heart": 0}),
                ),
            ],
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

        sim = Simulation(mg_config, seed=42)

        # Initial state: chest should be unaligned
        objs = sim.grid_objects()
        chests = [obj for obj in objs.values() if obj.get("type_name") == "chest"]
        assert len(chests) == 1
        assert chests[0].get("commons_id") is None, "Chest should start unaligned"

        # Agent changes vibe to heart_a
        sim.agent(0).set_action("change_vibe_heart_a")
        sim.step()

        # Agent moves east (to empty space)
        sim.agent(0).set_action("move_east")
        sim.step()

        # Agent moves east again onto the chest - should trigger alignment
        sim.agent(0).set_action("move_east")
        sim.step()

        # Check that chest is now aligned to agent's commons
        objs = sim.grid_objects()
        chests = [obj for obj in objs.values() if obj.get("type_name") == "chest"]
        assert chests[0].get("commons_id") == 0, "Chest should be aligned to agent's commons (id=0)"


class TestFreezeMutation:
    """Test Freeze mutations."""

    def test_freeze_target_agent(self):
        """Test freezing a target agent."""
        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "agent.blue", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=2,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(),
            ),
            objects={
                "wall": WallConfig(),
            },
            agents=[
                AgentConfig(
                    team_id=0,
                    inventory=InventoryConfig(initial={"energy": 10, "heart": 5}),
                    activation_handlers=[
                        ActivationHandler(
                            name="freeze_on_charger",
                            filters=[
                                VibeFilter(target="actor", vibe="charger"),
                            ],
                            mutations=[
                                FreezeMutation(target="target", duration=10),
                            ],
                        ),
                    ],
                ),
                AgentConfig(
                    team_id=1,
                    inventory=InventoryConfig(initial={"energy": 10, "heart": 5}),
                    activation_handlers=[
                        ActivationHandler(
                            name="freeze_on_charger",
                            filters=[
                                VibeFilter(target="actor", vibe="charger"),
                            ],
                            mutations=[
                                FreezeMutation(target="target", duration=10),
                            ],
                        ),
                    ],
                ),
            ],
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(mg_config, seed=42)

        # Verify initial state
        assert not get_agent_frozen_status(sim, 0), "Agent 0 should not start frozen"
        assert not get_agent_frozen_status(sim, 1), "Agent 1 should not start frozen"

        # Agent 0 changes to charger vibe
        sim.agent(0).set_action("change_vibe_charger")
        sim.agent(1).set_action("noop")
        sim.step()

        # Agent 0 moves into Agent 1 - should trigger freeze via activation handler
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Agent 1 should now be frozen
        assert get_agent_frozen_status(sim, 1), "Agent 1 should be frozen after activation"


class TestHandlerOrder:
    """Test that handlers are checked in registration order."""

    def test_first_handler_wins(self):
        """Test that handlers are checked in the order they are registered."""
        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", "chest", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 0, "heart": 5}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": ChestConfig(
                    name="chest",
                    activation_handlers=[
                        # First handler - gives 10 energy (should win)
                        ActivationHandler(
                            name="first_handler",
                            filters=[
                                VibeFilter(target="actor", vibe="charger"),
                            ],
                            mutations=[
                                ResourceDeltaMutation(target="actor", deltas={"energy": 10}),
                            ],
                        ),
                        # Second handler - gives 100 energy (should be skipped)
                        ActivationHandler(
                            name="second_handler",
                            filters=[
                                VibeFilter(target="actor", vibe="charger"),
                            ],
                            mutations=[
                                ResourceDeltaMutation(target="actor", deltas={"energy": 100}),
                            ],
                        ),
                    ],
                ),
            },
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(mg_config, seed=42)

        # Set charger vibe
        sim.agent(0).set_action("change_vibe_charger")
        sim.step()

        # Move onto chest
        sim.agent(0).set_action("move_east")
        sim.step()

        inv_after = get_agent_inventory(sim, 0)
        # Should get 10 from first handler (registration order)
        assert inv_after.get("energy", 0) == 10, "First handler should trigger (registration order)"


class TestCombinedFilters:
    """Test handlers with multiple filters (all must pass)."""

    def test_all_filters_must_pass(self):
        """Test that all filters in a handler must pass for mutations to apply."""
        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.agent", "chest", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(),
            ),
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 0, "heart": 5}),
            ),
            objects={
                "wall": WallConfig(),
                "chest": ChestConfig(
                    name="chest",
                    activation_handlers=[
                        ActivationHandler(
                            name="vibe_and_resource",
                            filters=[
                                VibeFilter(target="actor", vibe="charger"),
                                ResourceFilter(target="actor", resources={"heart": 10}),  # Requires 10 heart
                            ],
                            mutations=[
                                ResourceDeltaMutation(target="actor", deltas={"energy": 100}),
                            ],
                        ),
                    ],
                ),
            },
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
        sim = Simulation(mg_config, seed=42)

        # Set charger vibe (but only 5 heart, need 10)
        sim.agent(0).set_action("change_vibe_charger")
        sim.step()

        # Move onto chest - should NOT trigger (has charger vibe but not enough heart)
        sim.agent(0).set_action("move_east")
        sim.step()

        inv_after = get_agent_inventory(sim, 0)
        assert inv_after.get("energy", 0) == 0, "Should not trigger - insufficient heart despite correct vibe"
