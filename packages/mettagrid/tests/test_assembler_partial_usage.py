from mettagrid.config.mettagrid_config import AssemblerConfig, MettaGridConfig, ProtocolConfig
from mettagrid.simulator import Action, Simulation


class TestAssemblerPartialUsage:
    """Test assembler partial usage during cooldown functionality."""

    def test_partial_usage_disabled(self):
        """Test that assemblers cannot be used during cooldown when allow_partial_usage is False."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["iron", "steel"]
        cfg.game.agent.inventory.initial = {"iron": 100, "steel": 0}

        # Configure assembler with partial usage disabled
        cfg.game.objects["assembler"] = AssemblerConfig(
            name="assembler",
            protocols=[ProtocolConfig(input_resources={"iron": 10}, output_resources={"steel": 5}, cooldown=10)],
            allow_partial_usage=False,  # Disable partial usage
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "Z", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "Z": "assembler"},
        )

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)
        agent = sim.agent(0)

        iron_idx = sim.resource_names.index("iron")
        steel_idx = sim.resource_names.index("steel")

        # First usage - move east to interact with assembler
        agent.set_action(Action(name="move_east"))
        sim.step()

        # Verify resource change using inventory property
        inventory = agent.inventory
        assert inventory.get("iron", 0) == 90, (
            f"Agent should have 90 iron after first use, got {inventory.get('iron', 0)}"
        )
        assert inventory.get("steel", 0) == 5, (
            f"Agent should have 5 steel after first use, got {inventory.get('steel', 0)}"
        )

        # Also verify via grid_objects (uses integer indices)
        grid_objects = sim.grid_objects()
        agent_obj = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        assert agent_obj["inventory"][iron_idx] == 90, "Agent should have 90 iron after first use"
        assert agent_obj["inventory"][steel_idx] == 5, "Agent should have 5 steel after first use"

        # Wait 5 ticks (50% cooldown)
        for _ in range(5):
            agent.set_action(Action(name="noop"))
            sim.step()

        # Try to use during cooldown (should fail with partial usage disabled)
        agent.set_action(Action(name="move_south"))
        sim.step()

        # Verify resources using inventory property
        inventory = agent.inventory
        assert inventory.get("iron", 0) == 90, (
            f"Iron should remain at 90 (usage blocked), got {inventory.get('iron', 0)}"
        )
        assert inventory.get("steel", 0) == 5, (
            f"Steel should remain at 5 (usage blocked), got {inventory.get('steel', 0)}"
        )

        # Also verify via grid_objects
        grid_objects = sim.grid_objects()
        agent_obj = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        assert agent_obj["inventory"][iron_idx] == 90, (
            f"Iron should remain at 90 (usage blocked), has {agent_obj['inventory'].get(iron_idx, 0)}"
        )
        assert agent_obj["inventory"][steel_idx] == 5, (
            f"Steel should remain at 5 (usage blocked), has {agent_obj['inventory'].get(steel_idx, 0)}"
        )

    def test_partial_usage_scaling(self):
        """Test resource scaling at different cooldown progress levels."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["iron", "steel"]
        cfg.game.agent.inventory.initial = {"iron": 100, "steel": 0}

        # Protocol: 20 iron -> 10 steel, 100 tick cooldown
        cfg.game.objects["assembler"] = AssemblerConfig(
            name="assembler",
            protocols=[ProtocolConfig(input_resources={"iron": 20}, output_resources={"steel": 10}, cooldown=100)],
            allow_partial_usage=True,
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "Z", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "Z": "assembler"},
        )

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)
        agent = sim.agent(0)

        iron_idx = sim.resource_names.index("iron")
        steel_idx = sim.resource_names.index("steel")

        # First full usage - move east to interact with assembler
        agent.set_action(Action(name="move_east"))
        sim.step()

        # Verify using inventory property
        inventory = agent.inventory
        iron_consumed = 100 - inventory.get("iron", 0)
        steel_produced = inventory.get("steel", 0)

        assert iron_consumed == 20, f"Should consume 20 iron at full usage, consumed {iron_consumed}"
        assert steel_produced == 10, f"Should produce 10 steel at full usage, produced {steel_produced}"

        # Also verify via grid_objects
        grid_objects = sim.grid_objects()
        agent_obj = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        assert agent_obj["inventory"][iron_idx] == 80, "Agent should have 80 iron after first use"
        assert agent_obj["inventory"][steel_idx] == 10, "Agent should have 10 steel after first use"

        # Verify assembler is on cooldown (should have 100 tick cooldown)
        assembler = next((obj for _obj_id, obj in grid_objects.items() if obj.get("type_name") == "assembler"), None)
        if assembler:
            cooldown_remaining = assembler.get("cooldown_remaining", 0)
            assert cooldown_remaining == 100, f"Assembler should have 100 tick cooldown, has {cooldown_remaining}"

        # Test at 12% progress (12 ticks into 100 tick cooldown)
        for _ in range(12):
            agent.set_action(Action(name="noop"))
            sim.step()

        # Verify cooldown has decreased to 88 ticks remaining
        grid_objects = sim.grid_objects()
        assembler = next((obj for _obj_id, obj in grid_objects.items() if obj.get("type_name") == "assembler"), None)
        if assembler:
            cooldown_remaining = assembler.get("cooldown_remaining", 0)
            assert cooldown_remaining == 88, f"After 12 ticks, cooldown should be 88, has {cooldown_remaining}"

        # Get inventory before partial usage
        inventory_before = agent.inventory
        iron_before = inventory_before.get("iron", 0)
        steel_before = inventory_before.get("steel", 0)

        # Try to use at 12% cooldown progress
        agent.set_action(Action(name="move_east"))
        sim.step()

        # Get inventory after
        inventory_after = agent.inventory
        iron_after = inventory_after.get("iron", 0)
        steel_after = inventory_after.get("steel", 0)

        # At 12% progress:
        # Input: 20 * 0.12 = 2.4, rounded up = 3
        # Output: 10 * 0.12 = 1.2, rounded down = 1
        iron_consumed_12 = iron_before - iron_after
        steel_produced_12 = steel_after - steel_before

        assert iron_consumed_12 == 3, (
            f"Should consume 3 iron at 12% progress (20*0.12 rounded up), consumed {iron_consumed_12}"
        )
        assert steel_produced_12 == 1, (
            f"Should produce 1 steel at 12% progress (10*0.12 rounded down), produced {steel_produced_12}"
        )

        # Verify that assembler does not trigger when partial usage would yield no output
        iron_before = inventory_after.get("iron", 0)
        steel_before = inventory_after.get("steel", 0)

        # At 1% progress:
        # Input: 20 * 0.01 = 0.2, rounded up = 1
        # Output: 10 * 0.01 = 0.1, rounded down = 0

        agent.set_action(Action(name="move_east"))
        sim.step()

        inventory_final = agent.inventory
        iron_consumed_01 = iron_before - inventory_final.get("iron", 0)
        steel_produced_01 = inventory_final.get("steel", 0) - steel_before

        assert iron_consumed_01 == 0 and steel_produced_01 == 0, (
            "Assembler should not activate when partial-usage output would be zero"
        )
