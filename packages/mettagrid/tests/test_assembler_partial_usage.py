import numpy as np

from mettagrid.config.mettagrid_config import AssemblerConfig, MettaGridConfig, RecipeConfig
from mettagrid.core import MettaGridCore
from mettagrid.mettagrid_c import dtype_actions
from mettagrid.test_support.actions import action_index
from mettagrid.test_support.orientation import Orientation


class TestAssemblerPartialUsage:
    """Test assembler partial usage during cooldown functionality."""

    def test_partial_usage_disabled(self):
        """Test that assemblers cannot be used during cooldown when allow_partial_usage is False."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["iron", "steel"]
        cfg.game.agent.initial_inventory = {"iron": 100, "steel": 0}

        # Configure assembler with partial usage disabled
        cfg.game.objects["assembler"] = AssemblerConfig(
            type_id=20,
            name="assembler",
            map_char="Z",
            recipes=[(["W"], RecipeConfig(input_resources={"iron": 10}, output_resources={"steel": 5}, cooldown=10))],
            allow_partial_usage=False,  # Disable partial usage
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "Z", "#"],
                ["#", "#", "#", "#"],
            ]
        )

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True

        env = MettaGridCore(cfg)
        obs, info = env.reset()

        iron_idx = env.resource_names.index("iron")
        steel_idx = env.resource_names.index("steel")
        noop_idx = env.action_names.index("noop")

        # First usage
        actions = np.array([action_index(env, "move", Orientation.EAST)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        grid_objects = env.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)

        assert agent["inventory"][iron_idx] == 90, "Agent should have 90 iron after first use"
        assert agent["inventory"][steel_idx] == 5, "Agent should have 5 steel after first use"

        # Wait 5 ticks (50% cooldown)
        for _ in range(5):
            actions = np.array([noop_idx], dtype=dtype_actions)
            obs, rewards, terminals, truncations, info = env.step(actions)

        # Try to use during cooldown (should fail)
        actions = np.array([action_index(env, "move", Orientation.SOUTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        grid_objects = env.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)

        # Resources should be unchanged
        assert agent["inventory"][iron_idx] == 90, (
            f"Iron should remain at 90 (usage blocked), has {agent['inventory'].get(iron_idx, 0)}"
        )
        assert agent["inventory"][steel_idx] == 5, (
            f"Steel should remain at 5 (usage blocked), has {agent['inventory'].get(steel_idx, 0)}"
        )

    def test_partial_usage_scaling(self):
        """Test resource scaling at different cooldown progress levels."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["iron", "steel"]
        cfg.game.agent.initial_inventory = {"iron": 100, "steel": 0}

        # Recipe: 20 iron -> 10 steel, 100 tick cooldown
        cfg.game.objects["assembler"] = AssemblerConfig(
            type_id=20,
            name="assembler",
            map_char="Z",
            recipes=[(["W"], RecipeConfig(input_resources={"iron": 20}, output_resources={"steel": 10}, cooldown=100))],
            allow_partial_usage=True,
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "Z", "#"],
                ["#", "#", "#", "#"],
            ]
        )

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True

        env = MettaGridCore(cfg)
        obs, info = env.reset()

        iron_idx = env.resource_names.index("iron")
        steel_idx = env.resource_names.index("steel")
        noop_idx = env.action_names.index("noop")

        # First full usage
        actions = np.array([action_index(env, "move", Orientation.EAST)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        grid_objects = env.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)

        iron_consumed = 100 - agent["inventory"][iron_idx]
        steel_produced = agent["inventory"][steel_idx]

        assert iron_consumed == 20, f"Should consume 20 iron at full usage, consumed {iron_consumed}"
        assert steel_produced == 10, f"Should produce 10 steel at full usage, produced {steel_produced}"

        # Verify assembler is on cooldown (should have 100 tick cooldown)
        assembler = next((obj for _obj_id, obj in grid_objects.items() if obj.get("type_name") == "assembler"), None)
        if assembler:
            cooldown_remaining = assembler.get("cooldown_remaining", 0)
            assert cooldown_remaining == 100, f"Assembler should have 100 tick cooldown, has {cooldown_remaining}"

        # Test at 12% progress (12 ticks into 100 tick cooldown)
        for _ in range(12):
            actions = np.array([noop_idx], dtype=dtype_actions)
            obs, rewards, terminals, truncations, info = env.step(actions)

        # Verify cooldown has decreased to 88 ticks remaining
        grid_objects = env.grid_objects()
        assembler = next((obj for _obj_id, obj in grid_objects.items() if obj.get("type_name") == "assembler"), None)
        if assembler:
            cooldown_remaining = assembler.get("cooldown_remaining", 0)
            assert cooldown_remaining == 88, f"After 12 ticks, cooldown should be 88, has {cooldown_remaining}"

        iron_before = agent["inventory"][iron_idx]
        steel_before = agent["inventory"][steel_idx]

        actions = np.array([action_index(env, "move", Orientation.EAST)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        grid_objects = env.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)

        # At 12% progress:
        # Input: 20 * 0.12 = 2.4, rounded up = 3
        # Output: 10 * 0.12 = 1.2, rounded down = 1
        iron_consumed_12 = iron_before - agent["inventory"][iron_idx]
        steel_produced_12 = agent["inventory"][steel_idx] - steel_before

        assert iron_consumed_12 == 3, (
            f"Should consume 3 iron at 12% progress (20*0.12 rounded up), consumed {iron_consumed_12}"
        )
        assert steel_produced_12 == 1, (
            f"Should produce 1 steel at 12% progress (10*0.12 rounded down), produced {steel_produced_12}"
        )

        # Verify that assembler does not trigger when partial usage would yield no output
        iron_before = agent["inventory"][iron_idx]
        steel_before = agent["inventory"][steel_idx]
        # At 1% progress:
        # Input: 20 * 0.01 = 0.2, rounded up = 1
        # Output: 10 * 0.01 = 0.1, rounded down = 0

        actions = np.array([action_index(env, "move", Orientation.EAST)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        grid_objects = env.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        iron_consumed_01 = iron_before - agent["inventory"][iron_idx]
        steel_produced_01 = agent["inventory"][steel_idx] - steel_before

        assert iron_consumed_01 == 0 and steel_produced_01 == 0, (
            "Assembler should not activate when partial-usage output would be zero"
        )
