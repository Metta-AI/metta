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
        obs, info = env.reset(seed=0)

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
        obs, info = env.reset(seed=0)

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
        # Output: 10 * 0.12 = 1.2, deterministic for seed=0 and yields 1
        iron_consumed_12 = iron_before - agent["inventory"][iron_idx]
        steel_produced_12 = agent["inventory"][steel_idx] - steel_before

        assert iron_consumed_12 == 3, (
            f"Should consume 3 iron at 12% progress (20*0.12 rounded up), consumed {iron_consumed_12}"
        )
        assert steel_produced_12 == 1, (
            f"Should produce 1 steel at 12% progress for seed=0, produced {steel_produced_12}"
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

        assert iron_consumed_01 == 1, "Partial usage should consume 1 iron at very low progress"
        assert steel_produced_01 == 0, "Should produce 0 steel at very low progress for seed=0"

    def test_partial_usage_fractional_probabilistic_yields(self):
        """Test probabilistic fractional outputs during partial usage."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "Z", "#"],
                ["#", "#", "#", "#"],
            ]
        )

        cfg.game.resource_names = ["iron", "steel"]
        cfg.game.agent.initial_inventory = {"iron": 100, "steel": 0}

        cfg.game.objects["assembler"] = AssemblerConfig(
            type_id=21,
            name="probabilistic_partial",
            recipes=[
                (
                    ["W"],
                    RecipeConfig(
                        input_resources={"iron": 8},
                        output_resources={"steel": 3},
                        cooldown=10,
                    ),
                )
            ],
            allow_partial_usage=True,
        )

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True

        env = MettaGridCore(cfg)
        env.reset(seed=0)

        move_east_idx = action_index(env, "move", Orientation.EAST)
        noop_idx = env.action_names.index("noop")
        iron_idx = env.resource_names.index("iron")
        steel_idx = env.resource_names.index("steel")

        def get_inventory_value(agent_obj: dict, resource_idx: int) -> int:
            inventory = agent_obj["inventory"]
            if isinstance(inventory, dict):
                return int(inventory.get(resource_idx, 0))
            return int(inventory[resource_idx])

        samples = []
        for seed in range(200):
            env.reset(seed=seed)

            first_use = np.array([move_east_idx], dtype=dtype_actions)
            env.step(first_use)

            noop_action = np.array([noop_idx], dtype=dtype_actions)
            for _ in range(5):
                env.step(noop_action)

            pre_agent = next(obj for _obj_id, obj in env.grid_objects().items() if "agent_id" in obj)
            iron_before = get_inventory_value(pre_agent, iron_idx)
            steel_before = get_inventory_value(pre_agent, steel_idx)

            env.step(first_use)

            post_agent = next(obj for _obj_id, obj in env.grid_objects().items() if "agent_id" in obj)
            iron_after = get_inventory_value(post_agent, iron_idx)
            steel_after = get_inventory_value(post_agent, steel_idx)

            iron_consumed = iron_before - iron_after
            steel_gained = steel_after - steel_before

            # Step() advances the timestep before the assembler runs, so 5 cooldown ticks yields ~60% progress
            # and the ceil-scaled input consumes 5 iron (ceil(8 * 0.6)).
            assert iron_consumed == 5, f"Partial usage should consume 5 iron, consumed {iron_consumed}"
            samples.append(steel_gained)

        assert all(sample in (1, 2) for sample in samples), "Partial usage should only yield 1 or 2 steel"
        assert any(sample == 1 for sample in samples), "Should observe at least one 1-steel yield"
        assert any(sample == 2 for sample in samples), "Should observe at least one 2-steel yield"

        mean_output = float(np.mean(samples))
        assert abs(mean_output - 1.8) < 0.1, f"Mean steel yield {mean_output} should approximate expected value 1.8"
