import pytest
from mettagrid.config.mettagrid_config import AssemblerConfig, MettaGridConfig, ProtocolConfig
from mettagrid.simulator import Action, Simulation


class TestProtocolSigmoidPricing:
    """Test sigmoid pricing feature for assembler protocols.

    Sigmoid pricing allows:
    - Linear phase (0 to sigmoid uses): cost scales from 0 (free) to 1 (full price)
    - Exponential phase (after sigmoid uses): cost = base * (1+inflation)^(n-sigmoid)
    """

    def test_sigmoid_linear_phase_first_use_is_free(self):
        """Test that first use of a protocol with sigmoid > 0 is free (cost = 0)."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["iron", "steel"]
        cfg.game.agent.initial_inventory = {"iron": 100, "steel": 0}

        # Protocol with sigmoid=5: first 5 uses scale from 0 to 1
        cfg.game.objects["assembler"] = AssemblerConfig(
            name="assembler",
            protocols=[
                ProtocolConfig(
                    input_resources={"iron": 10},
                    output_resources={"steel": 1},
                    cooldown=0,
                    sigmoid=5,
                    inflation=0.0,
                )
            ],
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

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # First use - should be FREE (activation_count=0, multiplier=0/5=0)
        agent.set_action(Action(name="move_east"))
        sim.step()

        inventory = agent.inventory
        # At activation_count=0 with sigmoid=5: multiplier = 0/5 = 0, so cost = ceil(10*0) = 0
        assert inventory.get("iron", 0) == 100, (
            f"First use should be free (cost=0), iron should be 100, got {inventory.get('iron', 0)}"
        )
        assert inventory.get("steel", 0) == 1, f"Should produce steel, got {inventory.get('steel', 0)}"

    def test_sigmoid_linear_phase_scaling(self):
        """Test that costs scale linearly during the sigmoid phase."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["iron", "steel"]
        cfg.game.agent.initial_inventory = {"iron": 100, "steel": 0}

        # Protocol with sigmoid=4: uses 0-3 scale from 0 to 0.75, use 4+ is full price
        cfg.game.objects["assembler"] = AssemblerConfig(
            name="assembler",
            protocols=[
                ProtocolConfig(
                    input_resources={"iron": 10},
                    output_resources={"steel": 1},
                    cooldown=0,
                    sigmoid=4,
                    inflation=0.0,
                )
            ],
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

        expected_costs = [
            # activation_count=0: multiplier = 0/4 = 0.0, cost = ceil(10*0.0) = 0
            0,
            # activation_count=1: multiplier = 1/4 = 0.25, cost = ceil(10*0.25) = 3
            3,
            # activation_count=2: multiplier = 2/4 = 0.5, cost = ceil(10*0.5) = 5
            5,
            # activation_count=3: multiplier = 3/4 = 0.75, cost = ceil(10*0.75) = 8
            8,
            # activation_count=4: multiplier = 1.0 (full price), cost = ceil(10*1.0) = 10
            10,
            # activation_count=5: still full price (no inflation), cost = 10
            10,
        ]

        iron_remaining = 100
        for i, expected_cost in enumerate(expected_costs):
            iron_before = agent.inventory.get("iron", 0)
            agent.set_action(Action(name="move_east"))
            sim.step()
            iron_after = agent.inventory.get("iron", 0)
            actual_cost = iron_before - iron_after

            assert actual_cost == expected_cost, (
                f"Use {i}: expected cost {expected_cost}, got {actual_cost}. "
                f"Iron before: {iron_before}, after: {iron_after}"
            )

            iron_remaining -= expected_cost
            assert iron_after == iron_remaining, f"Use {i}: iron should be {iron_remaining}, got {iron_after}"

    def test_sigmoid_inflation_exponential_phase(self):
        """Test that costs increase exponentially after sigmoid uses when inflation > 0."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["iron", "steel"]
        cfg.game.agent.initial_inventory = {"iron": 1000, "steel": 0}

        # Protocol with sigmoid=2 and 50% inflation
        # After 2 uses, cost increases by 50% each use
        cfg.game.objects["assembler"] = AssemblerConfig(
            name="assembler",
            protocols=[
                ProtocolConfig(
                    input_resources={"iron": 10},
                    output_resources={"steel": 1},
                    cooldown=0,
                    sigmoid=2,
                    inflation=0.5,  # 50% inflation
                )
            ],
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

        sim = Simulation(cfg)
        agent = sim.agent(0)

        expected_costs = [
            # activation_count=0: multiplier = 0/2 = 0.0, cost = 0
            0,
            # activation_count=1: multiplier = 1/2 = 0.5, cost = ceil(10*0.5) = 5
            5,
            # activation_count=2: multiplier = (1+0.5)^0 = 1.0, cost = 10
            10,
            # activation_count=3: multiplier = (1+0.5)^1 = 1.5, cost = ceil(10*1.5) = 15
            15,
            # activation_count=4: multiplier = (1+0.5)^2 = 2.25, cost = ceil(10*2.25) = 23
            23,
            # activation_count=5: multiplier = (1+0.5)^3 = 3.375, cost = ceil(10*3.375) = 34
            34,
        ]

        for i, expected_cost in enumerate(expected_costs):
            iron_before = agent.inventory.get("iron", 0)
            agent.set_action(Action(name="move_east"))
            sim.step()
            iron_after = agent.inventory.get("iron", 0)
            actual_cost = iron_before - iron_after

            assert actual_cost == expected_cost, (
                f"Use {i}: expected cost {expected_cost}, got {actual_cost}. "
                f"Iron before: {iron_before}, after: {iron_after}"
            )

    def test_sigmoid_zero_means_no_discount(self):
        """Test that sigmoid=0 means no discount phase (full price from start)."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["iron", "steel"]
        cfg.game.agent.initial_inventory = {"iron": 100, "steel": 0}

        # Protocol with sigmoid=0: no discount, full price from start
        cfg.game.objects["assembler"] = AssemblerConfig(
            name="assembler",
            protocols=[
                ProtocolConfig(
                    input_resources={"iron": 10},
                    output_resources={"steel": 1},
                    cooldown=0,
                    sigmoid=0,
                    inflation=0.0,
                )
            ],
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

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # All uses should cost full price (10)
        for i in range(5):
            iron_before = agent.inventory.get("iron", 0)
            agent.set_action(Action(name="move_east"))
            sim.step()
            iron_after = agent.inventory.get("iron", 0)
            actual_cost = iron_before - iron_after

            assert actual_cost == 10, f"Use {i}: expected full cost 10, got {actual_cost}"

    def test_sigmoid_with_multiple_resources(self):
        """Test that sigmoid pricing applies to all input resources."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["iron", "copper", "steel"]
        cfg.game.agent.initial_inventory = {"iron": 100, "copper": 100, "steel": 0}

        # Protocol requiring multiple resources with sigmoid=2
        cfg.game.objects["assembler"] = AssemblerConfig(
            name="assembler",
            protocols=[
                ProtocolConfig(
                    input_resources={"iron": 10, "copper": 20},
                    output_resources={"steel": 1},
                    cooldown=0,
                    sigmoid=2,
                    inflation=0.0,
                )
            ],
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

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # Use 0: multiplier = 0/2 = 0, both costs should be 0
        agent.set_action(Action(name="move_east"))
        sim.step()

        inventory = agent.inventory
        assert inventory.get("iron", 0) == 100, f"Use 0: iron should be 100, got {inventory.get('iron', 0)}"
        assert inventory.get("copper", 0) == 100, f"Use 0: copper should be 100, got {inventory.get('copper', 0)}"

        # Use 1: multiplier = 1/2 = 0.5, iron cost = 5, copper cost = 10
        agent.set_action(Action(name="move_east"))
        sim.step()

        inventory = agent.inventory
        assert inventory.get("iron", 0) == 95, f"Use 1: iron should be 95, got {inventory.get('iron', 0)}"
        assert inventory.get("copper", 0) == 90, f"Use 1: copper should be 90, got {inventory.get('copper', 0)}"

        # Use 2: multiplier = 1.0, iron cost = 10, copper cost = 20
        agent.set_action(Action(name="move_east"))
        sim.step()

        inventory = agent.inventory
        assert inventory.get("iron", 0) == 85, f"Use 2: iron should be 85, got {inventory.get('iron', 0)}"
        assert inventory.get("copper", 0) == 70, f"Use 2: copper should be 70, got {inventory.get('copper', 0)}"

    def test_sigmoid_default_values(self):
        """Test that default sigmoid=0 and inflation=0 work correctly (no change in behavior)."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["iron", "steel"]
        cfg.game.agent.initial_inventory = {"iron": 100, "steel": 0}

        # Protocol with default sigmoid and inflation (both 0)
        cfg.game.objects["assembler"] = AssemblerConfig(
            name="assembler",
            protocols=[
                ProtocolConfig(
                    input_resources={"iron": 10},
                    output_resources={"steel": 1},
                    cooldown=0,
                    # sigmoid and inflation use default values (0)
                )
            ],
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

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # Should cost full price (10) each time
        for _ in range(3):
            iron_before = agent.inventory.get("iron", 0)
            agent.set_action(Action(name="move_east"))
            sim.step()
            iron_after = agent.inventory.get("iron", 0)
            assert iron_before - iron_after == 10, "Default config should use full price"


class TestProtocolConfigValidation:
    """Test that ProtocolConfig properly validates sigmoid and inflation fields."""

    def test_sigmoid_non_negative(self):
        """Test that sigmoid must be non-negative."""
        # Valid - should not raise
        ProtocolConfig(sigmoid=0)
        ProtocolConfig(sigmoid=5)
        ProtocolConfig(sigmoid=100)

        # Invalid - should raise
        with pytest.raises(ValueError):
            ProtocolConfig(sigmoid=-1)

    def test_inflation_non_negative(self):
        """Test that inflation must be non-negative."""
        # Valid - should not raise
        ProtocolConfig(inflation=0.0)
        ProtocolConfig(inflation=0.5)
        ProtocolConfig(inflation=1.0)

        # Invalid - should raise
        with pytest.raises(ValueError):
            ProtocolConfig(inflation=-0.1)
