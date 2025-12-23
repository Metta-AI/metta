import math

from mettagrid.config.mettagrid_config import (
    InventoryConfig,
    MarketConfig,
    MarketTerminalConfig,
    MettaGridConfig,
    ResourceLimitsConfig,
)
from mettagrid.config.vibes import Vibe
from mettagrid.simulator import Simulation


class TestMarket:
    """Test market buy and sell functionality."""

    def test_market_sell(self):
        """Test that selling to market works correctly."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        # Vibes with names matching resource names are tradeable at the market
        cfg.game.vibe_names = ["gold", "heart"]
        cfg.game.agent.inventory.initial = {"gold": 10, "heart": 0}

        # Market with sell terminal on south side
        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "south": MarketTerminalConfig(sell=True, amount=1),
            },
            inventory=InventoryConfig(
                initial={"gold": 100},  # Market starts with gold stock
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        # Vibes with names matching resource names are tradeable at the market
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")
        heart_idx = sim.resource_names.index("heart")

        # Set vibe to gold (to indicate we're trading gold)
        sim.agent(0).set_action("change_vibe_gold")
        sim.step()

        # Move north into market - should sell 1 gold
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        market = next(obj for _obj_id, obj in grid_objects.items() if obj["type_name"] == "market")

        # Agent sold 1 gold, should have 9 left
        assert agent["inventory"].get(gold_idx, 0) == 9, (
            f"Agent should have 9 gold after selling 1. Has {agent['inventory'].get(gold_idx, 0)}"
        )
        # Market bought 1 gold, should have 101
        assert market["inventory"].get(gold_idx, 0) == 101, (
            f"Market should have 101 gold. Has {market['inventory'].get(gold_idx, 0)}"
        )
        # Agent should have received hearts: price = 100/sqrt(101) ~ 10 (calculated AFTER adding to inventory)
        expected_price = int(round(100 / math.sqrt(101)))  # Price after sale
        assert agent["inventory"].get(heart_idx, 0) == expected_price, (
            f"Agent should have {expected_price} hearts. Has {agent['inventory'].get(heart_idx, 0)}"
        )

    def test_market_buy(self):
        """Test that buying from market works correctly."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        cfg.game.vibe_names = ["gold", "heart"]
        cfg.game.agent.inventory.initial = {"gold": 0, "heart": 100}

        # Market with buy terminal on south side
        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "south": MarketTerminalConfig(sell=False, amount=1),  # Buy terminal
            },
            inventory=InventoryConfig(
                initial={"gold": 100},
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")
        heart_idx = sim.resource_names.index("heart")

        # Set vibe to gold
        sim.agent(0).set_action("change_vibe_gold")
        sim.step()

        # Move north into market - should buy 1 gold
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        market = next(obj for _obj_id, obj in grid_objects.items() if obj["type_name"] == "market")

        # Agent bought 1 gold
        assert agent["inventory"].get(gold_idx, 0) == 1, (
            f"Agent should have 1 gold after buying. Has {agent['inventory'].get(gold_idx, 0)}"
        )
        # Market sold 1 gold
        assert market["inventory"].get(gold_idx, 0) == 99, (
            f"Market should have 99 gold. Has {market['inventory'].get(gold_idx, 0)}"
        )
        # Agent paid hearts: price = 100/sqrt(100) = 10
        expected_price = int(round(100 / math.sqrt(100)))
        expected_hearts = 100 - expected_price
        assert agent["inventory"].get(heart_idx, 0) == expected_hearts, (
            f"Agent should have {expected_hearts} hearts. Has {agent['inventory'].get(heart_idx, 0)}"
        )

    def test_market_multiple_trades(self):
        """Test that terminal amount controls number of trades."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        cfg.game.vibe_names = ["gold", "heart"]
        cfg.game.agent.inventory.initial = {"gold": 10, "heart": 0}

        # Market with sell terminal allowing 3 trades
        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "south": MarketTerminalConfig(sell=True, amount=3),
            },
            inventory=InventoryConfig(
                initial={"gold": 100},
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")
        heart_idx = sim.resource_names.index("heart")

        # Set vibe to gold
        sim.agent(0).set_action("change_vibe_gold")
        sim.step()

        # Move north into market - should sell 3 gold
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        market = next(obj for _obj_id, obj in grid_objects.items() if obj["type_name"] == "market")

        # Agent sold 3 gold
        assert agent["inventory"].get(gold_idx, 0) == 7, (
            f"Agent should have 7 gold after selling 3. Has {agent['inventory'].get(gold_idx, 0)}"
        )
        # Market bought 3 gold
        assert market["inventory"].get(gold_idx, 0) == 103, (
            f"Market should have 103 gold. Has {market['inventory'].get(gold_idx, 0)}"
        )
        # Agent received hearts for each trade at different prices
        assert agent["inventory"].get(heart_idx, 0) > 0, "Agent should have received hearts"

    def test_market_cannot_trade_hearts(self):
        """Test that hearts cannot be traded."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        cfg.game.vibe_names = ["gold", "heart"]
        cfg.game.agent.inventory.initial = {"gold": 10, "heart": 50}

        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "south": MarketTerminalConfig(sell=True, amount=1),
            },
            inventory=InventoryConfig(
                initial={"gold": 100},
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        heart_idx = sim.resource_names.index("heart")

        # Set vibe to heart (trying to sell hearts)
        sim.agent(0).set_action("change_vibe_heart")
        sim.step()

        # Move north into market - should NOT trade
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)

        # Agent should still have 50 hearts (no trade happened)
        assert agent["inventory"].get(heart_idx, 0) == 50, (
            f"Agent should still have 50 hearts. Has {agent['inventory'].get(heart_idx, 0)}"
        )

    def test_market_no_terminal_on_side(self):
        """Test that no trade happens if terminal doesn't exist on that side."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        cfg.game.vibe_names = ["gold", "heart"]
        cfg.game.agent.inventory.initial = {"gold": 10, "heart": 0}

        # Only north terminal, agent comes from south
        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "north": MarketTerminalConfig(sell=True, amount=1),  # Only on north side
            },
            inventory=InventoryConfig(
                initial={"gold": 100},
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")

        # Set vibe to gold
        sim.agent(0).set_action("change_vibe_gold")
        sim.step()

        # Move north into market - should NOT trade (terminal is on north, agent enters from south)
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)

        # Agent should still have 10 gold (no trade happened)
        assert agent["inventory"].get(gold_idx, 0) == 10, (
            f"Agent should still have 10 gold. Has {agent['inventory'].get(gold_idx, 0)}"
        )

    def test_market_price_changes(self):
        """Test that price changes as inventory changes (100/sqrt(inventory))."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        cfg.game.vibe_names = ["gold", "heart"]
        # Note: Inventory is capped at 255 due to SharedInventoryLimit
        cfg.game.agent.inventory.initial = {"gold": 0, "heart": 255}

        # Market with very low initial gold stock
        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "south": MarketTerminalConfig(sell=False, amount=1),  # Buy terminal
            },
            inventory=InventoryConfig(
                initial={"gold": 4},  # Low stock = high price
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")
        heart_idx = sim.resource_names.index("heart")

        # Set vibe to gold
        sim.agent(0).set_action("change_vibe_gold")
        sim.step()

        # Move north into market - should buy 1 gold at high price
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)

        # Price should be 100/sqrt(4) = 50
        expected_price = int(round(100 / math.sqrt(4)))
        expected_hearts = 255 - expected_price  # Agent started with 255 (max inventory)
        assert agent["inventory"].get(gold_idx, 0) == 1, (
            f"Agent should have 1 gold. Has {agent['inventory'].get(gold_idx, 0)}"
        )
        assert agent["inventory"].get(heart_idx, 0) == expected_hearts, (
            f"Agent should have {expected_hearts} hearts (paid {expected_price}). "
            f"Has {agent['inventory'].get(heart_idx, 0)}"
        )

    def test_market_cannot_afford(self):
        """Test that agent cannot buy if they don't have enough hearts."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        cfg.game.vibe_names = ["gold", "heart"]
        cfg.game.agent.inventory.initial = {"gold": 0, "heart": 5}  # Only 5 hearts

        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "south": MarketTerminalConfig(sell=False, amount=1),  # Buy terminal
            },
            inventory=InventoryConfig(
                initial={"gold": 100},  # Price = 100/sqrt(100) = 10
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")
        heart_idx = sim.resource_names.index("heart")

        # Set vibe to gold
        sim.agent(0).set_action("change_vibe_gold")
        sim.step()

        # Move north into market - should NOT buy (can't afford)
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)

        # Agent should still have 0 gold and 5 hearts
        assert agent["inventory"].get(gold_idx, 0) == 0, (
            f"Agent should have 0 gold. Has {agent['inventory'].get(gold_idx, 0)}"
        )
        assert agent["inventory"].get(heart_idx, 0) == 5, (
            f"Agent should still have 5 hearts. Has {agent['inventory'].get(heart_idx, 0)}"
        )

    def test_market_out_of_stock(self):
        """Test that agent cannot buy if market is out of stock."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        cfg.game.vibe_names = ["gold", "heart"]
        # Note: Inventory is capped at 255 due to SharedInventoryLimit
        cfg.game.agent.inventory.initial = {"gold": 0, "heart": 255}

        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "south": MarketTerminalConfig(sell=False, amount=1),  # Buy terminal
            },
            inventory=InventoryConfig(
                initial={"gold": 0},  # No stock!
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")
        heart_idx = sim.resource_names.index("heart")

        # Set vibe to gold
        sim.agent(0).set_action("change_vibe_gold")
        sim.step()

        # Move north into market - should NOT buy (out of stock)
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)

        # Agent should still have 0 gold and 255 hearts (no trade happened)
        assert agent["inventory"].get(gold_idx, 0) == 0, (
            f"Agent should have 0 gold. Has {agent['inventory'].get(gold_idx, 0)}"
        )
        assert agent["inventory"].get(heart_idx, 0) == 255, (
            f"Agent should still have 255 hearts. Has {agent['inventory'].get(heart_idx, 0)}"
        )

    def test_market_buy_fails_when_agent_inventory_full(self):
        """Test that buying fails atomically when agent inventory is full.

        This catches a bug where hearts were taken but resource wasn't given
        because the agent's inventory was full for that resource type.
        The market should check capacity BEFORE taking payment.
        """
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        cfg.game.vibe_names = ["gold", "heart"]
        # Agent has full gold inventory (at limit) but plenty of hearts
        cfg.game.agent.inventory.initial = {"gold": 5, "heart": 100}
        # Gold limit is 5, so agent can't receive more gold
        cfg.game.agent.inventory.limits = {
            "gold": ResourceLimitsConfig(limit=5, resources=["gold"]),
            "heart": ResourceLimitsConfig(limit=255, resources=["heart"]),
        }

        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "south": MarketTerminalConfig(sell=False, amount=1),  # Buy terminal
            },
            inventory=InventoryConfig(
                initial={"gold": 100},  # Market has plenty of stock
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")
        heart_idx = sim.resource_names.index("heart")

        # Set vibe to gold
        sim.agent(0).set_action("change_vibe_gold")
        sim.step()

        # Move north into market - should NOT buy (agent inventory full)
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        market = next(obj for _obj_id, obj in grid_objects.items() if obj["type_name"] == "market")

        # Agent should still have 5 gold (couldn't receive more)
        assert agent["inventory"].get(gold_idx, 0) == 5, (
            f"Agent should still have 5 gold (inventory full). Has {agent['inventory'].get(gold_idx, 0)}"
        )
        # CRITICAL: Agent should NOT have lost hearts (no partial transaction)
        assert agent["inventory"].get(heart_idx, 0) == 100, (
            f"Agent should still have 100 hearts (transaction should fail atomically). "
            f"Has {agent['inventory'].get(heart_idx, 0)}"
        )
        # Market should still have 100 gold (no trade happened)
        assert market["inventory"].get(gold_idx, 0) == 100, (
            f"Market should still have 100 gold. Has {market['inventory'].get(gold_idx, 0)}"
        )

    def test_market_sell_fails_when_agent_heart_inventory_full(self):
        """Test that selling fails atomically when agent can't receive hearts.

        This catches a bug where resource was taken but hearts weren't given
        because the agent's heart inventory was full.
        The market should check heart capacity BEFORE taking the resource.
        """
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        cfg.game.vibe_names = ["gold", "heart"]
        # Agent has gold to sell but heart inventory is full
        cfg.game.agent.inventory.initial = {"gold": 10, "heart": 50}
        cfg.game.agent.inventory.limits = {
            "gold": ResourceLimitsConfig(limit=255, resources=["gold"]),
            # Heart limit is 50, exactly what agent has - can't receive more
            "heart": ResourceLimitsConfig(limit=50, resources=["heart"]),
        }

        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "south": MarketTerminalConfig(sell=True, amount=1),  # Sell terminal
            },
            inventory=InventoryConfig(
                initial={"gold": 100},
                limits={"gold": ResourceLimitsConfig(limit=1000, resources=["gold"])},
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")
        heart_idx = sim.resource_names.index("heart")

        # Set vibe to gold
        sim.agent(0).set_action("change_vibe_gold")
        sim.step()

        # Move north into market - should NOT sell (agent can't receive hearts)
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        market = next(obj for _obj_id, obj in grid_objects.items() if obj["type_name"] == "market")

        # CRITICAL: Agent should NOT have lost gold (no partial transaction)
        assert agent["inventory"].get(gold_idx, 0) == 10, (
            f"Agent should still have 10 gold (transaction should fail atomically). "
            f"Has {agent['inventory'].get(gold_idx, 0)}"
        )
        # Agent should still have 50 hearts (couldn't receive more)
        assert agent["inventory"].get(heart_idx, 0) == 50, (
            f"Agent should still have 50 hearts. Has {agent['inventory'].get(heart_idx, 0)}"
        )
        # Market should still have 100 gold (no trade happened)
        assert market["inventory"].get(gold_idx, 0) == 100, (
            f"Market should still have 100 gold. Has {market['inventory'].get(gold_idx, 0)}"
        )

    def test_market_sell_fails_when_market_inventory_full(self):
        """Test that selling fails atomically when market can't receive resource.

        The market should check its own capacity BEFORE taking the agent's resource.
        """
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold", "heart"]
        cfg.game.vibe_names = ["gold", "heart"]
        cfg.game.agent.inventory.initial = {"gold": 10, "heart": 0}

        cfg.game.objects["market"] = MarketConfig(
            terminals={
                "south": MarketTerminalConfig(sell=True, amount=1),  # Sell terminal
            },
            inventory=InventoryConfig(
                initial={"gold": 100},  # Market is at capacity
                limits={"gold": ResourceLimitsConfig(limit=100, resources=["gold"])},  # Limit = current
            ),
            currency_resource="heart",
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "M", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "M": "market"},
        )

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.vibes = [Vibe("🪙", "gold"), Vibe("❤️", "heart")]
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")
        heart_idx = sim.resource_names.index("heart")

        # Set vibe to gold
        sim.agent(0).set_action("change_vibe_gold")
        sim.step()

        # Move north into market - should NOT sell (market inventory full)
        sim.agent(0).set_action("move_north")
        sim.step()

        # Check results
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        market = next(obj for _obj_id, obj in grid_objects.items() if obj["type_name"] == "market")

        # Agent should still have 10 gold (no trade happened)
        assert agent["inventory"].get(gold_idx, 0) == 10, (
            f"Agent should still have 10 gold. Has {agent['inventory'].get(gold_idx, 0)}"
        )
        # Agent should still have 0 hearts (no payment received)
        assert agent["inventory"].get(heart_idx, 0) == 0, (
            f"Agent should still have 0 hearts. Has {agent['inventory'].get(heart_idx, 0)}"
        )
        # Market should still have 100 gold
        assert market["inventory"].get(gold_idx, 0) == 100, (
            f"Market should still have 100 gold. Has {market['inventory'].get(gold_idx, 0)}"
        )
