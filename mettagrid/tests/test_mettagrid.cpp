#include <gtest/gtest.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "actions/attack.hpp"
#include "actions/get_output.hpp"
#include "actions/put_recipe_items.hpp"
#include "mettagrid_c.hpp"
#include "objects/agent.hpp"
#include "objects/constants.hpp"
#include "objects/converter.hpp"

namespace py = pybind11;

class MettaGridTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Python interpreter
    Py_Initialize();
  }

  void TearDown() override {
    // Finalize Python interpreter
    Py_Finalize();
  }

  // Helper function to create test configurations
  py::dict create_test_configs() {
    py::dict agent_cfg;
    agent_cfg["freeze_duration"] = 100;
    py::dict agent_rewards;
    agent_rewards["heart"] = 1.0;
    agent_rewards["ore.red"] = 0.125;  // Pick a power of 2 so floating point precision issues don't matter
    agent_cfg["rewards"] = agent_rewards;

    py::dict group_cfg;
    group_cfg["max_inventory"] = 123;
    py::dict group_rewards;
    group_rewards["ore.red"] = 0.0;    // Should override agent ore.red reward
    group_rewards["ore.green"] = 0.5;  // New reward
    group_cfg["rewards"] = group_rewards;

    py::dict configs;
    configs["agent_cfg"] = agent_cfg;
    configs["group_cfg"] = group_cfg;
    return configs;
  }
};

TEST_F(MettaGridTest, AgentCreation) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  // Test agent creation
  std::unique_ptr<Agent> agent(MettaGrid::create_agent(0, 0, "green", 1, group_cfg, agent_cfg));
  ASSERT_NE(agent, nullptr);

  // Verify merged configuration
  EXPECT_EQ(agent->freeze_duration, 100);  // Group config overrides agent

  // Verify merged rewards
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::heart], 1.0);      // Agent reward preserved
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::ore_red], 0.0);    // Group reward overrides
  EXPECT_FLOAT_EQ(agent->resource_rewards[InventoryItem::ore_green], 0.5);  // Group reward added
}

TEST_F(MettaGridTest, UpdateInventory) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  std::unique_ptr<Agent> agent(MettaGrid::create_agent(0, 0, "green", 1, group_cfg, agent_cfg));
  ASSERT_NE(agent, nullptr);
  float reward = 0;
  agent->init(&reward);

  // Test adding items
  int delta = agent->update_inventory(InventoryItem::heart, 5);

  EXPECT_EQ(delta, 5);
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 5);

  // Test removing items
  delta = agent->update_inventory(InventoryItem::heart, -2);
  EXPECT_EQ(delta, -2);
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 3);

  // Test hitting zero
  delta = agent->update_inventory(InventoryItem::heart, -10);
  EXPECT_EQ(delta, -3);  // Should only remove what's available
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 0);

  // Test hitting max_items limit
  agent->update_inventory(InventoryItem::heart, 50);
  delta = agent->update_inventory(InventoryItem::heart, 200);  // max_items is 123
  EXPECT_EQ(delta, 73);                                        // Should only add up to max_items
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 123);

  // Test multiple items
  delta = agent->update_inventory(InventoryItem::ore_red, 10);
  EXPECT_EQ(delta, 10);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 10);
  EXPECT_EQ(agent->inventory[InventoryItem::heart], 123);  // Other items unchanged
}

TEST_F(MettaGridTest, AttackAction) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  // Create two agents - one attacker and one target
  Agent* attacker = MettaGrid::create_agent(2, 0, "red", 1, group_cfg, agent_cfg);
  Agent* target = MettaGrid::create_agent(0, 0, "blue", 2, group_cfg, agent_cfg);
  ASSERT_NE(attacker, nullptr);
  ASSERT_NE(target, nullptr);
  float attacker_reward = 0;
  attacker->init(&attacker_reward);
  float target_reward = 0;
  target->init(&target_reward);

  // Give attacker a laser
  attacker->update_inventory(InventoryItem::laser, 1);
  EXPECT_EQ(attacker->inventory[InventoryItem::laser], 1);

  // Give target some items to steal
  target->update_inventory(InventoryItem::heart, 2);
  target->update_inventory(InventoryItem::battery, 3);
  EXPECT_EQ(target->inventory[InventoryItem::heart], 2);
  EXPECT_EQ(target->inventory[InventoryItem::battery], 3);

  // Assert the attacker is facing the correct direction
  EXPECT_EQ(attacker->orientation, Orientation::Up);

  // Create a grid and add the agents to it. Then set up an attack action handler and call it.
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }
  Grid grid(10, 10, layer_for_type_id);
  grid.add_object(attacker);
  grid.add_object(target);

  ActionConfig attack_cfg;
  Attack attack(attack_cfg);
  attack.init(&grid);

  // Perform attack (arg 5 targets directly in front)
  bool success = attack.handle_action(attacker->id, 5, 0);
  EXPECT_TRUE(success);

  // Verify laser was consumed
  EXPECT_EQ(attacker->inventory[InventoryItem::laser], 0);

  // Verify target was frozen
  EXPECT_GT(target->frozen, 0);

  // Verify target's inventory was stolen
  EXPECT_EQ(target->inventory[InventoryItem::heart], 0);
  EXPECT_EQ(target->inventory[InventoryItem::battery], 0);
  EXPECT_EQ(attacker->inventory[InventoryItem::heart], 2);
  EXPECT_EQ(attacker->inventory[InventoryItem::battery], 3);

  // grid will delete the agents
}

TEST_F(MettaGridTest, PutRecipeItems) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  Agent* agent = MettaGrid::create_agent(1, 0, "red", 1, group_cfg, agent_cfg);
  ASSERT_NE(agent, nullptr);
  float reward = 0;
  agent->init(&reward);

  // Create a grid and add the agent
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }
  Grid grid(10, 10, layer_for_type_id);
  grid.add_object(agent);

  // Create a generator that takes red ore and outputs batteries
  ObjectConfig generator_cfg;
  generator_cfg["hp"] = 30;
  generator_cfg["input_ore.red"] = 1;
  generator_cfg["output_battery"] = 1;
  // Set the max_output to 0 so it won't consume things we put in it.
  generator_cfg["max_output"] = 0;
  generator_cfg["conversion_ticks"] = 1;
  generator_cfg["cooldown"] = 10;
  generator_cfg["initial_items"] = 0;

  EventManager event_manager;
  Converter* generator = new Converter(0, 0, generator_cfg, ObjectType::GeneratorT);
  grid.add_object(generator);
  generator->set_event_manager(&event_manager);

  // Give agent some items
  agent->update_inventory(InventoryItem::ore_red, 1);
  agent->update_inventory(InventoryItem::ore_blue, 1);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 1);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_blue], 1);

  // Create put_recipe_items action handler
  ActionConfig put_cfg;
  PutRecipeItems put(put_cfg);
  put.init(&grid);

  // Test putting matching items
  bool success = put.handle_action(agent->id, 0, 0);
  EXPECT_TRUE(success);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 0);      // One red ore consumed
  EXPECT_EQ(agent->inventory[InventoryItem::ore_blue], 1);     // Blue ore unchanged
  EXPECT_EQ(generator->inventory[InventoryItem::ore_red], 1);  // One red ore added to generator

  // Test putting non-matching items
  success = put.handle_action(agent->id, 0, 0);
  EXPECT_FALSE(success);                                        // Should fail since we only have blue ore left
  EXPECT_EQ(agent->inventory[InventoryItem::ore_blue], 1);      // Blue ore unchanged
  EXPECT_EQ(generator->inventory[InventoryItem::ore_blue], 0);  // No blue ore in generator

  // grid will delete the agent and generator
}

TEST_F(MettaGridTest, GetOutput) {
  auto configs = create_test_configs();
  auto agent_cfg = configs["agent_cfg"].cast<py::dict>();
  auto group_cfg = configs["group_cfg"].cast<py::dict>();

  Agent* agent = MettaGrid::create_agent(1, 0, "red", 1, group_cfg, agent_cfg);
  ASSERT_NE(agent, nullptr);
  float reward = 0;
  agent->init(&reward);

  // Create a grid and add the agent
  std::vector<Layer> layer_for_type_id;
  for (const auto& layer : ObjectLayers) {
    layer_for_type_id.push_back(layer.second);
  }
  Grid grid(10, 10, layer_for_type_id);
  grid.add_object(agent);

  // Create a generator that takes red ore and outputs batteries
  ObjectConfig generator_cfg;
  generator_cfg["hp"] = 30;
  generator_cfg["input_ore.red"] = 1;
  generator_cfg["output_battery"] = 1;
  // Set the max_output to 0 so it won't consume things we put in it.
  generator_cfg["max_output"] = 1;
  generator_cfg["conversion_ticks"] = 1;
  generator_cfg["cooldown"] = 10;
  generator_cfg["initial_items"] = 1;

  EventManager event_manager;
  Converter* generator = new Converter(0, 0, generator_cfg, ObjectType::GeneratorT);
  grid.add_object(generator);
  generator->set_event_manager(&event_manager);

  // Give agent some items
  agent->update_inventory(InventoryItem::ore_red, 1);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 1);

  // Create put_recipe_items action handler
  ActionConfig get_cfg;
  GetOutput get(get_cfg);
  get.init(&grid);

  // Test getting output
  bool success = get.handle_action(agent->id, 0, 0);
  EXPECT_TRUE(success);
  EXPECT_EQ(agent->inventory[InventoryItem::ore_red], 1);      // Still have red ore
  EXPECT_EQ(agent->inventory[InventoryItem::battery], 1);      // Also have a battery
  EXPECT_EQ(generator->inventory[InventoryItem::battery], 0);  // Generator gave away its battery

  // grid will delete the agent and generator
}
